# main.py
# ═══════════════════════════════════════════════════════════════
# MASTER AUTOMATION — Runs your entire trading system
#
# WHAT IT DOES AUTOMATICALLY:
#   ✅ Connects to MT5
#   ✅ Generates signals every hour
#   ✅ Records paper trades automatically
#   ✅ Retrains model every Monday
#   ✅ Saves all results
#   ✅ Shows live dashboard
#   ✅ Sends desktop alerts when signal fires
#
# HOW TO RUN:
#   python main.py
#   Leave it running — does everything automatically
#   Press Ctrl+C to stop
# ═══════════════════════════════════════════════════════════════

import os
import time
import schedule
import subprocess
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
from data_source import get_data, connect_mt5, disconnect_mt5
from indicators import add_all_indicators
from live_signals import generate_signal, load_model
from paper_trading import record_trade, close_trade, show_stats


# ═══════════════════════════════════════════════════════════════
# CONFIGURATION — Change these to your preferences
# ═══════════════════════════════════════════════════════════════

PAIRS              = ["EURUSD", "GBPUSD", "USDJPY"]
TIMEFRAME          = "H1"
SIGNAL_INTERVAL    = 60        # Check signals every 60 minutes
MIN_CONFIDENCE     = 0.60      # Only trade 60%+ confidence
PRIMARY_PAIR       = "GBPUSD"  # Most reliable model
RETRAIN_DAY        = "monday"  # Retrain every Monday
LOG_FILE           = "data/automation_log.txt"


# ═══════════════════════════════════════════════════════════════
# PART 1 — LOGGING
# ═══════════════════════════════════════════════════════════════

def log(message: str, level: str = "INFO") -> None:
    """Write to log file and print to screen."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line      = f"[{timestamp}] [{level}] {message}"
    print(line)

    os.makedirs("data", exist_ok=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


# ═══════════════════════════════════════════════════════════════
# PART 2 — AUTO CLOSE TRADES THAT HIT SL OR TP
# ═══════════════════════════════════════════════════════════════

def check_and_close_trades() -> None:
    """
    Automatically check if any open paper trades
    have hit their Take Profit or Stop Loss.
    Closes them automatically.
    """
    journal_file = "data/paper_trades.csv"
    if not os.path.exists(journal_file):
        return

    df = pd.read_csv(journal_file)
    open_trades = df[df['outcome'] == 'OPEN']

    if open_trades.empty:
        return

    log(f"Checking {len(open_trades)} open trades...")

    for _, trade in open_trades.iterrows():
        pair       = trade['pair']
        signal     = trade['signal']
        entry      = float(trade['entry'])
        stop_loss  = float(trade['stop_loss'])
        take_profit = float(trade['take_profit'])
        trade_id   = int(trade['id'])

        # Get current price from MT5
        df_price = get_data(pair=pair, timeframe="M1", bars=5)
        if df_price.empty:
            continue

        current_price = float(df_price['close'].iloc[-1])
        high_price    = float(df_price['high'].max())
        low_price     = float(df_price['low'].min())

        # Check if TP or SL was hit
        if signal == "BUY":
            if high_price >= take_profit:
                close_trade(trade_id, take_profit, "WIN",
                           "Auto-closed: TP hit")
                log(f"🏆 {pair} BUY WIN! TP hit at {take_profit}",
                    "WIN")
                _send_alert(f"WIN! {pair} BUY hit Take Profit!")

            elif low_price <= stop_loss:
                close_trade(trade_id, stop_loss, "LOSS",
                           "Auto-closed: SL hit")
                log(f"❌ {pair} BUY LOSS. SL hit at {stop_loss}",
                    "LOSS")
                _send_alert(f"LOSS. {pair} BUY hit Stop Loss.")

        elif signal == "SELL":
            if low_price <= take_profit:
                close_trade(trade_id, take_profit, "WIN",
                           "Auto-closed: TP hit")
                log(f"🏆 {pair} SELL WIN! TP hit at {take_profit}",
                    "WIN")
                _send_alert(f"WIN! {pair} SELL hit Take Profit!")

            elif high_price >= stop_loss:
                close_trade(trade_id, stop_loss, "LOSS",
                           "Auto-closed: SL hit")
                log(f"❌ {pair} SELL LOSS. SL hit at {stop_loss}",
                    "LOSS")
                _send_alert(f"LOSS. {pair} SELL hit Stop Loss.")

        else:
            log(f"  {pair}: Current={current_price:.5f} "
                f"TP={take_profit} SL={stop_loss}")


# ═══════════════════════════════════════════════════════════════
# PART 3 — AUTO GENERATE AND RECORD SIGNALS
# ═══════════════════════════════════════════════════════════════

def generate_and_record_signals() -> None:
    """
    Generate signals for all pairs.
    Automatically records high-confidence signals
    as paper trades.
    """
    now = datetime.now()
    log(f"═══ SIGNAL CHECK #{_get_cycle()} — {now.strftime('%H:%M')} ═══")

    # Skip if market is closed (weekend)
    if now.weekday() >= 5:
        log("Market closed (weekend). Skipping signal check.")
        return

    signals_generated = []

    for pair in PAIRS:
        try:
            result = load_model(pair)
            if result is None:
                log(f"No model for {pair} — run train_model.py",
                    "WARN")
                continue

            model, encoder, features = result

            signal = generate_signal(
                pair          = pair,
                model         = model,
                encoder       = encoder,
                feature_names = features,
                timeframe     = TIMEFRAME,
                bars          = 200
            )

            if "error" in signal:
                log(f"Signal error for {pair}: {signal['error']}",
                    "ERROR")
                continue

            conf     = signal['confidence']
            sig_type = signal['signal']
            entry    = signal['entry_price']

            log(f"  {pair}: {sig_type} | "
                f"Confidence: {conf*100:.1f}% | "
                f"Entry: {entry:.5f}")

            signals_generated.append(signal)

            # Auto-record if confidence is high enough
            # AND it's not already open for this pair
            if conf >= MIN_CONFIDENCE:
                if not _has_open_trade(pair):
                    record_trade(
                        pair        = pair,
                        signal      = sig_type,
                        entry       = entry,
                        stop_loss   = signal['stop_loss'],
                        take_profit = signal['take_profit'],
                        confidence  = conf,
                        timeframe   = TIMEFRAME
                    )
                    log(f"  ✅ Auto-recorded {pair} {sig_type} "
                        f"@ {entry:.5f}", "TRADE")
                    _send_alert(
                        f"New Signal: {pair} {sig_type} "
                        f"@ {entry:.5f} ({conf*100:.0f}% conf)"
                    )
                else:
                    log(f"  ⏭️  {pair} already has open trade — skipping")

        except Exception as e:
            log(f"Error processing {pair}: {e}", "ERROR")

    # Save signal summary
    _save_signal_summary(signals_generated)


# ═══════════════════════════════════════════════════════════════
# PART 4 — AUTO RETRAIN MODEL
# ═══════════════════════════════════════════════════════════════

def auto_retrain() -> None:
    """
    Retrain all models with latest data.
    Runs automatically every Monday.
    """
    log("═══ WEEKLY RETRAIN STARTING ═══", "RETRAIN")

    try:
        result = subprocess.run(
            ["python", "train_model.py"],
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout
        )

        if result.returncode == 0:
            log("✅ Weekly retrain completed successfully",
                "RETRAIN")
            # Extract accuracy from output
            for line in result.stdout.split('\n'):
                if 'walk-forward accuracy' in line:
                    log(f"  {line.strip()}", "RETRAIN")
        else:
            log(f"❌ Retrain failed: {result.stderr[:200]}",
                "ERROR")

    except subprocess.TimeoutExpired:
        log("❌ Retrain timed out after 10 minutes", "ERROR")
    except Exception as e:
        log(f"❌ Retrain error: {e}", "ERROR")


# ═══════════════════════════════════════════════════════════════
# PART 5 — WEEKLY PERFORMANCE REPORT
# ═══════════════════════════════════════════════════════════════

def weekly_report() -> None:
    """Generate and log weekly performance summary."""
    log("═══ WEEKLY PERFORMANCE REPORT ═══", "REPORT")

    journal_file = "data/paper_trades.csv"
    if not os.path.exists(journal_file):
        log("No trades recorded yet.", "REPORT")
        return

    df     = pd.read_csv(journal_file)
    closed = df[df['outcome'].isin(['WIN', 'LOSS', 'TIMEOUT'])]

    if closed.empty:
        log("No closed trades this week yet.", "REPORT")
        return

    # Last 7 days only
    closed['date_opened'] = pd.to_datetime(closed['date_opened'])
    week_ago = datetime.now() - timedelta(days=7)
    this_week = closed[closed['date_opened'] >= week_ago]

    if this_week.empty:
        log("No closed trades in last 7 days.", "REPORT")
        return

    total    = len(this_week)
    wins     = len(this_week[this_week['outcome'] == 'WIN'])
    losses   = len(this_week[this_week['outcome'] == 'LOSS'])
    win_rate = wins / total * 100
    pnl      = this_week['pnl_pct'].astype(float).sum()

    log(f"  Trades   : {total}", "REPORT")
    log(f"  Wins     : {wins}", "REPORT")
    log(f"  Losses   : {losses}", "REPORT")
    log(f"  Win Rate : {win_rate:.1f}%", "REPORT")
    log(f"  Total PnL: {pnl:+.3f}%", "REPORT")

    if win_rate >= 50:
        log("  ✅ GOOD WEEK — model performing well", "REPORT")
    else:
        log("  ⚠️  Tough week — consider retraining", "REPORT")


# ═══════════════════════════════════════════════════════════════
# PART 6 — DESKTOP ALERT
# ═══════════════════════════════════════════════════════════════

def _send_alert(message: str) -> None:
    """
    Send a Windows desktop notification.
    You will see a popup when a signal fires.
    """
    try:
        # Windows toast notification
        subprocess.Popen([
            "powershell", "-Command",
            f"""
            Add-Type -AssemblyName System.Windows.Forms
            $notify = New-Object System.Windows.Forms.NotifyIcon
            $notify.Icon = [System.Drawing.SystemIcons]::Information
            $notify.Visible = $true
            $notify.ShowBalloonTip(
                5000,
                'Forex AI Signal',
                '{message}',
                [System.Windows.Forms.ToolTipIcon]::Info
            )
            Start-Sleep -Seconds 6
            $notify.Dispose()
            """
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except:
        pass  # Alerts are optional — don't crash if they fail


# ═══════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════

def _has_open_trade(pair: str) -> bool:
    """Check if there is already an open paper trade for this pair."""
    journal_file = "data/paper_trades.csv"
    if not os.path.exists(journal_file):
        return False
    df = pd.read_csv(journal_file)
    open_trades = df[(df['pair'] == pair) &
                     (df['outcome'] == 'OPEN')]
    return len(open_trades) > 0


_cycle_count = 0
def _get_cycle() -> int:
    global _cycle_count
    _cycle_count += 1
    return _cycle_count


def _save_signal_summary(signals: list) -> None:
    """Save signal summary to CSV."""
    if not signals:
        return
    os.makedirs("data", exist_ok=True)
    rows = []
    for s in signals:
        if "error" not in s:
            rows.append({
                'timestamp':  s['timestamp'],
                'pair':       s['pair'],
                'signal':     s['signal'],
                'confidence': s['confidence'],
                'entry':      s['entry_price'],
                'stop_loss':  s['stop_loss'],
                'take_profit': s['take_profit'],
            })
    if rows:
        df = pd.DataFrame(rows)
        path = "data/all_signals.csv"
        df.to_csv(
            path,
            mode='a',
            header=not os.path.exists(path),
            index=False
        )


# ═══════════════════════════════════════════════════════════════
# PART 7 — SCHEDULE ALL TASKS
# ═══════════════════════════════════════════════════════════════

def setup_schedule() -> None:
    """
    Set up all automated tasks.

    SCHEDULE:
    → Every 60 minutes: check signals + close trades
    → Every Monday 9:00 AM: retrain model
    → Every Friday 6:00 PM: weekly report
    """
    # Signals every hour
    schedule.every(SIGNAL_INTERVAL).minutes.do(
        generate_and_record_signals
    )

    # Check open trades every 15 minutes
    schedule.every(15).minutes.do(check_and_close_trades)

    # Weekly retrain every Monday at 9 AM
    schedule.every().monday.at("09:00").do(auto_retrain)

    # Weekly report every Friday at 6 PM
    schedule.every().friday.at("18:00").do(weekly_report)

    log("Schedule configured:")
    log(f"  Signals    : every {SIGNAL_INTERVAL} minutes")
    log(f"  Trade check: every 15 minutes")
    log(f"  Retrain    : every Monday 9:00 AM")
    log(f"  Report     : every Friday 6:00 PM")


# ═══════════════════════════════════════════════════════════════
# MAIN — RUN EVERYTHING
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # Install schedule if needed
    try:
        import schedule
    except ImportError:
        os.system("pip install schedule")
        import schedule

    print("=" * 55)
    print("  FOREX AI — MASTER AUTOMATION")
    print("=" * 55)
    print(f"  Started : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Pairs   : {', '.join(PAIRS)}")
    print(f"  TF      : {TIMEFRAME}")
    print(f"  Min Conf: {MIN_CONFIDENCE*100:.0f}%")
    print(f"  Log     : {LOG_FILE}")
    print("=" * 55)
    print("\nPress Ctrl+C to stop\n")

    # Connect MT5
    mt5_ok = connect_mt5()
    if not mt5_ok:
        log("MT5 not connected — using yfinance fallback", "WARN")

    # Run immediately on startup
    log("Running initial signal check...")
    generate_and_record_signals()
    check_and_close_trades()

    # Set up recurring schedule
    setup_schedule()

    log("Automation running. All tasks scheduled.")
    log(f"Dashboard: run 'streamlit run dashboard.py' in new terminal")

    # Keep running forever
    try:
        while True:
            schedule.run_pending()
            time.sleep(30)  # Check schedule every 30 seconds

    except KeyboardInterrupt:
        log("Automation stopped by user.")
        if mt5_ok:
            disconnect_mt5()
        print("\n✅ Stopped cleanly.")