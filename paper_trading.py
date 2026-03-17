# paper_trading.py
# ═══════════════════════════════════════════════════════════════
# PURPOSE: Track paper trades to validate AI signals
#          before going live with real money
#
# HOW IT WORKS:
#   1. AI generates signal
#   2. You record it here as a paper trade
#   3. When market moves, you record the result
#   4. After 50+ trades, you have real evidence
#      whether your AI works in live conditions
#
# PAPER TRADING RULES:
#   - Treat it EXACTLY like real money
#   - Never skip a signal because it looks wrong
#   - Record every trade no matter the result
#   - Do this for minimum 3 months
# ═══════════════════════════════════════════════════════════════

import os
import pandas as pd
from datetime import datetime


JOURNAL_FILE = "data/paper_trades.csv"
os.makedirs("data", exist_ok=True)


# ═══════════════════════════════════════════════════════════════
# PART 1 — RECORD A NEW TRADE
# ═══════════════════════════════════════════════════════════════

def record_trade(
    pair:        str,
    signal:      str,
    entry:       float,
    stop_loss:   float,
    take_profit: float,
    confidence:  float,
    timeframe:   str = "D1"
) -> None:
    """
    Record a new paper trade when signal fires.
    Call this when AI generates a signal.
    """
    trade = {
        'id':           _next_id(),
        'date_opened':  datetime.now().strftime('%Y-%m-%d %H:%M'),
        'date_closed':  '',
        'pair':         pair,
        'signal':       signal,
        'timeframe':    timeframe,
        'entry':        entry,
        'stop_loss':    stop_loss,
        'take_profit':  take_profit,
        'confidence':   confidence,
        'exit_price':   '',
        'outcome':      'OPEN',
        'pnl_pips':     '',
        'pnl_pct':      '',
        'notes':        ''
    }

    df = _load_journal()
    df = pd.concat(
        [df, pd.DataFrame([trade])],
        ignore_index=True
    )
    df.to_csv(JOURNAL_FILE, index=False)

    print(f"\n✅ Paper trade recorded!")
    print(f"   ID        : #{trade['id']}")
    print(f"   Pair      : {pair}")
    print(f"   Signal    : {signal}")
    print(f"   Entry     : {entry}")
    print(f"   Stop Loss : {stop_loss}")
    print(f"   TP        : {take_profit}")
    print(f"   Confidence: {confidence*100:.1f}%")


# ═══════════════════════════════════════════════════════════════
# PART 2 — CLOSE A TRADE WITH RESULT
# ═══════════════════════════════════════════════════════════════

def close_trade(
    trade_id:    int,
    exit_price:  float,
    outcome:     str,
    notes:       str = ""
) -> None:
    """
    Close a paper trade with its result.
    
    outcome: 'WIN' | 'LOSS' | 'TIMEOUT'
    Call this when price hits TP or SL.
    """
    df = _load_journal()

    if trade_id not in df['id'].values:
        print(f"❌ Trade #{trade_id} not found")
        return

    idx = df[df['id'] == trade_id].index[0]
    row = df.loc[idx]

    # Calculate P&L
    entry  = float(row['entry'])
    signal = row['signal']

    if signal == 'BUY':
        pnl_pct = (exit_price - entry) / entry * 100
    else:
        pnl_pct = (entry - exit_price) / entry * 100

    df.loc[idx, 'date_closed'] = datetime.now().strftime(
        '%Y-%m-%d %H:%M'
    )
    df.loc[idx, 'exit_price'] = exit_price
    df.loc[idx, 'outcome']    = outcome
    df.loc[idx, 'pnl_pct']    = round(pnl_pct, 4)
    df.loc[idx, 'notes']      = notes

    df.to_csv(JOURNAL_FILE, index=False)

    icon = "✅" if outcome == 'WIN' else "❌"
    print(f"\n{icon} Trade #{trade_id} closed!")
    print(f"   Outcome   : {outcome}")
    print(f"   Exit      : {exit_price}")
    print(f"   P&L       : {pnl_pct:+.3f}%")


# ═══════════════════════════════════════════════════════════════
# PART 3 — VIEW STATISTICS
# ═══════════════════════════════════════════════════════════════

def show_stats() -> None:
    """Show your paper trading performance statistics."""
    df = _load_journal()

    if df.empty:
        print("No trades recorded yet.")
        return

    closed = df[df['outcome'].isin(['WIN', 'LOSS', 'TIMEOUT'])]
    open_t = df[df['outcome'] == 'OPEN']

    print(f"\n{'═'*50}")
    print(f"  PAPER TRADING STATISTICS")
    print(f"{'═'*50}")
    print(f"  Total Trades : {len(df)}")
    print(f"  Open Trades  : {len(open_t)}")
    print(f"  Closed Trades: {len(closed)}")

    if len(closed) == 0:
        print("  No closed trades yet.")
        return

    wins     = len(closed[closed['outcome'] == 'WIN'])
    losses   = len(closed[closed['outcome'] == 'LOSS'])
    timeouts = len(closed[closed['outcome'] == 'TIMEOUT'])
    win_rate = wins / len(closed) * 100

    print(f"\n  Results:")
    print(f"  ✅ Wins    : {wins}")
    print(f"  ❌ Losses  : {losses}")
    print(f"  ⏱️  Timeouts: {timeouts}")
    print(f"  Win Rate  : {win_rate:.1f}%")

    # P&L stats
    closed_pnl = closed['pnl_pct'].astype(float)
    total_pnl  = closed_pnl.sum()
    avg_win    = closed_pnl[closed['outcome']=='WIN'].mean()
    avg_loss   = closed_pnl[closed['outcome']=='LOSS'].mean()

    print(f"\n  P&L:")
    print(f"  Total      : {total_pnl:+.3f}%")
    print(f"  Avg Win    : {avg_win:+.3f}%")
    print(f"  Avg Loss   : {avg_loss:+.3f}%")

    # Per pair breakdown
    print(f"\n  Per Pair:")
    for pair in closed['pair'].unique():
        pair_trades = closed[closed['pair'] == pair]
        pair_wins   = len(pair_trades[pair_trades['outcome']=='WIN'])
        pair_wr     = pair_wins / len(pair_trades) * 100
        pair_pnl    = pair_trades['pnl_pct'].astype(float).sum()
        print(f"  {pair}: {pair_wins}/{len(pair_trades)} "
              f"wins ({pair_wr:.0f}%) | P&L: {pair_pnl:+.2f}%")

    # Confidence analysis
    print(f"\n  Confidence vs Accuracy:")
    for threshold, label in [
        (0.7, "HIGH (≥70%)"),
        (0.5, "MED (50-70%)"),
        (0.0, "LOW (<50%)")
    ]:
        if threshold == 0.7:
            subset = closed[
                closed['confidence'].astype(float) >= 0.7
            ]
        elif threshold == 0.5:
            subset = closed[
                (closed['confidence'].astype(float) >= 0.5) &
                (closed['confidence'].astype(float) < 0.7)
            ]
        else:
            subset = closed[
                closed['confidence'].astype(float) < 0.5
            ]

        if len(subset) > 0:
            s_wins = len(subset[subset['outcome'] == 'WIN'])
            s_wr   = s_wins / len(subset) * 100
            print(f"  {label}: {s_wins}/{len(subset)} "
                  f"({s_wr:.0f}% win rate)")

    print(f"{'═'*50}")


# ═══════════════════════════════════════════════════════════════
# PART 4 — VIEW OPEN TRADES
# ═══════════════════════════════════════════════════════════════

def show_open_trades() -> None:
    """Show all currently open paper trades."""
    df   = _load_journal()
    open_t = df[df['outcome'] == 'OPEN']

    if open_t.empty:
        print("No open trades.")
        return

    print(f"\n{'─'*55}")
    print(f"  OPEN PAPER TRADES ({len(open_t)})")
    print(f"{'─'*55}")

    for _, row in open_t.iterrows():
        icon = "🟢" if row['signal'] == 'BUY' else "🔴"
        print(f"\n  {icon} #{row['id']} {row['pair']} "
              f"{row['signal']}")
        print(f"     Opened    : {row['date_opened']}")
        print(f"     Entry     : {row['entry']}")
        print(f"     Stop Loss : {row['stop_loss']}")
        print(f"     Take Profit: {row['take_profit']}")
        print(f"     Confidence: "
              f"{float(row['confidence'])*100:.0f}%")


# ═══════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════

def _load_journal() -> pd.DataFrame:
    if os.path.exists(JOURNAL_FILE):
        return pd.read_csv(JOURNAL_FILE)
    return pd.DataFrame(columns=[
        'id', 'date_opened', 'date_closed', 'pair',
        'signal', 'timeframe', 'entry', 'stop_loss',
        'take_profit', 'confidence', 'exit_price',
        'outcome', 'pnl_pips', 'pnl_pct', 'notes'
    ])


def _next_id() -> int:
    df = _load_journal()
    return 1 if df.empty else int(df['id'].max()) + 1


# ═══════════════════════════════════════════════════════════════
# INTERACTIVE MENU
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":

    print("=" * 50)
    print("  PAPER TRADING JOURNAL")
    print("=" * 50)

    while True:
        print("\nOptions:")
        print("  1 → Record today's signals as trades")
        print("  2 → Close a trade with result")
        print("  3 → View open trades")
        print("  4 → View statistics")
        print("  5 → Exit")

        choice = input("\nEnter choice (1-5): ").strip()

        if choice == "1":
            print("\nRecording today's signals...")

            # Today's signals from live_signals output
            signals = [
                {
                    "pair":        "GBPUSD",
                    "signal":      "BUY",
                    "entry":       1.33543,
                    "stop_loss":   1.33237,
                    "take_profit": 1.34155,
                    "confidence":  0.845
                },
                {
                    "pair":        "USDJPY",
                    "signal":      "BUY",
                    "entry":       158.867,
                    "stop_loss":   158.573,
                    "take_profit": 159.455,
                    "confidence":  0.656
                },
            ]

            for s in signals:
                record_trade(**s)

        elif choice == "2":
            show_open_trades()
            trade_id   = int(input("\nEnter trade ID to close: "))
            exit_price = float(input("Exit price: "))
            outcome    = input("Outcome (WIN/LOSS/TIMEOUT): ").upper()
            notes      = input("Notes (optional): ")
            close_trade(trade_id, exit_price, outcome, notes)

        elif choice == "3":
            show_open_trades()

        elif choice == "4":
            show_stats()

        elif choice == "5":
            print("\nGoodbye!")
            break

        else:
            print("Invalid choice")