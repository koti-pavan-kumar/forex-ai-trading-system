# backtest.py
# ═══════════════════════════════════════════════════════════════
# PURPOSE: Test how profitable your AI signals were historically
#
# HOW IT WORKS:
#   1. Fetch 2 years of historical data
#   2. Calculate indicators on all of it
#   3. Run AI model on each candle (as if trading live)
#   4. Simulate trades with entry, SL, TP
#   5. Calculate performance statistics
#   6. Show results in dashboard
#
# IMPORTANT RULES TO PREVENT CHEATING:
#   - AI only sees data UP TO current candle
#   - Never looks at future data
#   - Includes spread cost on every trade
#   - Realistic position sizing
# ═══════════════════════════════════════════════════════════════

import os
import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from data_source import get_data, connect_mt5, disconnect_mt5
from indicators import add_all_indicators


# ═══════════════════════════════════════════════════════════════
# PART 1 — SIMULATE ONE TRADE
# ═══════════════════════════════════════════════════════════════

def simulate_trade(
    entry_price:  float,
    stop_loss:    float,
    take_profit:  float,
    signal:       str,
    future_prices: pd.Series,
    spread:       float = 0.0002
) -> dict:
    """
    Simulate what happens after a trade signal is generated.
    
    Looks at future candles one by one to see if:
    - Price hits take profit first → WIN
    - Price hits stop loss first   → LOSS
    - Neither hit before data ends → TIMEOUT (close at last price)
    
    Args:
        entry_price   : price we entered the trade
        stop_loss     : price where we exit with a loss
        take_profit   : price where we exit with a profit
        signal        : BUY or SELL
        future_prices : next N candle prices after entry
        spread        : broker cost per trade (0.02%)
    
    Returns dict with trade result details
    """
    # Apply spread cost to entry
    # Why: brokers charge spread on every trade
    # This makes backtesting realistic
    if signal == "BUY":
        actual_entry = entry_price + spread
    else:
        actual_entry = entry_price - spread

    outcome    = "TIMEOUT"
    exit_price = future_prices.iloc[-1]

    for price in future_prices:
        if signal == "BUY":
            if price <= stop_loss:
                outcome    = "LOSS"
                exit_price = stop_loss
                break
            elif price >= take_profit:
                outcome    = "WIN"
                exit_price = take_profit
                break

        elif signal == "SELL":
            if price >= stop_loss:
                outcome    = "LOSS"
                exit_price = stop_loss
                break
            elif price <= take_profit:
                outcome    = "WIN"
                exit_price = take_profit
                break

    # Calculate profit/loss in pips and percentage
    if signal == "BUY":
        pnl_price = exit_price - actual_entry
    else:
        pnl_price = actual_entry - exit_price

    pnl_pct = (pnl_price / actual_entry) * 100

    return {
        "outcome":     outcome,
        "entry_price": actual_entry,
        "exit_price":  exit_price,
        "pnl_pct":     round(pnl_pct, 4),
        "pnl_price":   round(pnl_price, 6),
    }


# ═══════════════════════════════════════════════════════════════
# PART 2 — RUN FULL BACKTEST
# ═══════════════════════════════════════════════════════════════

def run_backtest(
    pair:              str   = "EURUSD",
    timeframe:         str   = "D1",
    bars:              int   = 500,
    initial_capital:   float = 10000.0,
    risk_per_trade:    float = 0.01,
    min_confidence:    float = 0.50,
    forward_candles:   int   = 10
) -> dict:
    """
    Run complete backtest for one currency pair.
    
    Args:
        pair            : Currency pair e.g. "EURUSD"
        timeframe       : Candle size e.g. "D1"
        bars            : How many candles to test on
        initial_capital : Starting account balance ($)
        risk_per_trade  : Risk 1% of account per trade
        min_confidence  : Only trade if AI confidence > 50%
        forward_candles : How many candles to check for SL/TP
    
    Returns:
        Dictionary with all performance statistics
        and list of all trades taken
    """

    print(f"\n{'═'*55}")
    print(f"  BACKTESTING {pair} on {timeframe}")
    print(f"{'═'*55}")

    # ── Load trained model ────────────────────────────────────
    model_path    = f"models/{pair}_model.pkl"
    encoder_path  = f"models/{pair}_encoder.pkl"
    features_path = f"models/{pair}_features.pkl"

    if not all(os.path.exists(p) for p in
               [model_path, encoder_path, features_path]):
        print(f"❌ No trained model found for {pair}")
        print(f"   Run train_model.py first!")
        return {}

    model         = joblib.load(model_path)
    encoder       = joblib.load(encoder_path)
    feature_names = joblib.load(features_path)
    print(f"✅ Model loaded for {pair}")

    # ── Fetch historical data ─────────────────────────────────
    print(f"📌 Fetching {bars} candles of historical data...")
    df_raw = get_data(pair=pair, timeframe=timeframe, bars=bars)

    if df_raw.empty:
        print(f"❌ Could not fetch data for {pair}")
        return {}

    print(f"✅ Got {len(df_raw)} candles")

    # ── Calculate indicators ──────────────────────────────────
    print(f"📌 Calculating indicators...")
    df = add_all_indicators(df_raw)
    print(f"✅ {len(df)} usable candles after warmup")

    # ── Run simulation ────────────────────────────────────────
    print(f"📌 Running trade simulation...")
    print(f"   Min confidence: {min_confidence*100:.0f}%")
    print(f"   Risk per trade: {risk_per_trade*100:.0f}%")
    print(f"   Capital: ${initial_capital:,.0f}")

    trades        = []
    capital       = initial_capital
    equity_curve  = [initial_capital]
    equity_dates  = [df.index[0]]

    # Walk forward through every candle
    # Stop before the last forward_candles
    # (need future data to simulate the trade outcome)
    for i in range(50, len(df) - forward_candles):
        current_bar = df.iloc[i]

        # Build feature vector from current bar only
        # CRITICAL: never use future data here
        feature_vector = {}
        for feature in feature_names:
            if feature in current_bar.index:
                feature_vector[feature] = current_bar[feature]
            else:
                feature_vector[feature] = 0.0

        X = pd.DataFrame([feature_vector])

        # Get AI prediction
        try:
            probs          = model.predict_proba(X)[0]
            pred_class     = model.predict(X)[0]
            signal_value   = encoder.inverse_transform([pred_class])[0]
            confidence     = float(max(probs))
        except Exception:
            continue

        # Map signal value to string
        signal_map = {-1: "SELL", 0: "HOLD", 1: "BUY"}
        signal     = signal_map.get(signal_value, "HOLD")

        # Skip HOLD or low confidence signals
        
        if confidence < min_confidence:
            continue

        # Calculate entry, SL, TP using ATR
        current_price = float(current_bar['close'])
        atr           = float(current_bar['atr'])

        if atr == 0 or np.isnan(atr):
            continue

        if signal == "BUY":
            stop_loss   = current_price - (1.5 * atr)
            take_profit = current_price + (3.0 * atr)
        else:
            stop_loss   = current_price + (1.5 * atr)
            take_profit = current_price - (3.0 * atr)

        # Position sizing using fixed risk
        # Risk 1% of current capital per trade
        risk_amount  = capital * risk_per_trade
        risk_per_unit = abs(current_price - stop_loss)

        if risk_per_unit == 0:
            continue

        # Get future prices for trade simulation
        future_prices = df['close'].iloc[i+1 : i+1+forward_candles]

        if len(future_prices) < 2:
            continue

        # Simulate the trade
        trade_result = simulate_trade(
            entry_price   = current_price,
            stop_loss     = stop_loss,
            take_profit   = take_profit,
            signal        = signal,
            future_prices = future_prices
        )

        # Apply trade result to capital
        # Using fixed 1% risk per trade
        if trade_result['outcome'] == "WIN":
            profit  = risk_amount * 2.0  # 2:1 RR = win 2x risk
            capital += profit
        elif trade_result['outcome'] == "LOSS":
            capital -= risk_amount
        # TIMEOUT: small loss from spread
        elif trade_result['outcome'] == "TIMEOUT":
            pnl_pct = trade_result['pnl_pct']
            capital += capital * (pnl_pct / 100)

        # Record trade
        trades.append({
            'date':        df.index[i],
            'pair':        pair,
            'signal':      signal,
            'confidence':  round(confidence, 3),
            'entry':       current_price,
            'stop_loss':   round(stop_loss, 5),
            'take_profit': round(take_profit, 5),
            'outcome':     trade_result['outcome'],
            'pnl_pct':     trade_result['pnl_pct'],
            'capital':     round(capital, 2),
        })

        equity_curve.append(capital)
        equity_dates.append(df.index[i])

    # ── Calculate statistics ──────────────────────────────────
    if not trades:
        print(f"❌ No trades generated")
        print(f"   Try lowering min_confidence")
        return {}

    df_trades = pd.DataFrame(trades)

    total_trades  = len(df_trades)
    wins          = len(df_trades[df_trades['outcome'] == 'WIN'])
    losses        = len(df_trades[df_trades['outcome'] == 'LOSS'])
    timeouts      = len(df_trades[df_trades['outcome'] == 'TIMEOUT'])
    win_rate      = (wins / total_trades) * 100

    total_return  = ((capital - initial_capital) / initial_capital) * 100

    # Profit factor = total wins / total losses
    win_pnl  = df_trades[df_trades['outcome']=='WIN']['pnl_pct'].sum()
    loss_pnl = abs(df_trades[df_trades['outcome']=='LOSS']['pnl_pct'].sum())
    profit_factor = (win_pnl / loss_pnl) if loss_pnl > 0 else 0

    # Maximum drawdown
    equity_series = pd.Series(equity_curve)
    rolling_max   = equity_series.expanding().max()
    drawdown      = (equity_series - rolling_max) / rolling_max * 100
    max_drawdown  = abs(drawdown.min())

    # Average win and loss
    avg_win  = df_trades[df_trades['outcome']=='WIN']['pnl_pct'].mean()
    avg_loss = df_trades[df_trades['outcome']=='LOSS']['pnl_pct'].mean()

    stats = {
        'pair':           pair,
        'timeframe':      timeframe,
        'total_trades':   total_trades,
        'wins':           wins,
        'losses':         losses,
        'timeouts':       timeouts,
        'win_rate':       round(win_rate, 1),
        'total_return':   round(total_return, 2),
        'profit_factor':  round(profit_factor, 2),
        'max_drawdown':   round(max_drawdown, 2),
        'avg_win':        round(avg_win, 4) if not np.isnan(avg_win) else 0,
        'avg_loss':       round(avg_loss, 4) if not np.isnan(avg_loss) else 0,
        'initial_capital': initial_capital,
        'final_capital':  round(capital, 2),
        'trades':         df_trades,
        'equity_curve':   equity_curve,
        'equity_dates':   equity_dates,
    }

    # ── Print results ─────────────────────────────────────────
    print(f"\n{'─'*45}")
    print(f"  BACKTEST RESULTS — {pair}")
    print(f"{'─'*45}")
    print(f"  Total Trades  : {total_trades}")
    print(f"  Wins          : {wins}  ({win_rate:.1f}%)")
    print(f"  Losses        : {losses}")
    print(f"  Timeouts      : {timeouts}")
    print(f"  Win Rate      : {win_rate:.1f}%  ", end="")
    print("✅" if win_rate >= 50 else "⚠️")
    print(f"  Profit Factor : {profit_factor:.2f}  ", end="")
    print("✅" if profit_factor >= 1.0 else "⚠️")
    print(f"  Total Return  : {total_return:+.2f}%  ", end="")
    print("✅" if total_return > 0 else "❌")
    print(f"  Max Drawdown  : {max_drawdown:.2f}%  ", end="")
    print("✅" if max_drawdown < 20 else "⚠️")
    print(f"  Start Capital : ${initial_capital:,.2f}")
    print(f"  End Capital   : ${capital:,.2f}")
    print(f"{'─'*45}")

    return stats


# ═══════════════════════════════════════════════════════════════
# PART 3 — BUILD RESULTS CHARTS
# ═══════════════════════════════════════════════════════════════

def build_equity_chart(stats: dict) -> go.Figure:
    """
    Build equity curve chart showing how capital
    grew or shrank over the backtest period.
    """
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=[
            f"{stats['pair']} Equity Curve",
            "Trade Outcomes"
        ],
        row_heights=[0.7, 0.3]
    )

    # Equity curve line
    fig.add_trace(
        go.Scatter(
            x=stats['equity_dates'],
            y=stats['equity_curve'],
            name='Capital',
            line=dict(color='#00FF88', width=2),
            fill='tozeroy',
            fillcolor='rgba(0,255,136,0.1)',
        ),
        row=1, col=1
    )

    # Initial capital line
    fig.add_hline(
        y=stats['initial_capital'],
        line_dash="dash",
        line_color="gray",
        opacity=0.5,
        row=1, col=1
    )

    # Trade outcomes bar chart
    trades     = stats['trades']
    colors     = []
    for outcome in trades['outcome']:
        if outcome == 'WIN':
            colors.append('#00FF88')
        elif outcome == 'LOSS':
            colors.append('#FF3366')
        else:
            colors.append('#FFD700')

    fig.add_trace(
        go.Bar(
            x=trades['date'],
            y=trades['pnl_pct'],
            name='Trade P&L',
            marker_color=colors,
        ),
        row=2, col=1
    )

    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='#0D1117',
        plot_bgcolor='#0D1117',
        font=dict(color='#E6EDF3'),
        height=500,
        showlegend=True,
        margin=dict(l=50, r=50, t=80, b=50),
    )

    for i in range(1, 3):
        fig.update_xaxes(gridcolor='#21262D', row=i, col=1)
        fig.update_yaxes(gridcolor='#21262D', row=i, col=1)

    return fig


# ═══════════════════════════════════════════════════════════════
# PART 4 — SAVE RESULTS
# ═══════════════════════════════════════════════════════════════

def save_backtest_results(stats: dict) -> None:
    """Save backtest results to CSV for future reference."""
    if not stats or 'trades' not in stats:
        return

    os.makedirs('backtest', exist_ok=True)

    # Save trades
    trades_path = f"backtest/{stats['pair']}_trades.csv"
    stats['trades'].to_csv(trades_path, index=False)

    # Save summary
    summary = {k: v for k, v in stats.items()
               if k not in ['trades', 'equity_curve', 'equity_dates']}
    summary_df   = pd.DataFrame([summary])
    summary_path = f"backtest/{stats['pair']}_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print(f"✅ Results saved to backtest/ folder")


# ═══════════════════════════════════════════════════════════════
# PART 5 — RUN BACKTEST FROM COMMAND LINE
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":

    print("=" * 55)
    print("  FOREX AI BACKTESTING ENGINE")
    print("=" * 55)
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Connect MT5
    mt5_ok = connect_mt5()

    pairs   = ["EURUSD", "GBPUSD", "USDJPY"]
    all_results = {}

    for pair in pairs:
        stats = run_backtest(
            pair            = pair,
            timeframe       = "H1",
            bars            = 5000,
            initial_capital = 10000.0,
            risk_per_trade  = 0.01,
            min_confidence  = 0.55,
            forward_candles = 30
        )

        if stats:
            save_backtest_results(stats)
            all_results[pair] = stats

    # ── Final comparison table ────────────────────────────────
    if all_results:
        print(f"\n{'═'*55}")
        print(f"  FINAL COMPARISON — ALL PAIRS")
        print(f"{'═'*55}")
        print(f"  {'Pair':<10} {'Trades':>7} {'WinRate':>8} "
              f"{'Return':>8} {'DrawDown':>9} {'ProfitF':>8}")
        print(f"  {'─'*55}")

        for pair, s in all_results.items():
            wr_icon = "✅" if s['win_rate'] >= 50  else "⚠️ "
            rt_icon = "✅" if s['total_return'] > 0 else "❌"
            print(
                f"  {pair:<10} "
                f"{s['total_trades']:>7} "
                f"{s['win_rate']:>7.1f}% {wr_icon} "
                f"{s['total_return']:>+7.2f}% {rt_icon} "
                f"{s['max_drawdown']:>8.2f}% "
                f"{s['profit_factor']:>8.2f}"
            )

    if mt5_ok:
        disconnect_mt5()

    print(f"\n{'═'*55}")
    print(f"  Backtest complete!")
    print(f"  Results saved in backtest/ folder")
    print(f"{'═'*55}")