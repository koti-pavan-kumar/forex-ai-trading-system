# live_signals.py
# ═══════════════════════════════════════════════════════════════
# PURPOSE: Generate live BUY/SELL/HOLD signals using trained AI
#
# HOW IT WORKS:
#   1. Fetch latest candle data
#   2. Calculate all indicators
#   3. Feed indicators into trained AI model
#   4. AI outputs: signal + confidence + entry/SL/TP prices
#   5. Print signal to screen
#   6. Repeat every 60 seconds
#
# REQUIRES: trained models in models/ folder
#           Run train_model.py first if not done
# ═══════════════════════════════════════════════════════════════

import os
import time
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from data_source import get_data, connect_mt5, disconnect_mt5
from indicators import add_all_indicators


# ═══════════════════════════════════════════════════════════════
# PART 1 — LOAD TRAINED MODELS
# ═══════════════════════════════════════════════════════════════

def load_model(pair: str) -> tuple:
    """
    Load the trained AI model for a currency pair.
    
    Returns:
        (model, encoder, feature_names) if found
        None if model file missing
    """
    model_path    = f"models/{pair}_model.pkl"
    encoder_path  = f"models/{pair}_encoder.pkl"
    features_path = f"models/{pair}_features.pkl"

    # Check all files exist
    for path in [model_path, encoder_path, features_path]:
        if not os.path.exists(path):
            print(f"❌ Model file missing: {path}")
            print(f"   Run train_model.py first!")
            return None

    model    = joblib.load(model_path)
    encoder  = joblib.load(encoder_path)
    features = joblib.load(features_path)

    return model, encoder, features


# ═══════════════════════════════════════════════════════════════
# PART 2 — GENERATE SIGNAL FOR ONE PAIR
# ═══════════════════════════════════════════════════════════════

def generate_signal(
    pair: str,
    model,
    encoder,
    feature_names: list,
    timeframe: str = "D1",
    bars: int = 200
) -> dict:
    """
    Generate a trading signal for one currency pair.
    
    Returns a dictionary with:
        signal      : BUY / SELL / HOLD
        confidence  : 0.0 to 1.0 (how sure the AI is)
        entry_price : current market price
        stop_loss   : recommended stop loss price
        take_profit : recommended take profit price
        risk_reward : ratio of reward to risk
        reasoning   : which indicators triggered the signal
    """

    # ── Fetch latest data ─────────────────────────────────────
    df_raw = get_data(pair=pair, timeframe=timeframe, bars=bars)
    if df_raw.empty:
        return {"error": f"Could not fetch data for {pair}"}

    # ── Calculate indicators ──────────────────────────────────
    df = add_all_indicators(df_raw)
    if df.empty or len(df) < 10:
        return {"error": f"Not enough data for {pair}"}

    # ── Get the LATEST candle indicators ──────────────────────
    # This is what the AI will look at to make its decision
    latest = df.iloc[-1]

    # Build feature vector (same features model was trained on)
    feature_vector = {}
    for feature in feature_names:
        if feature in latest.index:
            feature_vector[feature] = latest[feature]
        else:
            feature_vector[feature] = 0.0

    X = pd.DataFrame([feature_vector])

    # ── AI makes prediction ───────────────────────────────────
    # predict_proba gives probability for each class
    # e.g. [0.2, 0.3, 0.5] = 20% SELL, 30% HOLD, 50% BUY
    probabilities  = model.predict_proba(X)[0]
    predicted_class = model.predict(X)[0]

    # Convert encoded class back to -1, 0, 1
    signal_value = encoder.inverse_transform([predicted_class])[0]

    # Map to human readable
    signal_map = {-1: "SELL", 0: "BUY", 1: "BUY"}
    signal = signal_map.get(signal_value, "HOLD")

    # Confidence = probability of the predicted class
    confidence = float(max(probabilities))

    # ── Calculate entry, SL, TP prices ───────────────────────
    current_price = float(latest['close'])
    atr           = float(latest['atr'])

    # ATR-based stop loss and take profit
    # Why ATR: it adapts to current market volatility
    # High volatility = wider stops (avoids being stopped out)
    # Low volatility  = tighter stops (less risk)

    if signal == "BUY":
        entry_price = current_price
        stop_loss   = round(current_price - (1.5 * atr), 5)
        take_profit = round(current_price + (3.0 * atr), 5)

    elif signal == "SELL":
        entry_price = current_price
        stop_loss   = round(current_price + (1.5 * atr), 5)
        take_profit = round(current_price - (3.0 * atr), 5)

    else:  # HOLD
        entry_price = current_price
        stop_loss   = 0.0
        take_profit = 0.0

    # Risk reward ratio
    if signal != "HOLD" and stop_loss != 0:
        risk   = abs(current_price - stop_loss)
        reward = abs(take_profit - current_price)
        risk_reward = round(reward / risk, 2) if risk > 0 else 0
    else:
        risk_reward = 0

    # ── Build reasoning from indicators ───────────────────────
    reasoning = []

    rsi = latest.get('rsi', 50)
    if rsi > 70:
        reasoning.append(f"RSI {rsi:.1f} = Overbought")
    elif rsi < 30:
        reasoning.append(f"RSI {rsi:.1f} = Oversold")
    else:
        reasoning.append(f"RSI {rsi:.1f} = Neutral")

    macd      = latest.get('macd', 0)
    macd_sig  = latest.get('macd_signal', 0)
    if macd > macd_sig:
        reasoning.append("MACD above signal = Bullish")
    else:
        reasoning.append("MACD below signal = Bearish")

    bb_pct = latest.get('bb_percent_b', 0.5)
    if bb_pct > 0.8:
        reasoning.append(f"BB %B {bb_pct:.2f} = Near upper band")
    elif bb_pct < 0.2:
        reasoning.append(f"BB %B {bb_pct:.2f} = Near lower band")

    price  = latest['close']
    ema50  = latest.get('ema50', price)
    if price > ema50:
        reasoning.append("Price above EMA50 = Uptrend")
    else:
        reasoning.append("Price below EMA50 = Downtrend")

    return {
        "pair":         pair,
        "signal":       signal,
        "confidence":   confidence,
        "entry_price":  entry_price,
        "stop_loss":    stop_loss,
        "take_profit":  take_profit,
        "risk_reward":  risk_reward,
        "rsi":          round(float(rsi), 1),
        "reasoning":    reasoning,
        "timestamp":    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "timeframe":    timeframe,
        "data_rows":    len(df)
    }


# ═══════════════════════════════════════════════════════════════
# PART 3 — PRINT SIGNAL IN A NICE FORMAT
# ═══════════════════════════════════════════════════════════════

def print_signal(signal: dict) -> None:
    """Print a trading signal in a clear, readable format."""

    if "error" in signal:
        print(f"❌ Error: {signal['error']}")
        return

    # Signal color indicators
    if signal['signal'] == "BUY":
        icon   = "🟢"
        action = "BUY  ↑"
    elif signal['signal'] == "SELL":
        icon   = "🔴"
        action = "SELL ↓"
    else:
        icon   = "🟡"
        action = "HOLD →"

    # Confidence level
    conf = signal['confidence']
    if conf >= 0.7:
        conf_label = "HIGH"
        conf_icon  = "🔥"
    elif conf >= 0.5:
        conf_label = "MEDIUM"
        conf_icon  = "✅"
    else:
        conf_label = "LOW"
        conf_icon  = "⚠️ "

    print(f"\n{'─' * 50}")
    print(f"  {icon} {signal['pair']} — {action}")
    print(f"{'─' * 50}")
    print(f"  Time       : {signal['timestamp']}")
    print(f"  Timeframe  : {signal['timeframe']}")
    print(f"  Confidence : {conf*100:.1f}% {conf_label} {conf_icon}")
    print(f"  Entry      : {signal['entry_price']:.5f}")

    if signal['signal'] != "HOLD":
        print(f"  Stop Loss  : {signal['stop_loss']:.5f}")
        print(f"  Take Profit: {signal['take_profit']:.5f}")
        print(f"  Risk/Reward: 1 : {signal['risk_reward']}")

    print(f"\n  📊 Indicator Reasoning:")
    for reason in signal['reasoning']:
        print(f"     • {reason}")
    print(f"{'─' * 50}")


# ═══════════════════════════════════════════════════════════════
# PART 4 — SAVE SIGNAL TO FILE
# ═══════════════════════════════════════════════════════════════

def save_signal(signal: dict) -> None:
    """
    Save signal to a CSV file for record keeping.
    Why: You want to track all signals over time
    to measure how accurate your AI is in real trading.
    """
    if "error" in signal:
        return

    os.makedirs('data', exist_ok=True)
    filepath = f"data/{signal['pair']}_signals.csv"

    row = {
        'timestamp':    signal['timestamp'],
        'pair':         signal['pair'],
        'signal':       signal['signal'],
        'confidence':   signal['confidence'],
        'entry_price':  signal['entry_price'],
        'stop_loss':    signal['stop_loss'],
        'take_profit':  signal['take_profit'],
        'risk_reward':  signal['risk_reward'],
        'rsi':          signal['rsi'],
    }

    df_row = pd.DataFrame([row])

    # Append to existing file or create new one
    if os.path.exists(filepath):
        df_row.to_csv(filepath, mode='a', header=False, index=False)
    else:
        df_row.to_csv(filepath, index=False)


# ═══════════════════════════════════════════════════════════════
# PART 5 — RUN ONCE: Generate signals for all pairs
# ═══════════════════════════════════════════════════════════════

def run_once(pairs: list, timeframe: str = "D1") -> list:
    """
    Generate signals for all pairs once.
    Returns list of signal dictionaries.
    """
    all_signals = []

    for pair in pairs:
        # Load model
        result = load_model(pair)
        if result is None:
            continue

        model, encoder, features = result

        # Generate signal
        signal = generate_signal(
            pair=pair,
            model=model,
            encoder=encoder,
            feature_names=features,
            timeframe=timeframe
        )

        # Print and save
        print_signal(signal)
        save_signal(signal)
        all_signals.append(signal)

    return all_signals


# ═══════════════════════════════════════════════════════════════
# PART 6 — RUN LIVE: Repeat every N seconds
# ═══════════════════════════════════════════════════════════════

def run_live(
    pairs: list,
    timeframe: str = "D1",
    interval_seconds: int = 60
) -> None:
    """
    Run signal generation continuously.
    Generates new signals every interval_seconds.
    
    Press Ctrl+C to stop.
    """
    print(f"\n{'═' * 50}")
    print(f"  🚀 LIVE SIGNAL SYSTEM RUNNING")
    print(f"{'═' * 50}")
    print(f"  Pairs     : {', '.join(pairs)}")
    print(f"  Timeframe : {timeframe}")
    print(f"  Refresh   : every {interval_seconds} seconds")
    print(f"  Stop      : Press Ctrl+C")
    print(f"{'═' * 50}")

    cycle = 0

    while True:
        cycle += 1
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        print(f"\n\n{'═' * 50}")
        print(f"  📡 SIGNAL UPDATE #{cycle} — {now}")
        print(f"{'═' * 50}")

        signals = run_once(pairs, timeframe)

        # Summary line
        print(f"\n  📋 SUMMARY:")
        for s in signals:
            if "error" not in s:
                icon = "🟢" if s['signal'] == "BUY" else \
                       "🔴" if s['signal'] == "SELL" else "🟡"
                print(f"     {icon} {s['pair']}: {s['signal']} "
                      f"({s['confidence']*100:.0f}% confidence)")

        print(f"\n  ⏱️  Next update in {interval_seconds} seconds...")
        print(f"  Press Ctrl+C to stop\n")

        time.sleep(interval_seconds)


# ═══════════════════════════════════════════════════════════════
# PART 7 — RUN THIS FILE
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":

    print("=" * 50)
    print("  FOREX AI LIVE SIGNAL GENERATOR")
    print("=" * 50)

    # Connect MT5
    mt5_ok = connect_mt5()

    pairs = ["EURUSD", "GBPUSD", "USDJPY"]

    print("\nChoose mode:")
    print("  1 = Run once (generate signals now)")
    print("  2 = Run live (keep updating every 60 sec)")

    choice = input("\nEnter 1 or 2: ").strip()

    if choice == "1":
        print("\n📊 Generating signals now...\n")
        run_once(pairs, timeframe="H1")

    elif choice == "2":
        run_live(
            pairs=pairs,
            timeframe="H1",
            interval_seconds=60
        )

    else:
        print("Invalid choice. Running once...")
        run_once(pairs, timeframe="H1")

    if mt5_ok:
        disconnect_mt5()

    print("\n✅ Done!")