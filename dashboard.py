# dashboard.py
# ═══════════════════════════════════════════════════════════════
# PURPOSE: Visual trading dashboard showing live AI signals
#
# HOW TO RUN:
#   streamlit run dashboard.py
#
# WHAT IT SHOWS:
#   - Live BUY/SELL/HOLD signals for all pairs
#   - Price charts with indicators
#   - RSI and MACD charts
#   - Signal history table
#   - Auto-refreshes every 60 seconds
# ═══════════════════════════════════════════════════════════════

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import time
import os
import joblib

from data_source import get_data, connect_mt5, disconnect_mt5
from indicators import add_all_indicators
from live_signals import generate_signal, load_model


# ═══════════════════════════════════════════════════════════════
# PAGE CONFIGURATION
# ═══════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Forex AI Signals",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Dark theme styling ────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background-color: #0D1117;
        color: #E6EDF3;
    }

    /* Metric cards */
    div[data-testid="metric-container"] {
        background-color: #161B22;
        border: 1px solid #30363D;
        border-radius: 8px;
        padding: 15px;
    }

    /* BUY signal color */
    .buy-signal {
        color: #00FF88;
        font-size: 28px;
        font-weight: bold;
    }

    /* SELL signal color */
    .sell-signal {
        color: #FF3366;
        font-size: 28px;
        font-weight: bold;
    }

    /* HOLD signal color */
    .hold-signal {
        color: #FFD700;
        font-size: 28px;
        font-weight: bold;
    }

    /* Signal card */
    .signal-card {
        background-color: #161B22;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        border-left: 4px solid #30363D;
    }

    /* Headers */
    h1, h2, h3 {
        color: #E6EDF3;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #161B22;
    }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def get_signal_color(signal: str) -> str:
    """Return color for signal type."""
    colors = {
        "BUY":  "#00FF88",
        "SELL": "#FF3366",
        "HOLD": "#FFD700"
    }
    return colors.get(signal, "#FFFFFF")


def get_signal_icon(signal: str) -> str:
    """Return emoji icon for signal type."""
    icons = {
        "BUY":  "🟢",
        "SELL": "🔴",
        "HOLD": "🟡"
    }
    return icons.get(signal, "⚪")


def build_candlestick_chart(df: pd.DataFrame, pair: str) -> go.Figure:
    """
    Build a professional candlestick chart with:
    - OHLCV candlesticks
    - EMA 9, 21, 50 overlays
    - Bollinger Bands
    - Volume bars
    - MACD subplot
    - RSI subplot
    """
    # Create subplots: price, volume, RSI, MACD
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.55, 0.15, 0.15, 0.15],
        subplot_titles=[
            f"{pair} Price",
            "Volume",
            "RSI (14)",
            "MACD (12,26,9)"
        ]
    )

    # ── Candlestick chart ─────────────────────────────────────
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name=pair,
            increasing_line_color='#00FF88',
            decreasing_line_color='#FF3366',
            increasing_fillcolor='#00FF88',
            decreasing_fillcolor='#FF3366',
        ),
        row=1, col=1
    )

    # ── Bollinger Bands ───────────────────────────────────────
    if 'bb_upper' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df['bb_upper'],
                name='BB Upper',
                line=dict(color='rgba(100,100,255,0.5)', width=1),
                showlegend=False
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df['bb_lower'],
                name='BB Lower',
                line=dict(color='rgba(100,100,255,0.5)', width=1),
                fill='tonexty',
                fillcolor='rgba(100,100,255,0.05)',
                showlegend=False
            ),
            row=1, col=1
        )

    # ── EMA Lines ─────────────────────────────────────────────
    ema_config = [
        ('ema9',  '#FF9500', 'EMA 9'),
        ('ema21', '#FF5E5E', 'EMA 21'),
        ('ema50', '#5E9EFF', 'EMA 50'),
    ]

    for col, color, name in ema_config:
        if col in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index, y=df[col],
                    name=name,
                    line=dict(color=color, width=1.5),
                ),
                row=1, col=1
            )

    # ── Volume bars ───────────────────────────────────────────
    if 'volume' in df.columns:
        colors = [
            '#00FF88' if df['close'].iloc[i] >= df['open'].iloc[i]
            else '#FF3366'
            for i in range(len(df))
        ]
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df['volume'],
                name='Volume',
                marker_color=colors,
                showlegend=False
            ),
            row=2, col=1
        )

    # ── RSI ───────────────────────────────────────────────────
    if 'rsi' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df['rsi'],
                name='RSI',
                line=dict(color='#FF9500', width=2),
            ),
            row=3, col=1
        )
        # Overbought/oversold lines
        fig.add_hline(
            y=70, line_dash="dash",
            line_color="red", opacity=0.5,
            row=3, col=1
        )
        fig.add_hline(
            y=30, line_dash="dash",
            line_color="green", opacity=0.5,
            row=3, col=1
        )
        fig.add_hline(
            y=50, line_dash="dot",
            line_color="gray", opacity=0.3,
            row=3, col=1
        )

    # ── MACD ──────────────────────────────────────────────────
    if 'macd' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df['macd'],
                name='MACD',
                line=dict(color='#5E9EFF', width=2),
            ),
            row=4, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df['macd_signal'],
                name='Signal',
                line=dict(color='#FF9500', width=1.5),
            ),
            row=4, col=1
        )

        # MACD Histogram
        hist_colors = [
            '#00FF88' if v >= 0 else '#FF3366'
            for v in df['macd_hist'].fillna(0)
        ]
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df['macd_hist'],
                name='Histogram',
                marker_color=hist_colors,
                showlegend=False
            ),
            row=4, col=1
        )

    # ── Chart layout ──────────────────────────────────────────
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='#0D1117',
        plot_bgcolor='#0D1117',
        font=dict(color='#E6EDF3', size=11),
        height=700,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor='rgba(0,0,0,0)',
        ),
        xaxis_rangeslider_visible=False,
        margin=dict(l=50, r=50, t=80, b=50),
    )

    # Grid styling
    for i in range(1, 5):
        fig.update_xaxes(
            gridcolor='#21262D',
            showgrid=True,
            row=i, col=1
        )
        fig.update_yaxes(
            gridcolor='#21262D',
            showgrid=True,
            row=i, col=1
        )

    return fig


def load_signal_history(pair: str) -> pd.DataFrame:
    """Load historical signals from CSV file."""
    filepath = f"data/{pair}_signals.csv"
    if os.path.exists(filepath):
        df = pd.read_csv(filepath)
        return df.tail(20)
    return pd.DataFrame()


# ═══════════════════════════════════════════════════════════════
# MAIN DASHBOARD
# ═══════════════════════════════════════════════════════════════

def main():

    # ── Header ────────────────────────────────────────────────
    st.markdown("""
        <h1 style='text-align: center; color: #E6EDF3;'>
            📈 Forex AI Signal Dashboard
        </h1>
        <p style='text-align: center; color: #8B949E;'>
            AI-powered trading signals • Real-time indicators
        </p>
        <hr style='border-color: #30363D;'>
    """, unsafe_allow_html=True)

    # ── Sidebar controls ──────────────────────────────────────
    with st.sidebar:
        st.markdown("## ⚙️ Settings")

        selected_pair = st.selectbox(
            "Currency Pair",
            ["EURUSD", "GBPUSD", "USDJPY"],
            index=0
        )

        timeframe = st.selectbox(
            "Timeframe",
            ["D1", "H1", "M15", "M5"],
            index=0
        )

        bars = st.slider(
            "Candles to show",
            min_value=50,
            max_value=500,
            value=100,
            step=50
        )

        auto_refresh = st.checkbox(
            "Auto refresh (60s)",
            value=False
        )

        st.markdown("---")
        st.markdown("### 📊 Market Status")

        now = datetime.now()
        weekday = now.weekday()

        if weekday >= 5:
            st.error("🔴 Market CLOSED (Weekend)")
        else:
            st.success("🟢 Market OPEN")

        st.markdown(f"**Time:** {now.strftime('%H:%M:%S')}")
        st.markdown(f"**Date:** {now.strftime('%Y-%m-%d')}")

        st.markdown("---")
        st.markdown("### ℹ️ Signal Guide")
        st.markdown("🟢 **BUY** — AI predicts price rise")
        st.markdown("🔴 **SELL** — AI predicts price fall")
        st.markdown("🟡 **HOLD** — No clear direction")
        st.markdown("")
        st.markdown("**Confidence levels:**")
        st.markdown("🔥 HIGH ≥ 70%")
        st.markdown("✅ MEDIUM ≥ 50%")
        st.markdown("⚠️  LOW < 50%")

    # ── Generate signals for ALL pairs ────────────────────────
    pairs = ["EURUSD", "GBPUSD", "USDJPY"]
    all_signals = {}

    with st.spinner("🔄 Fetching live data and generating signals..."):
        for pair in pairs:
            result = load_model(pair)
            if result:
                model, encoder, features = result
                signal = generate_signal(
                    pair=pair,
                    model=model,
                    encoder=encoder,
                    feature_names=features,
                    timeframe=timeframe,
                    bars=bars
                )
                all_signals[pair] = signal

    # ── Signal cards row ──────────────────────────────────────
    st.markdown("## 🎯 Current Signals")

    cols = st.columns(3)

    for idx, pair in enumerate(pairs):
        with cols[idx]:
            if pair in all_signals and "error" not in all_signals[pair]:
                sig = all_signals[pair]
                color = get_signal_color(sig['signal'])
                icon  = get_signal_icon(sig['signal'])

                # Confidence badge
                conf = sig['confidence'] * 100
                if conf >= 70:
                    conf_badge = "🔥 HIGH"
                elif conf >= 50:
                    conf_badge = "✅ MEDIUM"
                else:
                    conf_badge = "⚠️ LOW"

                st.markdown(f"""
                <div style='
                    background-color: #161B22;
                    border-radius: 10px;
                    padding: 20px;
                    border-left: 5px solid {color};
                    margin-bottom: 10px;
                '>
                    <h3 style='color: #E6EDF3; margin:0;'>{pair}</h3>
                    <h2 style='color: {color}; margin:5px 0;'>
                        {icon} {sig['signal']}
                    </h2>
                    <p style='color: #8B949E; margin:0;'>
                        Confidence: <b style='color:{color}'>{conf:.1f}%</b>
                        {conf_badge}
                    </p>
                    <hr style='border-color:#30363D; margin:10px 0;'>
                    <p style='color: #E6EDF3; margin:2px 0; font-size:13px;'>
                        Entry: <b>{sig['entry_price']:.5f}</b>
                    </p>
                </div>
                """, unsafe_allow_html=True)

                if sig['signal'] != "HOLD":
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric(
                            "Stop Loss",
                            f"{sig['stop_loss']:.5f}",
                            delta=f"{sig['stop_loss'] - sig['entry_price']:.5f}"
                        )
                    with col2:
                        st.metric(
                            "Take Profit",
                            f"{sig['take_profit']:.5f}",
                            delta=f"{sig['take_profit'] - sig['entry_price']:.5f}"
                        )

                    st.markdown(
                        f"**Risk/Reward:** 1 : {sig['risk_reward']}"
                    )

            else:
                st.error(f"❌ {pair}: No signal available")

    st.markdown("---")

    # ── Main Chart ────────────────────────────────────────────
    st.markdown(f"## 📊 {selected_pair} Chart")

    with st.spinner(f"Loading {selected_pair} chart..."):
        df_raw = get_data(
            pair=selected_pair,
            timeframe=timeframe,
            bars=bars
        )

        if not df_raw.empty:
            df_chart = add_all_indicators(df_raw)

            if not df_chart.empty:
                fig = build_candlestick_chart(df_chart, selected_pair)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("Could not calculate indicators")
        else:
            st.error("Could not fetch chart data")

    # ── Indicator Details ─────────────────────────────────────
    st.markdown("---")
    st.markdown(f"## 🔬 {selected_pair} Indicator Details")

    if selected_pair in all_signals and "error" not in all_signals[selected_pair]:
        sig = all_signals[selected_pair]

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            rsi_val = sig['rsi']
            rsi_color = (
                "🔴" if rsi_val > 70
                else "🟢" if rsi_val < 30
                else "🟡"
            )
            st.metric("RSI (14)", f"{rsi_val:.1f}", f"{rsi_color}")

        with col2:
            if not df_raw.empty and not df_chart.empty:
                latest = df_chart.iloc[-1]
                st.metric(
                    "MACD",
                    f"{latest['macd']:.6f}",
                    f"Signal: {latest['macd_signal']:.6f}"
                )

        with col3:
            if not df_raw.empty and not df_chart.empty:
                st.metric(
                    "BB %B",
                    f"{latest['bb_percent_b']:.3f}",
                    "Above 1=OB, Below 0=OS"
                )

        with col4:
            if not df_raw.empty and not df_chart.empty:
                st.metric(
                    "ATR",
                    f"{latest['atr']:.5f}",
                    "Volatility measure"
                )

        # Reasoning
        st.markdown("### 💭 AI Reasoning")
        reason_cols = st.columns(len(sig['reasoning']))
        for i, reason in enumerate(sig['reasoning']):
            with reason_cols[i]:
                st.info(reason)

    # ── Signal History Table ──────────────────────────────────
    st.markdown("---")
    st.markdown("## 📋 Signal History")

    history_tabs = st.tabs(["EURUSD", "GBPUSD", "USDJPY"])

    for idx, pair in enumerate(pairs):
        with history_tabs[idx]:
            history = load_signal_history(pair)

            if not history.empty:
                # Color code the signal column
                def highlight_signal(val):
                    if val == "BUY":
                        return "background-color: #0d2818; color: #00FF88"
                    elif val == "SELL":
                        return "background-color: #2d0a14; color: #FF3366"
                    return "background-color: #2d2a00; color: #FFD700"

                styled = history.style.applymap(
                    highlight_signal,
                    subset=['signal']
                )
                st.dataframe(styled, use_container_width=True)
            else:
                st.info(
                    f"No signal history yet for {pair}. "
                    f"Run live_signals.py to generate signals."
                )
    # ── Backtest Results Tab ──────────────────────────────────
    st.markdown("---")
    st.markdown("## 📈 Backtest Performance")

    bt_pairs = ["EURUSD", "GBPUSD", "USDJPY"]
    bt_data  = []

    for pair in bt_pairs:
        summary_path = f"backtest/{pair}_summary.csv"
        if os.path.exists(summary_path):
            df_sum = pd.read_csv(summary_path)
            bt_data.append(df_sum.iloc[0])

    if bt_data:
        # Summary metrics row
        bt_cols = st.columns(3)
        for idx, row in enumerate(bt_data):
            with bt_cols[idx]:
                ret = float(row['total_return'])
                wr  = float(row['win_rate'])
                dd  = float(row['max_drawdown'])
                pf  = float(row['profit_factor'])

                ret_color = "#00FF88" if ret > 0 else "#FF3366"

                st.markdown(f"""
                <div style='
                    background-color: #161B22;
                    border-radius: 10px;
                    padding: 15px;
                    border-left: 4px solid {ret_color};
                '>
                    <h4 style='color:#E6EDF3; margin:0'>
                        {row['pair']} Backtest
                    </h4>
                    <p style='color:{ret_color}; font-size:20px;
                              font-weight:bold; margin:5px 0'>
                        {ret:+.1f}% Return
                    </p>
                    <p style='color:#8B949E; margin:2px 0; font-size:13px'>
                        Win Rate: {wr:.1f}%
                    </p>
                    <p style='color:#8B949E; margin:2px 0; font-size:13px'>
                        Max Drawdown: {dd:.1f}%
                    </p>
                    <p style='color:#8B949E; margin:2px 0; font-size:13px'>
                        Profit Factor: {pf:.2f}
                    </p>
                    <p style='color:#8B949E; margin:2px 0; font-size:13px'>
                        Total Trades: {int(row['total_trades'])}
                    </p>
                </div>
                """, unsafe_allow_html=True)

        # Equity curve for selected pair
        st.markdown(f"### 📉 {selected_pair} Equity Curve")
        trades_path = f"backtest/{selected_pair}_trades.csv"

        if os.path.exists(trades_path):
            df_trades = pd.read_csv(trades_path)
            df_trades['date'] = pd.to_datetime(df_trades['date'])

            # Build equity curve from capital column
            fig_bt = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.1,
                subplot_titles=[
                    f"{selected_pair} Capital Growth",
                    "Trade P&L %"
                ],
                row_heights=[0.65, 0.35]
            )

            # Capital line
            fig_bt.add_trace(
                go.Scatter(
                    x=df_trades['date'],
                    y=df_trades['capital'],
                    name='Capital ($)',
                    line=dict(color='#00FF88', width=2),
                    fill='tozeroy',
                    fillcolor='rgba(0,255,136,0.05)',
                ),
                row=1, col=1
            )

            # Individual trade bars
            bar_colors = [
                '#00FF88' if o == 'WIN'
                else '#FF3366' if o == 'LOSS'
                else '#FFD700'
                for o in df_trades['outcome']
            ]

            fig_bt.add_trace(
                go.Bar(
                    x=df_trades['date'],
                    y=df_trades['pnl_pct'],
                    name='Trade P&L',
                    marker_color=bar_colors,
                ),
                row=2, col=1
            )

            fig_bt.update_layout(
                template='plotly_dark',
                paper_bgcolor='#0D1117',
                plot_bgcolor='#0D1117',
                font=dict(color='#E6EDF3'),
                height=450,
                margin=dict(l=50, r=50, t=60, b=50),
            )
            for i in range(1, 3):
                fig_bt.update_xaxes(
                    gridcolor='#21262D', row=i, col=1)
                fig_bt.update_yaxes(
                    gridcolor='#21262D', row=i, col=1)

            st.plotly_chart(fig_bt, use_container_width=True)

            # Trade breakdown
            col1, col2, col3 = st.columns(3)
            wins     = len(df_trades[df_trades['outcome']=='WIN'])
            losses   = len(df_trades[df_trades['outcome']=='LOSS'])
            timeouts = len(df_trades[df_trades['outcome']=='TIMEOUT'])

            with col1:
                st.metric(
                    "✅ Wins",
                    wins,
                    f"{wins/len(df_trades)*100:.1f}%"
                )
            with col2:
                st.metric(
                    "❌ Losses",
                    losses,
                    f"{losses/len(df_trades)*100:.1f}%"
                )
            with col3:
                st.metric(
                    "⏱️ Timeouts",
                    timeouts,
                    f"{timeouts/len(df_trades)*100:.1f}%"
                )
        else:
            st.info(
                "Run python backtest.py first "
                "to generate backtest data"
            )
    else:
        st.info(
            "No backtest data found. "
            "Run: python backtest.py"
        )
    # ── Footer ────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(f"""
        <p style='text-align: center; color: #8B949E; font-size: 12px;'>
            Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} •
            Data: yfinance (MT5 when connected) •
            ⚠️ For educational purposes only. Not financial advice.
        </p>
    """, unsafe_allow_html=True)

    # ── Auto refresh ──────────────────────────────────────────
    if auto_refresh:
        time.sleep(60)
        st.rerun()


# ── Run the dashboard ─────────────────────────────────────────
if __name__ == "__main__":
    main()