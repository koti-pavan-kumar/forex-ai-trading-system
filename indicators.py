# indicators.py - UPGRADED VERSION
# ═══════════════════════════════════════════════════════════════
# VERSION 2: 50+ features for better AI accuracy
# ═══════════════════════════════════════════════════════════════

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from data_source import get_data, connect_mt5, disconnect_mt5

try:
    import pandas_ta as pta
    PANDAS_TA = True
except ImportError:
    PANDAS_TA = False
    print("⚠️  pandas_ta not installed. Run: pip install pandas-ta")


# ═══════════════════════════════════════════════════════════════
# ORIGINAL INDICATORS (kept from v1)
# ═══════════════════════════════════════════════════════════════

def calculate_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    delta    = df['close'].diff()
    gains    = delta.where(delta > 0, 0)
    losses   = -delta.where(delta < 0, 0)
    avg_gain = gains.ewm(span=period, adjust=False).mean()
    avg_loss = losses.ewm(span=period, adjust=False).mean()
    rs       = avg_gain / avg_loss
    return (100 - (100 / (1 + rs))).rename('rsi')


def calculate_macd(df, fast=12, slow=26, signal=9):
    ema_fast    = df['close'].ewm(span=fast,   adjust=False).mean()
    ema_slow    = df['close'].ewm(span=slow,   adjust=False).mean()
    macd_line   = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram   = macd_line - signal_line
    return pd.DataFrame({
        'macd':       macd_line,
        'macd_signal': signal_line,
        'macd_hist':  histogram
    })


def calculate_bollinger_bands(df, period=20, std_dev=2.0):
    middle    = df['close'].rolling(window=period).mean()
    std       = df['close'].rolling(window=period).std()
    upper     = middle + (std_dev * std)
    lower     = middle - (std_dev * std)
    percent_b = (df['close'] - lower) / (upper - lower)
    bandwidth = (upper - lower) / middle
    return pd.DataFrame({
        'bb_upper':     upper,
        'bb_middle':    middle,
        'bb_lower':     lower,
        'bb_percent_b': percent_b,
        'bb_bandwidth': bandwidth
    })


def calculate_emas(df):
    return pd.DataFrame({
        'ema9':  df['close'].ewm(span=9,  adjust=False).mean(),
        'ema21': df['close'].ewm(span=21, adjust=False).mean(),
        'ema50': df['close'].ewm(span=50, adjust=False).mean(),
    })


def calculate_atr(df, period=14):
    high  = df['high']
    low   = df['low']
    close = df['close']
    tr1   = high - low
    tr2   = abs(high - close.shift(1))
    tr3   = abs(low  - close.shift(1))
    tr    = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean().rename('atr')


def calculate_stochastic(df, k_period=14, d_period=3):
    lowest  = df['low'].rolling(window=k_period).min()
    highest = df['high'].rolling(window=k_period).max()
    k = 100 * (df['close'] - lowest) / (highest - lowest)
    d = k.rolling(window=d_period).mean()
    return pd.DataFrame({'stoch_k': k, 'stoch_d': d})


# ═══════════════════════════════════════════════════════════════
# NEW INDICATORS (added in v2)
# ═══════════════════════════════════════════════════════════════

def calculate_rsi_multi(df: pd.DataFrame) -> pd.DataFrame:
    """
    RSI at multiple timeframes.
    Why: RSI at different speeds gives
    different signals. Short RSI catches
    quick reversals. Long RSI shows
    overall exhaustion.
    """
    return pd.DataFrame({
        'rsi_7':  calculate_rsi(df, 7),
        'rsi_14': calculate_rsi(df, 14),
        'rsi_21': calculate_rsi(df, 21),
    })


def calculate_ema_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    EMA crossover signals and distances.
    Why: The crossover moment is more
    important than the EMA value itself.
    Distance shows how stretched price is.
    """
    ema9  = df['close'].ewm(span=9,   adjust=False).mean()
    ema21 = df['close'].ewm(span=21,  adjust=False).mean()
    ema50 = df['close'].ewm(span=50,  adjust=False).mean()
    ema200= df['close'].ewm(span=200, adjust=False).mean()

    return pd.DataFrame({
        'ema9':          ema9,
        'ema21':         ema21,
        'ema50':         ema50,
        'ema200':        ema200,

        # Crossover signals (1 = bullish cross, -1 = bearish)
        'ema9_21_cross': np.sign(ema9 - ema21),
        'ema21_50_cross': np.sign(ema21 - ema50),

        # Distance from price to each EMA (%)
        'dist_ema9':  (df['close'] - ema9)  / ema9  * 100,
        'dist_ema21': (df['close'] - ema21) / ema21 * 100,
        'dist_ema50': (df['close'] - ema50) / ema50 * 100,

        # Is price above each EMA? (1=yes, 0=no)
        'above_ema50':  (df['close'] > ema50).astype(int),
        'above_ema200': (df['close'] > ema200).astype(int),
    })


def calculate_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Price momentum at multiple lookbacks.
    Why: Momentum tells us HOW FAST
    price is moving and in which direction.
    Combined lookbacks give full picture.
    """
    close = df['close']

    return pd.DataFrame({
        # Returns at different periods
        'return_1':  close.pct_change(1)  * 100,
        'return_3':  close.pct_change(3)  * 100,
        'return_5':  close.pct_change(5)  * 100,
        'return_10': close.pct_change(10) * 100,
        'return_20': close.pct_change(20) * 100,

        # Is momentum accelerating or decelerating?
        'momentum_accel': close.pct_change(1) - close.pct_change(1).shift(1),

        # Rolling highs and lows (breakout detection)
        'high_10': (df['high'] == df['high'].rolling(10).max()).astype(int),
        'low_10':  (df['low']  == df['low'].rolling(10).min()).astype(int),
        'high_20': (df['high'] == df['high'].rolling(20).max()).astype(int),
        'low_20':  (df['low']  == df['low'].rolling(20).min()).astype(int),
    })


def calculate_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Multiple volatility measures.
    Why: Volatility regime tells AI
    whether to expect big or small moves.
    High volatility = wider stops needed.
    Low volatility = potential breakout coming.
    """
    returns = df['close'].pct_change()
    atr     = calculate_atr(df)

    return pd.DataFrame({
        # Historical volatility at different windows
        'vol_5':  returns.rolling(5).std()  * 100,
        'vol_10': returns.rolling(10).std() * 100,
        'vol_20': returns.rolling(20).std() * 100,

        # ATR normalized by price
        'atr_pct': atr / df['close'] * 100,

        # Volatility ratio (is vol expanding or contracting?)
        'vol_ratio': returns.rolling(5).std() /
                     returns.rolling(20).std(),

        # Candle characteristics
        'candle_body':  abs(df['close'] - df['open']),
        'candle_upper': df['high'] - df[['open', 'close']].max(axis=1),
        'candle_lower': df[['open', 'close']].min(axis=1) - df['low'],
        'candle_range': df['high'] - df['low'],

        # Body as % of total range
        'body_ratio': abs(df['close'] - df['open']) /
                      (df['high'] - df['low'] + 1e-10),
    })


def calculate_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Trend strength and direction features.
    Why: Knowing if we are in a strong
    trend vs choppy market changes
    how the AI should interpret signals.
    """
    close  = df['close']
    high   = df['high']
    low    = df['low']

    # ADX (trend strength 0-100)
    # High ADX = strong trend
    # Low ADX  = choppy/sideways
    plus_dm  = high.diff()
    minus_dm = -low.diff()
    plus_dm[plus_dm   < 0] = 0
    minus_dm[minus_dm < 0] = 0

    atr14    = calculate_atr(df, 14)
    plus_di  = 100 * plus_dm.ewm(span=14).mean()  / atr14
    minus_di = 100 * minus_dm.ewm(span=14).mean() / atr14
    dx       = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx      = dx.ewm(span=14).mean()

    # Linear regression slope (trend direction)
    def rolling_slope(series, window=10):
        slopes = []
        for i in range(len(series)):
            if i < window:
                slopes.append(np.nan)
            else:
                y = series.iloc[i-window:i].values
                x = np.arange(window)
                slope = np.polyfit(x, y, 1)[0]
                slopes.append(slope)
        return pd.Series(slopes, index=series.index)

    return pd.DataFrame({
        'adx':      adx,
        'plus_di':  plus_di,
        'minus_di': minus_di,

        # Trend direction score (-1 to +1)
        'trend_score': (plus_di - minus_di) / (plus_di + minus_di + 1e-10),

        # Price position in recent range
        'price_percentile_20': (
            (close - close.rolling(20).min()) /
            (close.rolling(20).max() - close.rolling(20).min() + 1e-10)
        ),
        'price_percentile_50': (
            (close - close.rolling(50).min()) /
            (close.rolling(50).max() - close.rolling(50).min() + 1e-10)
        ),
    })


def calculate_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Time-based features.
    Why: Markets behave differently
    on different days and months.
    Monday often reverses Friday moves.
    December has year-end positioning.
    Cyclical encoding preserves continuity
    (Monday is close to Sunday, not far).
    """
    idx = df.index

    # Day of week (0=Monday, 4=Friday)
    dow = pd.Series(idx.dayofweek, index=idx)

    # Month
    month = pd.Series(idx.month, index=idx)

    # Cyclical encoding using sin/cos
    # Why: Avoids model thinking Friday(4) is
    # far from Monday(0) when they are adjacent
    return pd.DataFrame({
        # Day of week as cyclical
        'dow_sin': np.sin(2 * np.pi * dow / 5),
        'dow_cos': np.cos(2 * np.pi * dow / 5),

        # Month as cyclical
        'month_sin': np.sin(2 * np.pi * month / 12),
        'month_cos': np.cos(2 * np.pi * month / 12),

        # Is it Monday? (often reversal day)
        'is_monday': (dow == 0).astype(int),

        # Is it Friday? (often trend day)
        'is_friday': (dow == 4).astype(int),

        # Quarter
        'quarter': pd.Series(idx.quarter, index=idx),
    })


# ═══════════════════════════════════════════════════════════════
# MAIN FUNCTION — Combine ALL indicators
# ═══════════════════════════════════════════════════════════════

def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    VERSION 2: Now generates 55+ features.
    
    Input:  raw OHLCV DataFrame
    Output: enriched DataFrame ready for AI
    """
    result = df.copy()

    # ── Original indicators ───────────────────────────────────
    result['rsi'] = calculate_rsi(df)
    result['atr'] = calculate_atr(df)

    macd_df  = calculate_macd(df)
    bb_df    = calculate_bollinger_bands(df)
    stoch_df = calculate_stochastic(df)

    result = pd.concat([result, macd_df,  bb_df,
                        stoch_df], axis=1)

    # ── New V2 indicators ─────────────────────────────────────
    rsi_multi_df  = calculate_rsi_multi(df)
    ema_df        = calculate_ema_signals(df)
    momentum_df   = calculate_momentum_features(df)
    volatility_df = calculate_volatility_features(df)
    trend_df      = calculate_trend_features(df)
    time_df       = calculate_time_features(df)

    result = pd.concat([
        result,
        rsi_multi_df,
        ema_df,
        momentum_df,
        volatility_df,
        trend_df,
        time_df,
    ], axis=1)

    # ── Remove duplicate columns ──────────────────────────────
    result = result.loc[:, ~result.columns.duplicated()]

    # ── Drop rows with NaN ────────────────────────────────────
    result = result.dropna()

    return result


# ═══════════════════════════════════════════════════════════════
# TEST
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":

    print("=" * 55)
    print("  INDICATORS V2 TEST")
    print("=" * 55)

    mt5_ok = connect_mt5()

    print("\n📌 Fetching EURUSD data...")
    df_raw = get_data(pair="EURUSD", timeframe="D1", bars=500)

    if df_raw.empty:
        print("❌ Could not fetch data")
        exit()

    print(f"✅ Got {len(df_raw)} raw candles")

    print("\n📌 Calculating V2 indicators...")
    df_ind = add_all_indicators(df_raw)

    print(f"✅ Done!")
    print(f"   Raw columns      : {len(df_raw.columns)}")
    print(f"   Indicator columns: {len(df_ind.columns)}")
    print(f"   Usable rows      : {len(df_ind)}")

    print(f"\n📋 ALL FEATURES ({len(df_ind.columns)} total):")
    print("─" * 45)
    for i, col in enumerate(df_ind.columns, 1):
        print(f"  {i:3}. {col}")

    latest = df_ind.iloc[-1]
    print(f"\n📊 LATEST VALUES:")
    print(f"  RSI 7/14/21  : {latest['rsi_7']:.1f} / "
          f"{latest['rsi_14']:.1f} / {latest['rsi_21']:.1f}")
    print(f"  ADX          : {latest['adx']:.1f}  ", end="")
    print("Strong trend" if latest['adx'] > 25 else "Weak/sideways")
    print(f"  Trend score  : {latest['trend_score']:.3f}  ", end="")
    print("Bullish" if latest['trend_score'] > 0 else "Bearish")
    print(f"  Volatility   : {latest['vol_20']:.4f}%")
    print(f"  EMA9/21 cross: {latest['ema9_21_cross']:.0f}  ", end="")
    print("Bullish" if latest['ema9_21_cross'] > 0 else "Bearish")

    if mt5_ok:
        disconnect_mt5()

    print("\n" + "=" * 55)
    print("  ✅ V2 Indicators ready!")
    print("  ✅ Run train_model.py next")
    print("=" * 55)