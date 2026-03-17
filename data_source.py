import os
import json
import urllib.request
import pandas as pd
import numpy as np
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# ── Try importing MT5 ─────────────────────────────────────────
try:
    import MetaTrader5 as mt5
    MT5_INSTALLED = True
except ImportError:
    MT5_INSTALLED = False
    print("⚠️  MT5 package not found. Will use Alpha Vantage/yfinance.")

import yfinance as yf


# ═══════════════════════════════════════════════════════════════
# PART 1 — MT5 CONNECTION
# ═══════════════════════════════════════════════════════════════

def connect_mt5() -> bool:
    if not MT5_INSTALLED:
        print("❌ MT5 not installed")
        return False

    login    = int(os.getenv("MT5_LOGIN", 0))
    password = os.getenv("MT5_PASSWORD", "")
    server   = os.getenv("MT5_SERVER", "")

    if login == 0 or not password or not server:
        print("❌ MT5 credentials missing in .env file")
        return False

    MT5_PATH = r"C:\Program Files\MetaTrader 5 IC Markets Global\terminal64.exe"

    success = mt5.initialize(
        path=MT5_PATH,
        login=login,
        password=password,
        server=server
    )

    if not success:
        error = mt5.last_error()
        print(f"❌ MT5 connection failed: {error}")
        print("   Is MT5 terminal open? Are you logged in?")
        mt5.shutdown()
        return False

    account = mt5.account_info()
    print(f"✅ MT5 Connected!")
    print(f"   Account : {account.login}")
    print(f"   Server  : {account.server}")
    print(f"   Balance : ${account.balance:,.2f}")
    return True


def disconnect_mt5():
    if MT5_INSTALLED:
        mt5.shutdown()
        print("🔌 MT5 disconnected cleanly")


# ═══════════════════════════════════════════════════════════════
# PART 2 — TIMEFRAME MAPPINGS
# ═══════════════════════════════════════════════════════════════

def get_mt5_timeframe(timeframe: str):
    if not MT5_INSTALLED:
        return None
    mapping = {
        "M1":  mt5.TIMEFRAME_M1,
        "M5":  mt5.TIMEFRAME_M5,
        "M15": mt5.TIMEFRAME_M15,
        "M30": mt5.TIMEFRAME_M30,
        "H1":  mt5.TIMEFRAME_H1,
        "H4":  mt5.TIMEFRAME_H4,
        "D1":  mt5.TIMEFRAME_D1,
    }
    return mapping.get(timeframe)


def get_yf_interval(timeframe: str) -> str:
    mapping = {
        "M1":  "1m",
        "M5":  "5m",
        "M15": "15m",
        "M30": "30m",
        "H1":  "1h",
        "H4":  "1h",
        "D1":  "1d",
    }
    return mapping.get(timeframe, "1m")


def get_yf_period(timeframe: str) -> str:
    mapping = {
        "M1":  "7d",
        "M5":  "60d",
        "M15": "60d",
        "M30": "60d",
        "H1":  "730d",
        "H4":  "730d",
        "D1":  "5y",
    }
    return mapping.get(timeframe, "7d")


YF_SYMBOLS = {
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "USDJPY": "USDJPY=X",
    "AUDUSD": "AUDUSD=X",
    "USDCAD": "USDCAD=X",
    "USDCHF": "USDCHF=X",
}


# ═══════════════════════════════════════════════════════════════
# PART 3 — FETCH FROM MT5
# ═══════════════════════════════════════════════════════════════

def fetch_from_mt5(pair: str, timeframe: str = "M1", bars: int = 500) -> pd.DataFrame:
    if not MT5_INSTALLED:
        return pd.DataFrame()

    tf = get_mt5_timeframe(timeframe)
    if tf is None:
        return pd.DataFrame()

    rates = mt5.copy_rates_from_pos(pair, tf, 0, bars)

    if rates is None or len(rates) == 0:
        error = mt5.last_error()
        print(f"❌ MT5 returned no data for {pair}: {error}")
        print(f"   Is {pair} visible in MT5 Market Watch?")
        return pd.DataFrame()

    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df = df.rename(columns={'time': 'timestamp'})
    df = df.set_index('timestamp')
    df.index = df.index.tz_localize(None)
    df = df.rename(columns={'tick_volume': 'volume'})
    df = df[['open', 'high', 'low', 'close', 'volume', 'spread']]

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    return df


# ═══════════════════════════════════════════════════════════════
# PART 4 — FETCH FROM ALPHA VANTAGE
# ═══════════════════════════════════════════════════════════════

def fetch_alpha_vantage(pair: str, timeframe: str = "H1", bars: int = 500) -> pd.DataFrame:
    """
    Alpha Vantage free API.
    Real OHLCV data with proper high/low values.
    Free tier: 25 requests per day.
    Get key at: https://www.alphavantage.co/support/P5957K15K9OUNMXM
    """
    from_currency = pair[:3]
    to_currency   = pair[3:]

    interval_map = {
        "M1":  "1min",
        "M5":  "5min",
        "M15": "15min",
        "H1":  "60min",
    }
    interval = interval_map.get(timeframe, "60min")
    api_key  = os.getenv("ALPHA_VANTAGE_KEY", "")

    if not api_key:
        return pd.DataFrame()

    url = (
        f"https://www.alphavantage.co/query?"
        f"function=FX_INTRADAY"
        f"&from_symbol={from_currency}"
        f"&to_symbol={to_currency}"
        f"&interval={interval}"
        f"&outputsize=full"
        f"&apikey={api_key}"
    )

    try:
        print(f"   Trying Alpha Vantage for {pair}...")
        with urllib.request.urlopen(url, timeout=10) as response:
            data = json.loads(response.read().decode())

        if "Error Message" in data:
            print(f"❌ Alpha Vantage error: {data['Error Message']}")
            return pd.DataFrame()

        if "Note" in data:
            print(f"⚠️  Alpha Vantage rate limit. Wait 1 minute.")
            return pd.DataFrame()

        ts_key = [k for k in data.keys() if "Time Series" in k]
        if not ts_key:
            print(f"❌ Alpha Vantage: no data found")
            return pd.DataFrame()

        ts_data = data[ts_key[0]]

        rows = []
        for timestamp, values in ts_data.items():
            rows.append({
                'timestamp': pd.to_datetime(timestamp),
                'open':   float(values['1. open']),
                'high':   float(values['2. high']),
                'low':    float(values['3. low']),
                'close':  float(values['4. close']),
                'volume': 0.0,
                'spread': 0.0
            })

        df = pd.DataFrame(rows)
        df = df.set_index('timestamp')
        df = df.sort_index()
        df.index = df.index.tz_localize(None)

        return df.tail(bars)

    except Exception as e:
        print(f"❌ Alpha Vantage error: {e}")
        return pd.DataFrame()


# ═══════════════════════════════════════════════════════════════
# PART 5 — FETCH FROM YFINANCE (LAST RESORT)
# ═══════════════════════════════════════════════════════════════

def fetch_from_yfinance(pair: str, timeframe: str = "M1", bars: int = 500) -> pd.DataFrame:
    symbol = YF_SYMBOLS.get(pair)
    if not symbol:
        print(f"❌ {pair} not in yfinance symbol list")
        return pd.DataFrame()

    interval = get_yf_interval(timeframe)
    period   = get_yf_period(timeframe)

    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)

        if df.empty:
            print(f"❌ yfinance returned empty data for {pair}")
            return pd.DataFrame()

        df.columns = df.columns.str.lower()
        df.index.name = 'timestamp'

        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        df['spread'] = 0.0
        df = df[['open', 'high', 'low', 'close', 'volume', 'spread']]

        return df.tail(bars)

    except Exception as e:
        print(f"❌ yfinance error for {pair}: {e}")
        return pd.DataFrame()


# ═══════════════════════════════════════════════════════════════
# PART 6 — MAIN FUNCTION (USE THIS EVERYWHERE)
# ═══════════════════════════════════════════════════════════════

def get_data(
    pair:      str = "EURUSD",
    timeframe: str = "H1",
    bars:      int = 500
) -> pd.DataFrame:
    """
    Call this function from ALL other files.
    Tries MT5 → Alpha Vantage → yfinance automatically.
    Returns identical DataFrame format regardless of source.
    """

    # ── Priority 1: MT5 ──────────────────────────────────────
    if MT5_INSTALLED:
        df = fetch_from_mt5(pair, timeframe, bars)
        if not df.empty:
            print(f"📡 [{pair}] {len(df)} candles from MT5 | "
                  f"Latest close: {df['close'].iloc[-1]:.5f}")
            return df
        print(f"⚠️  MT5 failed → trying Alpha Vantage")

    # ── Priority 2: Alpha Vantage ─────────────────────────────
    if os.getenv("ALPHA_VANTAGE_KEY"):
        df = fetch_alpha_vantage(pair, timeframe, bars)
        if not df.empty:
            sample = df.tail(10)
            if not (sample['open'] == sample['high']).all():
                print(f"📡 [{pair}] {len(df)} candles from Alpha Vantage | "
                      f"Latest close: {df['close'].iloc[-1]:.5f}")
                return df
        print(f"⚠️  Alpha Vantage failed → trying yfinance")

    # ── Priority 3: yfinance ──────────────────────────────────
    df = fetch_from_yfinance(pair, timeframe, bars)
    if not df.empty:
        print(f"📡 [{pair}] {len(df)} candles from yfinance | "
              f"Latest close: {df['close'].iloc[-1]:.5f}")
        return df

    print(f"❌ FAILED: Could not get data for {pair}")
    return pd.DataFrame()


# ═══════════════════════════════════════════════════════════════
# PART 7 — TEST
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":

    print("=" * 55)
    print("  FOREX DATA SOURCE — CONNECTION TEST")
    print("=" * 55)

    print("\n📌 Step 1: Testing MT5 connection...")
    mt5_ok = connect_mt5()
    print(f"   Result: {'✅ Connected' if mt5_ok else '❌ Not available'}")

    print("\n📌 Step 2: Fetching data for all pairs...")
    pairs = ["EURUSD", "GBPUSD", "USDJPY"]
    results = {}

    for pair in pairs:
        print(f"\n   Testing {pair}...")
        df = get_data(pair=pair, timeframe="H1", bars=100)
        results[pair] = df

        if not df.empty:
            latest = df.iloc[-1]
            oldest = df.iloc[0]
            print(f"   ✅ Success!")
            print(f"      Candles : {len(df)}")
            print(f"      From    : {oldest.name}")
            print(f"      To      : {latest.name}")
            print(f"      Open    : {latest['open']:.5f}")
            print(f"      High    : {latest['high']:.5f}")
            print(f"      Low     : {latest['low']:.5f}")
            print(f"      Close   : {latest['close']:.5f}")
            same = (latest['open'] == latest['high'] == 
                    latest['low'] == latest['close'])
            print(f"      Quality : {'❌ Flat data' if same else '✅ Real OHLCV'}")
        else:
            print(f"   ❌ Failed")

    print("\n📌 Step 3: Cleaning up...")
    if mt5_ok:
        disconnect_mt5()

    print("\n" + "=" * 55)
    success = sum(1 for df in results.values() if not df.empty)
    print(f"  RESULT: {success}/3 pairs fetched")
    print("=" * 55)