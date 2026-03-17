# test_api.py
# Tests Alpha Vantage API key and D1 yfinance data quality

import urllib.request
import json
import yfinance as yf

# ── TEST 1: Alpha Vantage ─────────────────────────────────────
print("=" * 50)
print("TEST 1: Alpha Vantage API Key")
print("=" * 50)

url = (
    "https://www.alphavantage.co/query?"
    "function=FX_INTRADAY"
    "&from_symbol=EUR"
    "&to_symbol=USD"
    "&interval=60min"
    "&outputsize=compact"
    "&apikey=P5957K15K9OUNMXM"
)

try:
    with urllib.request.urlopen(url, timeout=15) as r:
        data = json.loads(r.read().decode())

    keys = list(data.keys())
    print(f"Keys found: {keys}")

    if "Time Series FX (60min)" in keys:
        ts = data["Time Series FX (60min)"]
        first_key = list(ts.keys())[0]
        first_bar = ts[first_key]
        print(f"\n✅ Alpha Vantage WORKING!")
        print(f"   Latest bar: {first_key}")
        print(f"   Open  : {first_bar['1. open']}")
        print(f"   High  : {first_bar['2. high']}")
        print(f"   Low   : {first_bar['3. low']}")
        print(f"   Close : {first_bar['4. close']}")

    elif "Note" in data:
        print("⚠️  Rate limit hit - wait 1 minute then retry")
        print(f"   Message: {data['Note'][:100]}")

    elif "Error Message" in data:
        print(f"❌ Wrong API key or bad request")
        print(f"   Error: {data['Error Message']}")

    elif "Information" in data:
        print(f"⚠️  API limit reached for today")
        print(f"   Message: {data['Information'][:150]}")

except Exception as e:
    print(f"❌ Connection error: {e}")


# ── TEST 2: yfinance Daily Data ───────────────────────────────
print("\n" + "=" * 50)
print("TEST 2: yfinance Daily (D1) Data Quality")
print("=" * 50)

pairs = {
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "USDJPY": "USDJPY=X"
}

for pair, symbol in pairs.items():
    try:
        df = yf.Ticker(symbol).history(period="2y", interval="1d")
        df.columns = df.columns.str.lower()

        if df.empty:
            print(f"❌ {pair}: No data returned")
            continue

        latest = df.iloc[-1]
        same = (latest['open'] == latest['high'] == 
                latest['low'] == latest['close'])

        print(f"\n{pair}:")
        print(f"   Rows  : {len(df)}")
        print(f"   Open  : {latest['open']:.5f}")
        print(f"   High  : {latest['high']:.5f}")
        print(f"   Low   : {latest['low']:.5f}")
        print(f"   Close : {latest['close']:.5f}")
        print(f"   Quality: {'❌ Flat data' if same else '✅ Real OHLCV'}")

    except Exception as e:
        print(f"❌ {pair} error: {e}")

print("\n" + "=" * 50)
print("Tests complete!")
print("=" * 50)