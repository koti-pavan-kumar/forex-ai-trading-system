# morning_check.py
# Run this every morning — takes 2 minutes
# Does everything you need for the day

from data_source import connect_mt5, disconnect_mt5
from live_signals import run_once
from paper_trading import show_open_trades, show_stats
from datetime import datetime

print("=" * 50)
print(f"  MORNING CHECK — {datetime.now().strftime('%d %b %Y')}")
print("=" * 50)

# Connect MT5
mt5_ok = connect_mt5()

# Show open trades
print("\n📌 Your Open Trades:")
show_open_trades()

# Generate today's signals
print("\n📌 Today's Signals:")
run_once(["EURUSD", "GBPUSD", "USDJPY"], timeframe="H1")

# Show stats
print("\n📌 Performance So Far:")
show_stats()

if mt5_ok:
    disconnect_mt5()

print("\n✅ Morning check complete!")
print("Run 'streamlit run dashboard.py' to see live charts")