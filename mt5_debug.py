# mt5_debug.py
# Finds exactly why MT5 is not connecting

import os
import sys
import subprocess
from dotenv import load_dotenv
load_dotenv()

print("=" * 55)
print("  MT5 CONNECTION DIAGNOSTIC")
print("=" * 55)

# ── CHECK 1: Python version and architecture ──────────────
print("\n📌 Check 1: Python Info")
print(f"   Version : {sys.version}")
print(f"   Platform: {sys.platform}")
print(f"   Arch    : {8 * struct.calcsize('P')} bit"
      if False else f"   Bits    : {sys.maxsize > 2**32 and '64-bit' or '32-bit'}")

# ── CHECK 2: MT5 package version ──────────────────────────
print("\n📌 Check 2: MT5 Package")
try:
    import MetaTrader5 as mt5
    print(f"   ✅ MT5 package installed")
    print(f"   Version: {mt5.__version__}")
except ImportError as e:
    print(f"   ❌ MT5 not installed: {e}")
    sys.exit(1)

# ── CHECK 3: Find MT5 terminal path ───────────────────────
print("\n📌 Check 3: MT5 Terminal Location")
possible_paths = [
    r"C:\Program Files\MetaTrader 5 IC Markets Global\terminal64.exe",
    r"C:\Program Files\MetaTrader 5\terminal64.exe",
    r"C:\Program Files (x86)\MetaTrader 5\terminal64.exe",
    os.path.expanduser(
        r"~\AppData\Roaming\MetaQuotes\Terminal"
    ),
]

found_path = None
for path in possible_paths:
    exists = os.path.exists(path)
    status = "✅ FOUND" if exists else "❌ not found"
    print(f"   {status}: {path}")
    if exists and path.endswith('.exe'):
        found_path = path

if not found_path:
    print("\n   🔍 Searching for terminal64.exe...")
    try:
        result = subprocess.run(
            ['where', 'terminal64.exe'],
            capture_output=True, text=True
        )
        if result.stdout:
            found_path = result.stdout.strip().split('\n')[0]
            print(f"   ✅ Found at: {found_path}")
        else:
            # Deep search
            search_dirs = [
                r"C:\Program Files",
                r"C:\Program Files (x86)",
                os.path.expanduser("~")
            ]
            for search_dir in search_dirs:
                for root, dirs, files in os.walk(search_dir):
                    if 'terminal64.exe' in files:
                        found_path = os.path.join(root, 'terminal64.exe')
                        print(f"   ✅ Found at: {found_path}")
                        break
                if found_path:
                    break
    except Exception as e:
        print(f"   Search error: {e}")

# ── CHECK 4: Is MT5 terminal running? ─────────────────────
print("\n📌 Check 4: MT5 Process Running?")
try:
    result = subprocess.run(
        ['tasklist', '/FI', 'IMAGENAME eq terminal64.exe'],
        capture_output=True, text=True
    )
    if 'terminal64.exe' in result.stdout:
        print("   ✅ MT5 terminal IS running")

        # Get process details
        lines = [l for l in result.stdout.split('\n')
                 if 'terminal64' in l.lower()]
        for line in lines:
            print(f"   Process: {line.strip()}")
    else:
        print("   ❌ MT5 terminal is NOT running")
        print("   → Please open MT5 and log in first!")
except Exception as e:
    print(f"   Error checking process: {e}")

# ── CHECK 5: Try connection WITHOUT path ──────────────────
print("\n📌 Check 5: Connect WITHOUT path specified")
login    = int(os.getenv("MT5_LOGIN", 0))
password = os.getenv("MT5_PASSWORD", "")
server   = os.getenv("MT5_SERVER", "")

print(f"   Login : {login}")
print(f"   Server: {server}")
print(f"   Pass  : {'*' * len(password)}")

result1 = mt5.initialize(
    login=login,
    password=password,
    server=server
)
print(f"   Result: {'✅ SUCCESS!' if result1 else '❌ Failed'}")
print(f"   Error : {mt5.last_error()}")
mt5.shutdown()

# ── CHECK 6: Try connection WITH path ─────────────────────
print("\n📌 Check 6: Connect WITH path specified")
if found_path and found_path.endswith('.exe'):
    print(f"   Using path: {found_path}")
    result2 = mt5.initialize(
        path=found_path,
        login=login,
        password=password,
        server=server
    )
    print(f"   Result: {'✅ SUCCESS!' if result2 else '❌ Failed'}")
    print(f"   Error : {mt5.last_error()}")
    if result2:
        info = mt5.account_info()
        print(f"   Account: {info.login}")
        print(f"   Balance: ${info.balance:,.2f}")
    mt5.shutdown()
else:
    print("   ⚠️  No valid path found to test")

# ── CHECK 7: Check Windows event log for MT5 errors ───────
print("\n📌 Check 7: MT5 Data Path")
appdata = os.getenv('APPDATA', '')
mt5_data = os.path.join(appdata, 'MetaQuotes', 'Terminal')

if os.path.exists(mt5_data):
    folders = os.listdir(mt5_data)
    print(f"   MT5 data folders found: {len(folders)}")
    for folder in folders:
        folder_path = os.path.join(mt5_data, folder)
        config_path = os.path.join(folder_path, 'config')
        origin_path = os.path.join(folder_path, 'origin.txt')

        if os.path.exists(origin_path):
            with open(origin_path, 'r') as f:
                origin = f.read().strip()
            print(f"\n   Terminal ID: {folder[:20]}...")
            print(f"   Origin     : {origin}")
else:
    print(f"   ❌ MT5 data folder not found at: {mt5_data}")

print("\n" + "=" * 55)
print("  DIAGNOSTIC COMPLETE")
print("=" * 55)
print("\n  Copy and paste ALL output above and share it.")