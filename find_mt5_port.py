# find_mt5_port.py
# Scans all common trading ports on ICMarkets demo server

import socket
import threading
import time

TARGET_IP = "170.75.202.214"  # demo.icmarkets.com

# All ports MT5/trading platforms commonly use
PORTS_TO_SCAN = [
    80, 443, 8080, 8443,
    8228, 8229, 8230,
    3000, 3001,
    8000, 8001,
    9000, 9001,
    10000, 10001,
    8888, 9999,
    1080, 1443,
    444, 445,
    7000, 7001,
    5000, 5001,
    4000, 4001,
    18812, 18813
]

open_ports = []
lock = threading.Lock()

def scan_port(ip, port):
    try:
        sock = socket.socket(
            socket.AF_INET, socket.SOCK_STREAM
        )
        sock.settimeout(2)
        result = sock.connect_ex((ip, port))
        sock.close()
        if result == 0:
            with lock:
                open_ports.append(port)
                print(f"   ✅ PORT {port} IS OPEN!")
    except:
        pass

print("=" * 55)
print(f"  SCANNING demo.icmarkets.com")
print(f"  IP: {TARGET_IP}")
print(f"  Testing {len(PORTS_TO_SCAN)} ports...")
print("=" * 55)

threads = []
for port in PORTS_TO_SCAN:
    t = threading.Thread(
        target=scan_port,
        args=(TARGET_IP, port)
    )
    threads.append(t)
    t.start()

for t in threads:
    t.join()

print("\n" + "=" * 55)
if open_ports:
    print(f"  OPEN PORTS FOUND: {sorted(open_ports)}")
    print(f"\n  These are the ports MT5 can use")
    print(f"  to connect to ICMarkets demo server")
else:
    print("  NO OPEN PORTS FOUND")
    print("  ICMarkets demo server is blocking")
    print("  all connections from your IP")
print("=" * 55)