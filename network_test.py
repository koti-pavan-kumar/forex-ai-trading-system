# network_test.py
import socket
import urllib.request

print("=" * 45)
print("  MT5 NETWORK CONNECTIVITY TEST")
print("=" * 45)

servers = [
    ("mt5.icmarkets.com",   443),
    ("mt5.icmarkets.com",   80),
    ("demo.icmarkets.com",  443),
    ("www.google.com",      443),
]

for host, port in servers:
    try:
        sock = socket.create_connection((host, port), timeout=5)
        sock.close()
        print(f"✅ {host}:{port} = REACHABLE")
    except Exception as e:
        print(f"❌ {host}:{port} = BLOCKED")

print("=" * 45)