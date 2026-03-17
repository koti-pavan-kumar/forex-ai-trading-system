# deep_network_test.py
import socket
import subprocess
import os

print("=" * 55)
print("  DEEP NETWORK DIAGNOSTIC")
print("=" * 55)

# Test 1: DNS Resolution
print("\n📌 Test 1: DNS Resolution")
domains = [
    "demo.icmarkets.com",
    "mt5.icmarkets.com", 
    "icmarkets.com",
    "trade.icmarkets.com",
    "mt-demo.icmarkets.com",
    "sc-demo.icmarkets.com",
    "icmarkets-sc-demo.com"
]

resolved = {}
for domain in domains:
    try:
        ip = socket.gethostbyname(domain)
        print(f"   ✅ {domain} → {ip}")
        resolved[domain] = ip
    except Exception as e:
        print(f"   ❌ {domain} → Cannot resolve")

# Test 2: Port testing on resolved IPs
print("\n📌 Test 2: Port Testing")
ports_to_test = [443, 80, 8228, 8229, 3000]

for domain, ip in resolved.items():
    for port in ports_to_test:
        try:
            sock = socket.create_connection(
                (ip, port), timeout=3
            )
            sock.close()
            print(f"   ✅ {domain}:{port} OPEN")
        except:
            pass

# Test 3: Traceroute to demo server
print("\n📌 Test 3: Route to demo.icmarkets.com")
try:
    result = subprocess.run(
        ["tracert", "-d", "-h", "5", "demo.icmarkets.com"],
        capture_output=True,
        text=True,
        timeout=20
    )
    print(result.stdout[:500])
except Exception as e:
    print(f"   Error: {e}")

# Test 4: Check MT5 log files for server address
print("\n📌 Test 4: MT5 Log - Finding Real Server")
appdata = os.getenv('APPDATA', '')
log_path = os.path.join(
    appdata, 'MetaQuotes', 'Terminal'
)

found_server = False
if os.path.exists(log_path):
    for folder in os.listdir(log_path):
        logs_dir = os.path.join(log_path, folder, 'logs')
        if os.path.exists(logs_dir):
            log_files = sorted(os.listdir(logs_dir))
            if log_files:
                latest = log_files[-1]
                log_file = os.path.join(logs_dir, latest)
                try:
                    with open(log_file, 'r',
                              errors='ignore') as f:
                        content = f.read()
                    
                    # Find connection attempts
                    lines = content.split('\n')
                    for line in lines:
                        if any(keyword in line.lower() for 
                               keyword in ['connect', 'server',
                                          'network', 'socket',
                                          'icmarkets']):
                            print(f"   LOG: {line.strip()[:100]}")
                            found_server = True
                except:
                    pass

if not found_server:
    print("   No relevant log entries found")

print("\n" + "=" * 55)
print("  DIAGNOSTIC COMPLETE")
print("=" * 55)