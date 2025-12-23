# create_test_alerts.py
import os, random
import pandas as pd
from datetime import datetime, timedelta

os.makedirs("data", exist_ok=True)

alert_names = [
    "UDA - Trojan Horse Traffic", "Port Scan Detected", "Malware Beaconing",
    "Suspicious DNS Query", "Brute Force Attempt", "Data Exfiltration Pattern"
]
severity_levels = ["Low","Medium","High","Critical"]
protocols = ["TCP port 22","TCP port 1492","UDP port 53","TCP port 443","TCP port 3389"]
start = datetime(2025,1,1,0,0,0)

rows = []
for i in range(10):
    dt = start + timedelta(minutes=random.randint(0,60*24*30))
    date = dt.strftime("%m/%d/%Y")
    time = dt.strftime("%I:%M:%S %p")
    src = f"192.168.{random.randint(1,30)}.{random.randint(2,250)}"
    dst = f"192.168.{random.randint(1,30)}.{random.randint(2,250)}"
    proto = random.choice(protocols)
    name = random.choice(alert_names)
    sev = random.choice(severity_levels)
    exec_summary = f"Host {src} shows {name.lower()} communication with {dst} over {proto}. Flows: {random.randint(1,20)} connections, {random.randint(100,2000)} bytes transferred, duration {random.randint(1,120)} sec."
    rows.append({
        "Date": date,
        "Time": time,
        "Alert Name": name,
        "Severity": sev,
        "Source Host": src,
        "Protocol/Port": proto,
        "Destination IP": dst,
        "Executive Summary": exec_summary
    })

df = pd.DataFrame(rows)
df.to_csv("data/test_alerts.csv", index=False)
print("Wrote data/test_alerts.csv with", len(df), "rows")
print(df.head().to_string(index=False))
