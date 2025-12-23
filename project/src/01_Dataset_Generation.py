import pandas as pd
import numpy as np
import random
import re
from datetime import datetime, timedelta

# HOW MANY SAMPLES?
num_samples = 1500

alert_names = [
    "UDA - Trojan Horse Traffic", "Port Scan Detected", "Malware Beaconing",
    "Suspicious DNS Query", "Brute Force Attempt", "Data Exfiltration Pattern",
    "Unauthorized Access Attempt", "Anomalous SSH Activity"
]

severity_levels = ["Low", "Medium", "High", "Critical"]
protocols = ["TCP", "UDP", "ICMP"]
ports = [22, 53, 80, 443, 1492, 8080, 3389, 5900]

keywords = {
    "malware": ["trojan", "malware", "beacon", "c2", "exfil"],
    "recon": ["scan", "recon", "brute force"],
    "policy": ["unauthorized", "access attempt", "ssh"],
}

action_map = {
    "Malware": ["Blocked", "Escalated"],
    "Reconnaissance": ["Investigated"],
    "Policy Violation": ["Escalated"],
    "False Positive": ["Closed"],
    "Benign Activity": ["No Action Required"]
}

data = []

start_date = datetime(2025, 1, 1)

def classify_status(summary, severity):
    s = summary.lower()
    if any(k in s for k in keywords["malware"]): return "Malicious"
    if severity == "Critical": return "Malicious"
    if severity == "Low": return "Legitimate"
    return "Malicious"

def classify_category(summary, severity):
    s = summary.lower()
    if any(k in s for k in keywords["malware"]): return "Malware"
    if any(k in s for k in keywords["recon"]): return "Reconnaissance"
    if any(k in s for k in keywords["policy"]): return "Policy Violation"
    if severity == "Low": return "Benign Activity"
    return "False Positive"

def classify_action(category):
    return random.choice(action_map[category])

for i in range(num_samples):
    date_time = start_date + timedelta(minutes=random.randint(0, 50000))
    date = date_time.strftime("%m/%d/%Y")
    time = date_time.strftime("%I:%M:%S %p")

    src = f"192.168.{random.randint(1,30)}.{random.randint(1,254)}"
    dst = f"192.168.{random.randint(1,30)}.{random.randint(1,254)}"

    proto = random.choice(protocols)
    port = random.choice(ports)

    # EXEC SUMMARY GENERATION
    base = random.choice(alert_names).lower()
    exec_summary = f"Host {src} shows {base} communication with {dst} over {proto} port {port}. "

    # Inject meaningful keywords
    if "trojan" in base or "malware" in base:
        exec_summary += "Possible trojan or malware beaconing detected."
    elif "scan" in base:
        exec_summary += "Multiple scanning attempts observed."
    elif "exfiltration" in base:
        exec_summary += "Possible data exfiltration pattern detected."
    elif "brute force" in base:
        exec_summary += "Brute force behavior observed."
    elif "unauthorized" in base:
        exec_summary += "Unauthorized access attempt detected."

    severity = random.choice(severity_levels)

    # LABELS (RULE BASED)
    status = classify_status(exec_summary, severity)
    category = classify_category(exec_summary, severity)
    action_taken = classify_action(category)

    row = [
        date, time, random.randint(10**17, 10**18-1),
        random.choice(alert_names),
        severity, src,
        f"{proto} port {port}", dst,
        exec_summary, status, category, action_taken
    ]

    data.append(row)

columns = ["Date","Time","Alert ID","Alert Name","Severity","Source Host",
           "Protocol/Port","Destination IP","Executive Summary",
           "Status","Category","Action Taken"]

df = pd.DataFrame(data, columns=columns)

df.to_csv("smart_siem_dataset.csv", index=False)

print("SMART DATASET GENERATED -> smart_siem_dataset.csv")
print(df.head())
