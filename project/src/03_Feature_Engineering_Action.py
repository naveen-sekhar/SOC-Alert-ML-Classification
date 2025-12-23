# feature_engineering_action.py
import pandas as pd
import numpy as np
import re

IN = "cleaned_smart_siem_dataset.csv"
OUT = "fe_smart_siem_dataset.csv"  # new file with features

df = pd.read_csv(IN)

# --- keyword sets ---
malware_kw = ["trojan", "malware", "beacon", "c2", "exfil", "ransom", "payload"]
recon_kw   = ["scan", "recon", "port scan", "scanning", "brute force", "bruteforce"]
policy_kw  = ["unauthorized", "unauthorised", "access attempt", "failed authentication", "login failed", "ssh", "unauthorized access"]

# normalize executive summary
df['exec_low'] = df['Executive Summary'].astype(str).str.lower()

# --- keyword flags and counts ---
def count_keywords(text, kws):
    return sum(1 for k in kws if k in text)

df['keyword_count'] = df['exec_low'].apply(lambda t: sum(count_keywords(t, kws) for kws in [malware_kw, recon_kw, policy_kw]))
df['has_malware_kw'] = df['exec_low'].apply(lambda t: int(any(k in t for k in malware_kw)))
df['has_recon_kw'] = df['exec_low'].apply(lambda t: int(any(k in t for k in recon_kw)))
df['has_policy_kw'] = df['exec_low'].apply(lambda t: int(any(k in t for k in policy_kw)))

# --- severity ordinal ---
sev_map = {"low":0, "medium":1, "high":2, "critical":3}
df['severity_ordinal'] = df['Severity'].astype(str).str.lower().map(sev_map).fillna(1).astype(int)

# --- high risk ports ---
high_risk = {22, 3389, 5900, 1492, 445, 3306, 1433}
def extract_port_num(x):
    try:
        m = re.search(r"(\d{1,5})", str(x))
        return int(m.group(1)) if m else -1
    except:
        return -1

# if Port exists as column, use it; else parse Protocol/Port
if 'Port' in df.columns:
    df['port_num'] = df['Port'].fillna(-1).astype(int)
else:
    df['port_num'] = df['Protocol'].astype(str) + " " + df.get('Protocol/Port', '').astype(str)
    df['port_num'] = df['Protocol/Port'].apply(lambda x: extract_port_num(x))

df['high_risk_port'] = df['port_num'].apply(lambda p: int(p in high_risk))

# --- off-hours flag ---
if 'Hour' in df.columns:
    df['off_hours'] = df['Hour'].apply(lambda h: int(h in [0,1,2,3,4,5]))
else:
    # try to parse Hour from datetime-like columns if present
    df['off_hours'] = 0

# --- parse flows / bytes / duration from exec summary if present ---
# Example patterns handled: "Flows: 14", "total of 784 bytes", "duration of 30 sec"
def extract_int(regex, text):
    m = re.search(regex, text)
    return int(m.group(1)) if m else 0

df['flow_count'] = df['exec_low'].apply(lambda t: extract_int(r'flows?:\s*([0-9]+)', t))
df['bytes_transferred'] = df['exec_low'].apply(lambda t: extract_int(r'([0-9]+)\s*bytes', t))
df['duration_sec'] = df['exec_low'].apply(lambda t: extract_int(r'duration\s*(?:of)?\s*([0-9]+)\s*sec', t))

# If zeroes and not present, you still get zeros (fine)
# --- final housekeeping: drop helper columns if desired ---
df = df.drop(columns=['exec_low'], errors='ignore')

# Save new dataset
df.to_csv(OUT, index=False)
print("Saved engineered dataset to:", OUT)
print(df[['keyword_count','has_malware_kw','has_recon_kw','has_policy_kw','severity_ordinal','port_num','high_risk_port','off_hours','flow_count','bytes_transferred','duration_sec']].head())
