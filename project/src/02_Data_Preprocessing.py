import pandas as pd
import numpy as np
import re
from datetime import datetime

# -------------------------------
# Load dataset
# -------------------------------
df = pd.read_csv("smart_siem_dataset.csv")

# -------------------------------
# 1. REMOVE NON-USEFUL COLUMNS
# -------------------------------
df = df.drop(columns=["Alert ID"])   # No predictive value

# -------------------------------
# 2. DATE & TIME PROCESSING
# -------------------------------
df['Datetime'] = pd.to_datetime(df['Date'] + " " + df['Time'])
df['Hour'] = df['Datetime'].dt.hour
df['Minute'] = df['Datetime'].dt.minute
df['DayOfWeek'] = df['Datetime'].dt.dayofweek
df['IsWeekend'] = df['DayOfWeek'].isin([5,6]).astype(int)

df = df.drop(columns=["Date", "Time", "Datetime"])

# -------------------------------
# 3. SPLIT PROTOCOL/PORT
# -------------------------------
def split_proto_port(x):
    match = re.search(r"([A-Za-z]+)\s+port\s+(\d+)", str(x))
    if match:
        return match.group(1), int(match.group(2))
    return "UNKNOWN", -1

df[['Protocol', 'Port']] = df['Protocol/Port'].apply(
    lambda x: pd.Series(split_proto_port(x))
)

df = df.drop(columns=["Protocol/Port"])

# -------------------------------
# 4. IP FEATURES
# -------------------------------
def is_private(ip):
    try:
        parts = list(map(int, ip.split(".")))
        return int(
            parts[0] == 10 or
            (parts[0] == 172 and 16 <= parts[1] <= 31) or
            (parts[0] == 192 and parts[1] == 168)
        )
    except:
        return 0

df['Src_Private'] = df['Source Host'].apply(is_private)
df['Dst_Private'] = df['Destination IP'].apply(is_private)

def same_subnet(a, b):
    try:
        return int(a.split(".")[:3] == b.split(".")[:3])
    except:
        return 0

df['Same_Subnet'] = [
    same_subnet(a, b)
    for a, b in zip(df['Source Host'], df['Destination IP'])
]

# IMPORTANT: Drop raw IP strings
df = df.drop(columns=["Source Host", "Destination IP"])

# -------------------------------
# 5. CLEAN CATEGORICAL VALUES
# -------------------------------
df['Alert Name'] = df['Alert Name'].astype(str)
df['Severity'] = df['Severity'].astype(str)
df['Protocol'] = df['Protocol'].astype(str)

# -------------------------------
# 6. CLEAN EXEC SUMMARY TEXT
# -------------------------------
df['Executive Summary'] = (
    df['Executive Summary']
    .astype(str)
    .str.replace(r'\s+', ' ', regex=True)
)

# -------------------------------
# 7. SAVE CLEANED DATASET
# -------------------------------
df.to_csv("cleaned_smart_siem_dataset.csv", index=False)

print("CLEANED DATA SAVED AS: cleaned_smart_siem_dataset.csv")
print(df.head())
