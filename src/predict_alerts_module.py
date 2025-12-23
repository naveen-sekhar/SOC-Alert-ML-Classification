# src/predict_alerts_module.py
# Reusable prediction module for: Status, Category, Action Taken.

import pandas as pd
import numpy as np
import re
import joblib
from pathlib import Path

def parse_datetime(date_str, time_str):
    try:
        return pd.to_datetime(str(date_str) + " " + str(time_str))
    except:
        return pd.NaT

def split_proto_port(s):
    m = re.search(r"([A-Za-z]+).*?(\d{1,5})", str(s))
    if m:
        return m.group(1).upper(), int(m.group(2))
    return "UNKNOWN", -1

def is_private(ip):
    try:
        a = list(map(int, str(ip).split(".")))
        return int(
            a[0] == 10 or
            (a[0] == 172 and 16 <= a[1] <= 31) or
            (a[0] == 192 and a[1] == 168)
        )
    except:
        return 0

def same_subnet(a, b):
    try:
        return int(str(a).split(".")[:3] == str(b).split(".")[:3])
    except:
        return 0

def extract_int(pattern, text):
    m = re.search(pattern, str(text))
    return int(m.group(1)) if m else 0

def preprocess_alerts(alerts):
    single = False
    if isinstance(alerts, dict):
        alerts = [alerts]; single = True

    df = pd.DataFrame(alerts)

    # ensure required cols exist
    for c in ["Date","Time","Protocol/Port","Source Host","Destination IP","Executive Summary","Alert Name","Severity"]:
        if c not in df.columns:
            df[c] = ""

    # Datetime features
    df['Datetime'] = df.apply(lambda r: parse_datetime(r['Date'], r['Time']), axis=1)
    df['Hour'] = df['Datetime'].dt.hour.fillna(0).astype(int)
    df['Minute'] = df['Datetime'].dt.minute.fillna(0).astype(int)
    df['DayOfWeek'] = df['Datetime'].dt.dayofweek.fillna(0).astype(int)
    df['IsWeekend'] = df['DayOfWeek'].isin([5,6]).astype(int)

    # Protocol / Port
    df[['Protocol','Port']] = df['Protocol/Port'].apply(lambda x: pd.Series(split_proto_port(x)))

    # IP features
    df['Src_Private'] = df['Source Host'].apply(is_private)
    df['Dst_Private'] = df['Destination IP'].apply(is_private)
    df['Same_Subnet'] = [same_subnet(a,b) for a,b in zip(df['Source Host'], df['Destination IP'])]

    # Text normalization
    df['exec_low'] = df['Executive Summary'].astype(str).str.lower()

    # Keyword features
    malware_kw = ["trojan","malware","beacon","c2","exfil","ransom","payload"]
    recon_kw   = ["scan","recon","port scan","scanning","brute force","bruteforce"]
    policy_kw  = ["unauthorized","access attempt","failed authentication","login failed","ssh"]

    df['keyword_count'] = df['exec_low'].apply(lambda t: sum(k in t for k in (malware_kw + recon_kw + policy_kw)))
    df['has_malware_kw'] = df['exec_low'].apply(lambda t: int(any(k in t for k in malware_kw)))
    df['has_recon_kw'] = df['exec_low'].apply(lambda t: int(any(k in t for k in recon_kw)))
    df['has_policy_kw'] = df['exec_low'].apply(lambda t: int(any(k in t for k in policy_kw)))

    # severity ordinal
    sev_map = {"low":0,"medium":1,"high":2,"critical":3}
    df['severity_ordinal'] = df['Severity'].astype(str).str.lower().map(sev_map).fillna(1).astype(int)

    # port numeric / high risk
    df['port_num'] = pd.to_numeric(df['Port'], errors='coerce').fillna(-1).astype(int)
    high_risk = {22,3389,5900,1492,445,3306,1433}
    df['high_risk_port'] = df['port_num'].apply(lambda p: int(p in high_risk))

    # off-hours
    df['off_hours'] = df['Hour'].apply(lambda h: int(h in [0,1,2,3,4,5]))

    # flows/bytes/duration extraction
    df['flow_count'] = df['exec_low'].apply(lambda t: extract_int(r"flows?:\s*([0-9]+)", t))
    df['bytes_transferred'] = df['exec_low'].apply(lambda t: extract_int(r"([0-9]+)\s*bytes", t))
    df['duration_sec'] = df['exec_low'].apply(lambda t: extract_int(r"duration.*?([0-9]+)", t))

    # drop raw helper columns (we keep engineered numeric columns)
    drop_cols = ["Date","Time","Protocol/Port","Source Host","Destination IP","Datetime","exec_low"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    return df, single

# ---------- Load models (paths assumed relative to project root) ----------
_models_loaded = False


def _resolve_path(rel: Path) -> Path:
    here = Path(__file__).resolve()
    search_roots = [here.parent, *here.parents]
    for cand in search_roots:
        for rel_candidate in (rel, Path("models") / rel):
            candidate = cand / rel_candidate
            if candidate.exists():
                return candidate
    raise FileNotFoundError(f"Missing model file: {rel}")


def _load_models():
    global _models_loaded, models_dict, label_encs, tfidf_main, svd_main, action_model, action_tfidf, action_svd, action_le, CAT_COLS, TEXT_COL
    if _models_loaded:
        return

    cat_root = Path("catboost_models")
    act_root = Path("catboost_action_improved")

    models_dict = joblib.load(_resolve_path(cat_root / "catboost_models_dict.joblib"))
    label_encs  = joblib.load(_resolve_path(cat_root / "label_encoders.joblib"))
    tfidf_main  = joblib.load(_resolve_path(cat_root / "tfidf.joblib"))
    svd_main    = joblib.load(_resolve_path(cat_root / "svd.joblib"))

    action_model = joblib.load(_resolve_path(act_root / "catboost_action_improved.joblib"))
    action_tfidf = joblib.load(_resolve_path(act_root / "tfidf.joblib"))
    action_svd   = joblib.load(_resolve_path(act_root / "svd.joblib"))
    action_le    = joblib.load(_resolve_path(act_root / "action_label_encoder.joblib"))

    CAT_COLS = ["Alert Name", "Severity", "Protocol"]
    TEXT_COL = "Executive Summary"
    _models_loaded = True

def predict_alerts(alerts):
    _load_models()
    df, single = preprocess_alerts(alerts)

    # Text for Status/Category
    X_text = tfidf_main.transform(df[TEXT_COL].astype(str))
    X_text_red = svd_main.transform(X_text)

    # Explicit expected numeric features — prevents accidental string columns
    expected_numeric = [
        "Hour","Minute","DayOfWeek","IsWeekend","Port","port_num","Src_Private",
        "Dst_Private","Same_Subnet","keyword_count","has_malware_kw","has_recon_kw",
        "has_policy_kw","severity_ordinal","high_risk_port","off_hours",
        "flow_count","bytes_transferred","duration_sec"
    ]
    numeric_cols = [c for c in expected_numeric if c in df.columns]

    # Categorical part
    if all(c in df.columns for c in CAT_COLS):
        X_cat = df[CAT_COLS].astype(str).reset_index(drop=True)
    else:
        # create placeholder if any cat col missing
        X_cat = pd.DataFrame({c: df.get(c, "") for c in CAT_COLS})

    X_num = df[numeric_cols].astype(float).reset_index(drop=True) if numeric_cols else pd.DataFrame(index=df.index)
    txt_df = pd.DataFrame(X_text_red, index=df.index, columns=[f"txt_{i}" for i in range(X_text_red.shape[1])]).reset_index(drop=True)

    X_all = pd.concat([X_cat.reset_index(drop=True), X_num, txt_df], axis=1)

    # Predict Status & Category
    status_model = models_dict["Status"]
    pred_status_enc = np.ravel(status_model.predict(X_all))
    pred_status = label_encs["Status"].inverse_transform(pred_status_enc.astype(int))

    category_model = models_dict["Category"]
    pred_category_enc = np.ravel(category_model.predict(X_all))
    pred_category = label_encs["Category"].inverse_transform(pred_category_enc.astype(int))

    # Action model (engineered)
    X_text_a = action_tfidf.transform(df[TEXT_COL].astype(str))
    X_text_a_red = action_svd.transform(X_text_a)
    txt_df_a = pd.DataFrame(X_text_a_red, index=df.index, columns=[f"txt_{i}" for i in range(X_text_a_red.shape[1])]).reset_index(drop=True)

    X_action = pd.concat([X_cat.reset_index(drop=True), X_num.reset_index(drop=True), txt_df_a], axis=1)

    pred_action_enc = np.ravel(action_model.predict(X_action))
    pred_action = action_le.inverse_transform(pred_action_enc.astype(int))

    out = {
        "Status": list(pred_status),
        "Category": list(pred_category),
        "Action Taken": list(pred_action)
    }

    return out if not single else {k:v[0] for k,v in out.items()}
