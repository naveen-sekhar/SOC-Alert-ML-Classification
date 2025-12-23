"""Compute evaluation metrics for the CatBoost models on a labeled CSV.

Default input: data/smart_siem_dataset.csv (synthetic dataset with ground-truth labels).
Outputs: confusion matrices (CSV) and classification reports (TXT) under data/metrics/.

Usage:
    python src/evaluate_metrics.py \
        --input data/smart_siem_dataset.csv \
        --output-dir data/metrics
"""

import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

import predict_alerts_module as pam

TARGETS = ["Status", "Category", "Action Taken"]


def compute_metrics(df: pd.DataFrame, preds: Dict[str, List[str]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    for target in TARGETS:
        y_true = df[target].astype(str)
        y_pred = pd.Series(preds[target]).astype(str)

        labels = sorted(list(set(y_true) | set(y_pred)))
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        cm_df = pd.DataFrame(cm, index=[f"true_{l}" for l in labels], columns=[f"pred_{l}" for l in labels])
        cm_path = out_dir / f"{target.lower().replace(' ', '_')}_confusion.csv"
        cm_df.to_csv(cm_path)

        report = classification_report(y_true, y_pred, labels=labels, digits=3)
        rep_path = out_dir / f"{target.lower().replace(' ', '_')}_report.txt"
        rep_path.write_text(report)

        acc = (y_true == y_pred).mean()
        print(f"{target}: accuracy={acc:.4f} | confusion -> {cm_path} | report -> {rep_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate CatBoost models on labeled data.")
    parser.add_argument("--input", default="data/smart_siem_dataset.csv", help="Path to labeled CSV with targets.")
    parser.add_argument("--output-dir", default="data/metrics", help="Directory to save metrics outputs.")
    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.output_dir)

    df_full = pd.read_csv(input_path)
    df_infer = df_full.drop(columns=TARGETS, errors="ignore")

    records = df_infer.to_dict(orient="records")
    preds = pam.predict_alerts(records)

    compute_metrics(df_full, preds, out_dir)


if __name__ == "__main__":
    main()
