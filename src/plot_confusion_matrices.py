"""Plot confusion matrices for CatBoost models on a labeled CSV.

Usage:
    python src/plot_confusion_matrices.py \
        --input data/smart_siem_dataset.csv \
        --output-dir data/metrics

Outputs PNG files per target under the output directory.
"""

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix

import predict_alerts_module as pam

TARGETS = ["Status", "Category", "Action Taken"]


def plot_cm(cm, labels, title, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(0.7 * len(labels) + 3, 0.7 * len(labels) + 3))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title(title, fontsize=14, pad=12)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    # Annotate cells
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black", fontsize=9)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot confusion matrices on labeled data.")
    parser.add_argument("--input", default="data/smart_siem_dataset.csv", help="Path to labeled CSV with targets.")
    parser.add_argument("--output-dir", default="data/metrics", help="Directory to save plots.")
    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.output_dir)

    df_full = pd.read_csv(input_path)
    df_infer = df_full.drop(columns=TARGETS, errors="ignore")

    records = df_infer.to_dict(orient="records")
    preds = pam.predict_alerts(records)

    for target in TARGETS:
        y_true = df_full[target].astype(str)
        y_pred = pd.Series(preds[target]).astype(str)
        labels = sorted(list(set(y_true) | set(y_pred)))
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        title = f"Confusion Matrix - {target}"
        path = out_dir / f"{target.lower().replace(' ', '_')}_confusion.png"
        plot_cm(cm, labels, title, path)
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
