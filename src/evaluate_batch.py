# evaluate_batch.py
# Batch prediction for SIEM alerts using predict_alerts_module.py
# Usage:
#   python evaluate_batch.py --input input.csv --output output.csv

import argparse
import pandas as pd
from predict_alerts_module import predict_alerts

# ---------------------------------------
# Argument Parser
# ---------------------------------------
def get_args():
    parser = argparse.ArgumentParser(description="Batch evaluate SIEM alerts CSV")
    parser.add_argument("--input", required=True, help="Path to input CSV file")
    parser.add_argument("--output", required=True, help="Path to save output CSV")
    return parser.parse_args()

# ---------------------------------------
# Batch Evaluation Logic
# ---------------------------------------
def evaluate_batch(input_path, output_path):

    print(f"Loading alerts from: {input_path}")
    df = pd.read_csv(input_path)

    # Convert df rows to list of dicts
    alerts = df.to_dict(orient="records")

    print("Running predictions on batch...")
    results = predict_alerts(alerts)  # returns dict of lists

    # Append predictions to dataframe
    df["Status"] = results["Status"]
    df["Category"] = results["Category"]
    df["Action Taken"] = results["Action Taken"]

    # Save output
    df.to_csv(output_path, index=False)

    print(f"Predictions saved to: {output_path}")
    print(df.head())

# ---------------------------------------
# Main
# ---------------------------------------
if __name__ == "__main__":
    args = get_args()
    evaluate_batch(args.input, args.output)
