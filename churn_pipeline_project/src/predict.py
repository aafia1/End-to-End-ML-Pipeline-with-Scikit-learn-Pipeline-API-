
"""
predict.py — Load exported pipeline and score new customers in batch.
"""
from __future__ import annotations
import argparse, os
import pandas as pd
from joblib import load

def main(args):
    model = load(args.model)
    df = pd.read_csv(args.csv)

    # Ensure the file doesn't include 'Churn' (this script is for inference)
    if "Churn" in df.columns:
        df = df.drop(columns=["Churn"])

    proba = model.predict_proba(df)[:, 1]
    pred = (proba >= args.threshold).astype(int)

    out = df.copy()
    out["churn_proba"] = proba
    out["churn_pred"] = pred

    out.to_csv(args.out, index=False)
    print(f"Wrote {len(out)} rows with predictions to {args.out}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True, help="Path to input CSV with feature columns (no Churn).")
    parser.add_argument("--model", type=str, default="models/best_pipeline.joblib", help="Path to saved pipeline.")
    parser.add_argument("--out", type=str, default="predictions.csv", help="Output CSV path.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold for churn=1.")
    args = parser.parse_args()
    main(args)
