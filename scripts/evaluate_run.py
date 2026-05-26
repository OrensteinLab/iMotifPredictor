#!/usr/bin/env python3
import argparse
from pathlib import Path

import pandas as pd

from imotifpredictor.utils.metrics import rank_prediction_columns_by_aupr


DEFAULT_EXCLUDE = ["Chromosome", "Start", "End", "Sequence", "Label", "Label_nuc"]


def main():
    parser = argparse.ArgumentParser(description="Compute AUPR for prediction columns in a CSV.")
    parser.add_argument("--csv", required=True, help="Path to input CSV")
    parser.add_argument("--label", default="Label_nuc", help="Binary label column (0/1). Default: Label_nuc")
    parser.add_argument("--out", required=True, help="Output CSV path for ranked AUPR table")
    parser.add_argument(
        "--exclude",
        nargs="*",
        default=DEFAULT_EXCLUDE,
        help="Columns to exclude from scoring (metadata, label, etc.)",
    )
    parser.add_argument("--nan_fill", type=float, default=0.0, help="Fill value for NaN scores. Default: 0.0")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("🚀 evaluate_run.py started")
    print("📂 Input:", str(csv_path))
    print("🎯 Label:", args.label)

    df = pd.read_csv(csv_path)
    print("✅ CSV loaded, shape:", df.shape)

    # Convert non-excluded cols to numeric where possible (safe + consistent with your current logic)
    for col in df.columns:
        if col not in args.exclude and col != args.label:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Keep your behavior: missing scores -> 0
    df.fillna(0, inplace=True)

    res = rank_prediction_columns_by_aupr(
        df=df,
        label_col=args.label,
        exclude_cols=args.exclude,
        nan_fill=args.nan_fill,
    )

    res.to_csv(out_path, index=False)
    print("✅ Saved:", str(out_path))
    print("📊 Result rows:", res.shape[0])
    if not res.empty:
        print("🏆 Top-5:")
        print(res.head(5).to_string(index=False))


if __name__ == "__main__":
    main()