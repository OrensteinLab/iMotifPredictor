#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import tensorflow as tf

from imotifpredictor.data.add_imip_score import one_hot_encode_sequences
from imotifpredictor.data.imotif_models import (
    get_imotif_variant_specs,
    default_imotif_model_path,
)


def _read_table(path: Path, fmt: str) -> pd.DataFrame:
    if fmt == "csv":
        return pd.read_csv(path)
    if fmt == "xlsx":
        return pd.read_excel(path)
    raise ValueError(f"Unsupported table format: {fmt}")


def _infer_format(path: Path, fmt: str) -> str:
    if fmt != "auto":
        return fmt
    suf = path.suffix.lower()
    if suf == ".csv":
        return "csv"
    if suf in [".xlsx", ".xls"]:
        return "xlsx"
    if suf in [".fasta", ".fa", ".fna"]:
        return "fasta"
    raise ValueError(f"Cannot infer format from suffix: {path.name}")


def _read_fasta_sequences(path: Path) -> pd.DataFrame:
    """
    Minimal FASTA reader (sequence-only).
    Produces a DataFrame with: ID, Sequence.
    """
    ids = []
    seqs = []

    cur_id: Optional[str] = None
    cur = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if cur_id is not None:
                    ids.append(cur_id)
                    seqs.append("".join(cur))
                cur_id = line[1:].strip() or f"seq_{len(ids)+1}"
                cur = []
            else:
                cur.append(line)

    if cur_id is not None:
        ids.append(cur_id)
        seqs.append("".join(cur))

    return pd.DataFrame({"ID": ids, "Sequence": seqs})


def main() -> None:
    specs = get_imotif_variant_specs()

    p = argparse.ArgumentParser(
        description=(
            "Predict iMotifPredictor scores using pretrained iMotifPredictor LSTM fusion models (.h5). "
            "For variants requiring epigenetic/iM-IP scalar inputs, input must be a table (CSV/XLSX) "
            "that already contains those numeric columns."
        )
    )
    p.add_argument("--input", required=True, help="Input file path (.csv/.xlsx/.fasta/.fa/.fna).")
    p.add_argument("--out", required=True, help="Output CSV path with added prediction column.")
    p.add_argument("--format", default="auto", choices=["auto", "csv", "xlsx", "fasta"], help="Input format.")
    p.add_argument("--variant", required=True, choices=sorted(specs.keys()), help="iMotifPredictor variant to use.")
    p.add_argument("--seq_col", default="Sequence", help="Sequence column name for CSV/XLSX (default: Sequence).")
    p.add_argument("--seq_len", type=int, default=124, help="Sequence length (default: 124).")
    p.add_argument("--batch_size", type=int, default=512, help="Batch size (default: 512).")
    p.add_argument("--model", default="", help="Optional explicit .h5 model path (overrides default).")
    p.add_argument("--repo_root", default="", help="Repository root for resolving default model paths.")
    p.add_argument(
        "--out_col",
        default="pred_iMotifPredictor",
        help="Output prediction column name (default: pred_iMotifPredictor).",
    )
    p.add_argument(
        "--drop_invalid_len",
        action="store_true",
        help="Drop rows where sequence length != --seq_len (recommended).",
    )
    p.add_argument(
        "--nan_fill",
        type=float,
        default=0.0,
        help="Fill NaN in required numeric features (default: 0.0).",
    )

    args = p.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fmt = _infer_format(in_path, args.format)
    spec = specs[args.variant]

    if fmt == "fasta" and not spec.supports_fasta:
        raise ValueError(
            f"Variant '{args.variant}' requires numeric features {spec.required_numeric_features} and "
            "cannot be run on FASTA."
        )

    # Resolve model path
    if args.model.strip():
        model_path = Path(args.model.strip())
    else:
        model_path = Path(
            default_imotif_model_path(
                variant=args.variant,
                repo_root=(args.repo_root.strip() or None),
            )
        )
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    print("🚀 predict_imotifpredictor.py started")
    print("📂 Input:", str(in_path))
    print("🧾 Format:", fmt)
    print("🧠 Variant:", args.variant)
    print("🧬 seq_len:", args.seq_len)
    print("🧩 Required features:", spec.required_numeric_features)
    print("🧠 Model:", str(model_path))
    print("💾 Output:", str(out_path))

    # Load input
    if fmt == "fasta":
        df = _read_fasta_sequences(in_path)
        seq_col = "Sequence"
    else:
        df = _read_table(in_path, fmt=fmt)
        seq_col = args.seq_col
        if seq_col not in df.columns:
            raise ValueError(f"Missing sequence column '{seq_col}'. Available: {list(df.columns)[:30]}...")

    # Drop invalid sequence lengths if requested
    df = df.copy()
    df[seq_col] = df[seq_col].astype(str)
    if args.drop_invalid_len:
        before = len(df)
        df = df[df[seq_col].str.len() == args.seq_len].copy()
        print(f"✅ Length filter: kept={len(df)} dropped={before-len(df)}")

    if df.empty:
        raise ValueError("No rows to predict after filtering.")

    # Validate/prepare numeric features (if any)
    feature_inputs = []
    if spec.required_numeric_features:
        for col in spec.required_numeric_features:
            if col not in df.columns:
                raise ValueError(f"Missing required feature column '{col}' for variant '{args.variant}'.")
            x = pd.to_numeric(df[col], errors="coerce").fillna(args.nan_fill).to_numpy(dtype=np.float32)
            feature_inputs.append(x.reshape(-1, 1))

    # Encode sequences
    X_seq = one_hot_encode_sequences(df[seq_col].tolist(), seq_len=args.seq_len)

    # Build model inputs
    if feature_inputs:
        model_inputs = [X_seq] + feature_inputs
    else:
        model_inputs = X_seq

    # Predict
    model = tf.keras.models.load_model(str(model_path))
    y = model.predict(model_inputs, batch_size=args.batch_size, verbose=0).reshape(-1).astype(np.float32)

    out_df = df.copy()
    out_df[args.out_col] = y
    out_df.to_csv(out_path, index=False)

    print("✅ Saved:", str(out_path))
    print("📊 Rows:", len(out_df), "Cols:", out_df.shape[1])
    print("📈 pred range:", float(np.min(y)), float(np.max(y)))


if __name__ == "__main__":
    main()