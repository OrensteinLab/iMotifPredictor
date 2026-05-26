#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import tensorflow as tf

from imotifpredictor.data.add_imip_score import default_imip_model_paths, one_hot_encode_sequences
from imotifpredictor.data.io_sequences import load_sequences_table


def _load_models(model_paths: List[str]) -> List[tf.keras.Model]:
    models = []
    for p in model_paths:
        mp = Path(p)
        if not mp.exists():
            raise FileNotFoundError(f"Model not found: {mp}")
        models.append(tf.keras.models.load_model(str(mp)))
    return models


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Predict iM-IP scores using pretrained iM-IP hybrid CNN-LSTM models. "
            "Supports CSV, XLSX, and FASTA inputs. Sequences not matching --seq_len are dropped."
        )
    )
    p.add_argument("--input", required=True, help="Input file path (.csv/.xlsx/.fasta/.fa/.fna).")
    p.add_argument("--out", required=True, help="Output CSV path with added prediction columns.")
    p.add_argument(
        "--format",
        default="auto",
        choices=["auto", "csv", "xlsx", "fasta"],
        help="Input format (default: auto, inferred from suffix).",
    )
    p.add_argument("--seq_col", default="Sequence", help="Sequence column name for CSV/XLSX (default: Sequence).")
    p.add_argument("--id_col", default="", help="Optional ID column name for CSV/XLSX (default: none).")
    p.add_argument("--seq_len", type=int, default=124, help="Sequence length (default: 124).")
    p.add_argument("--batch_size", type=int, default=512, help="Batch size (default: 512).")

    p.add_argument(
        "--models",
        default="",
        help=(
            "Comma-separated list of .h5 model paths (order: HEK,U2OS,MCF7 recommended). "
            "If omitted, defaults to models/iM-IP/{HEK,U2OS,MCF7} hybrid CNN-LSTM baseline."
        ),
    )
    p.add_argument("--repo_root", default="", help="Repository root for resolving default model paths.")

    p.add_argument("--add_mean", action="store_true", help="Add pred_hybrid_cnn_lstm_mean (ensemble mean).")
    p.add_argument("--mean_col", default="pred_hybrid_cnn_lstm_mean", help="Mean column name.")
    p.add_argument("--prefix", default="pred_iM-IP", help="Prefix for per-model prediction columns.")

    args = p.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.models.strip():
        model_paths = [x.strip() for x in args.models.split(",") if x.strip()]
    else:
        model_paths = default_imip_model_paths(repo_root=args.repo_root.strip() or None)

    print("🚀 predict_imip_score.py started")
    print("📂 Input:", str(in_path))
    print("💾 Output:", str(out_path))
    print("🧾 Format:", args.format)
    print("🧬 seq_len:", args.seq_len)

    # Load sequences (drop invalid lengths)
    df_seq, rep = load_sequences_table(
        input_path=str(in_path),
        fmt=args.format,
        seq_col=args.seq_col,
        id_col=(args.id_col.strip() or None),
        seq_len=args.seq_len,
        drop_invalid_len=True,
    )
    print(f"✅ Loaded: total={rep.total_rows} kept={rep.kept_rows} dropped={rep.dropped_rows}")

    X = one_hot_encode_sequences(df_seq["Sequence"].tolist(), seq_len=args.seq_len)

    models = _load_models(model_paths)

    per_model_preds = []
    for model in models:
        y = model.predict(X, batch_size=args.batch_size, verbose=0).reshape(-1).astype(np.float32)
        per_model_preds.append(y)

    colnames = []
    for mp in model_paths:
        stem = Path(mp).stem
        colnames.append(f"{args.prefix}_{stem}")

    out_df = df_seq.copy()
    for c, y in zip(colnames, per_model_preds):
        out_df[c] = y

    if args.add_mean:
        out_df[args.mean_col] = np.mean(np.stack(per_model_preds, axis=0), axis=0)

    out_df.to_csv(out_path, index=False)
    print("✅ Saved:", str(out_path))
    print("📊 Rows:", len(out_df), "Cols:", out_df.shape[1])


if __name__ == "__main__":
    main()