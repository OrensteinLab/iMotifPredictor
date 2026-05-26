#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from imotifpredictor.data.io_sequences import load_sequences_table
from imotifpredictor.data.add_imip_score import add_imip_predictions
from imotifpredictor.data.imotif_models import get_imotif_variant_specs
from imotifpredictor.models.imotifpredictor_lstm_fusion import predict_imotif_from_table


def main() -> None:
    specs = get_imotif_variant_specs()

    p = argparse.ArgumentParser(
        description=(
            "End-to-end prediction pipeline.\n"
            "1) Load sequences (CSV/XLSX/FASTA) and enforce fixed length (default: 124).\n"
            "2) Optionally add iM-IP per-model scores and pred_hybrid_cnn_lstm_mean.\n"
            "3) Predict iMotifPredictor score for a selected variant.\n"
            "Output is a CSV table with added columns."
        )
    )

    p.add_argument("--input", required=True, help="Input file path (.csv/.xlsx/.fasta/.fa/.fna).")
    p.add_argument("--out", required=True, help="Output CSV path.")

    p.add_argument("--format", default="auto", choices=["auto", "csv", "xlsx", "fasta"], help="Input format.")
    p.add_argument("--seq_col", default="Sequence", help="Sequence column name for CSV/XLSX (default: Sequence).")
    p.add_argument("--id_col", default="", help="Optional ID column name for CSV/XLSX (default: none).")
    p.add_argument("--seq_len", type=int, default=124, help="Sequence length (default: 124).")
    p.add_argument("--drop_invalid_len", action="store_true", help="Drop rows with invalid sequence length.")

    # iM-IP
    p.add_argument(
        "--add_imip",
        action="store_true",
        help="Add iM-IP predictions and pred_hybrid_cnn_lstm_mean using pretrained iM-IP models.",
    )
    p.add_argument("--imip_models", default="", help="Comma-separated list of iM-IP .h5 paths (optional).")
    p.add_argument("--imip_prefix", default="pred_iM-IP", help="Prefix for per-model iM-IP columns.")
    p.add_argument("--imip_mean_col", default="pred_hybrid_cnn_lstm_mean", help="iM-IP mean column name.")
    p.add_argument(
        "--imip_source_cols",
        default="",
        help=(
            "Optional comma-separated existing column names to average into --imip_mean_col "
            "(for example, three precomputed iM-IP score columns). "
            "If omitted, iM-IP models are executed."
        ),
    )
    p.add_argument("--repo_root", default="", help="Repository root for resolving default model paths.")

    # iMotifPredictor
    p.add_argument("--variant", required=True, choices=sorted(specs.keys()), help="iMotifPredictor variant.")
    p.add_argument("--imotif_out_col", default="pred_iMotifPredictor", help="Output column name.")
    p.add_argument("--imotif_model", default="", help="Optional explicit iMotifPredictor .h5 path (overrides default).")

    # runtime
    p.add_argument("--batch_size", type=int, default=512, help="Batch size (default: 512).")
    p.add_argument("--nan_fill", type=float, default=0.0, help="Fill value for NaN in numeric features.")

    args = p.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("🚀 predict_pipeline.py started")
    print("📂 Input:", str(in_path))
    print("💾 Output:", str(out_path))
    print("🧬 seq_len:", args.seq_len)
    print("🧠 Variant:", args.variant)

    # 1) load sequences
    df, rep = load_sequences_table(
        input_path=str(in_path),
        fmt=args.format,
        seq_col=args.seq_col,
        id_col=(args.id_col.strip() or None),
        seq_len=args.seq_len,
        drop_invalid_len=args.drop_invalid_len,
    )
    print(f"✅ Loaded: total={rep.total_rows} kept={rep.kept_rows} dropped={rep.dropped_rows}")

    # 2) optionally add iM-IP mean
    if args.add_imip:
        imip_models = [x.strip() for x in args.imip_models.split(",") if x.strip()] if args.imip_models.strip() else None
        imip_source_cols = [x.strip() for x in args.imip_source_cols.split(",") if x.strip()] if args.imip_source_cols.strip() else None
        df = add_imip_predictions(
            df=df,
            seq_col="Sequence",
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            model_paths=imip_models,
            repo_root=(args.repo_root.strip() or None),
            prefix=args.imip_prefix,
            add_mean=True,
            mean_col=args.imip_mean_col,
            source_cols=imip_source_cols,
            nan_fill=args.nan_fill,
        )
        print("✅ Added iM-IP predictions (+ mean)")

    # 3) iMotifPredictor prediction
    df = predict_imotif_from_table(
        df=df,
        variant=args.variant,
        seq_col="Sequence",
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        repo_root=(args.repo_root.strip() or None),
        model_path=(args.imotif_model.strip() or None),
        out_col=args.imotif_out_col,
        nan_fill=args.nan_fill,
    )
    print("✅ Added iMotifPredictor prediction")

    df.to_csv(out_path, index=False)
    print("✅ Saved:", str(out_path))
    print("📊 Rows:", len(df), "Cols:", df.shape[1])


if __name__ == "__main__":
    main()
