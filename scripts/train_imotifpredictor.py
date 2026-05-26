#!/usr/bin/env python3
"""
CLI entrypoint for chunk-wise training of iMotifPredictor (LSTM + late fusion).

Produces .h5 model files in the specified output directory.
"""

from __future__ import annotations

import argparse

from imotifpredictor.constants import FEATURE_VARIANTS
from imotifpredictor.train.train_imotifpredictor import ChunkTrainConfig, run_chunk_training


def main() -> None:
    p = argparse.ArgumentParser(description="Train iMotifPredictor (LSTM + late fusion) on chunk CSV files.")
    p.add_argument("--directory", type=str, required=True, help="Directory containing chunk CSV files.")
    p.add_argument("--output_dir", type=str, required=True, help="Output directory for saved .h5 models.")
    p.add_argument("--start", type=str, required=True, help="Start chunk filename (inclusive).")
    p.add_argument("--end", type=str, required=True, help="End chunk filename (inclusive).")

    p.add_argument(
        "--feature_variant",
        type=str,
        choices=list(FEATURE_VARIANTS.keys()) + ["all"],
        default="all",
        help="Feature-variant identifier.",
    )
    p.add_argument("--balancing", type=str, choices=["baseline", "class_weight"], required=True)
    p.add_argument("--seq_len", type=int, default=124)
    p.add_argument("--epochs_per_chunk", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--shuffle_files", action="store_true")
    p.add_argument("--skip_chr1", action="store_true")

    p.add_argument("--lstm_units", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--dense_units", type=int, default=32)
    p.add_argument("--learning_rate", type=float, default=1e-3)

    a = p.parse_args()

    cfg = ChunkTrainConfig(
        directory=a.directory,
        output_dir=a.output_dir,
        start=a.start,
        end=a.end,
        feature_variant=a.feature_variant,
        balancing=a.balancing,
        seq_len=a.seq_len,
        epochs_per_chunk=a.epochs_per_chunk,
        batch_size=a.batch_size,
        shuffle_files=a.shuffle_files,
        skip_chr1=a.skip_chr1,
        lstm_units=a.lstm_units,
        dropout=a.dropout,
        dense_units=a.dense_units,
        learning_rate=a.learning_rate,
    )

    run_chunk_training(cfg)


if __name__ == "__main__":
    main()