#!/usr/bin/env python3
from __future__ import annotations

import argparse

from imotifpredictor.train.train_imip_seqonly import ImipSeqOnlyRunConfig, run_train_imip_seqonly


def parse_args() -> ImipSeqOnlyRunConfig:
    p = argparse.ArgumentParser(description="Train iM-IP-seq hybrid CNN→LSTM (sequence-only).")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--input_csv", type=str, help="Path to a single CSV file.")
    src.add_argument("--input_dir", type=str, help="Directory containing chunk CSV files (*.csv).")

    p.add_argument("--output_dir", type=str, required=True)

    p.add_argument("--seq_len", type=int, default=124)
    p.add_argument("--train_chrom", type=str, default=None)
    p.add_argument("--test_chrom", type=str, default=None)

    p.add_argument("--balancing", choices=["baseline", "class_weight"], default="baseline")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--shuffle_files", action="store_true")

    p.add_argument("--seed", type=int, default=1337)

    # hyperparams (paper defaults)
    p.add_argument("--filters1", type=int, default=32)
    p.add_argument("--filters2", type=int, default=64)
    p.add_argument("--kernel_size", type=int, default=5)
    p.add_argument("--pool_size", type=int, default=2)
    p.add_argument("--lstm_units", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--dense_units", type=int, default=32)
    p.add_argument("--learning_rate", type=float, default=1e-3)

    a = p.parse_args()
    return ImipSeqOnlyRunConfig(
        input_csv=a.input_csv,
        input_dir=a.input_dir,
        output_dir=a.output_dir,
        seq_len=a.seq_len,
        train_chrom=a.train_chrom,
        test_chrom=a.test_chrom,
        balancing=a.balancing,
        epochs=a.epochs,
        batch_size=a.batch_size,
        shuffle_files=a.shuffle_files,
        seed=a.seed,
        filters1=a.filters1,
        filters2=a.filters2,
        kernel_size=a.kernel_size,
        pool_size=a.pool_size,
        lstm_units=a.lstm_units,
        dropout=a.dropout,
        dense_units=a.dense_units,
        learning_rate=a.learning_rate,
    )


def main() -> None:
    cfg = parse_args()
    run_train_imip_seqonly(cfg)


if __name__ == "__main__":
    main()