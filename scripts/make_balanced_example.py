#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict, List

import pandas as pd


DEFAULT_PRIORITY_COLS = [
    "Chromosome",
    "Start",
    "End",
    "Sequence",
    "Label_nuc",
    "pred_hybrid_cnn_lstm_mean",
    "H3K9me3",
    "atac_signals",
    "H3K4me1",
    "H3K27ac",
    "H3K36me3",
    "H3K4me3",
]


def _parse_keep_cols(arg: str) -> List[str]:
    if not arg.strip():
        return []
    return [c.strip() for c in arg.split(",") if c.strip()]


def _resolve_columns(input_csv: Path, keep_cols: List[str]) -> List[str]:
    header = pd.read_csv(input_csv, nrows=0)
    cols = list(header.columns)

    if keep_cols:
        missing = [c for c in keep_cols if c not in cols]
        if missing:
            raise ValueError(f"Missing requested columns: {missing}")
        return keep_cols

    # Auto mode: keep common/important columns that exist in the file.
    selected = [c for c in DEFAULT_PRIORITY_COLS if c in cols]
    if not selected:
        # Fallback: keep all columns if no priority columns are found.
        return cols
    return selected


def _reservoir_add(reservoir: List[dict], row: dict, seen_count: int, k: int, rng: random.Random) -> None:
    if len(reservoir) < k:
        reservoir.append(row)
        return
    j = rng.randint(1, seen_count)
    if j <= k:
        reservoir[j - 1] = row


def make_balanced_example(
    input_csv: Path,
    output_csv: Path,
    label_col: str,
    pos_label: int,
    neg_label: int,
    n_per_class: int,
    chunksize: int,
    seed: int,
    keep_cols: List[str],
) -> None:
    if n_per_class <= 0:
        raise ValueError("--n_per_class must be > 0")

    usecols = _resolve_columns(input_csv, keep_cols=keep_cols)
    if label_col not in usecols:
        usecols.append(label_col)

    rng = random.Random(seed)
    res: Dict[int, List[dict]] = {pos_label: [], neg_label: []}
    seen: Dict[int, int] = {pos_label: 0, neg_label: 0}

    for chunk in pd.read_csv(input_csv, usecols=usecols, chunksize=chunksize):
        y = pd.to_numeric(chunk[label_col], errors="coerce")
        for cls in (pos_label, neg_label):
            cls_chunk = chunk.loc[y == cls]
            if cls_chunk.empty:
                continue
            for row in cls_chunk.to_dict(orient="records"):
                seen[cls] += 1
                _reservoir_add(res[cls], row=row, seen_count=seen[cls], k=n_per_class, rng=rng)

    if len(res[pos_label]) == 0 or len(res[neg_label]) == 0:
        raise ValueError(
            f"No rows found for one or both classes: "
            f"{neg_label} -> {len(res[neg_label])}, {pos_label} -> {len(res[pos_label])}"
        )

    out_rows = res[neg_label] + res[pos_label]
    rng.shuffle(out_rows)
    out_df = pd.DataFrame(out_rows)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_csv, index=False)

    print("[DONE] Balanced example created")
    print(f"[INFO] Input: {input_csv}")
    print(f"[INFO] Output: {output_csv}")
    print(f"[INFO] Columns: {list(out_df.columns)}")
    print(f"[INFO] Sampled class counts: {label_col}={neg_label} -> {len(res[neg_label])}, {label_col}={pos_label} -> {len(res[pos_label])}")
    print(f"[INFO] Seen in source: {label_col}={neg_label} -> {seen[neg_label]}, {label_col}={pos_label} -> {seen[pos_label]}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create a small balanced example CSV from a very large input CSV "
            "using chunked streaming (memory efficient)."
        )
    )
    parser.add_argument("--input_csv", required=True, help="Path to large input CSV.")
    parser.add_argument(
        "--output_csv",
        default="data/example/example_windows.csv",
        help="Output path for example CSV. Default: data/example/example_windows.csv",
    )
    parser.add_argument("--label_col", default="Label_nuc", help="Binary label column. Default: Label_nuc")
    parser.add_argument("--neg_label", type=int, default=0, help="Negative class value. Default: 0")
    parser.add_argument("--pos_label", type=int, default=1, help="Positive class value. Default: 1")
    parser.add_argument(
        "--n_per_class",
        type=int,
        default=25,
        help="Number of rows to sample per class. Default: 25",
    )
    parser.add_argument("--chunksize", type=int, default=200000, help="CSV chunksize. Default: 200000")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed.")
    parser.add_argument(
        "--keep_cols",
        default="",
        help=(
            "Optional comma-separated list of columns to keep. "
            "If omitted, the script keeps common prediction-related columns if present."
        ),
    )
    args = parser.parse_args()

    make_balanced_example(
        input_csv=Path(args.input_csv),
        output_csv=Path(args.output_csv),
        label_col=args.label_col,
        pos_label=args.pos_label,
        neg_label=args.neg_label,
        n_per_class=args.n_per_class,
        chunksize=args.chunksize,
        seed=args.seed,
        keep_cols=_parse_keep_cols(args.keep_cols),
    )


if __name__ == "__main__":
    main()
