from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import pandas as pd


DNA_BASES: Sequence[str] = ("A", "C", "G", "T")


@dataclass(frozen=True)
class MutationScanConfig:
    seq_len: int = 124
    seq_col: str = "Sequence"
    out_seq_col: str = "Mutated_Sequence"
    include_wt_rows: bool = True  # include rows where Original==Mutated


def _validate_sequence(seq: str, seq_len: int) -> str:
    s = str(seq).upper()
    if len(s) != seq_len:
        raise ValueError(f"Expected sequence length {seq_len}, got {len(s)}")
    return s


def generate_mutation_table(sequence: str, cfg: MutationScanConfig) -> pd.DataFrame:
    """
    Generate a mutation table for a single 124-nt sequence.

    Output columns (matching the manuscript plotting scripts):
      - Position (1-based)
      - Original_Nucleotide
      - Mutated_Nucleotide
      - Mutated_Sequence
    """
    seq = _validate_sequence(sequence, cfg.seq_len)

    rows: List[dict] = []

    for i in range(cfg.seq_len):
        orig = seq[i]

        # WT row (orig->orig) is useful as baseline for WT-MUT heatmaps
        if cfg.include_wt_rows:
            rows.append(
                {
                    "Position": i + 1,
                    "Original_Nucleotide": orig,
                    "Mutated_Nucleotide": orig,
                    cfg.out_seq_col: seq,
                }
            )

        for mut in DNA_BASES:
            if mut == orig:
                continue

            mutated = seq[:i] + mut + seq[i + 1 :]
            rows.append(
                {
                    "Position": i + 1,
                    "Original_Nucleotide": orig,
                    "Mutated_Nucleotide": mut,
                    cfg.out_seq_col: mutated,
                }
            )

    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate all single-nucleotide variants (SNVs) for a 124-nt sequence."
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--sequence", type=str, help="A single DNA sequence (length must match --seq_len).")
    src.add_argument("--input_csv", type=str, help="CSV containing a sequence column (default: Sequence).")

    p.add_argument("--seq_col", type=str, default="Sequence", help="Sequence column name for --input_csv.")
    p.add_argument("--out", type=str, required=True, help="Output CSV path.")
    p.add_argument("--seq_len", type=int, default=124, help="Sequence length (default: 124).")
    p.add_argument(
        "--no_wt_rows",
        action="store_true",
        help="Do not include WT baseline rows (Original_Nucleotide == Mutated_Nucleotide).",
    )
    p.add_argument(
        "--out_seq_col",
        type=str,
        default="Mutated_Sequence",
        help="Name of the mutated sequence column (default: Mutated_Sequence).",
    )
    return p.parse_args()


def main() -> None:
    a = parse_args()
    cfg = MutationScanConfig(
        seq_len=a.seq_len,
        seq_col=a.seq_col,
        out_seq_col=a.out_seq_col,
        include_wt_rows=(not a.no_wt_rows),
    )

    out_path = Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if a.sequence is not None:
        seq = _validate_sequence(a.sequence, cfg.seq_len)
    else:
        df = pd.read_csv(a.input_csv)
        if cfg.seq_col not in df.columns:
            raise ValueError(f"Missing column '{cfg.seq_col}' in {a.input_csv}")
        seq = _validate_sequence(df[cfg.seq_col].iloc[0], cfg.seq_len)

    mut_df = generate_mutation_table(seq, cfg)
    mut_df.to_csv(out_path, index=False)
    print(f"[DONE] Saved mutation table: {out_path}")
    print(f"[INFO] Rows: {len(mut_df):,} | Unique positions: {mut_df['Position'].nunique():,}")


if __name__ == "__main__":
    main()