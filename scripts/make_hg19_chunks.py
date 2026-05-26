#!/usr/bin/env python3
from __future__ import annotations

import argparse

from imotifpredictor.data.hg19_chunking import chunk_fasta_windows


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Create chunked CSV files from a FASTA file containing genomic windows "
            "with headers formatted as chrom:start-end."
        )
    )
    p.add_argument("--fasta_file", required=True, help="Path to input FASTA windows file.")
    p.add_argument("--out_dir", required=True, help="Output directory for chunk CSV files.")
    p.add_argument("--rows_per_file", type=int, default=100000, help="Number of rows per chunk file.")
    p.add_argument("--out_prefix", default="chunk_", help="Output file prefix.")
    p.add_argument("--seq_len", type=int, default=124, help="Expected sequence length.")
    p.add_argument("--keep_n", action="store_true", help="Keep sequences containing 'N'.")
    args = p.parse_args()

    chunk_fasta_windows(
        fasta_file=args.fasta_file,
        out_dir=args.out_dir,
        rows_per_file=args.rows_per_file,
        out_prefix=args.out_prefix,
        seq_len=args.seq_len,
        skip_n=(not args.keep_n),
    )


if __name__ == "__main__":
    main()
