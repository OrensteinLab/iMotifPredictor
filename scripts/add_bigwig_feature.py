#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from imotifpredictor.data.bigwig_features import add_bigwig_feature_to_csv, load_chrom_mapping


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Add one epigenetic feature column to CSV window files by averaging "
            "signal from a bigWig track over [Start, End) coordinates."
        )
    )

    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--input_csv", type=str, help="Input CSV path.")
    src.add_argument("--input_dir", type=str, help="Input directory containing chunk CSV files.")

    p.add_argument("--bigwig", required=True, help="bigWig file path.")
    p.add_argument("--feature_col", required=True, help="Output feature column name (e.g., H3K9me3).")

    p.add_argument("--output_csv", default="", help="Output CSV path for --input_csv mode.")
    p.add_argument("--output_dir", default="", help="Output directory for --input_dir mode.")
    p.add_argument("--in_place", action="store_true", help="Overwrite input file(s).")
    p.add_argument("--glob", default="chunk_*.csv", help="File pattern for --input_dir mode.")

    p.add_argument("--chrom_col", default="Chromosome")
    p.add_argument("--start_col", default="Start")
    p.add_argument("--end_col", default="End")

    p.add_argument(
        "--chrom_map",
        choices=["none", "hg38_to_ncbi"],
        default="none",
        help="Built-in chromosome mapping name.",
    )
    p.add_argument(
        "--chrom_map_json",
        default="",
        help="Optional JSON mapping file. Overrides --chrom_map.",
    )
    p.add_argument("--nan_fill", type=float, default=0.0, help="Fill value for NaN-only windows.")
    p.add_argument("--unknown_chrom_fill", type=float, default=0.0, help="Fill value for missing chromosomes.")

    args = p.parse_args()
    mapping = load_chrom_mapping(
        mapping_json=(args.chrom_map_json.strip() or None),
        mapping_name=args.chrom_map,
    )

    if args.input_csv:
        in_csv = Path(args.input_csv)
        if args.in_place:
            out_csv = in_csv
        else:
            out_csv = Path(args.output_csv) if args.output_csv.strip() else in_csv.with_suffix(".with_feature.csv")

        add_bigwig_feature_to_csv(
            input_csv=str(in_csv),
            output_csv=str(out_csv),
            bigwig_path=args.bigwig,
            feature_col=args.feature_col,
            chrom_col=args.chrom_col,
            start_col=args.start_col,
            end_col=args.end_col,
            chrom_mapping=mapping,
            nan_fill=args.nan_fill,
            unknown_chrom_fill=args.unknown_chrom_fill,
        )
        print(f"[DONE] Updated: {out_csv}")
        return

    # input_dir mode
    in_dir = Path(args.input_dir)
    files = sorted(in_dir.glob(args.glob))
    if not files:
        raise FileNotFoundError(f"No files matched '{args.glob}' in {in_dir}")

    if (not args.in_place) and (not args.output_dir.strip()):
        raise ValueError("For --input_dir mode, use either --in_place or provide --output_dir.")

    out_dir = Path(args.output_dir) if args.output_dir.strip() else in_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    for fp in files:
        out_csv = fp if args.in_place else (out_dir / fp.name)
        add_bigwig_feature_to_csv(
            input_csv=str(fp),
            output_csv=str(out_csv),
            bigwig_path=args.bigwig,
            feature_col=args.feature_col,
            chrom_col=args.chrom_col,
            start_col=args.start_col,
            end_col=args.end_col,
            chrom_mapping=mapping,
            nan_fill=args.nan_fill,
            unknown_chrom_fill=args.unknown_chrom_fill,
        )
        print(f"[OK] {fp.name} -> {out_csv}")

    print(f"[DONE] Processed {len(files)} files.")


if __name__ == "__main__":
    main()
