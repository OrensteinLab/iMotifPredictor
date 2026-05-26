#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import fnmatch
import hashlib
from pathlib import Path
from typing import Iterable, Iterator


def iter_files(root: Path, include: str, exclude: str | None) -> Iterator[Path]:
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(root).as_posix()
        if not fnmatch.fnmatch(rel, include):
            continue
        if exclude and fnmatch.fnmatch(rel, exclude):
            continue
        yield p


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(chunk_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def build_manifest(
    root: Path,
    include: str,
    exclude: str | None,
    out_tsv: Path,
    out_checksums: Path,
) -> int:
    files = list(iter_files(root=root, include=include, exclude=exclude))
    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    out_checksums.parent.mkdir(parents=True, exist_ok=True)

    with out_tsv.open("w", newline="", encoding="utf-8") as f_tsv, out_checksums.open(
        "w", encoding="utf-8"
    ) as f_sum:
        writer = csv.writer(f_tsv, delimiter="\t")
        writer.writerow(["relative_path", "size_bytes", "sha256"])

        for p in files:
            rel = p.relative_to(root).as_posix()
            size = p.stat().st_size
            digest = sha256_file(p)
            writer.writerow([rel, size, digest])
            f_sum.write(f"{digest}  {rel}\n")

    return len(files)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a reproducibility manifest (TSV + checksums.txt) for large datasets "
            "stored outside the repository."
        )
    )
    parser.add_argument("--root", required=True, help="Dataset root directory.")
    parser.add_argument(
        "--include",
        default="*.csv",
        help="Glob pattern (relative path) for files to include. Default: *.csv",
    )
    parser.add_argument(
        "--exclude",
        default="",
        help="Optional glob pattern (relative path) to exclude.",
    )
    parser.add_argument(
        "--out_tsv",
        default="data/manifest.tsv",
        help="Output TSV path. Default: data/manifest.tsv",
    )
    parser.add_argument(
        "--out_checksums",
        default="data/checksums.sha256",
        help="Output checksum file path. Default: data/checksums.sha256",
    )
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist or is not a directory: {root}")

    n = build_manifest(
        root=root,
        include=args.include,
        exclude=(args.exclude.strip() or None),
        out_tsv=Path(args.out_tsv),
        out_checksums=Path(args.out_checksums),
    )
    print(f"[DONE] Files indexed: {n}")
    print(f"[DONE] Manifest: {args.out_tsv}")
    print(f"[DONE] Checksums: {args.out_checksums}")


if __name__ == "__main__":
    main()
