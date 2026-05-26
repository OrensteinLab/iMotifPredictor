from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterator, Tuple


HEADER_COLS = ["Chromosome", "Start", "End", "Sequence"]
LETTERS = "abcdefghijklmnopqrstuvwxyz"


def idx_to_code(i: int) -> str:
    """
    Convert a zero-based chunk index into a two-letter code: aa, ab, ..., zz.
    """
    if i < 0 or i >= 26 * 26:
        raise ValueError("Chunk index is out of the supported range [0, 675].")
    return LETTERS[i // 26] + LETTERS[i % 26]


def fasta_iter(path: Path) -> Iterator[Tuple[str, str]]:
    """
    Iterate over FASTA records and return tuples of (header_without_gt, sequence).
    """
    header = None
    seq_chunks = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(seq_chunks)
                header = line[1:]
                seq_chunks = []
            else:
                seq_chunks.append(line)

    if header is not None:
        yield header, "".join(seq_chunks)


def parse_hg_header(header: str) -> Tuple[str, int, int]:
    """
    Parse a FASTA header in the expected format: chrom:start-end.
    """
    chrom, coords = header.split(":")
    start, end = coords.split("-")
    return chrom, int(start), int(end)


def chunk_fasta_windows(
    fasta_file: str,
    out_dir: str,
    rows_per_file: int = 100_000,
    out_prefix: str = "chunk_",
    seq_len: int = 124,
    skip_n: bool = True,
) -> int:
    """
    Convert a FASTA file of fixed windows into chunked CSV files.

    Each output file contains:
      Chromosome, Start, End, Sequence
    """
    fasta_path = Path(fasta_file)
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    file_idx = 0
    row_in_current = 0
    out_f = None
    writer = None

    def open_new_chunk(idx: int):
        code = idx_to_code(idx)
        fname = output_dir / f"{out_prefix}{code}.csv"
        f = fname.open("w", newline="", encoding="utf-8")
        w = csv.writer(f)
        w.writerow(HEADER_COLS)
        print(f"[INFO] Opened {fname}")
        return f, w

    try:
        for h, seq in fasta_iter(fasta_path):
            seq = seq.upper()
            if len(seq) != seq_len:
                continue
            if skip_n and ("N" in seq):
                continue

            if writer is None or row_in_current >= rows_per_file:
                if out_f is not None:
                    out_f.close()
                out_f, writer = open_new_chunk(file_idx)
                file_idx += 1
                row_in_current = 0

            chrom, start, end = parse_hg_header(h)
            writer.writerow([chrom, start, end, seq])
            row_in_current += 1
    finally:
        if out_f is not None:
            out_f.close()

    print(f"[DONE] Created {file_idx} chunk files in {output_dir}")
    return file_idx
