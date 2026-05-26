from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Tuple

import pandas as pd


InputFormat = Literal["auto", "csv", "xlsx", "fasta"]


@dataclass
class LoadSequencesReport:
    total_rows: int
    kept_rows: int
    dropped_rows: int
    seq_len: int
    input_format: str


def _detect_format(path: Path) -> InputFormat:
    suf = path.suffix.lower()
    if suf in [".csv"]:
        return "csv"
    if suf in [".xlsx", ".xls"]:
        return "xlsx"
    if suf in [".fa", ".fasta", ".fna"]:
        return "fasta"
    raise ValueError(f"Cannot infer format from suffix '{path.suffix}'. Use --format explicitly.")


def _read_fasta(path: Path) -> pd.DataFrame:
    """
    Minimal FASTA reader.
    Returns DataFrame with columns: ID, Sequence
    """
    records = []
    cur_id = None
    cur_seq_parts = []

    def flush():
        nonlocal cur_id, cur_seq_parts
        if cur_id is None:
            return
        seq = "".join(cur_seq_parts).strip()
        records.append({"ID": cur_id, "Sequence": seq})
        cur_id = None
        cur_seq_parts = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                flush()
                cur_id = line[1:].strip() or "record"
                cur_seq_parts = []
            else:
                cur_seq_parts.append(line)
        flush()

    return pd.DataFrame(records)


def load_sequences_table(
    input_path: str,
    fmt: InputFormat = "auto",
    seq_col: str = "Sequence",
    id_col: Optional[str] = None,
    seq_len: int = 124,
    drop_invalid_len: bool = True,
) -> Tuple[pd.DataFrame, LoadSequencesReport]:
    """
    Load sequences from CSV / XLSX / FASTA into a DataFrame with a 'Sequence' column.

    Behavior:
      - If drop_invalid_len=True, rows whose Sequence length != seq_len are dropped.
      - For FASTA, output columns are: ID, Sequence.
      - For CSV/XLSX, Sequence is read from seq_col; if id_col provided and exists, it is preserved as 'ID'.
    """
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Input not found: {path}")

    if fmt == "auto":
        fmt = _detect_format(path)

    if fmt == "csv":
        df = pd.read_csv(path, on_bad_lines="skip", engine="python")
        if seq_col not in df.columns:
            raise ValueError(f"Missing sequence column '{seq_col}' in CSV.")
        out = pd.DataFrame({"Sequence": df[seq_col].astype(str)})
        if id_col and id_col in df.columns:
            out.insert(0, "ID", df[id_col].astype(str))
        input_format = "csv"

    elif fmt == "xlsx":
        df = pd.read_excel(path)  # requires openpyxl for .xlsx
        if seq_col not in df.columns:
            raise ValueError(f"Missing sequence column '{seq_col}' in XLSX.")
        out = pd.DataFrame({"Sequence": df[seq_col].astype(str)})
        if id_col and id_col in df.columns:
            out.insert(0, "ID", df[id_col].astype(str))
        input_format = "xlsx"

    elif fmt == "fasta":
        out = _read_fasta(path)
        if out.empty:
            raise ValueError("No FASTA records were found.")
        input_format = "fasta"

    else:
        raise ValueError(f"Unsupported format: {fmt}")

    total = int(len(out))
    if drop_invalid_len:
        mask = out["Sequence"].astype(str).str.len() == int(seq_len)
        out = out.loc[mask].copy()

    kept = int(len(out))
    dropped = total - kept

    report = LoadSequencesReport(
        total_rows=total,
        kept_rows=kept,
        dropped_rows=dropped,
        seq_len=int(seq_len),
        input_format=input_format,
    )
    return out, report