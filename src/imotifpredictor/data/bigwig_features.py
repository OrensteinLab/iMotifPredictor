from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd

try:
    import pyBigWig
except ImportError as exc:  # pragma: no cover
    pyBigWig = None
    _PYBIGWIG_IMPORT_ERROR = exc
else:
    _PYBIGWIG_IMPORT_ERROR = None


# Common case: window files use chrN naming, but some bigWig tracks use NCBI accessions.
HG38_TO_NCBI_CHROM_MAP: Dict[str, str] = {
    "chr1": "NC_000001.11",
    "chr2": "NC_000002.12",
    "chr3": "NC_000003.12",
    "chr4": "NC_000004.12",
    "chr5": "NC_000005.10",
    "chr6": "NC_000006.12",
    "chr7": "NC_000007.14",
    "chr8": "NC_000008.11",
    "chr9": "NC_000009.12",
    "chr10": "NC_000010.11",
    "chr11": "NC_000011.10",
    "chr12": "NC_000012.12",
    "chr13": "NC_000013.11",
    "chr14": "NC_000014.9",
    "chr15": "NC_000015.10",
    "chr16": "NC_000016.10",
    "chr17": "NC_000017.11",
    "chr18": "NC_000018.10",
    "chr19": "NC_000019.10",
    "chr20": "NC_000020.11",
    "chr21": "NC_000021.9",
    "chr22": "NC_000022.11",
    "chrX": "NC_000023.11",
    "chrY": "NC_000024.10",
    "chrM": "NC_012920.1",
}


def load_chrom_mapping(mapping_json: Optional[str], mapping_name: str = "none") -> Dict[str, str]:
    if mapping_json:
        with open(mapping_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError("Chromosome mapping JSON must be an object: {\"from\": \"to\", ...}")
        return {str(k): str(v) for k, v in data.items()}

    if mapping_name == "none":
        return {}
    if mapping_name == "hg38_to_ncbi":
        return dict(HG38_TO_NCBI_CHROM_MAP)
    raise ValueError(f"Unknown mapping_name: {mapping_name}")


def _ensure_pybigwig() -> None:
    if pyBigWig is None:
        raise ImportError(
            "pyBigWig is required for bigWig feature extraction. "
            "Install it with: pip install pyBigWig"
        ) from _PYBIGWIG_IMPORT_ERROR


def _mean_signal(bw, chrom: str, start: int, end: int, nan_fill: float) -> float:
    if start >= end:
        return nan_fill
    vals = np.asarray(bw.values(chrom, int(start), int(end)), dtype=float)
    if vals.size == 0:
        return nan_fill
    if np.isnan(vals).all():
        return nan_fill
    return float(np.nanmean(vals))


def add_bigwig_feature_to_df(
    df: pd.DataFrame,
    bigwig_path: str,
    feature_col: str,
    chrom_col: str = "Chromosome",
    start_col: str = "Start",
    end_col: str = "End",
    chrom_mapping: Optional[Dict[str, str]] = None,
    nan_fill: float = 0.0,
    unknown_chrom_fill: float = 0.0,
) -> pd.DataFrame:
    """
    Add a scalar feature column by averaging bigWig signal over [start, end) windows.
    """
    _ensure_pybigwig()
    chrom_mapping = chrom_mapping or {}
    out = df.copy()

    for c in [chrom_col, start_col, end_col]:
        if c not in out.columns:
            raise ValueError(f"Missing required column: {c}")

    bw = pyBigWig.open(str(bigwig_path))
    try:
        bw_chroms = set(bw.chroms().keys())
        feature_vals = []

        for _, row in out.iterrows():
            chrom = str(row[chrom_col])
            start = int(row[start_col])
            end = int(row[end_col])

            mapped = chrom_mapping.get(chrom, chrom)
            if mapped not in bw_chroms:
                feature_vals.append(float(unknown_chrom_fill))
                continue

            feature_vals.append(_mean_signal(bw, mapped, start, end, nan_fill=nan_fill))

        out[feature_col] = np.asarray(feature_vals, dtype=np.float32)
    finally:
        bw.close()

    return out


def add_bigwig_feature_to_csv(
    input_csv: str,
    output_csv: str,
    bigwig_path: str,
    feature_col: str,
    chrom_col: str = "Chromosome",
    start_col: str = "Start",
    end_col: str = "End",
    chrom_mapping: Optional[Dict[str, str]] = None,
    nan_fill: float = 0.0,
    unknown_chrom_fill: float = 0.0,
) -> None:
    df = pd.read_csv(input_csv)
    out = add_bigwig_feature_to_df(
        df=df,
        bigwig_path=bigwig_path,
        feature_col=feature_col,
        chrom_col=chrom_col,
        start_col=start_col,
        end_col=end_col,
        chrom_mapping=chrom_mapping,
        nan_fill=nan_fill,
        unknown_chrom_fill=unknown_chrom_fill,
    )
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)
