from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


@dataclass
class SeqOnlyEvalReport:
    label_col: str
    num_rows: int
    positives: int
    negatives: int
    pos_rate: float


def _validate_binary_labels(df: pd.DataFrame, label_col: str) -> np.ndarray:
    if label_col not in df.columns:
        raise ValueError(f"Missing label column: {label_col}")
    y = pd.to_numeric(df[label_col], errors="coerce")
    mask = y.isin([0, 1])
    y = y.loc[mask].astype(int).to_numpy()
    if y.size == 0:
        raise ValueError(f"No valid binary labels in column '{label_col}'")
    return y


def compute_aupr(y_true: np.ndarray, y_score: np.ndarray, nan_fill: float = 0.0) -> float:
    """
    Compute AUPR as Average Precision (AP).
    This is the primary metric used for genome-wide evaluation under extreme class imbalance.
    """
    s = np.asarray(y_score, dtype=float)
    if np.isnan(s).any():
        s = np.nan_to_num(s, nan=nan_fill)
    return float(average_precision_score(y_true, s))


def evaluate_imip_seqonly_table(
    df: pd.DataFrame,
    label_col: str = "Label_nuc",
    exclude_cols: Sequence[str] = ("Chromosome", "Start", "End", "Sequence", "Label", "Label_nuc"),
    nan_fill: float = 0.0,
    only_numeric: bool = True,
    min_unique: int = 2,
) -> Tuple[pd.DataFrame, SeqOnlyEvalReport]:
    """
    Evaluate candidate prediction columns for the iM-IP-seq sequence-only setting and rank them by AUPR.

    Output columns:
      - Column
      - AUPR
      - Positives
      - Negatives
    """
    y = _validate_binary_labels(df, label_col)
    positives = int((y == 1).sum())
    negatives = int((y == 0).sum())

    rep = SeqOnlyEvalReport(
        label_col=label_col,
        num_rows=int(y.size),
        positives=positives,
        negatives=negatives,
        pos_rate=float(positives / (positives + negatives)),
    )

    rows: List[Dict[str, object]] = []
    for col in df.columns:
        if col == label_col or col in exclude_cols:
            continue
        if only_numeric and (not pd.api.types.is_numeric_dtype(df[col])):
            continue

        s = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)

        if pd.Series(s).nunique(dropna=True) < min_unique:
            continue

        aupr = compute_aupr(y, s, nan_fill=nan_fill)
        rows.append(
            {
                "Column": col,
                "AUPR": aupr,
                "Positives": positives,
                "Negatives": negatives,
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        out = pd.DataFrame(columns=["Column", "AUPR", "Positives", "Negatives"])
    else:
        out = out.sort_values("AUPR", ascending=False).reset_index(drop=True)

    return out, rep