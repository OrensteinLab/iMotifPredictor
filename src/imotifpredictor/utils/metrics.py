from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve


@dataclass
class MetricRow:
    column: str
    aupr: float
    num_pos: int
    num_neg: int


def _make_valid_binary_mask(label_series: pd.Series) -> pd.Series:
    """
    Return a boolean mask selecting rows with valid binary labels {0,1}.
    """
    y = pd.to_numeric(label_series, errors="coerce")
    return y.isin([0, 1])


def compute_aupr_for_scores(
    y_true: np.ndarray,
    y_score: np.ndarray,
    nan_fill: float = 0.0,
) -> float:
    """
    Compute AUPR as Average Precision (AP).
    """
    s = np.asarray(y_score, dtype=float)
    if np.isnan(s).any():
        s = np.nan_to_num(s, nan=nan_fill)
    return float(average_precision_score(y_true, s))


def compute_pr_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    nan_fill: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (precision, recall) arrays for plotting a PR curve.
    """
    s = np.asarray(y_score, dtype=float)
    if np.isnan(s).any():
        s = np.nan_to_num(s, nan=nan_fill)
    precision, recall, _ = precision_recall_curve(y_true, s)
    return precision, recall


def rank_prediction_columns_by_aupr(
    df: pd.DataFrame,
    label_col: str,
    exclude_cols: Sequence[str],
    only_numeric: bool = True,
    min_unique: int = 2,
    nan_fill: float = 0.0,
) -> pd.DataFrame:
    """
    Compute AUPR (Average Precision) for each candidate prediction column in df,
    excluding metadata columns and the label column.

    The function aligns labels and scores by applying a shared valid-row mask
    based on the label column containing only {0,1} values.

    Output columns:
      - Column
      - AUPR
      - Positives
      - Negatives
    """
    if label_col not in df.columns:
        raise ValueError(f"label_col '{label_col}' not found in DataFrame")

    valid_mask = _make_valid_binary_mask(df[label_col])
    if int(valid_mask.sum()) == 0:
        raise ValueError(f"No valid binary labels found in '{label_col}'")

    dfv = df.loc[valid_mask].copy()
    y_true = pd.to_numeric(dfv[label_col], errors="coerce").astype(int).to_numpy()

    num_pos = int((y_true == 1).sum())
    num_neg = int((y_true == 0).sum())

    rows: List[Dict[str, object]] = []

    for col in dfv.columns:
        if col == label_col or col in exclude_cols:
            continue

        if only_numeric and (not pd.api.types.is_numeric_dtype(dfv[col])):
            continue

        s = pd.to_numeric(dfv[col], errors="coerce").to_numpy(dtype=float)

        # Skip constant/near-constant columns
        if pd.Series(s).nunique(dropna=True) < min_unique:
            continue

        aupr = compute_aupr_for_scores(y_true, s, nan_fill=nan_fill)
        rows.append(
            {
                "Column": col,
                "AUPR": aupr,
                "Positives": num_pos,
                "Negatives": num_neg,
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=["Column", "AUPR", "Positives", "Negatives"])

    return out.sort_values("AUPR", ascending=False).reset_index(drop=True)