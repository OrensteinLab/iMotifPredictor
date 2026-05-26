from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd
import tensorflow as tf


def one_hot_encode_sequences(seqs: List[str], seq_len: int = 124) -> np.ndarray:
    """
    One-hot encode DNA sequences into (N, seq_len, 4).

    Mapping:
      A -> [1,0,0,0]
      C -> [0,1,0,0]
      G -> [0,0,1,0]
      T -> [0,0,0,1]
      Non-ACGT -> [0,0,0,0]
    """
    if any(len(s) != seq_len for s in seqs):
        raise ValueError(f"All sequences must have length {seq_len}.")

    mapping = {"A": 0, "C": 1, "G": 2, "T": 3}
    X = np.zeros((len(seqs), seq_len, 4), dtype=np.float32)
    for i, seq in enumerate(seqs):
        for j, base in enumerate(seq.upper()):
            idx = mapping.get(base)
            if idx is not None:
                X[i, j, idx] = 1.0
    return X


def default_imip_model_paths(repo_root: Optional[str] = None) -> List[str]:
    """
    Return default iM-IP model paths (HEK/U2OS/MCF7) relative to the repository root.

    Expected layout:
      models/iM-IP/
        - iM-IP_HEK_hybrid_cnn_lstm_baseline.h5
        - iM-IP_U2OS_hybrid_cnn_lstm_baseline.h5
        - iM-IP_MCF7_hybrid_cnn_lstm_baseline.h5
    """
    root = Path(repo_root) if repo_root is not None else Path.cwd()
    base = root / "models" / "iM-IP"
    return [
        str(base / "iM-IP_HEK_hybrid_cnn_lstm_baseline.h5"),
        str(base / "iM-IP_U2OS_hybrid_cnn_lstm_baseline.h5"),
        str(base / "iM-IP_MCF7_hybrid_cnn_lstm_baseline.h5"),
    ]


@dataclass
class AddImipScoreConfig:
    input_csv: str
    output_csv: str
    model_paths: List[str]

    seq_col: str = "Sequence"
    seq_len: int = 124
    batch_size: int = 512
    out_col: str = "pred_hybrid_cnn_lstm_mean"


def _validate_sequence_column(df: pd.DataFrame, seq_col: str) -> pd.Series:
    if seq_col not in df.columns:
        raise ValueError(f"Missing sequence column: {seq_col}")
    return df[seq_col].astype(str)


def _resolve_model_paths(model_paths: Optional[Sequence[str]], repo_root: Optional[str]) -> List[str]:
    if model_paths is None:
        resolved = default_imip_model_paths(repo_root=repo_root)
    else:
        resolved = [str(p) for p in model_paths]

    if len(resolved) == 0:
        raise ValueError("No iM-IP model paths were provided.")

    for p in resolved:
        if not Path(p).exists():
            raise FileNotFoundError(f"Model not found: {p}")
    return resolved


def add_imip_predictions(
    df: pd.DataFrame,
    seq_col: str = "Sequence",
    seq_len: int = 124,
    batch_size: int = 512,
    model_paths: Optional[Sequence[str]] = None,
    repo_root: Optional[str] = None,
    prefix: str = "pred_iM-IP",
    add_mean: bool = True,
    mean_col: str = "pred_hybrid_cnn_lstm_mean",
    source_cols: Optional[Sequence[str]] = None,
    nan_fill: float = 0.0,
) -> pd.DataFrame:
    """
    Add iM-IP prediction columns to an in-memory table.

    Two modes are supported:
      1) `source_cols` mode: compute the mean score from existing columns in `df`.
      2) model-inference mode: run pretrained iM-IP models, add per-model columns, and
         optionally add an ensemble mean column.

    Notes:
      - Rows with invalid sequence length are dropped.
      - Numeric source columns are parsed with `to_numeric(..., errors="coerce")`
        and missing values are replaced by `nan_fill`.
    """
    out = df.copy()
    seqs = _validate_sequence_column(out, seq_col=seq_col)
    valid_mask = seqs.str.len() == seq_len
    out = out.loc[valid_mask].copy()
    if out.empty:
        raise ValueError("No valid rows after filtering by sequence length.")

    if source_cols is not None and len(source_cols) > 0:
        missing = [c for c in source_cols if c not in out.columns]
        if missing:
            raise ValueError(f"Missing source columns for iM-IP mean: {missing}")

        vals = []
        for c in source_cols:
            s = pd.to_numeric(out[c], errors="coerce").fillna(nan_fill).to_numpy(dtype=np.float32)
            vals.append(s)
        if add_mean:
            out[mean_col] = np.mean(np.stack(vals, axis=0), axis=0)
        return out

    paths = _resolve_model_paths(model_paths=model_paths, repo_root=repo_root)
    X = one_hot_encode_sequences(out[seq_col].tolist(), seq_len=seq_len)

    preds = []
    colnames = []
    for mp in paths:
        model = tf.keras.models.load_model(mp)
        y = model.predict(X, batch_size=batch_size, verbose=0).reshape(-1).astype(np.float32)
        preds.append(y)
        colnames.append(f"{prefix}_{Path(mp).stem}")

    for c, y in zip(colnames, preds):
        out[c] = y

    if add_mean:
        out[mean_col] = np.mean(np.stack(preds, axis=0), axis=0)
    return out


def add_imip_score(cfg: AddImipScoreConfig) -> None:
    """
    Add an iM-IP-derived scalar score column to a CSV file.
    """
    df = pd.read_csv(cfg.input_csv, on_bad_lines="skip", engine="python")
    df = add_imip_predictions(
        df=df,
        seq_col=cfg.seq_col,
        seq_len=cfg.seq_len,
        batch_size=cfg.batch_size,
        model_paths=cfg.model_paths,
        repo_root=None,
        prefix="pred_iM-IP",
        add_mean=True,
        mean_col=cfg.out_col,
    )

    out_path = Path(cfg.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
