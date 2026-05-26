from __future__ import annotations

import json
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import average_precision_score

from imotifpredictor.models.imip_seqonly_cnn_lstm import build_hybrid_cnn_lstm, DEFAULT_SEQ_LEN

REQUIRED_COLUMNS = ["Sequence", "Label_nuc"]


def set_global_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def ensure_dir(path: Union[str, Path]) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def list_csv_files(input_dir: Union[str, Path]) -> List[Path]:
    d = Path(input_dir)
    if not d.exists():
        raise FileNotFoundError(f"input_dir does not exist: {d}")
    files = sorted([p for p in d.iterdir() if p.suffix.lower() == ".csv"])
    if not files:
        raise FileNotFoundError(f"No .csv files found in: {d}")
    return files


def one_hot_encode_sequences(seqs: Iterable[str], seq_len: int) -> np.ndarray:
    seqs = list(seqs)
    if any(len(s) != seq_len for s in seqs):
        bad = [i for i, s in enumerate(seqs) if len(s) != seq_len][:5]
        raise ValueError(f"All sequences must have length {seq_len}. Bad indices (first 5): {bad}")

    mapping = {"A": 0, "C": 1, "G": 2, "T": 3}
    X = np.zeros((len(seqs), seq_len, 4), dtype=np.float32)
    for i, seq in enumerate(seqs):
        for j, base in enumerate(seq.upper()):
            idx = mapping.get(base)
            if idx is not None:
                X[i, j, idx] = 1.0
    return X


def validate_dataframe(df: pd.DataFrame, require_chromosome: bool) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if require_chromosome and "Chromosome" not in df.columns:
        raise ValueError("Chromosome filtering requested, but 'Chromosome' column is missing.")


def filter_by_chromosome(df: pd.DataFrame, chrom: Optional[str]) -> pd.DataFrame:
    if chrom is None:
        return df
    c = chrom.lower()
    return df[df["Chromosome"].astype(str).str.lower() == c]


def make_train_test_from_df(
    df: pd.DataFrame,
    seq_len: int,
    train_chrom: Optional[str],
    test_chrom: Optional[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    require_chr = (train_chrom is not None) or (test_chrom is not None)
    validate_dataframe(df, require_chromosome=require_chr)

    seqs = df["Sequence"].astype(str)
    labels = pd.to_numeric(df["Label_nuc"], errors="coerce")
    valid = (seqs.str.len() == seq_len) & labels.notna()
    df = df.loc[valid].copy()
    if df.empty:
        raise ValueError("No valid rows after filtering by seq_len and label parsing.")

    if require_chr:
        train_df = filter_by_chromosome(df, train_chrom)
        test_df = filter_by_chromosome(df, test_chrom)
        if train_df.empty:
            raise ValueError(f"No rows found for train_chrom={train_chrom}")
        if test_df.empty:
            raise ValueError(f"No rows found for test_chrom={test_chrom}")

        X_train = one_hot_encode_sequences(train_df["Sequence"], seq_len)
        y_train = train_df["Label_nuc"].astype(int).to_numpy()
        X_test = one_hot_encode_sequences(test_df["Sequence"], seq_len)
        y_test = test_df["Label_nuc"].astype(int).to_numpy()
        return X_train, y_train, X_test, y_test

    idx = np.arange(len(df))
    np.random.shuffle(idx)
    split = int(0.8 * len(idx))
    tr, te = idx[:split], idx[split:]

    X_train = one_hot_encode_sequences(df.iloc[tr]["Sequence"], seq_len)
    y_train = df.iloc[tr]["Label_nuc"].astype(int).to_numpy()
    X_test = one_hot_encode_sequences(df.iloc[te]["Sequence"], seq_len)
    y_test = df.iloc[te]["Label_nuc"].astype(int).to_numpy()
    return X_train, y_train, X_test, y_test


def compute_sample_weights(y: np.ndarray, mode: str) -> np.ndarray:
    mode = mode.lower()
    if mode == "baseline":
        return np.ones_like(y, dtype=np.float32)
    if mode != "class_weight":
        raise ValueError(f"Unknown balancing mode: {mode}")

    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2:
        return np.ones_like(y, dtype=np.float32)

    n = len(y)
    k = len(classes)
    w = {int(c): float(n) / (k * int(cnt)) for c, cnt in zip(classes, counts)}
    return np.array([w[int(lbl)] for lbl in y], dtype=np.float32)


def evaluate_aupr(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    y_true = y_true.astype(int)
    y_prob = y_prob.astype(float)
    return {
        "pos_rate": float(np.mean(y_true)),
        "aupr": float(average_precision_score(y_true, y_prob)),
    }


@dataclass
class ImipSeqOnlyRunConfig:
    input_csv: Optional[str]
    input_dir: Optional[str]
    output_dir: str

    seq_len: int = DEFAULT_SEQ_LEN
    train_chrom: Optional[str] = None
    test_chrom: Optional[str] = None

    balancing: str = "baseline"
    epochs: int = 1
    batch_size: int = 512
    shuffle_files: bool = False

    seed: int = 1337

    filters1: int = 32
    filters2: int = 64
    kernel_size: int = 5
    pool_size: int = 2
    lstm_units: int = 128
    dropout: float = 0.5
    dense_units: int = 32
    learning_rate: float = 1e-3


def load_dataframe(cfg: ImipSeqOnlyRunConfig) -> pd.DataFrame:
    if cfg.input_csv:
        path = Path(cfg.input_csv)
        if not path.exists():
            raise FileNotFoundError(f"input_csv does not exist: {path}")
        return pd.read_csv(path, on_bad_lines="skip", engine="python")

    assert cfg.input_dir is not None
    files = list_csv_files(cfg.input_dir)
    if cfg.shuffle_files:
        random.shuffle(files)

    dfs = []
    for f in files:
        try:
            dfs.append(pd.read_csv(f, on_bad_lines="skip", engine="python"))
        except Exception as e:
            print(f"[WARN] Failed reading {f.name}: {e}")

    if not dfs:
        raise RuntimeError("No readable CSV files were loaded.")
    return pd.concat(dfs, ignore_index=True)


def run_train_imip_seqonly(cfg: ImipSeqOnlyRunConfig) -> Path:
    set_global_seed(cfg.seed)
    outdir = ensure_dir(cfg.output_dir)

    (outdir / "config.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")

    t0 = time.time()
    print("[INFO] Loading data...")
    df = load_dataframe(cfg)
    print(f"[INFO] Loaded {len(df):,} rows")

    print("[INFO] Creating train/test split...")
    X_train, y_train, X_test, y_test = make_train_test_from_df(
        df=df,
        seq_len=cfg.seq_len,
        train_chrom=cfg.train_chrom,
        test_chrom=cfg.test_chrom,
    )

    print(
        f"[INFO] Train N={len(y_train):,} pos={int((y_train==1).sum()):,} neg={int((y_train==0).sum()):,} | "
        f"Test N={len(y_test):,} pos={int((y_test==1).sum()):,} neg={int((y_test==0).sum()):,}"
    )

    sw = compute_sample_weights(y_train, cfg.balancing)

    print("[INFO] Building model...")
    model = build_hybrid_cnn_lstm(
        seq_len=cfg.seq_len,
        filters1=cfg.filters1,
        filters2=cfg.filters2,
        kernel_size=cfg.kernel_size,
        pool_size=cfg.pool_size,
        lstm_units=cfg.lstm_units,
        dropout=cfg.dropout,
        dense_units=cfg.dense_units,
        learning_rate=cfg.learning_rate,
    )

    print("[INFO] Training...")
    history = model.fit(
        X_train,
        y_train,
        sample_weight=sw,
        validation_data=(X_test, y_test),
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        shuffle=True,
        verbose=2,
    )
    pd.DataFrame(history.history).to_csv(outdir / "history.csv", index=False)

    print("[INFO] Evaluating (AUPR)...")
    y_prob = model.predict(X_test, batch_size=cfg.batch_size, verbose=0).reshape(-1)
    metrics = evaluate_aupr(y_test, y_prob)
    (outdir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # ✅ SAVE AS H5 (to match your existing models)
    model_path = outdir / "model.h5"
    model.save(model_path, save_format="h5")

    elapsed = time.time() - t0
    print(f"[DONE] Saved model: {model_path}")
    print(f"[DONE] Metrics: {json.dumps(metrics, indent=2)}")
    print(f"[DONE] Total time: {elapsed:.2f} sec")
    return model_path