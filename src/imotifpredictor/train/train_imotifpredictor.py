"""
Chunk-wise training loop for iMotifPredictor (LSTM + late fusion).

Behavior is aligned with the provided training script:
  - Iterate over chunk CSV files within a filename range (inclusive)
  - Optionally skip chunks containing chr1 (detected by reading the Chromosome column)
  - One-hot encode sequences of fixed length
  - Optionally include scalar features (epigenetic and/or iM-IP score) as separate inputs
  - Train per chunk for a fixed number of epochs
  - Support baseline weights or per-chunk class_weight sample weights
  - Save model as HDF5 (.h5)

Evaluation on chr1 is not performed in this module; this module performs training only.
"""

from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight

from imotifpredictor.constants import FEATURE_VARIANTS
from imotifpredictor.models.imotifpredictor_lstm_fusion import build_imotifpredictor_lstm_fusion


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


def select_files_in_range(directory: str, start_name: str, end_name: str, skip_chr1: bool = True) -> List[str]:
    """
    Select chunk files between start_name and end_name (inclusive), based on sorted filenames.
    Optionally skip chunk files containing chr1 (detected via Chromosome column).
    """
    all_files = sorted(os.listdir(directory))
    start_index = next(i for i, name in enumerate(all_files) if name == start_name)
    end_index = next(i for i, name in enumerate(all_files) if name == end_name) + 1
    candidate_files = all_files[start_index:end_index]

    if not skip_chr1:
        return candidate_files

    filtered: List[str] = []
    for fname in candidate_files:
        filepath = os.path.join(directory, fname)
        try:
            df_chr = pd.read_csv(filepath, usecols=["Chromosome"])
            chroms = df_chr["Chromosome"].astype(str).str.lower().unique()
            if "chr1" not in chroms:
                filtered.append(fname)
        except Exception:
            # If Chromosome cannot be read, keep behavior conservative: skip file.
            continue

    return filtered


def build_input_list(
    X_seq: np.ndarray,
    X_features: Optional[np.ndarray],
    feature_names: List[str],
):
    """
    Build model inputs in the same format as the original script:
      - Sequence-only: X_seq
      - Sequence + scalars: [X_seq, f1, f2, ...] where each f_i has shape (N,1)
    """
    if not feature_names:
        return X_seq

    if X_features is None:
        raise ValueError("Feature names provided but X_features is None.")

    return [X_seq] + [X_features[:, i].reshape(-1, 1) for i in range(len(feature_names))]


def compute_sample_weight(y: np.ndarray, balancing: str) -> np.ndarray:
    """
    Compute per-sample weights.
      - baseline: uniform weights (all ones)
      - class_weight: balanced weights computed from labels within the current chunk
    """
    balancing = balancing.lower()
    if balancing == "baseline":
        return np.ones_like(y, dtype=np.float32)

    if balancing != "class_weight":
        raise ValueError(f"Unknown balancing mode: {balancing}")

    if len(np.unique(y)) <= 1:
        return np.ones_like(y, dtype=np.float32)

    cw = compute_class_weight(class_weight="balanced", classes=np.unique(y), y=y)
    cw_map = {cls: w for cls, w in zip(np.unique(y), cw)}
    return np.array([cw_map[label] for label in y], dtype=np.float32)


@dataclass
class ChunkTrainConfig:
    directory: str
    output_dir: str
    start: str
    end: str

    feature_variant: str  # key in FEATURE_VARIANTS or "all"
    balancing: str        # "baseline" or "class_weight"

    seq_len: int = 124
    epochs_per_chunk: int = 1
    batch_size: int = 512

    shuffle_files: bool = False
    skip_chr1: bool = False

    lstm_units: int = 128
    dropout: float = 0.5
    dense_units: int = 32
    learning_rate: float = 1e-3


def run_chunk_training(cfg: ChunkTrainConfig) -> None:
    os.makedirs(cfg.output_dir, exist_ok=True)

    chunk_files = select_files_in_range(cfg.directory, cfg.start, cfg.end, skip_chr1=cfg.skip_chr1)
    if cfg.shuffle_files:
        random.shuffle(chunk_files)

    if cfg.feature_variant == "all":
        variants_to_train = FEATURE_VARIANTS
    else:
        if cfg.feature_variant not in FEATURE_VARIANTS:
            raise ValueError(
                f"Unknown feature_variant: {cfg.feature_variant}. "
                f"Valid options: {list(FEATURE_VARIANTS.keys())} or 'all'."
            )
        variants_to_train = {cfg.feature_variant: FEATURE_VARIANTS[cfg.feature_variant]}

    t_start = time.time()

    for variant_name, feature_list in variants_to_train.items():
        model = build_imotifpredictor_lstm_fusion(
            seq_len=cfg.seq_len,
            scalar_feature_names=feature_list,
            lstm_units=cfg.lstm_units,
            dropout=cfg.dropout,
            dense_units=cfg.dense_units,
            learning_rate=cfg.learning_rate,
        )

        trained = 0
        skipped = 0

        for fname in chunk_files:
            path = os.path.join(cfg.directory, fname)

            try:
                df = pd.read_csv(path, on_bad_lines="skip", engine="python")
            except Exception:
                skipped += 1
                continue

            required_cols = ["Sequence", "Label_nuc"] + feature_list
            if any(col not in df.columns for col in required_cols):
                skipped += 1
                continue

            seq_series = df["Sequence"].astype(str)
            label_series = pd.to_numeric(df["Label_nuc"], errors="coerce")
            valid_mask = (seq_series.str.len() == cfg.seq_len) & label_series.notna()

            feat_values: Optional[np.ndarray] = None
            if feature_list:
                feat_df = df[feature_list].apply(pd.to_numeric, errors="coerce")
                feat_values = feat_df.to_numpy(dtype=np.float32)
                valid_mask &= np.isfinite(feat_values).all(axis=1)

            if int(valid_mask.sum()) == 0:
                skipped += 1
                continue

            seqs = seq_series[valid_mask].tolist()
            y = label_series[valid_mask].astype(int).to_numpy()

            X_seq = one_hot_encode_sequences(seqs, seq_len=cfg.seq_len)
            X_features = feat_values[valid_mask] if feature_list and feat_values is not None else None

            idx = np.arange(len(y))
            np.random.shuffle(idx)
            X_seq = X_seq[idx]
            y = y[idx]
            if X_features is not None:
                X_features = X_features[idx]

            input_data = build_input_list(X_seq, X_features, feature_list)
            sample_weight = compute_sample_weight(y, cfg.balancing)

            model.fit(
                input_data,
                y,
                sample_weight=sample_weight,
                batch_size=cfg.batch_size,
                epochs=cfg.epochs_per_chunk,
                verbose=2,
                shuffle=True,
            )
            trained += 1

        tag = f"lstm_only_{variant_name}_{cfg.balancing}_{cfg.start}_to_{cfg.end}"
        save_path = os.path.join(cfg.output_dir, f"{tag}.h5")
        model.save(save_path, save_format="h5")

        # Summary output kept minimal and machine-readable
        print(f"variant={variant_name} trained_chunks={trained} skipped_chunks={skipped} model_path={save_path}")

    elapsed = time.time() - t_start
    print(f"total_seconds={elapsed:.2f}")