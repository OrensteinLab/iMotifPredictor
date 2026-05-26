"""
iMotifPredictor model definition.

Architecture:
  - One-hot DNA input (seq_len x 4)
  - LSTM(128, tanh) -> Dropout(0.5)
  - Late fusion by concatenating scalar features after the sequence representation
  - Dense(32, ReLU) -> Dense(1, sigmoid)

The model is compiled for binary classification.
"""

from __future__ import annotations

import tensorflow as tf
import numpy as np
import pandas as pd
import tensorflow as tf

from imotifpredictor.data.add_imip_score import one_hot_encode_sequences
from imotifpredictor.data.imotif_models import get_imotif_variant_specs, default_imotif_model_path


from pathlib import Path
from typing import Optional


def build_imotifpredictor_lstm_fusion(
    seq_len: int = 124,
    scalar_feature_names: list[str] | None = None,
    lstm_units: int = 128,
    dropout: float = 0.5,
    dense_units: int = 32,
    learning_rate: float = 1e-3,
) -> tf.keras.Model:
    scalar_feature_names = scalar_feature_names or []

    seq_in = tf.keras.layers.Input(shape=(seq_len, 4), name="sequence_input")
    x = tf.keras.layers.LSTM(lstm_units, activation="tanh", name="lstm")(seq_in)
    x = tf.keras.layers.Dropout(dropout, name="dropout")(x)

    scalar_inputs = [tf.keras.layers.Input(shape=(1,), name=f) for f in scalar_feature_names]
    if scalar_inputs:
        x = tf.keras.layers.Concatenate(name="late_fusion")([x] + scalar_inputs)

    x = tf.keras.layers.Dense(dense_units, activation="relu", name="dense_32")(x)
    out = tf.keras.layers.Dense(1, activation="sigmoid", name="output")(x)

    inputs = [seq_in] + scalar_inputs if scalar_inputs else seq_in
    model = tf.keras.Model(inputs=inputs, outputs=out, name="imotifpredictor_lstm_fusion")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc"),
        ],
    )
    return model

def predict_imotif_from_table(
    df: pd.DataFrame,
    variant: str,
    seq_col: str = "Sequence",
    seq_len: int = 124,
    batch_size: int = 512,
    repo_root: Optional[str] = None,
    model_path: Optional[str] = None,
    out_col: str = "pred_iMotifPredictor",
    nan_fill: float = 0.0,
) -> pd.DataFrame:
    """
    Predict iMotifPredictor score for a selected variant from an input table.

    For variants requiring numeric features, the corresponding columns must exist in df.
    """
    specs = get_imotif_variant_specs()
    if variant not in specs:
        raise ValueError(f"Unknown variant: {variant}. Allowed: {sorted(specs.keys())}")

    spec = specs[variant]
    if seq_col not in df.columns:
        raise ValueError(f"Missing sequence column '{seq_col}'")

    df = df.copy()
    df[seq_col] = df[seq_col].astype(str)
    df = df[df[seq_col].str.len() == seq_len].copy()
    if df.empty:
        raise ValueError("No rows with valid sequence length after filtering.")

    X_seq = one_hot_encode_sequences(df[seq_col].tolist(), seq_len=seq_len)

    feature_inputs = []
    for col in spec.required_numeric_features:
        if col not in df.columns:
            raise ValueError(f"Missing required feature column '{col}' for variant '{variant}'")
        x = pd.to_numeric(df[col], errors="coerce").fillna(nan_fill).to_numpy(dtype=np.float32)
        feature_inputs.append(x.reshape(-1, 1))

    if model_path:
        mp = Path(model_path)
    else:
        mp = Path(default_imotif_model_path(variant=variant, repo_root=repo_root))
    if not mp.exists():
        raise FileNotFoundError(f"Model not found: {mp}")

    model = tf.keras.models.load_model(str(mp))

    inputs = [X_seq] + feature_inputs if feature_inputs else X_seq
    y = model.predict(inputs, batch_size=batch_size, verbose=0).reshape(-1).astype(np.float32)

    df[out_col] = y
    return df