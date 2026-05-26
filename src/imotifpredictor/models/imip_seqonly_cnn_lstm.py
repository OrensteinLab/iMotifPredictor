from __future__ import annotations

import tensorflow as tf

DEFAULT_SEQ_LEN = 124


def build_hybrid_cnn_lstm(
    seq_len: int = DEFAULT_SEQ_LEN,
    filters1: int = 32,
    filters2: int = 64,
    kernel_size: int = 5,
    pool_size: int = 2,
    lstm_units: int = 128,
    dropout: float = 0.5,
    dense_units: int = 32,
    learning_rate: float = 1e-3,
) -> tf.keras.Model:
    """
    Hybrid CNN→LSTM sequence-only model for iM-IP-seq (iM propensity score).

    Architecture (paper):
      Input (124×4 one-hot)
      Conv1D(32,k=5,ReLU,same) → MaxPool(2)
      Conv1D(64,k=5,ReLU,same) → MaxPool(2)
      LSTM(128,tanh)
      Dropout(0.5)
      Dense(32,ReLU)
      Dense(1,sigmoid)
    """
    x_in = tf.keras.layers.Input(shape=(seq_len, 4), name="sequence_input")

    x = tf.keras.layers.Conv1D(filters1, kernel_size, activation="relu", padding="same", name="cnn_1")(x_in)
    x = tf.keras.layers.MaxPooling1D(pool_size=pool_size, name="pool_1")(x)

    x = tf.keras.layers.Conv1D(filters2, kernel_size, activation="relu", padding="same", name="cnn_2")(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=pool_size, name="pool_2")(x)

    x = tf.keras.layers.LSTM(lstm_units, activation="tanh", name="lstm")(x)
    x = tf.keras.layers.Dropout(dropout, name="dropout")(x)

    x = tf.keras.layers.Dense(dense_units, activation="relu", name="dense_32")(x)
    out = tf.keras.layers.Dense(1, activation="sigmoid", name="output")(x)

    model = tf.keras.Model(inputs=x_in, outputs=out, name="hybrid_cnn_lstm_imipseq")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model