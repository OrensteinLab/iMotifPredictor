from __future__ import annotations

import argparse
import datetime as _dt
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

# server/no display
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import load_model

import logomaker


BASES = ("A", "C", "G", "T")


@dataclass(frozen=True)
class IGConfig:
    seq_len: int = 124
    steps: int = 50
    target_class_idx: int = 0  # used only if model output has >1 columns
    window_seq: int = 0  # 0 => full-length IG; >0 => sliding window IG
    font_name: str = "Arial Rounded MT Bold"
    figsize: Tuple[int, int] = (15, 5)
    dpi: int = 300


def safe_name(s: str) -> str:
    s = str(s).strip().lower()
    s = re.sub(r"[^a-z0-9._-]+", "_", s)
    return s[:120] if len(s) > 120 else s


def one_hot_encode(sequence: str) -> np.ndarray:
    """
    One-hot encoding with N/other mapped to [0,0,0,0].
    Output shape: (L, 4) with columns [A, C, G, T].
    """
    mapping = {
        "A": [1, 0, 0, 0],
        "C": [0, 1, 0, 0],
        "G": [0, 0, 1, 0],
        "T": [0, 0, 0, 1],
        "N": [0, 0, 0, 0],
    }
    s = str(sequence).upper()
    return np.asarray([mapping.get(b, [0, 0, 0, 0]) for b in s], dtype=np.float32)


def integrated_gradients(
    inputs_Lx4: np.ndarray,
    model: tf.keras.Model,
    target_class_idx: int = 0,
    baseline: Optional[np.ndarray] = None,
    steps: int = 50,
) -> np.ndarray:
    """
    Integrated Gradients attribution for a single input (no batch).

    This implementation:
      - interpolates between baseline and input across (steps+1) points
      - computes gradients at each point
      - averages gradients and multiplies by (input - baseline)

    Parameters
    ----------
    inputs_Lx4 : np.ndarray
        Input of shape (L,4).
    model : tf.keras.Model
        Keras model mapping (1,L,4) -> scalar or vector.
    target_class_idx : int
        Used only if model output has shape (1,K) with K>1.
    baseline : Optional[np.ndarray]
        Baseline of shape (L,4). Defaults to zeros.
    steps : int
        Number of integration steps.

    Returns
    -------
    np.ndarray
        Attribution array of shape (L,4).
    """
    x = np.asarray(inputs_Lx4, dtype=np.float32)
    if baseline is None:
        b = np.zeros_like(x, dtype=np.float32)
    else:
        b = np.asarray(baseline, dtype=np.float32)

    interpolated = [b + (float(i) / steps) * (x - b) for i in range(steps + 1)]

    grads = []
    for xi in interpolated:
        x_tf = tf.convert_to_tensor(xi[None, :, :], dtype=tf.float32)
        with tf.GradientTape() as tape:
            tape.watch(x_tf)
            preds = model(x_tf, training=False)

            # robust target selection
            if len(preds.shape) == 2 and preds.shape[1] > 1:
                target = preds[:, target_class_idx]
            else:
                target = tf.reshape(preds, (preds.shape[0],))  # (1,)

        grad = tape.gradient(target, x_tf)  # (1,L,4)
        grads.append(grad.numpy()[0])       # (L,4)

    avg_grads = np.mean(grads, axis=0)      # (L,4)
    ig = (x - b) * avg_grads
    return ig.astype(np.float32)


def _save_logo(attributions_Lx4: np.ndarray, out_png: Path, title: str, cfg: IGConfig) -> None:
    df = pd.DataFrame(attributions_Lx4, columns=["A", "C", "G", "T"])

    fig, ax = plt.subplots(figsize=cfg.figsize)
    logo = logomaker.Logo(
        df,
        ax=ax,
        shade_below=0.5,
        fade_below=0.5,
        font_name=cfg.font_name,
    )

    logo.style_spines(spines=["left", "bottom"], visible=True)
    logo.ax.set_ylabel("Attribution score", labelpad=-1, fontsize=20)
    logo.ax.set_xticks([])
    logo.ax.axhline(0, color="black", linewidth=0.5)
    logo.ax.tick_params(axis="y", labelsize=20)
    logo.ax.set_title(title, fontsize=18)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=cfg.dpi, bbox_inches="tight")
    plt.close(fig)


def run_ig_full_length(
    sequence: str,
    model: tf.keras.Model,
    out_png: Path,
    model_label: str,
    cfg: IGConfig,
) -> None:
    s = str(sequence).upper()
    X = one_hot_encode(s)
    if X.shape[0] != cfg.seq_len:
        raise ValueError(f"Sequence length must be {cfg.seq_len}. Got {X.shape[0]}")

    attr = integrated_gradients(
        inputs_Lx4=X,
        model=model,
        target_class_idx=cfg.target_class_idx,
        baseline=None,
        steps=cfg.steps,
    )

    _save_logo(attr, out_png, title=f"Integrated Gradients – {model_label}", cfg=cfg)


def run_ig_sliding_window(
    sequence: str,
    model: tf.keras.Model,
    window_seq: int,
    out_png: Path,
    model_label: str,
    cfg: IGConfig,
) -> None:
    s = str(sequence).upper()
    X_full = one_hot_encode(s)
    L = X_full.shape[0]

    if window_seq <= 0:
        raise ValueError("window_seq must be > 0 for sliding-window mode")
    if window_seq > L:
        raise ValueError(f"window_seq ({window_seq}) > sequence length ({L})")

    attributions = np.zeros((L, 4), dtype=np.float32)
    count = np.zeros((L,), dtype=np.float32)

    for i in range(0, L - window_seq + 1):
        window = X_full[i : i + window_seq]
        window_attr = integrated_gradients(
            inputs_Lx4=window,
            model=model,
            target_class_idx=cfg.target_class_idx,
            baseline=None,
            steps=cfg.steps,
        )
        attributions[i : i + window_seq] += window_attr
        count[i : i + window_seq] += 1.0

    count[count == 0] = 1.0
    attributions = attributions / count[:, None]

    _save_logo(attributions, out_png, title=f"Integrated Gradients – {model_label}", cfg=cfg)


def _parse_models_arg(models: str) -> Dict[str, str]:
    """
    Parse "name=path,name2=path2" into dict.
    """
    out: Dict[str, str] = {}
    for part in models.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError("Expected --models format: name=path,name2=path2")
        name, path = part.split("=", 1)
        out[name.strip()] = path.strip()
    if not out:
        raise ValueError("No models parsed from --models")
    return out


def main() -> None:
    p = argparse.ArgumentParser(
        description="Run Integrated Gradients on a single WT sequence and save logomaker-style attribution logos."
    )
    p.add_argument("--sequence", required=True, help="WT sequence (length must be 124 by default).")
    p.add_argument(
        "--models",
        required=True,
        help="Comma-separated list in the format name=path,name2=path2 (Keras .h5).",
    )
    p.add_argument("--out_dir", default="IG_results", help="Base output directory.")
    p.add_argument("--seq_len", type=int, default=124, help="Sequence length (default: 124).")
    p.add_argument("--steps", type=int, default=50, help="IG steps (default: 50).")
    p.add_argument("--target_class_idx", type=int, default=0, help="Target index for multi-output models.")
    p.add_argument(
        "--window_seq",
        type=int,
        default=0,
        help="Sliding window length. 0 => full-length IG (default).",
    )
    p.add_argument("--batch_note", default="", help="Optional tag appended to output folder name.")
    args = p.parse_args()

    cfg = IGConfig(
        seq_len=args.seq_len,
        steps=args.steps,
        target_class_idx=args.target_class_idx,
        window_seq=args.window_seq,
    )

    models = _parse_models_arg(args.models)

    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"IG_WT_{stamp}"
    if args.batch_note.strip():
        tag = f"{tag}_{safe_name(args.batch_note)}"

    out_root = Path(args.out_dir) / tag
    out_root.mkdir(parents=True, exist_ok=True)

    # traceability
    readme = out_root / "README.txt"
    with open(readme, "w", encoding="utf-8") as f:
        f.write(f"WT_SEQUENCE length: {len(args.sequence)}\n")
        f.write(f"SEQ_LEN: {cfg.seq_len}\n")
        f.write(f"WINDOW_SEQ: {cfg.window_seq}\n")
        f.write(f"IG_STEPS: {cfg.steps}\n")
        f.write(f"TARGET_CLASS_IDX: {cfg.target_class_idx}\n")
        f.write("MODELS:\n")
        for k, v in models.items():
            f.write(f"  {k}: {v}\n")

    for model_name, model_path in models.items():
        model_label = model_name
        safe_model = safe_name(model_name)

        model_outdir = out_root / safe_model
        model_outdir.mkdir(parents=True, exist_ok=True)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_name} -> {model_path}")

        print(f"[INFO] Loading model '{model_label}' from: {model_path}")
        model = load_model(model_path)

        out_png = model_outdir / f"IG_WT_{safe_model}.png"
        if cfg.window_seq and cfg.window_seq > 0:
            run_ig_sliding_window(
                sequence=args.sequence,
                model=model,
                window_seq=cfg.window_seq,
                out_png=out_png,
                model_label=model_label,
                cfg=cfg,
            )
        else:
            run_ig_full_length(
                sequence=args.sequence,
                model=model,
                out_png=out_png,
                model_label=model_label,
                cfg=cfg,
            )

        print(f"[OK] Saved: {out_png}")

    print(f"[DONE] Results folder: {out_root}")


if __name__ == "__main__":
    main()