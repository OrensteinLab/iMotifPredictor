from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    import tensorflow as tf
except Exception:  # pragma: no cover
    tf = None


@dataclass
class SeedConfig:
    seed: int = 1337
    set_python_hash_seed: bool = True


def set_global_seed(cfg: SeedConfig) -> None:
    """
    Set random seeds for best-effort reproducibility.

    Notes:
      - Full determinism is not guaranteed across TensorFlow/CUDA kernels.
      - This function is intended to reduce run-to-run variance when possible.
    """
    if cfg.set_python_hash_seed:
        os.environ["PYTHONHASHSEED"] = str(cfg.seed)

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    if tf is not None:
        try:
            tf.random.set_seed(cfg.seed)
        except Exception:
            pass