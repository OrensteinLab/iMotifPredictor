from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str = "imotifpredictor",
    log_file: Optional[str] = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Create a console logger and optionally a file logger.

    The returned logger is safe to call multiple times without duplicating handlers.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.handlers:
        return logger

    fmt = logging.Formatter("[%(asctime)s] %(levelname)s %(message)s")

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    sh.setLevel(level)
    logger.addHandler(sh)

    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        fh.setLevel(level)
        logger.addHandler(fh)

    logger.propagate = False
    return logger