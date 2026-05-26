from __future__ import annotations

from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Sequence, Union

import pandas as pd


def read_csv_safe(
    path: Union[str, Path],
    usecols: Optional[Sequence[str]] = None,
    chunksize: Optional[int] = None,
) -> Union[pd.DataFrame, Iterator[pd.DataFrame]]:
    """
    Read a CSV with settings suitable for large genomics tables.

    - Uses on_bad_lines="skip" to avoid crashing on malformed lines.
    - If chunksize is provided, returns an iterator over DataFrame chunks.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")

    kwargs = dict(on_bad_lines="skip", engine="python")
    if usecols is not None:
        kwargs["usecols"] = list(usecols)

    if chunksize is not None and chunksize > 0:
        return pd.read_csv(path, chunksize=chunksize, **kwargs)

    return pd.read_csv(path, **kwargs)


def list_csv_files(directory: Union[str, Path]) -> List[Path]:
    """
    List *.csv files in a directory (sorted).
    """
    d = Path(directory)
    if not d.exists():
        raise FileNotFoundError(f"Directory not found: {d}")
    files = sorted([p for p in d.iterdir() if p.suffix.lower() == ".csv"])
    if not files:
        raise FileNotFoundError(f"No CSV files found in: {d}")
    return files