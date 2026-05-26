#!/usr/bin/env python3
from __future__ import annotations

"""
Backward-compatible alias for a historical filename typo.

Prefer:
  python scripts/predict_pipeline.py ...
"""

from predict_pipeline import main


if __name__ == "__main__":
    print("[WARN] 'predict_pipline.py' is deprecated. Use 'predict_pipeline.py'.")
    main()
