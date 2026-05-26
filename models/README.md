# Pretrained Models

This directory contains pretrained Keras `.h5` model weights used by the prediction scripts.

## 1) iM-IP Models (`models/iM-IP/`)

Sequence-only hybrid CNN-LSTM models trained per cell type.

Files:
- `iM-IP_HEK_hybrid_cnn_lstm_baseline.h5`
- `iM-IP_U2OS_hybrid_cnn_lstm_baseline.h5`
- `iM-IP_MCF7_hybrid_cnn_lstm_baseline.h5`

Input:
- `Sequence` (fixed length 124; one-hot encoded internally)

Output:
- Scalar iM propensity score per sequence

Typical usage:
- `scripts/predict_imip_score.py`
- `scripts/predict_pipeline.py --add_imip`

## 2) iMotifPredictor Models (`models/iMotifPredictor/`)

LSTM late-fusion models with multiple feature variants.

Files and required input columns:
- `iMotifPredictor_HEK_seq_only_baseline.h5`
  - Required: `Sequence`
- `iMotifPredictor_HEK_seq_plus_imip_baseline.h5`
  - Required: `Sequence`, `pred_hybrid_cnn_lstm_mean`
- `iMotifPredictor_HEK_seq_plus_epi6_baseline.h5`
  - Required: `Sequence`, `H3K9me3`, `atac_signals`, `H3K4me1`, `H3K27ac`, `H3K36me3`, `H3K4me3`
- `iMotifPredictor_HEK_seq_plus_epi6_plus_imip_baseline.h5`
  - Required: all `seq_plus_epi6` columns + `pred_hybrid_cnn_lstm_mean`
- `iMotifPredictor_HEK_seq_plus_epi2_wdlps_baseline.h5`
  - Required: `Sequence`, `H3K9me3`, `atac_signals`
- `iMotifPredictor_HEK_seq_plus_epi2_wdlps_plus_imip_baseline.h5`
  - Required: `Sequence`, `H3K9me3`, `atac_signals`, `pred_hybrid_cnn_lstm_mean`

Typical usage:
- `scripts/predict_imotifpredictor.py`
- `scripts/predict_pipeline.py`

## Integrity and Naming

- Checksums are stored in `models/checksums.txt`.
- Model filenames are resolved by code in:
  - `src/imotifpredictor/data/imotif_models.py`
- If you rename files, update variant-to-filename mapping accordingly.
