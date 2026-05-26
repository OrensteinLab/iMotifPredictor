# iMotifPredictor

Official repository for the paper:

**iMotifPredictor: i-motif prediction by multi-data integration**

Accepted as a full paper at ACM BCB 2026.

`iMotifPredictor` is a deep-learning framework for genome-wide i-motif prediction by integrating:
- DNA sequence (124-nt windows)
- sequence-derived iM-IP score
- epigenetic scalar features

The repository includes inference/training scripts, pretrained models, and interpretability utilities.

## Model Overview

### iM-IP (sequence-only)
- Architecture: hybrid `CNN -> CNN -> LSTM -> Dense -> Sigmoid`
- Input: one-hot encoded DNA sequence (`124 x 4`)
- Output: scalar iM propensity score
- Pretrained models: `models/iM-IP/*.h5` (HEK, U2OS, MCF7)

### iMotifPredictor (LSTM fusion)
- Architecture: `LSTM -> Dropout -> (Late Fusion with scalar features) -> Dense -> Sigmoid`
- Input: DNA sequence (`124 x 4`) plus optional scalar features
- Output: probability score for i-motif formation
- Pretrained variants: `models/iMotifPredictor/*.h5`

#### iMotifPredictor Variants and Required Columns
- `seq_only`: `Sequence`
- `seq_plus_imip`: `Sequence`, `pred_hybrid_cnn_lstm_mean`
- `seq_plus_epi6`: `Sequence`, `H3K9me3`, `atac_signals`, `H3K4me1`, `H3K27ac`, `H3K36me3`, `H3K4me3`
- `seq_plus_epi6_plus_imip`: `Sequence` + all `seq_plus_epi6` columns + `pred_hybrid_cnn_lstm_mean`
- `seq_plus_epi2_wdlps`: `Sequence`, `H3K9me3`, `atac_signals`
- `seq_plus_epi2_wdlps_plus_imip`: `Sequence`, `H3K9me3`, `atac_signals`, `pred_hybrid_cnn_lstm_mean`

## Repository Layout

- `src/imotifpredictor/`: Python package
- `scripts/`: CLI entrypoints for training, inference, evaluation, and preprocessing
- `models/`: pretrained `.h5` models
- `data/example/`: minimal input example

## Clone

```bash
git clone https://github.com/OrensteinLab/iMotifPredictor.git
cd iMotifPredictor
```

## Installation

Create and activate a clean Python environment first.

### Option A: `venv`

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

### Option B: Conda

```bash
conda create -n imotif python=3.9 -y
conda activate imotif
pip install -r requirements.txt
pip install -e .
```

## Verified Quick Start (Example Data)

The following commands were validated on `data/example/example_windows.csv`.

```bash
python scripts/predict_imotifpredictor.py \
  --input data/example/example_windows.csv \
  --out outputs/example_seq_only_predictions.csv \
  --variant seq_only \
  --format csv \
  --seq_col Sequence \
  --batch_size 32
```

Sanity check:

```bash
python - <<'PY'
import pandas as pd
df = pd.read_csv("outputs/example_seq_only_predictions.csv")
print(df.shape)
print(df[["Label_nuc", "pred_iMotifPredictor"]].head())
PY
```

## Quick Inference

### 1) Predict iM-IP score from sequence

```bash
python scripts/predict_imip_score.py \
  --input data/example/example_windows.csv \
  --out outputs/imip_predictions.csv \
  --format csv \
  --add_mean
```

### 2) Predict iMotifPredictor score

```bash
python scripts/predict_imotifpredictor.py \
  --input data/example/example_windows.csv \
  --out outputs/imotif_predictions.csv \
  --variant seq_only \
  --format csv
```

To list available variants:

```bash
python scripts/predict_imotifpredictor.py --help
```

### 3) End-to-end pipeline

```bash
python scripts/predict_pipeline.py \
  --input data/example/example_windows.csv \
  --out outputs/pipeline_predictions.csv \
  --variant seq_plus_imip \
  --add_imip
```

If you already have three iM-IP columns in the input table, compute the mean directly:

```bash
python scripts/predict_pipeline.py \
  --input data/example/example_windows.csv \
  --out outputs/pipeline_predictions.csv \
  --variant seq_plus_imip \
  --add_imip \
  --imip_source_cols pred_a,pred_b,pred_c
```

## Training

### Train sequence-only iM-IP model

```bash
python scripts/train_imip_seqonly.py \
  --input_csv /path/to/train.csv \
  --output_dir outputs/train_imip \
  --epochs 1 \
  --batch_size 512
```

### Train iMotifPredictor on chunked CSV files

```bash
python scripts/train_imotifpredictor.py \
  --directory /path/to/chunks \
  --output_dir outputs/train_imotif \
  --start chunk_aa.csv \
  --end chunk_zz.csv \
  --feature_variant all \
  --balancing baseline
```

## Evaluation

```bash
python scripts/evaluate_run.py \
  --csv outputs/pipeline_predictions.csv \
  --label Label_nuc \
  --out outputs/aupr_ranked.csv
```

## Preprocessing for Large FASTA Windows

To split a large FASTA windows file into chunked CSV files:

```bash
python scripts/make_hg19_chunks.py \
  --fasta_file /path/to/windows.fa \
  --out_dir /path/to/chunks \
  --rows_per_file 100000 \
  --seq_len 124
```

This creates files like `chunk_aa.csv`, `chunk_ab.csv`, ...

To add an epigenetic feature from a bigWig track:

```bash
python scripts/add_bigwig_feature.py \
  --input_dir /path/to/chunks \
  --glob "chunk_*.csv" \
  --bigwig /path/to/H3K9me3.bw \
  --feature_col H3K9me3 \
  --in_place
```

If chromosome names differ between CSV and bigWig, use built-in mapping:

```bash
python scripts/add_bigwig_feature.py \
  --input_dir /path/to/chunks \
  --bigwig /path/to/track.bw \
  --feature_col H3K9me3 \
  --chrom_map hg38_to_ncbi \
  --in_place
```

## Large Data Policy 

This repository does **not** include full raw/derived genome-scale datasets (e.g., thousands of chunk files).

For reproducibility, provide externally hosted data plus:
- generation scripts (included here)
- file manifest
- checksums

Manifest helper:

```bash
python scripts/build_data_manifest.py \
  --root /path/to/chunks \
  --include "*.csv" \
  --out_tsv data/manifest.tsv \
  --out_checksums data/checksums.sha256
```
