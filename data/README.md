# Data Guide

This repository ships only a small example file (`data/example/example_windows.csv`).
Full genome-scale datasets are expected to be hosted externally.

## Expected Table Columns

Minimum required columns depend on the task:

- Sequence-only inference/training:
  - `Sequence`
  - `Label_nuc` (for supervised training/evaluation)

- Genome-window metadata (recommended):
  - `Chromosome`, `Start`, `End`

- iMotifPredictor feature variants:
  - iM-IP mean column: `pred_hybrid_cnn_lstm_mean`
  - epigenetic columns (variant-dependent), e.g.:
    - `H3K9me3`
    - `atac_signals`
    - `H3K4me1`
    - `H3K27ac`
    - `H3K36me3`
    - `H3K4me3`

## Chunked Data Convention

For large-scale runs, chunk files should follow:
- filename format: `chunk_aa.csv`, `chunk_ab.csv`, ...
- each file contains up to 100,000 rows
- canonical columns: `Chromosome, Start, End, Sequence`

## Generate Chunks from FASTA Windows

```bash
python scripts/make_hg19_chunks.py \
  --fasta_file /path/to/windows.fa \
  --out_dir /path/to/chunks \
  --rows_per_file 100000 \
  --seq_len 124
```

## Add Epigenetic Features from bigWig

Compute mean signal over each genomic window (`Start`/`End`) and add it as a new column:

```bash
python scripts/add_bigwig_feature.py \
  --input_dir /path/to/chunks \
  --glob "chunk_*.csv" \
  --bigwig /path/to/H3K9me3.bw \
  --feature_col H3K9me3 \
  --in_place
```

If the bigWig uses NCBI chromosome accessions while CSV uses `chr*` names:

```bash
python scripts/add_bigwig_feature.py \
  --input_dir /path/to/chunks \
  --bigwig /path/to/H3K9me3.bw \
  --feature_col H3K9me3 \
  --chrom_map hg38_to_ncbi \
  --in_place
```

## Reproducibility Manifest

```bash
python scripts/build_data_manifest.py \
  --root /path/to/chunks \
  --include "*.csv" \
  --out_tsv data/manifest.tsv \
  --out_checksums data/checksums.sha256
```

The manifest and checksums are the recommended minimal artifacts for large external datasets.
