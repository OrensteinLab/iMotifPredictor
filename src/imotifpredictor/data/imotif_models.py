from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


# Keep these names stable (used by CLI choices)
IMOTIF_VARIANTS = [
    "seq_only",
    "seq_plus_imip",
    "seq_plus_epi6",
    "seq_plus_epi2_wdlps",
    "seq_plus_epi2_wdlps_plus_imip",
    "seq_plus_epi6_plus_imip",
]


# Feature definitions (must match training)
EPIGENETIC_FEATURES_HEK6 = ["H3K9me3", "atac_signals", "H3K4me1", "H3K27ac", "H3K36me3", "H3K4me3"]
EPIGENETIC_FEATURES_WDLPS2 = ["H3K9me3", "atac_signals"]
IMIP_SCALAR_COL = "pred_hybrid_cnn_lstm_mean"


@dataclass(frozen=True)
class VariantSpec:
    variant: str
    model_filename: str
    required_numeric_features: List[str]
    supports_fasta: bool


def get_imotif_variant_specs() -> Dict[str, VariantSpec]:
    """
    Central registry: variant -> (model filename, required input feature columns, FASTA support).
    """
    return {
        "seq_only": VariantSpec(
            variant="seq_only",
            model_filename="iMotifPredictor_HEK_seq_only_baseline.h5",
            required_numeric_features=[],
            supports_fasta=True,
        ),
        "seq_plus_imip": VariantSpec(
            variant="seq_plus_imip",
            model_filename="iMotifPredictor_HEK_seq_plus_imip_baseline.h5",
            required_numeric_features=[IMIP_SCALAR_COL],
            supports_fasta=False,
        ),
        "seq_plus_epi6": VariantSpec(
            variant="seq_plus_epi6",
            model_filename="iMotifPredictor_HEK_seq_plus_epi6_baseline.h5",
            required_numeric_features=EPIGENETIC_FEATURES_HEK6,
            supports_fasta=False,
        ),
        "seq_plus_epi2_wdlps": VariantSpec(
            variant="seq_plus_epi2_wdlps",
            model_filename="iMotifPredictor_HEK_seq_plus_epi2_wdlps_baseline.h5",
            required_numeric_features=EPIGENETIC_FEATURES_WDLPS2,
            supports_fasta=False,
        ),
        "seq_plus_epi2_wdlps_plus_imip": VariantSpec(
            variant="seq_plus_epi2_wdlps_plus_imip",
            model_filename="iMotifPredictor_HEK_seq_plus_epi2_wdlps_plus_imip_baseline.h5",
            required_numeric_features=EPIGENETIC_FEATURES_WDLPS2 + [IMIP_SCALAR_COL],
            supports_fasta=False,
        ),
        "seq_plus_epi6_plus_imip": VariantSpec(
            variant="seq_plus_epi6_plus_imip",
            model_filename="iMotifPredictor_HEK_seq_plus_epi6_plus_imip_baseline.h5",
            required_numeric_features=EPIGENETIC_FEATURES_HEK6 + [IMIP_SCALAR_COL],
            supports_fasta=False,
        ),
    }


def default_imotif_model_path(
    variant: str,
    repo_root: Optional[str] = None,
) -> str:
    """
    Resolve the default .h5 model path for a given iMotifPredictor variant.
    """
    specs = get_imotif_variant_specs()
    if variant not in specs:
        raise ValueError(f"Unknown variant: {variant}. Allowed: {sorted(specs.keys())}")

    root = Path(repo_root) if repo_root else Path.cwd()
    return str(root / "models" / "iMotifPredictor" / specs[variant].model_filename)