"""
Project-wide constants for iMotifPredictor training.

Defines epigenetic feature sets and feature-variant configurations used for ablation
and for the final iMotifPredictor model.
"""

EPIGENETIC_FEATURES_HEK = [
    "H3K9me3",
    "atac_signals",
    "H3K4me1",
    "H3K27ac",
    "H3K36me3",
    "H3K4me3",
]

EPIGENETIC_FEATURES_WDLPS = [
    "H3K9me3",
    "atac_signals",
]

# iM-IP score column name used as a scalar feature
IMIP_SCALAR_FEATURE = "pred_hybrid_cnn_lstm_mean"

FEATURE_VARIANTS = {
    "sequence_only": [],
    "sequence_plus_scalar": [IMIP_SCALAR_FEATURE],
    "sequence_plus_epigenetic_hek": EPIGENETIC_FEATURES_HEK,
    # Full HEK feature set used for the main iMotifPredictor configuration
    "sequence_plus_epigenetic_hek_scalar": EPIGENETIC_FEATURES_HEK + [IMIP_SCALAR_FEATURE],
    "sequence_plus_epigenetic_wdlps": EPIGENETIC_FEATURES_WDLPS,
    "sequence_plus_epigenetic_wdlps_scalar": EPIGENETIC_FEATURES_WDLPS + [IMIP_SCALAR_FEATURE],
}