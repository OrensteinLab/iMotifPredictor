#!/usr/bin/env python3
import argparse

from imotifpredictor.data.add_imip_score import (
    AddImipScoreConfig,
    add_imip_score,
    default_imip_model_paths,
)


def main():
    p = argparse.ArgumentParser(
        description="Add pred_hybrid_cnn_lstm_mean by averaging iM-IP model predictions."
    )
    p.add_argument("--input_csv", required=True, help="Input CSV containing Sequence column.")
    p.add_argument("--output_csv", required=True, help="Output CSV path.")

    p.add_argument(
        "--models",
        default="",
        help=(
            "Comma-separated list of .h5 model paths. "
            "If omitted, defaults to models/iM-IP/{HEK,U2OS,MCF7} hybrid CNN-LSTM baseline."
        ),
    )
    p.add_argument("--repo_root", default="", help="Repository root used for resolving default model paths.")

    p.add_argument("--seq_col", default="Sequence")
    p.add_argument("--seq_len", type=int, default=124)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--out_col", default="pred_hybrid_cnn_lstm_mean")

    a = p.parse_args()

    if a.models.strip():
        model_paths = [x.strip() for x in a.models.split(",") if x.strip()]
    else:
        model_paths = default_imip_model_paths(repo_root=a.repo_root.strip() or None)

    cfg = AddImipScoreConfig(
        input_csv=a.input_csv,
        output_csv=a.output_csv,
        model_paths=model_paths,
        seq_col=a.seq_col,
        seq_len=a.seq_len,
        batch_size=a.batch_size,
        out_col=a.out_col,
    )

    add_imip_score(cfg)
    print(f"Saved: {a.output_csv}")
    print("Models:")
    for m in model_paths:
        print("  -", m)


if __name__ == "__main__":
    main()