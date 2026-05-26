from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Iterable, Iterator, Tuple


@dataclass(frozen=True)
class SyntheticConfig:
    seq_len: int = 124
    core_min: int = 2
    core_max: int = 10
    loop_min: int = 1
    loop_max: int = 12


def create_sequence(
    core1: int,
    loop1: int,
    core2: int,
    loop2: int,
    core3: int,
    loop3: int,
    core4: int,
    seq_len: int,
) -> str:
    """
    Create a synthetic 124-nt sequence with a central i-motif-like pattern.

    Pattern:
        C^core1 N^loop1 C^core2 N^loop2 C^core3 N^loop3 C^core4

    The sequence is then centered within a fixed-length window by padding 'N'
    equally on both sides (floor/ceil split if needed).
    """
    motif = (
        ("C" * core1) + ("N" * loop1) +
        ("C" * core2) + ("N" * loop2) +
        ("C" * core3) + ("N" * loop3) +
        ("C" * core4)
    )

    if len(motif) > seq_len:
        # Cannot be padded to seq_len
        return ""

    pad_total = seq_len - len(motif)
    left_pad = pad_total // 2
    right_pad = pad_total - left_pad  # ensures total length == seq_len

    return ("N" * left_pad) + motif + ("N" * right_pad)


def iter_parameter_grid(cfg: SyntheticConfig) -> Iterator[Tuple[int, int, int, int, int, int, int]]:
    core_lengths = range(cfg.core_min, cfg.core_max + 1)
    loop_lengths = range(cfg.loop_min, cfg.loop_max + 1)

    for core1, core2, core3, core4 in product(core_lengths, repeat=4):
        for loop1, loop2, loop3 in product(loop_lengths, repeat=3):
            yield core1, loop1, core2, loop2, core3, loop3, core4


def write_synthetic_csv(out_csv: Path, cfg: SyntheticConfig) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    header = [
        "Sequence",
        "Core1_Length", "Loop1_Length",
        "Core2_Length", "Loop2_Length",
        "Core3_Length", "Loop3_Length",
        "Core4_Length",
    ]

    written = 0
    total = 0

    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)

        for core1, loop1, core2, loop2, core3, loop3, core4 in iter_parameter_grid(cfg):
            total += 1
            seq = create_sequence(core1, loop1, core2, loop2, core3, loop3, core4, seq_len=cfg.seq_len)
            if not seq:
                continue
            if len(seq) != cfg.seq_len:
                continue

            w.writerow([seq, core1, loop1, core2, loop2, core3, loop3, core4])
            written += 1

    print(f"[DONE] Wrote {written:,} sequences to: {out_csv}")
    print(f"[INFO] Total parameter combinations scanned: {total:,}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate synthetic 124-nt sequences with centered C-cores and N-loops (paper interpretability grid)."
    )
    p.add_argument("--out", required=True, help="Output CSV path.")
    p.add_argument("--seq_len", type=int, default=124, help="Fixed sequence length (default: 124).")

    p.add_argument("--core_min", type=int, default=2, help="Minimum core length (default: 2).")
    p.add_argument("--core_max", type=int, default=10, help="Maximum core length (default: 10).")

    p.add_argument("--loop_min", type=int, default=1, help="Minimum loop length (default: 1).")
    p.add_argument("--loop_max", type=int, default=12, help="Maximum loop length (default: 12).")

    return p.parse_args()


def main() -> None:
    a = parse_args()
    cfg = SyntheticConfig(
        seq_len=a.seq_len,
        core_min=a.core_min,
        core_max=a.core_max,
        loop_min=a.loop_min,
        loop_max=a.loop_max,
    )
    write_synthetic_csv(Path(a.out), cfg)


if __name__ == "__main__":
    main()