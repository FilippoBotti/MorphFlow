"""
Compatibility entrypoint for the canonical split-safe MorphAny3D dataset generator.

By default this creates exactly 50,000 examples:
  - 40,000 train
  - 5,000 validation
  - 5,000 test

The implementation lives in generate_morphany3d_split_dataset.py so the dataset
layout, split logic and latent saving code stay in one place.
"""

from __future__ import annotations

import sys
from typing import List, Optional, Sequence

try:
    from .generate_morphany3d_split_dataset import main as split_main
except ImportError:
    from generate_morphany3d_split_dataset import main as split_main


DEFAULT_ASSETS_DIR = "/home/filippo/datasets/3d/flux_outputs"
DEFAULT_OUTPUT_DIR = "/home/filippo/datasets/3d/morphing_dataset_flux"
DEFAULT_TRAIN_SAMPLES = 40_000
DEFAULT_VAL_SAMPLES = 5_000
DEFAULT_TEST_SAMPLES = 5_000


def _has_option(argv: Sequence[str], option: str) -> bool:
    return any(arg == option or arg.startswith(f"{option}=") for arg in argv)


def _with_default(argv: List[str], option: str, value: object) -> List[str]:
    if _has_option(argv, option):
        return argv
    return [*argv, option, str(value)]


def build_forwarded_argv(argv: Sequence[str]) -> List[str]:
    forwarded = list(argv)

    if "-h" in forwarded or "--help" in forwarded:
        return forwarded

    defaults = [
        ("--assets-dir", DEFAULT_ASSETS_DIR),
        ("--output-dir", DEFAULT_OUTPUT_DIR),
        ("--target-train-samples", DEFAULT_TRAIN_SAMPLES),
        ("--target-val-samples", DEFAULT_VAL_SAMPLES),
        ("--target-test-samples", DEFAULT_TEST_SAMPLES),
    ]

    for option, value in defaults:
        forwarded = _with_default(forwarded, option, value)

    return forwarded


def main(argv: Optional[Sequence[str]] = None) -> None:
    split_main(build_forwarded_argv(sys.argv[1:] if argv is None else argv))


if __name__ == "__main__":
    main()
