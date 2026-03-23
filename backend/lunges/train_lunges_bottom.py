"""
train_lunges_bottom.py – Train TCN for lunge bottom-phase classification.

Labels: correct | torso_lean_forward | knee_over_toe | not_deep_enough
Feature set: 42 dims/frame (30 xyz + 12 computed scalars)

Usage:
  python lunges/train_lunges_bottom.py \
    --data dataset/lunges/dataset_bottom \
    --out  lunges/models/lunges_bottom_tcn.pt
"""

from __future__ import annotations

if __package__ in {None, ""}:
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import argparse

from shared.training_utils import NPZDataset, run_bottom_training
from lunges.features import BOTTOM_FEATURE_DIM, extract_bottom_features

def build_lunge_dataset(samples, label_map, window_size):
    return NPZDataset(
        samples,
        label_map,
        window_size,
        extract_bottom_features,
        BOTTOM_FEATURE_DIM,
    )

_FEAT_DETAIL = {"key_joint_xyz": 30, "scalars": 12, "total": BOTTOM_FEATURE_DIM}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train lunge bottom classifier TCN")
    parser.add_argument("--data", required=True, help="dataset/lunges/dataset_bottom")
    parser.add_argument("--out", required=True, help="lunges/models/lunges_bottom_tcn.pt")
    parser.add_argument("--T", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=75)
    parser.add_argument("--bs", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--ch", type=int, default=64)
    parser.add_argument("--val_ratio", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=921)
    parser.add_argument("--cpu", action="store_true")
    run_bottom_training(
        parser.parse_args(),
        build_lunge_dataset,
        BOTTOM_FEATURE_DIM,
        "lunges_bottom",
        _FEAT_DETAIL,
    )


if __name__ == "__main__":
    main()
