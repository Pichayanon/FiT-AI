"""
train_squat_stand.py – Train TCN for squat standing-stance classification.

Labels: good_stand | stand_too_narrow | stand_too_wide
Feature set: 16 dims/frame (4 width ratios + 6 X-positions + 3 angles + extras)

Usage:
  python squat/train_squat_stand.py \
    --data dataset/squat/dataset_standing \
    --out  squat/models/squat_stand_tcn.pt
"""

from __future__ import annotations

if __package__ in {None, ""}:
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import argparse

from shared.training_utils import NPZDataset, run_bottom_training
from squat.features import STAND_FEATURE_DIM, extract_stand_features

def build_standing_dataset(samples, label_map, window_size):
    return NPZDataset(
        samples,
        label_map,
        window_size,
        extract_stand_features,
        STAND_FEATURE_DIM,
    )

_FEAT_DETAIL = {
    "width_ratios": 4, "x_positions": 6, "angles": 3,
    "distances": 2, "feet_over_shoulder": 1, "total": STAND_FEATURE_DIM,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train squat standing classifier TCN")
    parser.add_argument("--data", required=True, help="dataset/squat/dataset_standing")
    parser.add_argument("--out", required=True, help="squat/models/squat_stand_tcn.pt")
    parser.add_argument("--T", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=75)
    parser.add_argument("--bs", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--ch", type=int, default=64)
    parser.add_argument("--val_ratio", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=921)
    parser.add_argument("--cpu", action="store_true")
    run_bottom_training(
        parser.parse_args(),
        build_standing_dataset,
        STAND_FEATURE_DIM,
        "squat_stand",
        _FEAT_DETAIL,
    )


if __name__ == "__main__":
    main()
