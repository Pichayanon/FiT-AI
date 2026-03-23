from __future__ import annotations

if __package__ in {None, ""}:
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from shared.training_utils import build_phase_training_parser, run_phase_training


def main() -> None:
    parser = build_phase_training_parser("Train lunge side-view phase TCN")
    run_phase_training(parser.parse_args())


if __name__ == "__main__":
    main()
