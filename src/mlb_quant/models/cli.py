"""CLI for the chronological MLB model tournament.

Argument parsing happens before any heavy imports or computation. Nothing in
this module trains a model, loads data, or touches disk until the
``tournament`` subcommand is explicitly invoked.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mlb-quant-models",
        description="MLB pitcher strikeout/outs/BF/pitch-count model commands.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    tournament_parser = subparsers.add_parser(
        "tournament",
        help="Run the full chronological model tournament (expensive: trains "
        "naive/ridge/poisson/tree/ensemble models across 2021-2024 walk-forward "
        "folds, runs the feature-family ablation, freezes winners, and scores "
        "the untouched 2025 holdout).",
    )
    tournament_parser.add_argument(
        "--input-path", type=Path,
        default=Path("data/processed/model_features_2021_2025.parquet"),
    )
    tournament_parser.add_argument("--artifact-root", type=Path, default=Path("models/artifacts"))
    tournament_parser.add_argument(
        "--report-path", type=Path,
        default=Path("reports/model_tournament_v0_1.md"),
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "tournament":
        # Imported lazily so `--help` never pulls in sklearn/pandas/joblib.
        from .tournament import run_tournament

        result = run_tournament(
            input_path=args.input_path,
            artifact_root=args.artifact_root,
            report_path=args.report_path,
        )
        print("winners:", result["winners"])
        print("holdout rows:", len(result["holdout"]))


if __name__ == "__main__":
    main()
