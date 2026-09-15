"""CLI for generating the point-in-time modeling feature table."""

from __future__ import annotations

import argparse
from pathlib import Path

from .engine import generate_feature_table


def main() -> None:
    parser = argparse.ArgumentParser(description="Build leakage-controlled MLB pitcher features.")
    parser.add_argument("--appearance-path", type=Path, default=Path("data/processed/starting_pitcher_games_2021_2025.parquet"))
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-path", type=Path, default=Path("data/processed/model_features_2021_2025.parquet"))
    parser.add_argument("--dictionary-path", type=Path, default=Path("reports/feature_dictionary_2021_2025.csv"))
    args = parser.parse_args()
    features = generate_feature_table(args.appearance_path, args.raw_root, args.output_path, args.dictionary_path)
    print(f"rows={len(features)} features={len(features.columns)} output={args.output_path}")


if __name__ == "__main__":
    main()