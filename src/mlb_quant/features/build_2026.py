from pathlib import Path

import joblib
import pandas as pd

from mlb_quant.features.engine import (
    build_point_in_time_features,
    build_raw_game_metrics,
    feature_dictionary,
    load_scaled_raw_paths,
)


HISTORICAL_APPEARANCE_PATH = Path(
    "data/processed/starting_pitcher_games_2021_2025.parquet"
)

LIVE_APPEARANCE_PATH = Path(
    "data/processed/live_2026/"
    "starting_pitcher_games_2026-03-25_2026-09-14.parquet"
)

RAW_ROOT = Path("data/raw")

OUTPUT_PATH = Path(
    "data/processed/model_features_2026_through_2026-09-14.parquet"
)

DICTIONARY_PATH = Path(
    "reports/feature_dictionary_2026_through_2026-09-14.csv"
)

MODEL_PATHS = [
    Path("models/artifacts/mlb_k_quant_production_2026.joblib"),
    Path("models/artifacts/mlb_outs_quant_production_2026.joblib"),
    Path("models/artifacts/mlb_bf_quant_production_2026.joblib"),
    Path("models/artifacts/mlb_pitches_quant_production_2026.joblib"),
]


def main() -> None:
    historical = pd.read_parquet(
        HISTORICAL_APPEARANCE_PATH
    )

    live = pd.read_parquet(
        LIVE_APPEARANCE_PATH
    )

    historical["game_date"] = pd.to_datetime(
        historical["game_date"]
    )

    live["game_date"] = pd.to_datetime(
        live["game_date"]
    )

    if "season" not in historical.columns:
        historical.insert(
            0,
            "season",
            historical["game_date"].dt.year,
        )

    if "season" not in live.columns:
        live.insert(
            0,
            "season",
            live["game_date"].dt.year,
        )

    if not live["season"].eq(2026).all():
        raise ValueError(
            "Live appearance file contains a non-2026 season."
        )

    appearances = pd.concat(
        [historical, live],
        ignore_index=True,
    )

    appearances = appearances.sort_values(
        [
            "game_date",
            "game_id",
            "pitcher_mlbam_id",
        ]
    ).reset_index(drop=True)

    duplicates = appearances.duplicated(
        ["game_id", "pitcher_mlbam_id"]
    ).sum()

    if duplicates:
        raise ValueError(
            f"Combined appearances contain "
            f"{duplicates} duplicate starter rows."
        )

    # IMPORTANT:
    # Only load 2026 raw pitch-level data.
    # Historical appearance rows are still included above so
    # cross-season fields such as days rest and previous-start
    # pitch count can carry across the 2025 -> 2026 boundary.
    raw_paths = load_scaled_raw_paths(
        RAW_ROOT,
        seasons=(2026,),
    )

    if not raw_paths:
        raise ValueError(
            "No 2026 raw Statcast chunks found."
        )

    print(
        f"Historical appearances: {len(historical)}"
    )
    print(
        f"2026 appearances: {len(live)}"
    )
    print(
        f"Combined appearances: {len(appearances)}"
    )
    print(
        f"2026 raw chunks: {len(raw_paths)}"
    )

    pitcher_metrics, type_metrics, team_metrics = (
        build_raw_game_metrics(
            raw_paths,
            appearances,
        )
    )

    print(
        f"Pitcher metric rows: {len(pitcher_metrics)}"
    )
    print(
        f"Pitch-type metric rows: {len(type_metrics)}"
    )
    print(
        f"Team metric rows: {len(team_metrics)}"
    )

    all_features = build_point_in_time_features(
        appearances,
        pitcher_metrics,
        type_metrics,
        team_metrics,
    )

    features = all_features[
        all_features["season"] == 2026
    ].copy()

    if len(features) != len(live):
        raise ValueError(
            f"2026 row mismatch: "
            f"appearances={len(live)} "
            f"features={len(features)}"
        )

    duplicate_features = features.duplicated(
        ["game_id", "pitcher_mlbam_id"]
    ).sum()

    if duplicate_features:
        raise ValueError(
            f"Found {duplicate_features} duplicate "
            f"2026 feature rows."
        )

    frozen_columns = None

    for path in MODEL_PATHS:
        bundle = joblib.load(path)
        columns = bundle["feature_columns"]

        if frozen_columns is None:
            frozen_columns = columns
        elif columns != frozen_columns:
            raise ValueError(
                f"Feature schema mismatch in {path}"
            )

    missing = [
        column
        for column in frozen_columns
        if column not in features.columns
    ]

    if missing:
        raise ValueError(
            f"Missing {len(missing)} production "
            f"features: {missing[:30]}"
        )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    DICTIONARY_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    features.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    feature_dictionary(features).to_csv(
        DICTIONARY_PATH,
        index=False,
    )

    model_frame = features[frozen_columns]
    row_missing = model_frame.isna().sum(axis=1)

    print()
    print("CORRECTED 2026 FEATURE BUILD COMPLETE")
    print(f"Rows: {len(features)}")
    print(f"Total columns: {len(features.columns)}")
    print(
        f"Frozen model features: {len(frozen_columns)}"
    )
    print(
        f"Missing frozen features: {len(missing)}"
    )
    print(
        "Date range:",
        features["game_date"].min(),
        "through",
        features["game_date"].max(),
    )
    print(
        "Overall NaN rate:",
        f"{model_frame.isna().to_numpy().mean() * 100:.2f}%",
    )
    print(
        "Mean missing features/start:",
        round(row_missing.mean(), 2),
    )
    print(
        "pitcher_days_rest missing:",
        f"{features['pitcher_days_rest'].isna().mean() * 100:.2f}%",
    )
    print(
        "pitches_previous_start missing:",
        f"{features['pitches_previous_start'].isna().mean() * 100:.2f}%",
    )
    print(f"Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()