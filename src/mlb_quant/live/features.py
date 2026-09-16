from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

from mlb_quant.features.engine import (
    build_point_in_time_features,
    build_raw_game_metrics,
)


HISTORICAL_APPEARANCE_PATH = Path(
    "data/processed/starting_pitcher_games_2021_2025.parquet"
)

LIVE_PROCESSED_DIR = Path(
    "data/processed/live_2026"
)

RAW_2026_DIR = Path(
    "data/raw/season=2026"
)

TOURNAMENT_CONFIG_PATH = Path(
    "reports/model_tournament_config_v0_1.json"
)


STARTER_FILE_RE = re.compile(
    r"^starting_pitcher_games_"
    r"(\d{4}-\d{2}-\d{2})_"
    r"(\d{4}-\d{2}-\d{2})\.parquet$"
)


RAW_FILE_RE = re.compile(
    r"^statcast_"
    r"(\d{4}-\d{2}-\d{2})_"
    r"(\d{4}-\d{2}-\d{2})\.parquet$"
)


# ---------------------------------------------------------
# SCHEMA NORMALIZATION
# ---------------------------------------------------------


def normalize_appearance_schema(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """
    Normalize starter/pregame appearance tables to the schema
    used by the historical feature engine.
    """

    frame = frame.copy()

    rename_map = {}

    if (
        "game_pk" in frame.columns
        and "game_id" not in frame.columns
    ):
        rename_map["game_pk"] = "game_id"

    if (
        "pitcher_hand" in frame.columns
        and "pitcher_handedness" not in frame.columns
    ):
        rename_map[
            "pitcher_hand"
        ] = "pitcher_handedness"

    if rename_map:
        frame = frame.rename(
            columns=rename_map
        )

    required = {
        "game_date",
        "game_id",
        "pitcher_mlbam_id",
    }

    missing = required - set(
        frame.columns
    )

    if missing:
        raise ValueError(
            "Appearance table missing required "
            f"columns: {sorted(missing)}"
        )

    frame["game_date"] = pd.to_datetime(
        frame["game_date"]
    ).dt.normalize()

    # The processed starter tables do not always store season.
    # The feature engine requires it for pitcher-season histories.
    if "season" not in frame.columns:
        frame["season"] = frame["game_date"].dt.year
    else:
        frame["season"] = pd.to_numeric(
            frame["season"],
            errors="coerce",
        )
        frame["season"] = frame["season"].fillna(
            frame["game_date"].dt.year
        )

    frame["season"] = frame["season"].astype(int)

    frame["game_id"] = pd.to_numeric(
        frame["game_id"],
        errors="raise",
    )

    frame["pitcher_mlbam_id"] = (
        pd.to_numeric(
            frame["pitcher_mlbam_id"],
            errors="raise",
        )
        .astype(int)
    )

    return frame


# ---------------------------------------------------------
# COMPLETED 2026 STARTER DATA
# ---------------------------------------------------------


def load_completed_2026(
    prediction_date: pd.Timestamp,
) -> pd.DataFrame:
    """
    Load every valid completed 2026 starter table available
    before the prediction date.

    Multiple overlapping processed files are allowed.
    Final rows are deduplicated by game_id + pitcher ID.
    """

    frames = []

    paths = sorted(
        LIVE_PROCESSED_DIR.glob(
            "starting_pitcher_games_*.parquet"
        )
    )

    for path in paths:
        match = STARTER_FILE_RE.fullmatch(
            path.name
        )

        if not match:
            continue

        start_date = pd.Timestamp(
            match.group(1)
        )

        end_date = pd.Timestamp(
            match.group(2)
        )

        # Only 2026 live data belongs here.
        if start_date.year != 2026:
            continue

        # A file beginning on or after prediction day
        # cannot contribute pregame information.
        if start_date >= prediction_date:
            continue

        frame = pd.read_parquet(
            path
        )

        # Ignore empty placeholder files returned by
        # unavailable Statcast queries.
        if frame.empty:
            continue

        frame = normalize_appearance_schema(
            frame
        )

        # Strict point-in-time filter even if a file's
        # filename accidentally overlaps the prediction day.
        frame = frame.loc[
            frame["game_date"]
            < prediction_date
        ].copy()

        if frame.empty:
            continue

        frames.append(
            frame
        )

    if not frames:
        raise RuntimeError(
            "No valid completed 2026 starter data "
            "exists before the prediction date."
        )

    completed = pd.concat(
        frames,
        ignore_index=True,
        sort=False,
    )

    completed = (
        completed
        .sort_values(
            [
                "game_date",
                "game_id",
                "pitcher_mlbam_id",
            ]
        )
        .drop_duplicates(
            [
                "game_id",
                "pitcher_mlbam_id",
            ],
            keep="last",
        )
        .reset_index(
            drop=True
        )
    )

    if completed.empty:
        raise RuntimeError(
            "Completed 2026 starter table became "
            "empty after filtering."
        )

    if (
        completed["game_date"]
        >= prediction_date
    ).any():
        raise RuntimeError(
            "POINT-IN-TIME FAILURE: completed starter "
            "data includes prediction date or later."
        )

    return completed


# ---------------------------------------------------------
# RAW STATCAST DISCOVERY
# ---------------------------------------------------------


def discover_raw_2026_paths(
    prediction_date: pd.Timestamp,
) -> list[Path]:
    """
    Return valid non-empty 2026 raw Statcast chunks whose
    entire date range ends before the prediction date.
    """

    valid_paths: list[Path] = []

    for path in sorted(
        RAW_2026_DIR.glob(
            "statcast_*.parquet"
        )
    ):
        match = RAW_FILE_RE.fullmatch(
            path.name
        )

        if not match:
            continue

        start_date = pd.Timestamp(
            match.group(1)
        )

        end_date = pd.Timestamp(
            match.group(2)
        )

        if start_date.year != 2026:
            continue

        # Critical leakage protection:
        # the entire raw chunk must end before the
        # prediction date.
        if end_date >= prediction_date:
            continue

        try:
            dates = pd.read_parquet(
                path,
                columns=[
                    "game_date",
                ],
            )

        except Exception as exc:
            raise RuntimeError(
                f"Could not validate raw Statcast "
                f"chunk {path}: {exc}"
            ) from exc

        # Ignore bogus zero-row downloads.
        if dates.empty:
            continue

        game_dates = pd.to_datetime(
            dates["game_date"]
        ).dt.normalize()

        if (
            game_dates
            >= prediction_date
        ).any():
            raise RuntimeError(
                "POINT-IN-TIME FAILURE: raw Statcast "
                f"chunk contains {prediction_date.date()} "
                f"or later: {path}"
            )

        valid_paths.append(
            path
        )

    if not valid_paths:
        raise RuntimeError(
            "No valid 2026 raw Statcast chunks "
            "were found."
        )

    return valid_paths


# ---------------------------------------------------------
# FROZEN MODEL FEATURE LIST
# ---------------------------------------------------------


def find_feature_columns(
    value,
) -> list[str] | None:
    """
    Recursively locate the frozen feature-column list inside
    the tournament configuration.
    """

    preferred_keys = (
        "feature_columns",
        "frozen_feature_columns",
        "model_feature_columns",
    )

    if isinstance(
        value,
        dict,
    ):
        for key in preferred_keys:
            candidate = value.get(
                key
            )

            if (
                isinstance(
                    candidate,
                    list,
                )
                and candidate
                and all(
                    isinstance(
                        item,
                        str,
                    )
                    for item in candidate
                )
            ):
                return candidate

        for nested in value.values():
            result = find_feature_columns(
                nested
            )

            if result is not None:
                return result

    elif isinstance(
        value,
        list,
    ):
        for nested in value:
            result = find_feature_columns(
                nested
            )

            if result is not None:
                return result

    return None


def load_frozen_feature_columns() -> list[str]:
    if not TOURNAMENT_CONFIG_PATH.exists():
        raise FileNotFoundError(
            "Tournament config not found: "
            f"{TOURNAMENT_CONFIG_PATH}"
        )

    config = json.loads(
        TOURNAMENT_CONFIG_PATH.read_text()
    )

    columns = find_feature_columns(
        config
    )

    if columns is None:
        raise RuntimeError(
            "Could not locate frozen feature columns "
            f"inside {TOURNAMENT_CONFIG_PATH}"
        )

    columns = list(
        dict.fromkeys(
            columns
        )
    )

    if len(columns) != 244:
        raise RuntimeError(
            "Frozen feature schema changed unexpectedly. "
            f"Expected 244 features, found {len(columns)}."
        )

    return columns


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--date",
        required=True,
        help="Prediction date YYYY-MM-DD.",
    )

    args = parser.parse_args()

    prediction_date = pd.Timestamp(
        args.date
    ).normalize()

    pregame_path = (
        LIVE_PROCESSED_DIR
        / (
            f"pregame_starters_"
            f"{args.date}.parquet"
        )
    )

    output_path = (
        LIVE_PROCESSED_DIR
        / (
            f"pregame_features_"
            f"{args.date}.parquet"
        )
    )

    if not HISTORICAL_APPEARANCE_PATH.exists():
        raise FileNotFoundError(
            "Historical starter table not found: "
            f"{HISTORICAL_APPEARANCE_PATH}"
        )

    if not pregame_path.exists():
        raise FileNotFoundError(
            "Pregame starter file not found: "
            f"{pregame_path}"
        )

    # -----------------------------------------------------
    # LOAD APPEARANCES
    # -----------------------------------------------------

    historical = pd.read_parquet(
        HISTORICAL_APPEARANCE_PATH
    )

    historical = normalize_appearance_schema(
        historical
    )

    completed_2026 = load_completed_2026(
        prediction_date
    )

    pregame = pd.read_parquet(
        pregame_path
    )

    pregame = normalize_appearance_schema(
        pregame
    )

    if not (
        pregame["game_date"]
        == prediction_date
    ).all():
        bad_dates = sorted(
            pregame.loc[
                pregame["game_date"]
                != prediction_date,
                "game_date",
            ]
            .dt.date
            .astype(str)
            .unique()
            .tolist()
        )

        raise RuntimeError(
            "Pregame starter file contains dates "
            f"other than {args.date}: {bad_dates}"
        )

    latest_completed = (
        completed_2026[
            "game_date"
        ].max()
    )

    if latest_completed >= prediction_date:
        raise RuntimeError(
            "POINT-IN-TIME FAILURE: latest completed "
            "starter data is not before prediction date."
        )

    # -----------------------------------------------------
    # COMBINE APPEARANCE HISTORY + TODAY'S PREGAME ROWS
    # -----------------------------------------------------

    appearances = pd.concat(
        [
            historical,
            completed_2026,
            pregame,
        ],
        ignore_index=True,
        sort=False,
    )

    appearances = (
        appearances
        .sort_values(
            [
                "game_date",
                "game_id",
                "pitcher_mlbam_id",
            ]
        )
        .drop_duplicates(
            [
                "game_id",
                "pitcher_mlbam_id",
            ],
            keep="last",
        )
        .reset_index(
            drop=True
        )
    )

    raw_paths = discover_raw_2026_paths(
        prediction_date
    )

    # -----------------------------------------------------
    # STATUS
    # -----------------------------------------------------

    print()
    print(
        "LIVE PREGAME FEATURE BUILD"
    )
    print(
        f"Prediction date: "
        f"{prediction_date.date()}"
    )
    print(
        f"Latest completed data: "
        f"{latest_completed.date()}"
    )
    print(
        f"Historical appearances: "
        f"{len(historical)}"
    )
    print(
        f"Completed 2026 starts: "
        f"{len(completed_2026)}"
    )
    print(
        f"Pregame starters: "
        f"{len(pregame)}"
    )
    print(
        f"2026 raw chunks: "
        f"{len(raw_paths)}"
    )
    print()

    # -----------------------------------------------------
    # RAW GAME METRICS
    #
    # build_raw_game_metrics returns:
    #
    # (
    #     pitcher_metrics,
    #     pitch_type_metrics,
    #     team_metrics,
    # )
    # -----------------------------------------------------

    (
        pitcher_metrics,
        pitch_type_metrics,
        team_metrics,
    ) = build_raw_game_metrics(
        raw_paths,
        appearances,
    )

    print(
        f"Pitcher metric rows: "
        f"{len(pitcher_metrics)}"
    )
    print(
        f"Pitch-type metric rows: "
        f"{len(pitch_type_metrics)}"
    )
    print(
        f"Team metric rows: "
        f"{len(team_metrics)}"
    )
    print()

    # -----------------------------------------------------
    # POINT-IN-TIME FEATURE ENGINE
    # -----------------------------------------------------

    features = build_point_in_time_features(
        appearances,
        pitcher_metrics,
        pitch_type_metrics,
        team_metrics,
    )

    # -----------------------------------------------------
    # EXTRACT ONLY TODAY'S PREGAME STARTERS
    # -----------------------------------------------------

    pregame_keys = (
        pregame[
            [
                "game_id",
                "pitcher_mlbam_id",
            ]
        ]
        .drop_duplicates()
    )

    live = pregame_keys.merge(
        features,
        on=[
            "game_id",
            "pitcher_mlbam_id",
        ],
        how="left",
        validate="one_to_one",
    )

    if len(live) != len(
        pregame_keys
    ):
        raise RuntimeError(
            "Pregame feature row count does not "
            "match unique pregame starter count."
        )

    # Make sure every starter actually found a
    # generated feature row.
    if "game_date" in live.columns:
        unmatched = (
            live["game_date"]
            .isna()
        )

        if unmatched.any():
            raise RuntimeError(
                f"{int(unmatched.sum())} pregame "
                "starters failed to generate features."
            )

    # -----------------------------------------------------
    # VALIDATE FROZEN PRODUCTION FEATURE SCHEMA
    # -----------------------------------------------------

    frozen_features = (
        load_frozen_feature_columns()
    )

    missing_frozen = [
        column
        for column in frozen_features
        if column not in live.columns
    ]

    if missing_frozen:
        raise RuntimeError(
            "Missing frozen production features: "
            f"{missing_frozen}"
        )

    if len(frozen_features) != 244:
        raise RuntimeError(
            "Expected exactly 244 frozen "
            "production features."
        )

    # -----------------------------------------------------
    # TARGET LEAKAGE CHECK
    # -----------------------------------------------------

    target_columns = [
        "strikeouts",
        "outs_recorded",
        "pitches_thrown",
        "batters_faced",
    ]

    for column in target_columns:
        if column not in live.columns:
            raise RuntimeError(
                f"Expected target column "
                f"'{column}' is missing."
            )

        if live[column].notna().any():
            bad = int(
                live[
                    column
                ].notna().sum()
            )

            raise RuntimeError(
                "POINT-IN-TIME FAILURE: "
                f"pregame target '{column}' "
                f"contains {bad} non-null values."
            )

    # -----------------------------------------------------
    # FEATURE-MISSINGNESS DIAGNOSTIC
    # -----------------------------------------------------

    live[
        "missing_features"
    ] = (
        live[
            frozen_features
        ]
        .isna()
        .sum(
            axis=1
        )
    )

    # -----------------------------------------------------
    # SAVE
    # -----------------------------------------------------

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    live.to_parquet(
        output_path,
        index=False,
    )

    # -----------------------------------------------------
    # FINAL VALIDATION OUTPUT
    # -----------------------------------------------------

    print(
        "PREGAME FEATURE BUILD COMPLETE"
    )

    print(
        f"Rows: {len(live)}"
    )

    print(
        f"Frozen features: "
        f"{len(frozen_features)}"
    )

    print(
        f"Missing frozen columns: "
        f"{len(missing_frozen)}"
    )

    print(
        "Mean missing features/start: "
        f"{live['missing_features'].mean():.2f}"
    )

    print(
        "Maximum missing features/start: "
        f"{int(live['missing_features'].max())}"
    )

    preview_columns = [
        column
        for column in [
            "pitcher_name",
            "team",
            "opponent",
            "home_away",
            "pitcher_starts_before_game",
            "pitches_previous_start",
            "season_k_pct",
            "season_outs_per_start",
            "missing_features",
        ]
        if column in live.columns
    ]

    if preview_columns:
        print()
        print(
            live[
                preview_columns
            ].to_string(
                index=False
            )
        )

    print()
    print(
        f"Output: {output_path}"
    )


if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()