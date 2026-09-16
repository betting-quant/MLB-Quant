from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from mlb_quant.models.tournament import _predict_final
from mlb_quant.models.distribution import (
    poisson_probability,
    negative_binomial_probability,
    safe_probability,
)


MODEL_PATHS = {
    "strikeouts": Path(
        "models/artifacts/mlb_k_quant_production_2026.joblib"
    ),
    "outs_recorded": Path(
        "models/artifacts/mlb_outs_quant_production_2026.joblib"
    ),
    "batters_faced": Path(
        "models/artifacts/mlb_bf_quant_production_2026.joblib"
    ),
    "pitches_thrown": Path(
        "models/artifacts/mlb_pitches_quant_production_2026.joblib"
    ),
}

DISTRIBUTION_PATH = Path(
    "models/artifacts/mlb_distribution_production_2026.joblib"
)


def probability_over(
    prediction: np.ndarray,
    line: float,
    method: str,
    fitted_object,
    config: dict,
) -> np.ndarray:
    if method == "poisson":
        probabilities = poisson_probability(
            prediction,
            line,
        )

    elif method == "negative_binomial":
        probabilities = negative_binomial_probability(
            prediction,
            line,
            fitted_object["alpha"],
        )

    elif method in {
        "empirical",
        "conditional_empirical",
    }:
        probabilities = fitted_object.probability_over(
            prediction,
            line,
            config["min_value"],
            config["max_value"],
        )

    else:
        raise ValueError(
            f"Unknown distribution method: {method}"
        )

    return safe_probability(probabilities)


def fair_american_odds(probability: float) -> float:
    p = float(
        np.clip(
            probability,
            1e-6,
            1 - 1e-6,
        )
    )

    if p >= 0.5:
        return -100.0 * p / (1.0 - p)

    return 100.0 * (1.0 - p) / p


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--date",
        required=True,
        help="Prediction date YYYY-MM-DD.",
    )

    args = parser.parse_args()

    feature_path = Path(
        f"data/processed/live_2026/"
        f"pregame_features_{args.date}.parquet"
    )

    point_output = Path(
        f"data/processed/live_2026/"
        f"pregame_predictions_{args.date}.parquet"
    )

    point_csv = Path(
        f"reports/live_predictions_{args.date}.csv"
    )

    probability_output = Path(
        f"data/processed/live_2026/"
        f"pregame_probabilities_{args.date}.parquet"
    )

    probability_csv = Path(
        f"reports/live_probabilities_{args.date}.csv"
    )

    if not feature_path.exists():
        raise FileNotFoundError(
            f"Pregame feature file not found: "
            f"{feature_path}"
        )

    frame = pd.read_parquet(
        feature_path
    )

    if frame.empty:
        raise ValueError(
            "Pregame feature table is empty."
        )

    frame["game_date"] = pd.to_datetime(
        frame["game_date"]
    )

    expected_date = pd.Timestamp(args.date)

    if not frame["game_date"].eq(
        expected_date
    ).all():
        raise ValueError(
            "Feature table contains another date."
        )

    # Pregame outcomes must be unknown.
    for target in [
        "strikeouts",
        "outs_recorded",
        "batters_faced",
        "pitches_thrown",
    ]:
        if frame[target].notna().any():
            raise ValueError(
                f"Pregame leakage detected: "
                f"{target} contains actual outcomes."
            )

    loaded_models = {}

    frozen_columns = None

    for target, path in MODEL_PATHS.items():
        bundle = joblib.load(path)

        columns = bundle["feature_columns"]

        if frozen_columns is None:
            frozen_columns = columns

        elif columns != frozen_columns:
            raise ValueError(
                f"Feature schema mismatch in {path}"
            )

        loaded_models[target] = bundle

    missing_columns = [
        column
        for column in frozen_columns
        if column not in frame.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing {len(missing_columns)} "
            f"production features: "
            f"{missing_columns[:30]}"
        )

    print("LIVE PRODUCTION PREDICTION")
    print(f"Date: {args.date}")
    print(f"Starters: {len(frame)}")
    print(
        f"Frozen features: {len(frozen_columns)}"
    )
    print()

    point = frame[
        [
            "game_date",
            "game_id",
            "pitcher_mlbam_id",
            "pitcher_name",
            "team",
            "opponent",
            "home_away",
            "pitcher_handedness",
            "pitcher_starts_before_game",
        ]
    ].copy()

    point["missing_features"] = (
        frame[frozen_columns]
        .isna()
        .sum(axis=1)
        .to_numpy()
    )

    for target, bundle in loaded_models.items():
        prediction = _predict_final(
            bundle,
            frame,
            target,
        )

        point[f"projected_{target}"] = (
            prediction
        )

    # Save point predictions.
    point_output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    point_csv.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    point.to_parquet(
        point_output,
        index=False,
    )

    point.to_csv(
        point_csv,
        index=False,
    )

    # --------------------------------------------------
    # Probability distributions
    # --------------------------------------------------

    distributions = joblib.load(
        DISTRIBUTION_PATH
    )

    probability_rows = []

    for target in [
        "strikeouts",
        "outs_recorded",
    ]:
        dist_bundle = distributions[target]

        method = dist_bundle["method"]
        fitted_object = dist_bundle["distribution"]
        config = dist_bundle["config"]

        prediction = point[
            f"projected_{target}"
        ].to_numpy(dtype=float)

        for line in config["lines"]:
            over = probability_over(
                prediction,
                float(line),
                method,
                fitted_object,
                config,
            )

            under = 1.0 - over

            for idx in range(len(point)):
                probability_rows.append(
                    {
                        "game_date": point.iloc[
                            idx
                        ]["game_date"],
                        "game_id": int(
                            point.iloc[idx][
                                "game_id"
                            ]
                        ),
                        "pitcher_mlbam_id": int(
                            point.iloc[idx][
                                "pitcher_mlbam_id"
                            ]
                        ),
                        "pitcher_name": point.iloc[
                            idx
                        ]["pitcher_name"],
                        "team": point.iloc[
                            idx
                        ]["team"],
                        "opponent": point.iloc[
                            idx
                        ]["opponent"],
                        "target": target,
                        "distribution_method": method,
                        "projection": float(
                            prediction[idx]
                        ),
                        "line": float(line),
                        "probability_over": float(
                            over[idx]
                        ),
                        "probability_under": float(
                            under[idx]
                        ),
                        "fair_odds_over": (
                            fair_american_odds(
                                over[idx]
                            )
                        ),
                        "fair_odds_under": (
                            fair_american_odds(
                                under[idx]
                            )
                        ),
                    }
                )

    probabilities = pd.DataFrame(
        probability_rows
    )

    probability_output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    probability_csv.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    probabilities.to_parquet(
        probability_output,
        index=False,
    )

    probabilities.to_csv(
        probability_csv,
        index=False,
    )

    # --------------------------------------------------
    # Terminal preview
    # --------------------------------------------------

    display = point[
        [
            "pitcher_name",
            "team",
            "opponent",
            "projected_strikeouts",
            "projected_outs_recorded",
            "projected_batters_faced",
            "projected_pitches_thrown",
            "pitcher_starts_before_game",
            "missing_features",
        ]
    ].copy()

    display = display.rename(
        columns={
            "projected_strikeouts": "K",
            "projected_outs_recorded": "Outs",
            "projected_batters_faced": "BF",
            "projected_pitches_thrown": "Pitches",
            "pitcher_starts_before_game": "Starts",
            "missing_features": "Missing",
        }
    )

    for column in [
        "K",
        "Outs",
        "BF",
        "Pitches",
    ]:
        display[column] = (
            display[column].round(2)
        )

    print("POINT PROJECTIONS")
    print()
    print(
        display.to_string(
            index=False
        )
    )

    print()
    print("PROBABILITY TABLE COMPLETE")
    print(
        f"Probability rows: "
        f"{len(probabilities)}"
    )
    print()
    print(
        f"Point predictions: {point_output}"
    )
    print(
        f"Point CSV: {point_csv}"
    )
    print(
        f"Probability predictions: "
        f"{probability_output}"
    )
    print(
        f"Probability CSV: "
        f"{probability_csv}"
    )


if __name__ == "__main__":
    main()