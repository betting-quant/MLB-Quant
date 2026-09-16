from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from mlb_quant.models.tournament import (
    _predict_final,
    score_predictions,
)
from mlb_quant.models.distribution import (
    probability_dataset,
    score_probability_frame,
    calibration_table,
)


FEATURE_PATH = Path(
    "data/processed/model_features_2026_through_2026-09-14.parquet"
)

DISTRIBUTION_PATH = Path(
    "models/artifacts/mlb_distribution_production_2026.joblib"
)

PREDICTION_OUTPUT = Path(
    "data/processed/forward_2026_predictions_v0_1.parquet"
)

PROBABILITY_OUTPUT = Path(
    "data/processed/forward_2026_probability_predictions_v0_1.parquet"
)

POINT_REPORT_CSV = Path(
    "reports/forward_2026_point_v0_1.csv"
)

PROB_REPORT_CSV = Path(
    "reports/forward_2026_probability_v0_1.csv"
)

PER_LINE_REPORT_CSV = Path(
    "reports/forward_2026_probability_by_line_v0_1.csv"
)

REPORT_PATH = Path(
    "reports/forward_2026_v0_1.md"
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


def weighted_calibration_error(table: pd.DataFrame) -> float:
    valid = table.dropna(
        subset=[
            "predicted_probability",
            "observed_rate",
        ]
    ).copy()

    if valid.empty or valid["n"].sum() == 0:
        return float("nan")

    error = (
        valid["predicted_probability"]
        - valid["observed_rate"]
    ).abs()

    weights = valid["n"] / valid["n"].sum()

    return float((error * weights).sum())


def monotonicity_violations(
    probability_frame: pd.DataFrame,
) -> int:
    pivot = probability_frame.pivot_table(
        index=[
            "season",
            "game_id",
            "pitcher_mlbam_id",
        ],
        columns="line",
        values="probability_over",
        aggfunc="first",
    )

    pivot = pivot.reindex(
        sorted(pivot.columns),
        axis=1,
    )

    values = pivot.to_numpy(dtype=float)

    if values.shape[1] < 2:
        return 0

    # As the prop line increases, P(Over) must never increase.
    changes = np.diff(values, axis=1)

    return int(
        np.nansum(changes > 1e-12)
    )


def main() -> None:
    frame = pd.read_parquet(FEATURE_PATH)

    frame["game_date"] = pd.to_datetime(
        frame["game_date"]
    )

    if not frame["season"].eq(2026).all():
        raise ValueError(
            "Forward-test file contains a non-2026 season."
        )

    print("2026 FORWARD TEST")
    print(f"Starts: {len(frame)}")
    print(
        "Date range:",
        frame["game_date"].min(),
        "through",
        frame["game_date"].max(),
    )
    print()

    point_results = []
    prediction_frames = []

    for target, artifact_path in MODEL_PATHS.items():
        bundle = joblib.load(artifact_path)

        missing = [
            column
            for column in bundle["feature_columns"]
            if column not in frame.columns
        ]

        if missing:
            raise ValueError(
                f"{target}: missing production features "
                f"{missing[:20]}"
            )

        prediction = _predict_final(
            bundle,
            frame,
            target,
        )

        score = score_predictions(
            frame[target],
            prediction,
        )

        point_results.append(
            {
                "target": target,
                "model": bundle["type"],
                **score,
            }
        )

        part = frame[
            [
                "season",
                "game_date",
                "game_id",
                "pitcher_mlbam_id",
                "opponent",
                "home_away",
                "pitcher_handedness",
            ]
        ].copy()

        part["target"] = target
        part["model"] = bundle["type"]
        part["actual"] = frame[target].to_numpy()
        part["prediction"] = prediction
        part["error"] = (
            part["prediction"]
            - part["actual"]
        )

        prediction_frames.append(part)

    point_results = pd.DataFrame(point_results)
    point_predictions = pd.concat(
        prediction_frames,
        ignore_index=True,
    )

    distribution_bundle = joblib.load(
        DISTRIBUTION_PATH
    )

    probability_results = []
    probability_frames = []
    per_line_results = []
    calibration_tables = {}

    for target in [
        "strikeouts",
        "outs_recorded",
    ]:
        bundle = distribution_bundle[target]

        point_frame = point_predictions[
            point_predictions["target"] == target
        ][
            [
                "season",
                "game_id",
                "pitcher_mlbam_id",
                "actual",
                "prediction",
            ]
        ].copy()

        probability_frame = probability_dataset(
            point_frame,
            bundle["method"],
            bundle["distribution"],
            bundle["config"],
        )

        probability_frame["target"] = target

        overall_score = score_probability_frame(
            probability_frame
        )

        calibration = calibration_table(
            probability_frame
        )

        calibration["target"] = target

        ece = weighted_calibration_error(
            calibration
        )

        violations = monotonicity_violations(
            probability_frame
        )

        probability_results.append(
            {
                "target": target,
                "method": bundle["method"],
                "brier": overall_score["brier"],
                "log_loss": overall_score["log_loss"],
                "n_binary_observations": overall_score["n"],
                "expected_calibration_error": ece,
                "probability_std": float(
                    probability_frame[
                        "probability_over"
                    ].std()
                ),
                "monotonicity_violations": violations,
            }
        )

        for line, group in probability_frame.groupby(
            "line"
        ):
            line_score = score_probability_frame(
                group
            )

            per_line_results.append(
                {
                    "target": target,
                    "method": bundle["method"],
                    "line": float(line),
                    "brier": line_score["brier"],
                    "log_loss": line_score["log_loss"],
                    "n": line_score["n"],
                    "mean_predicted_over": float(
                        group[
                            "probability_over"
                        ].mean()
                    ),
                    "actual_over_rate": float(
                        group["actual_over"].mean()
                    ),
                    "calibration_error": float(
                        group[
                            "probability_over"
                        ].mean()
                        - group[
                            "actual_over"
                        ].mean()
                    ),
                }
            )

        calibration_tables[target] = calibration
        probability_frames.append(
            probability_frame
        )

    probability_results = pd.DataFrame(
        probability_results
    )

    per_line_results = pd.DataFrame(
        per_line_results
    )

    all_probability_predictions = pd.concat(
        probability_frames,
        ignore_index=True,
    )

    PREDICTION_OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    POINT_REPORT_CSV.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    point_predictions.to_parquet(
        PREDICTION_OUTPUT,
        index=False,
    )

    all_probability_predictions.to_parquet(
        PROBABILITY_OUTPUT,
        index=False,
    )

    point_results.to_csv(
        POINT_REPORT_CSV,
        index=False,
    )

    probability_results.to_csv(
        PROB_REPORT_CSV,
        index=False,
    )

    per_line_results.to_csv(
        PER_LINE_REPORT_CSV,
        index=False,
    )

    report = [
        "# MLB 2026 Forward Test v0.1",
        "",
        "Production models were trained on 2021-2025 only.",
        "",
        "2026 outcomes were not used for model fitting, "
        "model selection, feature selection, or "
        "distribution fitting.",
        "",
        f"Starts evaluated: {len(frame)}",
        "",
        "## Point-model performance",
        "",
        point_results.to_markdown(index=False),
        "",
        "## Probability performance",
        "",
        probability_results.to_markdown(index=False),
        "",
        "## Probability performance by line",
        "",
        per_line_results.to_markdown(index=False),
        "",
    ]

    for target, table in calibration_tables.items():
        report.extend(
            [
                f"## {target} calibration",
                "",
                table.to_markdown(index=False),
                "",
            ]
        )

    REPORT_PATH.write_text(
        "\n".join(report),
        encoding="utf-8",
    )

    print("POINT MODEL RESULTS")
    print(
        point_results.to_string(
            index=False
        )
    )

    print()
    print("PROBABILITY RESULTS")
    print(
        probability_results.to_string(
            index=False
        )
    )

    print()
    print("FORWARD TEST COMPLETE")
    print(f"Point predictions: {PREDICTION_OUTPUT}")
    print(
        f"Probability predictions: "
        f"{PROBABILITY_OUTPUT}"
    )
    print(f"Report: {REPORT_PATH}")


if __name__ == "__main__":
    main()