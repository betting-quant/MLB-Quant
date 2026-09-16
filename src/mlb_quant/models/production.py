from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

from mlb_quant.models.tournament import _train_final_model
from mlb_quant.models.distribution import fit_distribution


FEATURE_PATH = Path("data/processed/model_features_2021_2025.parquet")
PREDICTION_PATH = Path("data/processed/model_tournament_predictions_v0_1.parquet")

TOURNAMENT_CONFIG_PATH = Path("reports/model_tournament_config_v0_1.json")
DISTRIBUTION_EVAL_PATH = Path("models/artifacts/mlb_distribution_v0_1.joblib")

ARTIFACT_ROOT = Path("models/artifacts")
REPORT_PATH = Path("reports/production_2026_config.json")


ARTIFACT_NAMES = {
    "strikeouts": "mlb_k_quant_production_2026.joblib",
    "outs_recorded": "mlb_outs_quant_production_2026.joblib",
    "batters_faced": "mlb_bf_quant_production_2026.joblib",
    "pitches_thrown": "mlb_pitches_quant_production_2026.joblib",
}


def main() -> None:
    frame = pd.read_parquet(FEATURE_PATH).sort_values(
        ["season", "game_date", "game_id", "pitcher_mlbam_id"]
    ).reset_index(drop=True)

    config = json.loads(TOURNAMENT_CONFIG_PATH.read_text())
    winners = config["winners"]
    feature_columns = config["feature_columns"]

    missing = [c for c in feature_columns if c not in frame.columns]
    if missing:
        raise ValueError(
            f"Production feature table is missing {len(missing)} frozen features: "
            f"{missing[:20]}"
        )

    production_train = frame[frame["season"] <= 2025].copy()

    if production_train.empty:
        raise ValueError("No 2021-2025 production training data found.")

    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)

    point_model_report = {}

    for target, filename in ARTIFACT_NAMES.items():
        winner = winners[target]

        bundle = _train_final_model(
            production_train,
            target,
            winner,
            feature_columns,
            max_train_season=2025,
        )

        output_path = ARTIFACT_ROOT / filename
        joblib.dump(bundle, output_path)

        point_model_report[target] = {
            "winner": winner,
            "artifact": str(output_path),
            "training_seasons": sorted(
                production_train["season"].dropna().astype(int).unique().tolist()
            ),
            "training_rows": int(len(production_train)),
            "feature_count": int(len(feature_columns)),
        }

        print(
            f"{target}: {winner} -> {output_path}"
        )

    # ---------------------------------------------------------
    # Production probability distributions
    #
    # Use ONLY genuinely out-of-sample residuals:
    # 2022, 2023, 2024 chronological validation predictions
    # plus the untouched 2025 holdout predictions.
    #
    # Distribution method itself was already selected before
    # incorporating 2025.
    # ---------------------------------------------------------

    historical_predictions = pd.read_parquet(PREDICTION_PATH)
    evaluation_distribution = joblib.load(DISTRIBUTION_EVAL_PATH)

    production_distributions = {}
    distribution_report = {}

    for target in ["strikeouts", "outs_recorded"]:
        eval_bundle = evaluation_distribution[target]

        method = eval_bundle["method"]
        point_model = eval_bundle["model"]
        target_config = eval_bundle["config"]

        residual_history = historical_predictions[
            (historical_predictions["target"] == target)
            & (historical_predictions["model"] == point_model)
            & (historical_predictions["season"] <= 2025)
        ].copy()

        residual_history = residual_history.sort_values(
            ["season", "game_id", "pitcher_mlbam_id"]
        ).reset_index(drop=True)

        duplicates = residual_history.duplicated(
            ["season", "game_id", "pitcher_mlbam_id"]
        ).sum()

        if duplicates:
            raise ValueError(
                f"{target}: found {duplicates} duplicate OOF residual rows."
            )

        if residual_history.empty:
            raise ValueError(
                f"{target}: no out-of-sample residual history found."
            )

        fitted_distribution = fit_distribution(
            method,
            residual_history,
        )

        production_distributions[target] = {
            "method": method,
            "model": point_model,
            "distribution": fitted_distribution,
            "config": target_config,
            "residual_seasons": sorted(
                residual_history["season"]
                .dropna()
                .astype(int)
                .unique()
                .tolist()
            ),
        }

        distribution_report[target] = {
            "method": method,
            "point_model": point_model,
            "residual_rows": int(len(residual_history)),
            "residual_seasons": sorted(
                residual_history["season"]
                .dropna()
                .astype(int)
                .unique()
                .tolist()
            ),
        }

        print(
            f"{target} distribution: {method} "
            f"using {len(residual_history)} OOS residuals"
        )

    distribution_path = (
        ARTIFACT_ROOT / "mlb_distribution_production_2026.joblib"
    )

    joblib.dump(
        production_distributions,
        distribution_path,
    )

    production_report = {
        "version": "production_2026_v0_1",
        "purpose": "Forward-looking 2026 MLB pitcher prop prediction",
        "architecture_selection": "Frozen from chronological 2021-2024 development and 2025 holdout evaluation",
        "production_training_through": 2025,
        "feature_columns_frozen_from_evaluation": True,
        "feature_count": len(feature_columns),
        "point_models": point_model_report,
        "distributions": distribution_report,
        "distribution_artifact": str(distribution_path),
    }

    REPORT_PATH.write_text(
        json.dumps(production_report, indent=2),
        encoding="utf-8",
    )

    print()
    print("PRODUCTION 2026 BUILD COMPLETE")
    print(f"Training rows: {len(production_train)}")
    print(f"Frozen features: {len(feature_columns)}")
    print(f"Distribution artifact: {distribution_path}")
    print(f"Report: {REPORT_PATH}")


if __name__ == "__main__":
    main()