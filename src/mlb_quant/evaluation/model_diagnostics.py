"""Post-hoc diagnostics for the frozen model tournament.

These functions only read saved artifacts, predictions, and the feature
table. They never fit, tune, or otherwise modify a model. Any computation
touching the 2025 holdout here is reporting, not model selection: the
holdout predictions were produced once by models already frozen on
2021-2024 (see ``mlb_quant.models.tournament``).
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

TARGET_TO_ARTIFACT = {
    "strikeouts": "mlb_k_quant_v0_1.joblib",
    "outs_recorded": "mlb_outs_quant_v0_1.joblib",
    "batters_faced": "mlb_bf_quant_v0_1.joblib",
    "pitches_thrown": "mlb_pitches_quant_v0_1.joblib",
}


def _mae_bias(group: pd.DataFrame) -> pd.Series:
    error = group["prediction"] - group["actual"]
    return pd.Series({"mae": error.abs().mean(), "bias": error.mean(), "n": len(group)})


def bucketed_errors(predictions: pd.DataFrame, features: pd.DataFrame, winners: dict[str, str]) -> pd.DataFrame:
    """Break winner-model errors down by season, hand, projection tier, and experience."""
    merged = predictions.merge(
        features[["season", "game_id", "pitcher_mlbam_id", "pitcher_handedness", "pitcher_starts_before_game"]],
        on=["season", "game_id", "pitcher_mlbam_id"],
        how="left",
    )
    rows = []
    for target, model_name in winners.items():
        subset = merged[(merged.target == target) & (merged.model == model_name)].copy()
        if subset.empty:
            continue
        subset["experience_bucket"] = np.where(subset["pitcher_starts_before_game"] < 5, "early_season", "established")
        subset["projection_tier"] = pd.qcut(subset["prediction"], 3, labels=["low", "medium", "high"], duplicates="drop")
        for bucket_type, column in [
            ("season", "season"),
            ("pitcher_handedness", "pitcher_handedness"),
            ("projection_tier", "projection_tier"),
            ("experience_bucket", "experience_bucket"),
        ]:
            for split_name, split_frame in subset.groupby("split", observed=True):
                for bucket_value, bucket_frame in split_frame.groupby(column, observed=True):
                    metrics = _mae_bias(bucket_frame)
                    rows.append({
                        "target": target, "model": model_name, "split": split_name,
                        "bucket_type": bucket_type, "bucket_value": str(bucket_value), **metrics,
                    })
    return pd.DataFrame(rows)


def residual_summary(predictions: pd.DataFrame, winners: dict[str, str]) -> pd.DataFrame:
    """Report residual shape and extreme-prediction behavior for each winner."""
    rows = []
    for target, model_name in winners.items():
        for split_name, group in predictions[(predictions.target == target) & (predictions.model == model_name)].groupby("split", observed=True):
            residual = group["prediction"] - group["actual"]
            std = residual.std() or np.nan
            extreme = (residual.abs() > 3 * std).sum() if std and not np.isnan(std) else 0
            rows.append({
                "target": target, "model": model_name, "split": split_name,
                "residual_mean": residual.mean(), "residual_std": residual.std(),
                "residual_skew": residual.skew(), "residual_p05": residual.quantile(0.05),
                "residual_p95": residual.quantile(0.95),
                "extreme_residual_count_gt3sd": int(extreme),
                "zero_clipped_predictions": int((group["prediction"] <= 0).sum()),
                "n": len(group),
            })
    return pd.DataFrame(rows)


def permutation_importances(
    artifact_root: Path,
    features: pd.DataFrame,
    feature_columns: list[str],
    winners: dict[str, str],
    holdout_season: int = 2025,
    top_n: int = 15,
    n_repeats: int = 3,
    random_state: int = 42,
    max_rows: int = 1500,
) -> pd.DataFrame:
    """Diagnostic permutation importance of frozen models on the holdout rows.

    This evaluates already-fitted pipelines; it does not refit or select
    features, and results are not fed back into model selection. A capped,
    randomly sampled subset of holdout rows keeps this diagnostic tractable;
    the sample is drawn without regard to prediction accuracy.
    """
    holdout = features[features.season == holdout_season]
    if len(holdout) > max_rows:
        holdout = holdout.sample(n=max_rows, random_state=random_state)
    rows = []
    for target, model_name in winners.items():
        bundle = joblib.load(Path(artifact_root) / TARGET_TO_ARTIFACT[target])
        if bundle["type"] not in {"ridge", "poisson", "hist_gradient_boosting"}:
            continue
        estimator = bundle["model"]
        result = permutation_importance(
            estimator, holdout[feature_columns], holdout[target],
            n_repeats=n_repeats, random_state=random_state, scoring="neg_mean_absolute_error",
            n_jobs=-1,
        )
        importance = pd.Series(result.importances_mean, index=feature_columns).sort_values(ascending=False)
        for rank, (feature, value) in enumerate(importance.head(top_n).items(), start=1):
            rows.append({"target": target, "model": model_name, "rank": rank, "feature": feature, "importance_mae_increase": float(value)})
    return pd.DataFrame(rows)


def overfitting_check(validation: pd.DataFrame, holdout: pd.DataFrame, winners: dict[str, str]) -> pd.DataFrame:
    """Compare the last chronological validation fold to the untouched 2025 holdout."""
    rows = []
    for target, model_name in winners.items():
        last_fold = validation[
            (validation.target == target) & (validation.model == model_name) & (validation.split == "train_2021_2023_validate_2024")
        ]
        holdout_row = holdout[(holdout.target == target) & (holdout.model == model_name)]
        if last_fold.empty or holdout_row.empty:
            continue
        last_mae = float(last_fold["mae"].iloc[0])
        holdout_mae = float(holdout_row["mae"].iloc[0])
        rows.append({
            "target": target, "model": model_name,
            "validation_2024_mae": last_mae, "holdout_2025_mae": holdout_mae,
            "gap": holdout_mae - last_mae,
            "gap_pct": (holdout_mae - last_mae) / last_mae * 100 if last_mae else np.nan,
        })
    return pd.DataFrame(rows)


def ablation_family_value(ablation: pd.DataFrame) -> pd.DataFrame:
    """Marginal MAE change moving through each cumulative feature family."""
    order = ["pitcher_history", "workload", "opponent", "recent", "full"]
    means = ablation.groupby(["target", "feature_family"])["mae"].mean().unstack()[order]
    deltas = means.diff(axis=1).drop(columns=["pitcher_history"])
    deltas.columns = [f"delta_vs_prior_adding_{name}" for name in deltas.columns]
    return means.join(deltas).reset_index()


def run_diagnostics(
    predictions_path: Path = Path("data/processed/model_tournament_predictions_v0_1.parquet"),
    features_path: Path = Path("data/processed/model_features_2021_2025.parquet"),
    config_path: Path = Path("reports/model_tournament_config_v0_1.json"),
    validation_path: Path = Path("reports/model_tournament_validation_v0_1.csv"),
    holdout_path: Path = Path("reports/model_tournament_holdout_2025_v0_1.csv"),
    ablation_path: Path = Path("reports/model_tournament_ablation_v0_1.csv"),
    artifact_root: Path = Path("models/artifacts"),
    output_path: Path = Path("reports/model_diagnostics_v0_1.md"),
) -> dict[str, pd.DataFrame]:
    config = json.loads(Path(config_path).read_text())
    winners = config["winners"]
    feature_columns = config["feature_columns"]
    predictions = pd.read_parquet(predictions_path)
    features = pd.read_parquet(features_path)
    validation = pd.read_csv(validation_path)
    holdout = pd.read_csv(holdout_path)
    ablation = pd.read_csv(ablation_path)

    buckets = bucketed_errors(predictions, features, winners)
    residuals = residual_summary(predictions, winners)
    importances = permutation_importances(artifact_root, features, feature_columns, winners)
    overfit = overfitting_check(validation, holdout, winners)
    family_value = ablation_family_value(ablation)

    buckets.to_csv(Path("reports/model_tournament_buckets_v0_1.csv"), index=False)
    residuals.to_csv(Path("reports/model_tournament_residuals_v0_1.csv"), index=False)
    importances.to_csv(Path("reports/model_tournament_permutation_importance_v0_1.csv"), index=False)
    overfit.to_csv(Path("reports/model_tournament_overfitting_check_v0_1.csv"), index=False)

    lines = ["# MLB Model Tournament v0.1 -- Diagnostics", "", "Diagnostic-only report computed from frozen, already-saved models and", "predictions. Nothing here retrains, re-tunes, or reselects a model, and", "the 2025 holdout was not used to change any modeling decision.", ""]
    lines += ["## Overfitting check (last walk-forward fold vs untouched 2025)", "", overfit.to_string(index=False), ""]
    lines += ["## Cumulative feature-family value (mean walk-forward MAE)", "", family_value.to_string(index=False), ""]
    lines += ["## Residual summary (winner models)", "", residuals.to_string(index=False), ""]
    lines += ["## Error buckets (winner models, mean MAE/bias)", "", buckets.to_string(index=False), ""]
    lines += ["## Permutation importance (top 15 features per target, holdout evaluation only)", "", importances.to_string(index=False), ""]
    Path(output_path).write_text("\n".join(lines) + "\n")
    return {
        "buckets": buckets, "residuals": residuals, "importances": importances,
        "overfitting": overfit, "family_value": family_value,
    }


if __name__ == "__main__":
    run_diagnostics()
