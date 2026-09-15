import hashlib
from pathlib import Path

import pandas as pd

from mlb_quant.evaluation.model_diagnostics import (
    bucketed_errors,
    overfitting_check,
    residual_summary,
)


def _synthetic_predictions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "season": [2024, 2024, 2025, 2025],
            "game_id": [1, 2, 3, 4],
            "pitcher_mlbam_id": [10, 11, 10, 11],
            "target": ["strikeouts"] * 4,
            "model": ["hist_gradient_boosting"] * 4,
            "split": ["train_2021_2023_validate_2024", "train_2021_2023_validate_2024", "holdout_2025", "holdout_2025"],
            "actual": [5, 6, 4, 7],
            "prediction": [5.5, 5.0, 4.5, 6.0],
        }
    )


def _synthetic_features() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "season": [2024, 2024, 2025, 2025],
            "game_id": [1, 2, 3, 4],
            "pitcher_mlbam_id": [10, 11, 10, 11],
            "pitcher_handedness": ["R", "L", "R", "L"],
            "pitcher_starts_before_game": [2, 12, 3, 20],
        }
    )


def test_bucketed_errors_uses_only_matching_split_rows():
    predictions = _synthetic_predictions()
    features = _synthetic_features()
    buckets = bucketed_errors(predictions, features, {"strikeouts": "hist_gradient_boosting"})
    holdout_rows = buckets[buckets.split == "holdout_2025"]
    assert set(holdout_rows.bucket_type) == {"season", "pitcher_handedness", "projection_tier", "experience_bucket"}
    experience = holdout_rows[holdout_rows.bucket_type == "experience_bucket"]
    assert set(experience.bucket_value) == {"early_season", "established"}


def test_residual_summary_flags_zero_clipped_and_extreme_predictions():
    predictions = _synthetic_predictions()
    summary = residual_summary(predictions, {"strikeouts": "hist_gradient_boosting"})
    assert (summary["n"] == 2).all()
    assert "zero_clipped_predictions" in summary.columns


def test_overfitting_check_compares_last_fold_to_holdout_only():
    validation = pd.DataFrame(
        {
            "target": ["strikeouts"],
            "model": ["hist_gradient_boosting"],
            "split": ["train_2021_2023_validate_2024"],
            "mae": [1.5],
        }
    )
    holdout = pd.DataFrame({"target": ["strikeouts"], "model": ["hist_gradient_boosting"], "mae": [1.65]})
    result = overfitting_check(validation, holdout, {"strikeouts": "hist_gradient_boosting"})
    row = result.iloc[0]
    assert row["validation_2024_mae"] == 1.5
    assert row["holdout_2025_mae"] == 1.65
    assert round(row["gap"], 4) == 0.15


def test_diagnostics_never_modifies_saved_model_artifacts():
    artifact_root = Path("models/artifacts")
    artifacts = sorted(artifact_root.glob("mlb_*_quant_v0_1.joblib"))
    assert artifacts, "expected frozen artifacts from the tournament run"
    before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in artifacts}
    from mlb_quant.evaluation import model_diagnostics  # noqa: F401  (import-only, no retraining)

    after = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in artifacts}
    assert before == after
