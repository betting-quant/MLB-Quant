import warnings

import numpy as np
import pandas as pd

from mlb_quant.models.tournament import (
    HOLDOUT_SEASON,
    TARGETS,
    _fold_feature_columns,
    _predict_final,
    _train_final_model,
    chronological_splits,
    make_estimator,
    model_feature_columns,
    score_predictions,
)


def test_chronological_splits_hold_out_2025():
    frame = pd.DataFrame({"season": [2021, 2022, 2023, 2024, 2025]})
    splits = chronological_splits(frame)
    assert [(split.train_seasons, split.validation_season) for split in splits] == [
        ((2021,), 2022),
        ((2021, 2022), 2023),
        ((2021, 2022, 2023), 2024),
    ]
    assert HOLDOUT_SEASON not in {split.validation_season for split in splits}


def test_model_features_exclude_targets_and_identifiers():
    frame = pd.DataFrame(
        {
            "season": [2021],
            "game_id": [1],
            "pitcher_mlbam_id": [2],
            "strikeouts": [3],
            "outs_recorded": [4],
            "batters_faced": [5],
            "pitches_thrown": [6],
            "season_k_pct": [0.2],
        }
    )
    columns = model_feature_columns(frame)
    assert columns == ["season_k_pct"]
    assert not set(TARGETS).intersection(columns)


def _synthetic_feature_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "season": [2021],
            "game_id": [1],
            "pitcher_mlbam_id": [2],
            "strikeouts": [3],
            "outs_recorded": [4],
            "batters_faced": [5],
            "pitches_thrown": [6],
            "season_k_pct": [0.2],
            "season_pitches_per_start": [90.0],
            "pitcher_days_rest": [4],
            "month": [4],
            "home_indicator": [1],
            "opponent_team_k_pct": [0.22],
            "last3_k_pct": [0.21],
            "pitch_FF_usage": [0.5],
            "velocity_sum": [700.0],
        }
    )


def test_feature_families_are_cumulative_supersets():
    frame = _synthetic_feature_frame()
    families = ["pitcher_history", "workload", "opponent", "recent", "full"]
    column_sets = [set(model_feature_columns(frame, family)) for family in families]
    for earlier, later in zip(column_sets, column_sets[1:]):
        assert earlier.issubset(later)
    assert column_sets[-1] == set(model_feature_columns(frame, "full"))


def test_pitcher_history_family_excludes_workload_opponent_recent_and_arsenal():
    frame = _synthetic_feature_frame()
    columns = set(model_feature_columns(frame, "pitcher_history"))
    assert "season_k_pct" in columns
    assert "season_pitches_per_start" not in columns
    assert "opponent_team_k_pct" not in columns
    assert "last3_k_pct" not in columns
    assert "pitch_FF_usage" not in columns


def test_full_family_recovers_every_numeric_feature_column():
    frame = _synthetic_feature_frame()
    full = set(model_feature_columns(frame, "full"))
    recent = set(model_feature_columns(frame, "recent"))
    arsenal_only = full - recent
    assert arsenal_only == {"pitch_FF_usage", "velocity_sum"}


def test_ensemble_can_be_trained_and_persisted_as_final_model():
    rng = np.random.default_rng(0)
    n = 40
    frame = pd.DataFrame(
        {
            "season": [2021] * 20 + [2025] * 20,
            "game_id": range(n),
            "pitcher_mlbam_id": [1] * n,
            "season_k_pct": rng.uniform(0.15, 0.35, n),
            "season_pitches_per_start": rng.uniform(70, 100, n),
        }
    )
    frame["strikeouts"] = frame["season_k_pct"] * 20 + rng.normal(0, 0.1, n)
    columns = ["season_k_pct", "season_pitches_per_start"]
    bundle = _train_final_model(frame, "strikeouts", "ensemble", columns)
    assert bundle["type"] == "ensemble"
    assert set(bundle["models"]) == {"ridge", "hist_gradient_boosting"}
    prediction = _predict_final(bundle, frame[frame.season == 2025], "strikeouts")
    assert len(prediction) == 20
    assert (prediction >= 0).all()


def test_score_predictions_never_warns_on_zero_actuals():
    actual = pd.Series([0, 0, 3, 5])
    predicted = np.array([0.0, 2.0, 2.5, 4.0])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        metrics = score_predictions(actual, predicted)
    assert np.isfinite(metrics["poisson_deviance"])
    assert np.isfinite(metrics["mae"])


def test_poisson_estimator_scales_features_and_allows_more_iterations():
    pipeline = make_estimator("poisson")
    step_names = [name for name, _ in pipeline.steps]
    assert step_names == ["imputer", "scale", "model"]
    assert pipeline.named_steps["model"].max_iter >= 1000


def test_fold_feature_columns_drops_all_null_training_columns():
    train = pd.DataFrame(
        {
            "season_k_pct": [0.2, 0.25, 0.3],
            "pitch_SC_velocity": [np.nan, np.nan, np.nan],
        }
    )
    columns = _fold_feature_columns(train, ["season_k_pct", "pitch_SC_velocity"])
    assert columns == ["season_k_pct"]


def test_fold_feature_columns_keeps_partially_populated_columns():
    train = pd.DataFrame(
        {
            "season_k_pct": [0.2, np.nan, 0.3],
        }
    )
    columns = _fold_feature_columns(train, ["season_k_pct"])
    assert columns == ["season_k_pct"]
