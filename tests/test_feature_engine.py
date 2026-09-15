import pandas as pd

from mlb_quant.features.engine import TARGET_COLUMNS, build_point_in_time_features


def _synthetic_inputs():
    appearances = pd.DataFrame(
        {
            "season": [2024, 2024, 2024],
            "game_date": ["2024-04-01", "2024-04-05", "2024-04-10"],
            "game_id": [1, 2, 3],
            "pitcher_mlbam_id": [10, 10, 10],
            "pitcher_name": ["Test", "Test", "Test"],
            "team": ["AAA"] * 3,
            "opponent": ["BBB"] * 3,
            "home_away": ["home"] * 3,
            "pitcher_handedness": ["R"] * 3,
            "pitches_thrown": [80, 90, 100],
            "batters_faced": [10, 10, 10],
            "strikeouts": [1, 2, 99],
            "outs_recorded": [15, 18, 27],
            "hits": [1, 1, 99],
            "walks": [1, 1, 99],
            "runs": [1, 1, 99],
            "source_rows": [80, 90, 100],
            "outs_method": ["test"] * 3,
        }
    )
    metrics = pd.DataFrame(
        {
            "season": [2024, 2024, 2024],
            "game_date": pd.to_datetime(["2024-04-01", "2024-04-05", "2024-04-10"]),
            "game_id": [1, 2, 3],
            "pitcher_mlbam_id": [10, 10, 10],
            "pitches": [80, 90, 100],
            "bf_metric": [10, 10, 10],
            "k_count": [1, 2, 99],
            "bb_count": [1, 1, 99],
            "hits_count": [1, 1, 99],
            "strike_count": [50, 55, 60],
            "called_strike_count": [20, 20, 20],
            "whiff_count": [10, 10, 10],
            "csw_count": [30, 30, 30],
            "swing_count": [40, 40, 40],
            "contact_count": [30, 30, 30],
            "zone_pitch_count": [40, 40, 40],
            "zone_swing_count": [30, 30, 30],
            "zone_contact_count": [20, 20, 20],
            "chase_swing_count": [10, 10, 10],
            "chase_pitch_count": [40, 40, 40],
            "first_pitch_strike_count": [5, 5, 5],
            "first_pitch_pa_count": [10, 10, 10],
            "velocity_sum": [7600, 8550, 9500],
            "velocity_count": [80, 90, 100],
            "terminal_pa_count": [10, 10, 10],
            "outs_recorded": [15, 18, 27],
        }
    )
    return appearances, metrics


def test_features_exclude_current_game_and_targets_are_preserved():
    appearances, metrics = _synthetic_inputs()
    features = build_point_in_time_features(appearances, metrics, minimum_starts=1)
    game_two = features.loc[features["game_id"] == 2].iloc[0]
    assert game_two["season_k_pct"] == 0.1
    assert game_two["last3_k_pct"] == 0.1
    assert game_two["season_outs_per_start"] == 15
    assert game_two["pitches_previous_start"] == 80
    assert set(TARGET_COLUMNS).issubset(features.columns)
    assert not {"hits", "walks", "runs", "source_rows", "outs_method"}.intersection(features.columns)


def test_future_game_mutation_cannot_change_prior_features():
    appearances, metrics = _synthetic_inputs()
    original = build_point_in_time_features(appearances, metrics, minimum_starts=1)
    metrics.loc[2, ["k_count", "bb_count", "pitches", "outs_recorded"]] = [0, 0, 1, 0]
    mutated = build_point_in_time_features(appearances, metrics, minimum_starts=1)
    feature_columns = [column for column in original.columns if column not in TARGET_COLUMNS]
    original_game_two = original.loc[original["game_id"] == 2, feature_columns].reset_index(drop=True)
    mutated_game_two = mutated.loc[mutated["game_id"] == 2, feature_columns].reset_index(drop=True)
    pd.testing.assert_frame_equal(original_game_two, mutated_game_two)


def test_early_history_is_flagged_and_not_filled_from_future():
    appearances, metrics = _synthetic_inputs()
    features = build_point_in_time_features(appearances, metrics, minimum_starts=3)
    first = features.iloc[0]
    assert first["season_insufficient_history"] == 1
    assert pd.isna(first["season_k_pct"])


def test_raw_current_game_velocity_totals_do_not_leak_into_features():
    """Regression: velocity_sum/velocity_count are current-game pitch totals
    (velocity_count ~= pitches_thrown for that same appearance) and must never
    survive unshifted into the point-in-time feature table."""
    appearances, metrics = _synthetic_inputs()
    features = build_point_in_time_features(appearances, metrics, minimum_starts=1)
    assert "velocity_sum" not in features.columns
    assert "velocity_count" not in features.columns

