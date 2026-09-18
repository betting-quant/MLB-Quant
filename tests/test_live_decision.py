from __future__ import annotations

import pandas as pd
import pytest

from mlb_quant.live.decision import (
    DEFAULT_STRESS_PP,
    base_signal_tier,
    downgrade_for_stale_data,
    effective_stress_pp,
    evaluate_market_rows,
    line_specific_extra_stress_pp,
    normalize_role_flags,
    selected_side_values,
    stress_probability,
)


def test_normalize_role_flags() -> None:
    assert normalize_role_flags(None) == "NONE"
    assert normalize_role_flags("") == "NONE"
    assert normalize_role_flags("   ") == "NONE"
    assert normalize_role_flags("nan") == "NONE"
    assert normalize_role_flags("NONE") == "NONE"
    assert (
        normalize_role_flags(
            "LOW_PROJECTED_PITCHES"
        )
        == "LOW_PROJECTED_PITCHES"
    )


def test_selected_side_values_uses_higher_ev() -> None:
    row = pd.Series(
        {
            "model_over_probability": 0.61,
            "model_under_probability": 0.39,
            "market_over_no_vig": 0.50,
            "market_under_no_vig": 0.50,
            "over_odds": -110,
            "under_odds": -110,
            "over_ev": 0.12,
            "under_ev": -0.08,
            "over_probability_edge": 0.11,
            "under_probability_edge": -0.11,
            "model_fair_over_odds": -156,
            "model_fair_under_odds": 156,
        }
    )

    side = selected_side_values(row)

    assert side["side"] == "OVER"
    assert side["probability"] == pytest.approx(
        0.61
    )
    assert side["ev"] == pytest.approx(
        0.12
    )
    assert side["opposite_ev"] == pytest.approx(
        -0.08
    )
    assert side["edge"] == pytest.approx(
        0.11
    )


def test_outs_line_specific_extra_stress() -> None:
    assert (
        line_specific_extra_stress_pp(
            "outs_recorded",
            15.5,
        )
        == pytest.approx(1.0)
    )

    assert (
        line_specific_extra_stress_pp(
            "outs_recorded",
            16.5,
        )
        == pytest.approx(1.0)
    )

    assert (
        line_specific_extra_stress_pp(
            "outs_recorded",
            18.5,
        )
        == pytest.approx(1.0)
    )

    assert (
        line_specific_extra_stress_pp(
            "outs_recorded",
            17.5,
        )
        == pytest.approx(0.0)
    )

    assert (
        line_specific_extra_stress_pp(
            "strikeouts",
            15.5,
        )
        == pytest.approx(0.0)
    )


def test_probability_stress_combines_base_and_line_adjustment() -> None:
    assert DEFAULT_STRESS_PP == pytest.approx(
        3.0
    )

    total_stress = effective_stress_pp(
        target="outs_recorded",
        line=15.5,
        base_stress_pp=DEFAULT_STRESS_PP,
    )

    assert total_stress == pytest.approx(
        4.0
    )

    stressed = stress_probability(
        probability=0.60,
        stress_pp=total_stress,
    )

    assert stressed == pytest.approx(
        0.56
    )

    assert stress_probability(
        probability=0.001,
        stress_pp=3.0,
    ) == pytest.approx(
        0.001
    )


def test_signal_tiers_and_stale_downgrade() -> None:
    assert (
        base_signal_tier(
            ev=0.20,
            edge=0.15,
            stressed_ev=0.12,
            role_flags=(
                "LOW_PROJECTED_PITCHES"
            ),
        )
        == "PASS_ROLE_RISK"
    )

    assert (
        base_signal_tier(
            ev=-0.01,
            edge=0.02,
            stressed_ev=-0.04,
            role_flags="NONE",
        )
        == "PASS"
    )

    assert (
        base_signal_tier(
            ev=0.04,
            edge=0.04,
            stressed_ev=-0.01,
            role_flags="NONE",
        )
        == "WATCH_FRAGILE"
    )

    assert (
        base_signal_tier(
            ev=0.08,
            edge=0.06,
            stressed_ev=0.03,
            role_flags="NONE",
        )
        == "STRONG_CANDIDATE"
    )

    assert (
        base_signal_tier(
            ev=0.04,
            edge=0.035,
            stressed_ev=0.01,
            role_flags="NONE",
        )
        == "CANDIDATE"
    )

    assert (
        downgrade_for_stale_data(
            "STRONG_CANDIDATE",
            2,
        )
        == "CANDIDATE_STALE_DATA"
    )

    assert (
        downgrade_for_stale_data(
            "PASS_ROLE_RISK",
            3,
        )
        == "PASS_ROLE_RISK"
    )


def test_evaluate_market_rows_applies_outs_stress() -> None:
    frame = pd.DataFrame(
        [
            {
                "pitcher_mlbam_id": 123456,
                "pitcher_name": "Test Pitcher",
                "team": "AAA",
                "opponent": "BBB",
                "target": "outs_recorded",
                "line": 15.5,
                "projection": 17.0,
                "over_odds": -110,
                "under_odds": -110,
                "model_over_probability": 0.60,
                "model_under_probability": 0.40,
                "market_over_no_vig": 0.50,
                "market_under_no_vig": 0.50,
                "over_probability_edge": 0.10,
                "under_probability_edge": -0.10,
                "over_ev": 0.10,
                "under_ev": -0.10,
                "model_fair_over_odds": -150,
                "model_fair_under_odds": 150,
                "role_flags": "NONE",
            }
        ]
    )

    output = evaluate_market_rows(
        frame=frame,
        base_stress_pp=3.0,
        as_of_date=pd.Timestamp(
            "2026-09-17"
        ),
        data_age_days=1,
    )

    assert len(output) == 1

    row = output.iloc[0]

    assert row["model_side"] == "OVER"

    assert row[
        "base_stress_pp"
    ] == pytest.approx(
        3.0
    )

    assert row[
        "line_calibration_stress_pp"
    ] == pytest.approx(
        1.0
    )

    assert row[
        "stress_pp"
    ] == pytest.approx(
        4.0
    )

    assert row[
        "stressed_probability"
    ] == pytest.approx(
        0.56
    )

    assert row[
        "data_as_of"
    ] == "2026-09-17"

    assert row[
        "data_age_days"
    ] == 1

    assert row[
        "role_flags"
    ] == "NONE"
