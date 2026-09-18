"""Robust live decision layer for MLB pitcher prop markets."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mlb_quant.market.pricing import expected_value


DEFAULT_STRESS_PP = 3.0

OUTS_EXTRA_STRESS_LINES = (
    15.5,
    16.5,
    18.5,
)

OUTS_EXTRA_STRESS_PP = 1.0


def normalize_role_flags(value: Any) -> str:
    """Normalize missing/blank role flags to NONE."""

    if value is None:
        return "NONE"

    try:
        if pd.isna(value):
            return "NONE"
    except (TypeError, ValueError):
        pass

    text = str(value).strip()

    if not text:
        return "NONE"

    if text.lower() in {
        "none",
        "nan",
        "<na>",
        "null",
    }:
        return "NONE"

    return text


def selected_side_values(
    row: pd.Series,
) -> dict[str, Any]:
    """Return the higher-EV side of a sportsbook row."""

    over_ev = float(row["over_ev"])
    under_ev = float(row["under_ev"])

    if over_ev >= under_ev:
        return {
            "side": "OVER",
            "probability": float(
                row["model_over_probability"]
            ),
            "market_probability": float(
                row["market_over_no_vig"]
            ),
            "odds": float(
                row["over_odds"]
            ),
            "ev": over_ev,
            "opposite_ev": under_ev,
            "edge": float(
                row["over_probability_edge"]
            ),
            "fair_odds": float(
                row["model_fair_over_odds"]
            ),
        }

    return {
        "side": "UNDER",
        "probability": float(
            row["model_under_probability"]
        ),
        "market_probability": float(
            row["market_under_no_vig"]
        ),
        "odds": float(
            row["under_odds"]
        ),
        "ev": under_ev,
        "opposite_ev": over_ev,
        "edge": float(
            row["under_probability_edge"]
        ),
        "fair_odds": float(
            row["model_fair_under_odds"]
        ),
    }


def line_specific_extra_stress_pp(
    target: str,
    line: float,
) -> float:
    """Return calibration-derived extra probability stress."""

    if str(target) != "outs_recorded":
        return 0.0

    numeric_line = float(line)

    for flagged_line in OUTS_EXTRA_STRESS_LINES:
        if np.isclose(
            numeric_line,
            flagged_line,
            atol=1e-8,
        ):
            return float(
                OUTS_EXTRA_STRESS_PP
            )

    return 0.0


def effective_stress_pp(
    target: str,
    line: float,
    base_stress_pp: float,
) -> float:
    """Total adverse probability stress."""

    return float(
        base_stress_pp
        + line_specific_extra_stress_pp(
            target=target,
            line=line,
        )
    )


def stress_probability(
    probability: float,
    stress_pp: float,
) -> float:
    """Apply adverse probability stress."""

    stressed = (
        float(probability)
        - float(stress_pp) / 100.0
    )

    return float(
        np.clip(
            stressed,
            0.001,
            0.999,
        )
    )


def base_signal_tier(
    ev: float,
    edge: float,
    stressed_ev: float,
    role_flags: str,
) -> str:
    """Assign decision tier before data-age adjustment."""

    role_flags = normalize_role_flags(
        role_flags
    )

    if role_flags != "NONE":
        return "PASS_ROLE_RISK"

    if ev <= 0:
        return "PASS"

    if stressed_ev <= 0:
        return "WATCH_FRAGILE"

    if (
        ev >= 0.06
        and edge >= 0.05
        and stressed_ev >= 0.015
    ):
        return "STRONG_CANDIDATE"

    if (
        ev >= 0.03
        and edge >= 0.03
    ):
        return "CANDIDATE"

    return "WATCH"


def downgrade_for_stale_data(
    tier: str,
    data_age_days: int,
) -> str:
    """Downgrade signals when model inputs are stale."""

    if int(data_age_days) <= 1:
        return tier

    downgrade = {
        "STRONG_CANDIDATE": (
            "CANDIDATE_STALE_DATA"
        ),
        "CANDIDATE": (
            "WATCH_STALE_DATA"
        ),
        "WATCH": (
            "WATCH_STALE_DATA"
        ),
        "WATCH_FRAGILE": (
            "WATCH_FRAGILE_STALE_DATA"
        ),
    }

    return downgrade.get(
        tier,
        tier,
    )


def evaluate_market_rows(
    frame: pd.DataFrame,
    base_stress_pp: float,
    as_of_date: pd.Timestamp,
    data_age_days: int,
) -> pd.DataFrame:
    """Evaluate every sportsbook market row."""

    records: list[
        dict[str, Any]
    ] = []

    for _, row in frame.iterrows():

        side = selected_side_values(
            row
        )

        role_flags = normalize_role_flags(
            row.get(
                "role_flags",
                "NONE",
            )
        )

        calibration_stress_pp = (
            line_specific_extra_stress_pp(
                target=str(
                    row["target"]
                ),
                line=float(
                    row["line"]
                ),
            )
        )

        applied_stress_pp = (
            effective_stress_pp(
                target=str(
                    row["target"]
                ),
                line=float(
                    row["line"]
                ),
                base_stress_pp=float(
                    base_stress_pp
                ),
            )
        )

        stressed_probability = (
            stress_probability(
                probability=float(
                    side["probability"]
                ),
                stress_pp=(
                    applied_stress_pp
                ),
            )
        )

        stressed_ev = float(
            expected_value(
                stressed_probability,
                float(
                    side["odds"]
                ),
            )
        )

        tier = base_signal_tier(
            ev=float(
                side["ev"]
            ),
            edge=float(
                side["edge"]
            ),
            stressed_ev=(
                stressed_ev
            ),
            role_flags=(
                role_flags
            ),
        )

        tier = downgrade_for_stale_data(
            tier=tier,
            data_age_days=(
                data_age_days
            ),
        )

        record = row.to_dict()

        record.update(
            {
                "role_flags": (
                    role_flags
                ),
                "model_side": (
                    side["side"]
                ),
                "selected_probability": float(
                    side["probability"]
                ),
                "selected_market_no_vig": float(
                    side[
                        "market_probability"
                    ]
                ),
                "selected_odds": float(
                    side["odds"]
                ),
                "selected_fair_odds": float(
                    side["fair_odds"]
                ),
                "selected_edge": float(
                    side["edge"]
                ),
                "selected_ev": float(
                    side["ev"]
                ),
                "opposite_side_ev": float(
                    side["opposite_ev"]
                ),
                "base_stress_pp": float(
                    base_stress_pp
                ),
                "line_calibration_stress_pp": float(
                    calibration_stress_pp
                ),
                "stress_pp": float(
                    applied_stress_pp
                ),
                "stressed_probability": float(
                    stressed_probability
                ),
                "stressed_ev": float(
                    stressed_ev
                ),
                "data_as_of": (
                    as_of_date
                    .date()
                    .isoformat()
                ),
                "data_age_days": int(
                    data_age_days
                ),
                "signal_tier": (
                    tier
                ),
            }
        )

        records.append(
            record
        )

    return pd.DataFrame(
        records
    )


def add_display_columns(
    output: pd.DataFrame,
) -> pd.DataFrame:
    """Add percentage display fields."""

    result = output.copy()

    probability_columns = {
        "selected_probability": (
            "selected_probability_pct"
        ),
        "selected_market_no_vig": (
            "selected_market_no_vig_pct"
        ),
        "stressed_probability": (
            "stressed_probability_pct"
        ),
    }

    for source, destination in (
        probability_columns.items()
    ):
        result[
            destination
        ] = (
            pd.to_numeric(
                result[source],
                errors="coerce",
            )
            * 100.0
        )

    result[
        "selected_edge_pp"
    ] = (
        pd.to_numeric(
            result["selected_edge"],
            errors="coerce",
        )
        * 100.0
    )

    result[
        "selected_ev_pct"
    ] = (
        pd.to_numeric(
            result["selected_ev"],
            errors="coerce",
        )
        * 100.0
    )

    result[
        "opposite_side_ev_pct"
    ] = (
        pd.to_numeric(
            result["opposite_side_ev"],
            errors="coerce",
        )
        * 100.0
    )

    result[
        "stressed_ev_pct"
    ] = (
        pd.to_numeric(
            result["stressed_ev"],
            errors="coerce",
        )
        * 100.0
    )

    return result


def sort_decisions(
    output: pd.DataFrame,
) -> pd.DataFrame:
    """Sort strongest decisions first."""

    tier_rank = {
        "STRONG_CANDIDATE": 0,
        "CANDIDATE": 1,
        "CANDIDATE_STALE_DATA": 2,
        "WATCH": 3,
        "WATCH_STALE_DATA": 4,
        "WATCH_FRAGILE": 5,
        "WATCH_FRAGILE_STALE_DATA": 6,
        "PASS_ROLE_RISK": 7,
        "PASS": 8,
    }

    result = output.copy()

    result[
        "_tier_rank"
    ] = (
        result[
            "signal_tier"
        ]
        .map(
            tier_rank
        )
        .fillna(
            99
        )
    )

    return (
        result.sort_values(
            [
                "_tier_rank",
                "selected_ev",
                "selected_edge",
            ],
            ascending=[
                True,
                False,
                False,
            ],
        )
        .drop(
            columns=[
                "_tier_rank"
            ]
        )
        .reset_index(
            drop=True
        )
    )


REQUIRED_COLUMNS = {
    "pitcher_mlbam_id",
    "pitcher_name",
    "team",
    "opponent",
    "target",
    "line",
    "projection",
    "over_odds",
    "under_odds",
    "model_over_probability",
    "model_under_probability",
    "market_over_no_vig",
    "market_under_no_vig",
    "over_probability_edge",
    "under_probability_edge",
    "over_ev",
    "under_ev",
    "model_fair_over_odds",
    "model_fair_under_odds",
    "role_flags",
}


def validate_market_frame(
    frame: pd.DataFrame,
) -> None:
    if frame.empty:
        raise ValueError(
            "Live market analysis is empty."
        )

    missing = (
        REQUIRED_COLUMNS
        - set(
            frame.columns
        )
    )

    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )


def parse_date(
    value: str,
    argument_name: str,
) -> pd.Timestamp:

    parsed = pd.Timestamp(
        value
    ).normalize()

    if (
        parsed.strftime(
            "%Y-%m-%d"
        )
        != value
    ):
        raise ValueError(
            f"{argument_name} must use YYYY-MM-DD."
        )

    return parsed


def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Apply robust MLB pitcher-prop "
            "decision rules to live market analysis."
        )
    )

    parser.add_argument(
        "--date",
        required=True,
        help=(
            "Slate date in YYYY-MM-DD format."
        ),
    )

    parser.add_argument(
        "--as-of",
        required=True,
        help=(
            "Latest completed-data date used "
            "by the model in YYYY-MM-DD format."
        ),
    )

    parser.add_argument(
        "--stress-pp",
        type=float,
        default=DEFAULT_STRESS_PP,
        help=(
            "Base adverse probability stress "
            "in percentage points. "
            f"Default: {DEFAULT_STRESS_PP:.1f}."
        ),
    )

    args = parser.parse_args()

    input_path = (
        Path("reports")
        / f"live_market_analysis_{args.date}.csv"
    )

    output_path = (
        Path("reports")
        / f"live_decisions_{args.date}.csv"
    )

    if not input_path.exists():
        raise FileNotFoundError(
            f"Market analysis not found: {input_path}"
        )

    frame = pd.read_csv(
        input_path
    )

    validate_market_frame(
        frame
    )

    slate_date = parse_date(
        args.date,
        "--date",
    )

    as_of_date = parse_date(
        args.as_of,
        "--as-of",
    )

    if as_of_date >= slate_date:
        raise ValueError(
            "As-of date must be before the slate date."
        )

    data_age_days = int(
        (
            slate_date
            - as_of_date
        ).days
    )

    if not (
        0.0
        < float(args.stress_pp)
        < 25.0
    ):
        raise ValueError(
            "--stress-pp must be greater "
            "than 0 and less than 25."
        )

    output = evaluate_market_rows(
        frame=frame,
        base_stress_pp=float(
            args.stress_pp
        ),
        as_of_date=(
            as_of_date
        ),
        data_age_days=(
            data_age_days
        ),
    )

    output = add_display_columns(
        output
    )

    output = sort_decisions(
        output
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.to_csv(
        output_path,
        index=False,
    )

    display_columns = [
        "pitcher_name",
        "team",
        "opponent",
        "target",
        "line",
        "projection",
        "model_side",
        "selected_odds",
        "selected_probability_pct",
        "selected_market_no_vig_pct",
        "selected_edge_pp",
        "selected_ev_pct",
        "base_stress_pp",
        "line_calibration_stress_pp",
        "stress_pp",
        "stressed_ev_pct",
        "data_age_days",
        "role_flags",
        "signal_tier",
    ]

    display = output[
        display_columns
    ].copy()

    numeric_round = [
        "projection",
        "selected_probability_pct",
        "selected_market_no_vig_pct",
        "selected_edge_pp",
        "selected_ev_pct",
        "base_stress_pp",
        "line_calibration_stress_pp",
        "stress_pp",
        "stressed_ev_pct",
    ]

    for column in numeric_round:
        display[
            column
        ] = (
            pd.to_numeric(
                display[column],
                errors="coerce",
            )
            .round(1)
        )

    print()
    print(
        "ROBUST LIVE DECISION ANALYSIS"
    )
    print()

    print(
        f"Slate date: {args.date}"
    )

    print(
        f"Data through: {args.as_of}"
    )

    print(
        f"Data age: {data_age_days} day(s)"
    )

    print(
        "Base probability stress: "
        f"{args.stress_pp:.1f} pp"
    )

    print(
        "Extra calibration stress: "
        "+1.0 pp on outs-recorded "
        "15.5 / 16.5 / 18.5"
    )

    print()

    print(
        display.to_string(
            index=False
        )
    )

    print()
    print(
        "SIGNAL COUNTS"
    )

    print(
        output[
            "signal_tier"
        ]
        .value_counts()
        .to_string()
    )

    print()
    print(
        f"Output: {output_path}"
    )


if __name__ == "__main__":
    main()
