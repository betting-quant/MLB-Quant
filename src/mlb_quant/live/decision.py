from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from mlb_quant.market.pricing import expected_value


def selected_side_values(row: pd.Series) -> dict:
    if row["over_ev"] >= row["under_ev"]:
        return {
            "side": "OVER",
            "probability": float(
                row["model_over_probability"]
            ),
            "market_probability": float(
                row["market_over_no_vig"]
            ),
            "odds": float(row["over_odds"]),
            "ev": float(row["over_ev"]),
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
        "odds": float(row["under_odds"]),
        "ev": float(row["under_ev"]),
        "edge": float(
            row["under_probability_edge"]
        ),
        "fair_odds": float(
            row["model_fair_under_odds"]
        ),
    }


def base_signal_tier(
    ev: float,
    edge: float,
    stressed_ev: float,
    role_flags: str,
) -> str:
    # Any workload/history warning blocks an automatic candidate.
    if role_flags != "NONE":
        return "PASS_ROLE_RISK"

    # Neither side is profitable according to the model.
    if ev <= 0:
        return "PASS"

    # Edge disappears after a modest adverse probability move.
    if stressed_ev <= 0:
        return "WATCH_FRAGILE"

    # Large raw edge that also survives stress.
    if (
        ev >= 0.06
        and edge >= 0.05
        and stressed_ev >= 0.015
    ):
        return "STRONG_CANDIDATE"

    # Meaningful edge that survives stress.
    if (
        ev >= 0.03
        and edge >= 0.03
    ):
        return "CANDIDATE"

    # Positive, but too thin to treat like a strong signal.
    return "WATCH"


def downgrade_for_stale_data(
    tier: str,
    data_age_days: int,
) -> str:
    if data_age_days <= 1:
        return tier

    downgrade = {
        "STRONG_CANDIDATE": "CANDIDATE_STALE_DATA",
        "CANDIDATE": "WATCH_STALE_DATA",
        "WATCH": "WATCH_STALE_DATA",
        "WATCH_FRAGILE": "WATCH_FRAGILE_STALE_DATA",
    }

    return downgrade.get(
        tier,
        tier,
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--date",
        required=True,
        help="Slate date YYYY-MM-DD.",
    )

    parser.add_argument(
        "--as-of",
        required=True,
        help=(
            "Latest completed-data date used by the model "
            "in YYYY-MM-DD format."
        ),
    )

    parser.add_argument(
        "--stress-pp",
        type=float,
        default=3.0,
        help=(
            "Adverse probability stress in percentage points. "
            "Default: 3.0."
        ),
    )

    args = parser.parse_args()

    input_path = Path(
        f"reports/live_market_analysis_{args.date}.csv"
    )

    output_path = Path(
        f"reports/live_decisions_{args.date}.csv"
    )

    if not input_path.exists():
        raise FileNotFoundError(
            f"Market analysis not found: {input_path}"
        )

    frame = pd.read_csv(
        input_path
    )

    if frame.empty:
        raise ValueError(
            "Live market analysis is empty."
        )

    required = {
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

    missing = required - set(
        frame.columns
    )

    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )

    slate_date = pd.Timestamp(
        args.date
    )

    as_of_date = pd.Timestamp(
        args.as_of
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

    stress = (
        args.stress_pp / 100.0
    )

    if not 0 < stress < 0.25:
        raise ValueError(
            "--stress-pp must be greater than 0 "
            "and less than 25."
        )

    records = []

    for _, row in frame.iterrows():
        side = selected_side_values(
            row
        )

        stressed_probability = max(
            0.001,
            side["probability"] - stress,
        )

        stressed_ev = expected_value(
            stressed_probability,
            side["odds"],
        )

        tier = base_signal_tier(
            ev=side["ev"],
            edge=side["edge"],
            stressed_ev=stressed_ev,
            role_flags=str(
                row["role_flags"]
            ),
        )

        tier = downgrade_for_stale_data(
            tier,
            data_age_days,
        )

        record = row.to_dict()

        record.update(
            {
                "model_side": side["side"],
                "selected_probability": (
                    side["probability"]
                ),
                "selected_market_no_vig": (
                    side["market_probability"]
                ),
                "selected_odds": (
                    side["odds"]
                ),
                "selected_fair_odds": (
                    side["fair_odds"]
                ),
                "selected_edge": (
                    side["edge"]
                ),
                "selected_ev": (
                    side["ev"]
                ),
                "stress_pp": (
                    args.stress_pp
                ),
                "stressed_probability": (
                    stressed_probability
                ),
                "stressed_ev": (
                    stressed_ev
                ),
                "data_as_of": (
                    as_of_date.date().isoformat()
                ),
                "data_age_days": (
                    data_age_days
                ),
                "signal_tier": tier,
            }
        )

        records.append(
            record
        )

    output = pd.DataFrame(
        records
    )

    output[
        "selected_probability_pct"
    ] = (
        output[
            "selected_probability"
        ]
        * 100
    )

    output[
        "selected_market_no_vig_pct"
    ] = (
        output[
            "selected_market_no_vig"
        ]
        * 100
    )

    output[
        "selected_edge_pp"
    ] = (
        output[
            "selected_edge"
        ]
        * 100
    )

    output[
        "selected_ev_pct"
    ] = (
        output[
            "selected_ev"
        ]
        * 100
    )

    output[
        "stressed_probability_pct"
    ] = (
        output[
            "stressed_probability"
        ]
        * 100
    )

    output[
        "stressed_ev_pct"
    ] = (
        output[
            "stressed_ev"
        ]
        * 100
    )

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

    output["_tier_rank"] = (
        output["signal_tier"]
        .map(tier_rank)
        .fillna(99)
    )

    output = output.sort_values(
        [
            "_tier_rank",
            "selected_ev",
        ],
        ascending=[
            True,
            False,
        ],
    ).drop(
        columns="_tier_rank"
    ).reset_index(
        drop=True
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.to_csv(
        output_path,
        index=False,
    )

    display = output[
        [
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
            "stressed_ev_pct",
            "data_age_days",
            "role_flags",
            "signal_tier",
        ]
    ].copy()

    numeric_round = [
        "projection",
        "selected_probability_pct",
        "selected_market_no_vig_pct",
        "selected_edge_pp",
        "selected_ev_pct",
        "stressed_ev_pct",
    ]

    for column in numeric_round:
        display[column] = (
            display[column]
            .round(1)
        )

    print()
    print("ROBUST LIVE DECISION ANALYSIS")
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
        f"Probability stress: "
        f"{args.stress_pp:.1f} pp"
    )

    print()

    print(
        display.to_string(
            index=False
        )
    )

    print()

    print("SIGNAL COUNTS")

    print(
        output[
            "signal_tier"
        ].value_counts().to_string()
    )

    print()

    print(
        f"Output: {output_path}"
    )


if __name__ == "__main__":
    main()