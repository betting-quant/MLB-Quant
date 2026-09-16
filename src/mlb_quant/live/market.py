from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from mlb_quant.market.pricing import analyze_market


TARGET_ALIASES = {
    "k": "strikeouts",
    "ks": "strikeouts",
    "strikeout": "strikeouts",
    "strikeouts": "strikeouts",
    "outs": "outs_recorded",
    "out": "outs_recorded",
    "outs_recorded": "outs_recorded",
}


def normalize_target(value: str) -> str:
    key = str(value).strip().lower()

    if key not in TARGET_ALIASES:
        raise ValueError(
            f"Unsupported target '{value}'. "
            "Use strikeouts or outs_recorded."
        )

    return TARGET_ALIASES[key]


def role_flags(row: pd.Series) -> str:
    flags = []

    if row["pitcher_starts_before_game"] < 4:
        flags.append("LIMITED_START_HISTORY")

    if row["projected_pitches_thrown"] < 60:
        flags.append("LOW_PROJECTED_PITCH_COUNT")

    if row["projected_batters_faced"] < 15:
        flags.append("LOW_PROJECTED_BF")

    if row["projected_outs_recorded"] < 9:
        flags.append("LOW_PROJECTED_OUTS")

    if row["missing_features"] >= 100:
        flags.append("HIGH_FEATURE_MISSINGNESS")

    if not flags:
        return "NONE"

    return "|".join(flags)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--date",
        required=True,
        help="Slate date YYYY-MM-DD.",
    )

    parser.add_argument(
        "--market-file",
        required=True,
        help="CSV containing sportsbook prop lines and odds.",
    )

    args = parser.parse_args()

    market_path = Path(args.market_file)

    probability_path = Path(
        f"data/processed/live_2026/"
        f"pregame_probabilities_{args.date}.parquet"
    )

    point_path = Path(
        f"data/processed/live_2026/"
        f"pregame_predictions_{args.date}.parquet"
    )

    output_path = Path(
        f"reports/live_market_analysis_{args.date}.csv"
    )

    if not market_path.exists():
        raise FileNotFoundError(
            f"Market file not found: {market_path}"
        )

    if not probability_path.exists():
        raise FileNotFoundError(
            f"Probability file not found: {probability_path}"
        )

    if not point_path.exists():
        raise FileNotFoundError(
            f"Point prediction file not found: {point_path}"
        )

    # --------------------------------------------------
    # LOAD SPORTSBOOK MARKET INPUT
    # --------------------------------------------------

    market = pd.read_csv(market_path)

    required = {
        "pitcher_mlbam_id",
        "target",
        "line",
        "over_odds",
        "under_odds",
    }

    missing_columns = required - set(market.columns)

    if missing_columns:
        raise ValueError(
            "Market CSV missing required columns: "
            f"{sorted(missing_columns)}"
        )

    # Allow the full generated market template to remain
    # in the CSV. Completely blank market rows are ignored.
    market["line"] = pd.to_numeric(
        market["line"],
        errors="coerce",
    )

    market["over_odds"] = pd.to_numeric(
        market["over_odds"],
        errors="coerce",
    )

    market["under_odds"] = pd.to_numeric(
        market["under_odds"],
        errors="coerce",
    )

    blank_market = (
        market["line"].isna()
        & market["over_odds"].isna()
        & market["under_odds"].isna()
    )

    market = market.loc[
        ~blank_market
    ].copy()

    if market.empty:
        raise ValueError(
            "No completed sportsbook market rows found."
        )

    # Partially entered rows should fail loudly.
    incomplete = (
        market["line"].isna()
        | market["over_odds"].isna()
        | market["under_odds"].isna()
    )

    if incomplete.any():
        print()
        print("INCOMPLETE MARKET ROWS")
        print(
            market.loc[
                incomplete,
                [
                    "pitcher_mlbam_id",
                    "target",
                    "line",
                    "over_odds",
                    "under_odds",
                ],
            ].to_string(index=False)
        )

        raise ValueError(
            f"{int(incomplete.sum())} market rows are only "
            "partially filled in."
        )

    market["pitcher_mlbam_id"] = pd.to_numeric(
        market["pitcher_mlbam_id"],
        errors="raise",
    ).astype(int)

    market["target"] = market[
        "target"
    ].map(normalize_target)

    # Name/team/opponent may exist in the human-editable
    # sportsbook template. The prediction data is treated as
    # authoritative, so drop these before merging to avoid
    # _x / _y suffixes.
    market = market.drop(
        columns=[
            column
            for column in [
                "pitcher_name",
                "team",
                "opponent",
            ]
            if column in market.columns
        ]
    )

    # --------------------------------------------------
    # LOAD MODEL OUTPUTS
    # --------------------------------------------------

    probabilities = pd.read_parquet(
        probability_path
    )

    points = pd.read_parquet(
        point_path
    )

    # --------------------------------------------------
    # PREPARE POINT-MODEL DATA
    # --------------------------------------------------

    point_columns = [
        "pitcher_mlbam_id",
        "pitcher_starts_before_game",
        "missing_features",
        "projected_strikeouts",
        "projected_outs_recorded",
        "projected_batters_faced",
        "projected_pitches_thrown",
    ]

    points = points[
        point_columns
    ].copy()

    points["pitcher_mlbam_id"] = (
        points["pitcher_mlbam_id"]
        .astype(int)
    )

    point_duplicates = points.duplicated(
        ["pitcher_mlbam_id"]
    ).sum()

    if point_duplicates:
        raise ValueError(
            f"Found {point_duplicates} duplicate "
            "pitchers in point predictions."
        )

    # --------------------------------------------------
    # PREPARE PROBABILITY DATA
    # --------------------------------------------------

    probability_columns = [
        "pitcher_mlbam_id",
        "pitcher_name",
        "team",
        "opponent",
        "target",
        "projection",
        "line",
        "probability_over",
        "probability_under",
        "distribution_method",
    ]

    probabilities = probabilities[
        probability_columns
    ].copy()

    probabilities["pitcher_mlbam_id"] = (
        probabilities["pitcher_mlbam_id"]
        .astype(int)
    )

    probabilities["line"] = pd.to_numeric(
        probabilities["line"]
    )

    probability_duplicates = (
        probabilities.duplicated(
            [
                "pitcher_mlbam_id",
                "target",
                "line",
            ]
        ).sum()
    )

    if probability_duplicates:
        raise ValueError(
            f"Found {probability_duplicates} duplicate "
            "pitcher/target/line probability rows."
        )

    # --------------------------------------------------
    # JOIN SPORTSBOOK MARKETS TO MODEL PROBABILITIES
    # --------------------------------------------------

    joined = market.merge(
        probabilities,
        on=[
            "pitcher_mlbam_id",
            "target",
            "line",
        ],
        how="left",
        validate="many_to_one",
    )

    unmatched = joined[
        "probability_over"
    ].isna()

    if unmatched.any():
        print()
        print("UNMATCHED MARKET ROWS")
        print(
            joined.loc[
                unmatched,
                [
                    "pitcher_mlbam_id",
                    "target",
                    "line",
                ],
            ].to_string(index=False)
        )

        raise ValueError(
            f"{int(unmatched.sum())} market rows "
            "could not be matched to model probabilities."
        )

    # --------------------------------------------------
    # ADD WORKLOAD / ROLE CONTEXT
    # --------------------------------------------------

    joined = joined.merge(
        points,
        on="pitcher_mlbam_id",
        how="left",
        validate="many_to_one",
    )

    missing_point_data = joined[
        "projected_pitches_thrown"
    ].isna()

    if missing_point_data.any():
        raise ValueError(
            f"{int(missing_point_data.sum())} market rows "
            "could not be matched to point predictions."
        )

    # --------------------------------------------------
    # PRICE EACH MARKET
    # --------------------------------------------------

    results = []

    for _, row in joined.iterrows():
        analysis = analyze_market(
            model_over_probability=row[
                "probability_over"
            ],
            over_odds=row[
                "over_odds"
            ],
            under_odds=row[
                "under_odds"
            ],
        )

        record = row.to_dict()

        record.update(
            analysis
        )

        record["role_flags"] = role_flags(
            pd.Series(record)
        )

        over_ev = float(
            record["over_ev"]
        )

        under_ev = float(
            record["under_ev"]
        )

        # ----------------------------------------------
        # DECISION SIDE
        #
        # Do not label a negative-EV side as a bet.
        # ----------------------------------------------

        if max(over_ev, under_ev) <= 0:
            record["higher_ev_side"] = "PASS"

            if over_ev >= under_ev:
                record["higher_ev"] = over_ev

                record[
                    "higher_probability_edge"
                ] = record[
                    "over_probability_edge"
                ]

            else:
                record["higher_ev"] = under_ev

                record[
                    "higher_probability_edge"
                ] = record[
                    "under_probability_edge"
                ]

        elif over_ev > under_ev:
            record["higher_ev_side"] = "OVER"

            record["higher_ev"] = over_ev

            record[
                "higher_probability_edge"
            ] = record[
                "over_probability_edge"
            ]

        else:
            record["higher_ev_side"] = "UNDER"

            record["higher_ev"] = under_ev

            record[
                "higher_probability_edge"
            ] = record[
                "under_probability_edge"
            ]

        results.append(
            record
        )

    output = pd.DataFrame(
        results
    )

    # --------------------------------------------------
    # HUMAN-READABLE PERCENTAGE COLUMNS
    # --------------------------------------------------

    output["over_ev_pct"] = (
        output["over_ev"] * 100
    )

    output["under_ev_pct"] = (
        output["under_ev"] * 100
    )

    output["higher_ev_pct"] = (
        output["higher_ev"] * 100
    )

    output[
        "over_probability_edge_pp"
    ] = (
        output[
            "over_probability_edge"
        ]
        * 100
    )

    output[
        "under_probability_edge_pp"
    ] = (
        output[
            "under_probability_edge"
        ]
        * 100
    )

    output[
        "higher_probability_edge_pp"
    ] = (
        output[
            "higher_probability_edge"
        ]
        * 100
    )

    output["sportsbook_hold_pct"] = (
        output["sportsbook_hold"] * 100
    )

    output[
        "model_over_probability_pct"
    ] = (
        output[
            "model_over_probability"
        ]
        * 100
    )

    output[
        "model_under_probability_pct"
    ] = (
        output[
            "model_under_probability"
        ]
        * 100
    )

    output[
        "market_over_no_vig_pct"
    ] = (
        output[
            "market_over_no_vig"
        ]
        * 100
    )

    output[
        "market_under_no_vig_pct"
    ] = (
        output[
            "market_under_no_vig"
        ]
        * 100
    )

    # --------------------------------------------------
    # SORT
    # --------------------------------------------------

    output = output.sort_values(
        [
            "higher_ev",
            "higher_probability_edge",
        ],
        ascending=[
            False,
            False,
        ],
    ).reset_index(
        drop=True
    )

    # --------------------------------------------------
    # SAVE FULL OUTPUT
    # --------------------------------------------------

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.to_csv(
        output_path,
        index=False,
    )

    # --------------------------------------------------
    # TERMINAL DISPLAY
    # --------------------------------------------------

    display_columns = [
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
        "market_over_no_vig",
        "model_fair_over_odds",
        "model_fair_under_odds",
        "over_ev_pct",
        "under_ev_pct",
        "higher_ev_side",
        "higher_probability_edge_pp",
        "higher_ev_pct",
        "role_flags",
    ]

    display = output[
        display_columns
    ].copy()

    display["projection"] = (
        display[
            "projection"
        ].round(2)
    )

    display[
        "model_over_probability"
    ] = (
        display[
            "model_over_probability"
        ]
        .mul(100)
        .round(1)
    )

    display[
        "market_over_no_vig"
    ] = (
        display[
            "market_over_no_vig"
        ]
        .mul(100)
        .round(1)
    )

    display[
        "model_fair_over_odds"
    ] = (
        display[
            "model_fair_over_odds"
        ]
        .round(0)
    )

    display[
        "model_fair_under_odds"
    ] = (
        display[
            "model_fair_under_odds"
        ]
        .round(0)
    )

    display[
        "over_ev_pct"
    ] = (
        display[
            "over_ev_pct"
        ]
        .round(1)
    )

    display[
        "under_ev_pct"
    ] = (
        display[
            "under_ev_pct"
        ]
        .round(1)
    )

    display[
        "higher_probability_edge_pp"
    ] = (
        display[
            "higher_probability_edge_pp"
        ]
        .round(1)
    )

    display[
        "higher_ev_pct"
    ] = (
        display[
            "higher_ev_pct"
        ]
        .round(1)
    )

    print()
    print("LIVE MARKET ANALYSIS")
    print()

    print(
        display.to_string(
            index=False
        )
    )

    print()
    print(
        f"Market rows analyzed: "
        f"{len(output)}"
    )

    print(
        "Positive-EV markets: "
        f"{int((output['higher_ev'] > 0).sum())}"
    )

    print(
        "Pass markets: "
        f"{int((output['higher_ev_side'] == 'PASS').sum())}"
    )

    print(
        f"Output: {output_path}"
    )


if __name__ == "__main__":
    main()