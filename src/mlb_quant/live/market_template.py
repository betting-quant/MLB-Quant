from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--date",
        required=True,
        help="Slate date YYYY-MM-DD.",
    )

    args = parser.parse_args()

    prediction_path = Path(
        f"data/processed/live_2026/"
        f"pregame_predictions_{args.date}.parquet"
    )

    output_path = Path(
        f"reports/market_input_{args.date}.csv"
    )

    if not prediction_path.exists():
        raise FileNotFoundError(
            f"Prediction file not found: {prediction_path}"
        )

    predictions = pd.read_parquet(
        prediction_path
    )

    if predictions.empty:
        raise ValueError(
            "Prediction file is empty."
        )

    rows = []

    for _, pitcher in predictions.iterrows():
        base = {
            "pitcher_mlbam_id": int(
                pitcher["pitcher_mlbam_id"]
            ),
            "pitcher_name": pitcher[
                "pitcher_name"
            ],
            "team": pitcher["team"],
            "opponent": pitcher["opponent"],
        }

        rows.append(
            {
                **base,
                "target": "strikeouts",
                "line": "",
                "over_odds": "",
                "under_odds": "",
                "sportsbook": "",
            }
        )

        rows.append(
            {
                **base,
                "target": "outs_recorded",
                "line": "",
                "over_odds": "",
                "under_odds": "",
                "sportsbook": "",
            }
        )

    template = pd.DataFrame(rows)

    template = template.sort_values(
        [
            "team",
            "pitcher_name",
            "target",
        ]
    ).reset_index(drop=True)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    template.to_csv(
        output_path,
        index=False,
    )

    print()
    print("MARKET INPUT TEMPLATE CREATED")
    print()
    print(f"Pitchers: {len(predictions)}")
    print(f"Market rows: {len(template)}")
    print(f"Output: {output_path}")
    print()

    print(
        template[
            [
                "pitcher_mlbam_id",
                "pitcher_name",
                "team",
                "opponent",
                "target",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()