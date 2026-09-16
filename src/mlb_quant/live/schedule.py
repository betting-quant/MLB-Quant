from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np
import pandas as pd


MLB_API = "https://statsapi.mlb.com/api/v1"


def _get_json(endpoint: str, params: dict | None = None) -> dict:
    url = f"{MLB_API}/{endpoint.lstrip('/')}"

    if params:
        url = f"{url}?{urlencode(params)}"

    with urlopen(url, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def _team_abbreviation(team_id: int) -> str:
    payload = _get_json(f"teams/{team_id}")
    teams = payload.get("teams", [])

    if not teams:
        raise ValueError(
            f"Could not resolve MLB team ID {team_id}"
        )

    return teams[0]["abbreviation"]


def _pitcher_hand(pitcher_id: int) -> str:
    payload = _get_json(f"people/{pitcher_id}")
    people = payload.get("people", [])

    if not people:
        raise ValueError(
            f"Could not resolve pitcher ID {pitcher_id}"
        )

    hand = people[0].get("pitchHand", {}).get("code")

    if hand not in {"R", "L"}:
        raise ValueError(
            f"Unexpected pitcher hand for {pitcher_id}: {hand}"
        )

    return hand


def _display_name(full_name: str) -> str:
    parts = full_name.strip().split()

    if len(parts) < 2:
        return full_name

    return f"{parts[-1]}, {' '.join(parts[:-1])}"


def fetch_probable_starters(game_date: str) -> pd.DataFrame:
    payload = _get_json(
        "schedule",
        {
            "sportId": 1,
            "date": game_date,
            "hydrate": "probablePitcher",
        },
    )

    rows = []
    missing_probables = []

    team_cache: dict[int, str] = {}
    hand_cache: dict[int, str] = {}

    for date_block in payload.get("dates", []):
        for game in date_block.get("games", []):
            game_id = int(game["gamePk"])

            away = game["teams"]["away"]
            home = game["teams"]["home"]

            away_team_id = int(away["team"]["id"])
            home_team_id = int(home["team"]["id"])

            if away_team_id not in team_cache:
                team_cache[away_team_id] = _team_abbreviation(
                    away_team_id
                )

            if home_team_id not in team_cache:
                team_cache[home_team_id] = _team_abbreviation(
                    home_team_id
                )

            away_abbr = team_cache[away_team_id]
            home_abbr = team_cache[home_team_id]

            sides = [
                (
                    "away",
                    away,
                    away_abbr,
                    home_abbr,
                ),
                (
                    "home",
                    home,
                    home_abbr,
                    away_abbr,
                ),
            ]

            for (
                home_away,
                side,
                team,
                opponent,
            ) in sides:
                probable = side.get("probablePitcher")

                if not probable or "id" not in probable:
                    missing_probables.append(
                        {
                            "game_id": game_id,
                            "team": team,
                            "opponent": opponent,
                            "home_away": home_away,
                        }
                    )
                    continue

                pitcher_id = int(probable["id"])
                pitcher_name = probable.get(
                    "fullName",
                    str(pitcher_id),
                )

                if pitcher_id not in hand_cache:
                    hand_cache[pitcher_id] = _pitcher_hand(
                        pitcher_id
                    )

                rows.append(
                    {
                        "game_date": game_date,
                        "game_id": game_id,
                        "pitcher_mlbam_id": pitcher_id,
                        "pitcher_name": _display_name(
                            pitcher_name
                        ),
                        "team": team,
                        "opponent": opponent,
                        "home_away": home_away,
                        "pitcher_handedness": (
                            hand_cache[pitcher_id]
                        ),
                        "pitches_thrown": np.nan,
                        "batters_faced": np.nan,
                        "strikeouts": np.nan,
                        "outs_recorded": np.nan,
                        "hits": np.nan,
                        "walks": np.nan,
                        "runs": np.nan,
                        "source_rows": 0,
                        "outs_method": "pregame_unplayed",
                    }
                )

    if missing_probables:
        print()
        print("WARNING: MISSING PROBABLE STARTERS")

        for item in missing_probables:
            print(
                f"  game {item['game_id']}: "
                f"{item['team']} vs {item['opponent']} "
                f"({item['home_away']})"
            )

    frame = pd.DataFrame(rows)

    if frame.empty:
        return frame

    frame = frame.sort_values(
        [
            "game_id",
            "home_away",
        ]
    ).reset_index(drop=True)

    duplicates = frame.duplicated(
        [
            "game_id",
            "pitcher_mlbam_id",
        ]
    ).sum()

    if duplicates:
        raise ValueError(
            f"Found {duplicates} duplicate probable starter rows."
        )

    return frame


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--date",
        required=True,
        help="Pregame MLB date in YYYY-MM-DD format.",
    )

    parser.add_argument(
        "--output",
        default=None,
    )

    args = parser.parse_args()

    starters = fetch_probable_starters(args.date)

    if starters.empty:
        print(
            f"No probable starters found for {args.date}."
        )
        return

    output_path = (
        Path(args.output)
        if args.output
        else Path(
            f"data/processed/live_2026/"
            f"pregame_starters_{args.date}.parquet"
        )
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    starters.to_parquet(
        output_path,
        index=False,
    )

    print()
    print("PREGAME STARTER ROWS")
    print()

    print(
        starters[
            [
                "game_id",
                "pitcher_name",
                "team",
                "opponent",
                "home_away",
                "pitcher_handedness",
            ]
        ].to_string(index=False)
    )

    print()
    print(f"Starter rows: {len(starters)}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()