from __future__ import annotations

import argparse
import json
import os
import re
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests


API_BASE = "https://api.the-odds-api.com/v4"
SPORT_KEY = "baseball_mlb"

SUPPORTED_MARKETS = (
    "pitcher_strikeouts",
    "pitcher_outs",
)

MARKET_TARGET_MAP = {
    "pitcher_strikeouts": "strikeouts",
    "pitcher_outs": "outs_recorded",
}

REPORTS_DIR = Path("reports")
LIVE_DIR = Path("data/processed/live_2026")

CACHE_ROOT = (
    LIVE_DIR
    / "tracking"
    / "odds_cache"
)

HISTORY_ROOT = (
    LIVE_DIR
    / "tracking"
    / "odds_history"
)


TEAM_MAP = {
    "arizona diamondbacks": "AZ",
    "atlanta braves": "ATL",
    "baltimore orioles": "BAL",
    "boston red sox": "BOS",
    "chicago cubs": "CHC",
    "chicago white sox": "CWS",
    "cincinnati reds": "CIN",
    "cleveland guardians": "CLE",
    "colorado rockies": "COL",
    "detroit tigers": "DET",
    "houston astros": "HOU",
    "kansas city royals": "KC",
    "los angeles angels": "LAA",
    "los angeles dodgers": "LAD",
    "miami marlins": "MIA",
    "milwaukee brewers": "MIL",
    "minnesota twins": "MIN",
    "new york mets": "NYM",
    "new york yankees": "NYY",
    "oakland athletics": "ATH",
    "athletics": "ATH",
    "philadelphia phillies": "PHI",
    "pittsburgh pirates": "PIT",
    "san diego padres": "SD",
    "san francisco giants": "SF",
    "seattle mariners": "SEA",
    "st louis cardinals": "STL",
    "st. louis cardinals": "STL",
    "tampa bay rays": "TB",
    "texas rangers": "TEX",
    "toronto blue jays": "TOR",
    "washington nationals": "WSH",
}


def require_api_key() -> str:
    key = (
        os.getenv("THE_ODDS_API_KEY")
        or os.getenv("ODDS_API_KEY")
    )

    if not key:
        raise RuntimeError(
            "No The Odds API key found. Set "
            "THE_ODDS_API_KEY before running."
        )

    return key.strip()


def normalize_text(value: Any) -> str:
    if value is None:
        return ""

    text = str(value).strip().lower()

    text = unicodedata.normalize(
        "NFKD",
        text,
    )

    text = "".join(
        character
        for character in text
        if not unicodedata.combining(
            character
        )
    )

    text = text.replace(
        "’",
        "'",
    )

    text = re.sub(
        r"[^a-z0-9]+",
        " ",
        text,
    )

    return re.sub(
        r"\s+",
        " ",
        text,
    ).strip()


def normalize_team_name(
    value: Any,
) -> str:
    raw = str(
        value or ""
    ).strip()

    upper = raw.upper()

    valid_abbreviations = {
        "AZ",
        "ATL",
        "BAL",
        "BOS",
        "CHC",
        "CWS",
        "CIN",
        "CLE",
        "COL",
        "DET",
        "HOU",
        "KC",
        "LAA",
        "LAD",
        "MIA",
        "MIL",
        "MIN",
        "NYM",
        "NYY",
        "ATH",
        "OAK",
        "PHI",
        "PIT",
        "SD",
        "SF",
        "SEA",
        "STL",
        "TB",
        "TEX",
        "TOR",
        "WSH",
    }

    if upper in valid_abbreviations:
        if upper == "OAK":
            return "ATH"
        return upper

    return TEAM_MAP.get(
        normalize_text(raw),
        "",
    )


def name_aliases(
    value: Any,
) -> set[str]:
    raw = str(
        value or ""
    ).strip()

    aliases: set[str] = set()

    normal = normalize_text(
        raw
    )

    if normal:
        aliases.add(
            normal
        )

    if "," in raw:
        pieces = [
            piece.strip()
            for piece in raw.split(
                ","
            )
            if piece.strip()
        ]

        if len(pieces) >= 2:
            reversed_name = (
                " ".join(
                    pieces[1:]
                )
                + " "
                + pieces[0]
            )

            aliases.add(
                normalize_text(
                    reversed_name
                )
            )

    cleaned_aliases = set()

    suffixes = {
        "jr",
        "sr",
        "ii",
        "iii",
        "iv",
        "v",
    }

    for alias in aliases:
        tokens = [
            token
            for token in alias.split()
            if token not in suffixes
        ]

        if tokens:
            cleaned_aliases.add(
                " ".join(tokens)
            )

    return aliases | cleaned_aliases


def names_match(
    api_name: str,
    model_name: str,
) -> bool:
    left = name_aliases(
        api_name
    )

    right = name_aliases(
        model_name
    )

    if left & right:
        return True

    for lhs in left:
        lhs_tokens = lhs.split()

        for rhs in right:
            rhs_tokens = rhs.split()

            if (
                len(lhs_tokens) >= 2
                and len(rhs_tokens) >= 2
                and set(lhs_tokens)
                == set(rhs_tokens)
            ):
                return True

            if (
                lhs_tokens
                and rhs_tokens
                and lhs_tokens[-1]
                == rhs_tokens[-1]
                and lhs_tokens[0][0:1]
                == rhs_tokens[0][0:1]
            ):
                return True

    return False


def find_prediction_file(
    date: str,
) -> Path:
    candidates = [
        REPORTS_DIR
        / f"live_predictions_{date}.csv",
        LIVE_DIR
        / f"pregame_predictions_{date}.parquet",
    ]

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        "Could not find today's prediction "
        f"file for {date}."
    )


def load_starters(
    date: str,
) -> pd.DataFrame:
    path = find_prediction_file(
        date
    )

    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(
            path
        )
    else:
        frame = pd.read_parquet(
            path
        )

    required = {
        "pitcher_name",
        "team",
        "opponent",
    }

    missing = (
        required
        - set(frame.columns)
    )

    if missing:
        raise RuntimeError(
            "Today's prediction file is "
            "missing required columns: "
            + ", ".join(
                sorted(missing)
            )
        )

    id_column = None

    for candidate in (
        "pitcher_mlbam_id",
        "pitcher_id",
        "mlbam_id",
    ):
        if candidate in frame.columns:
            id_column = candidate
            break

    columns = [
        "pitcher_name",
        "team",
        "opponent",
    ]

    if id_column:
        columns.append(
            id_column
        )

    starters = (
        frame[columns]
        .copy()
        .drop_duplicates()
        .reset_index(
            drop=True
        )
    )

    if (
        id_column
        and id_column
        != "pitcher_mlbam_id"
    ):
        starters = starters.rename(
            columns={
                id_column:
                    "pitcher_mlbam_id"
            }
        )

    if "pitcher_mlbam_id" not in starters.columns:
        starters[
            "pitcher_mlbam_id"
        ] = pd.NA

    starters["team"] = (
        starters["team"]
        .map(
            normalize_team_name
        )
    )

    starters["opponent"] = (
        starters["opponent"]
        .map(
            normalize_team_name
        )
    )

    starters = starters.loc[
        starters["team"].ne("")
        & starters[
            "opponent"
        ].ne("")
    ].copy()

    return starters


def api_get(
    url: str,
    params: dict[str, Any],
) -> tuple[Any, dict[str, str]]:
    response = requests.get(
        url,
        params=params,
        timeout=30,
    )

    if response.status_code >= 400:
        raise RuntimeError(
            "The Odds API request failed "
            f"({response.status_code}): "
            f"{response.text[:500]}"
        )

    return (
        response.json(),
        dict(
            response.headers
        ),
    )


def cache_path(
    date: str,
    event_id: str,
) -> Path:
    directory = (
        CACHE_ROOT
        / date
    )

    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    return (
        directory
        / f"{event_id}.json"
    )


def cache_is_fresh(
    path: Path,
    cache_minutes: int,
) -> bool:
    if not path.exists():
        return False

    age_seconds = (
        time.time()
        - path.stat().st_mtime
    )

    return (
        age_seconds
        <= cache_minutes * 60
    )


def fetch_events(
    api_key: str,
) -> list[dict[str, Any]]:
    url = (
        f"{API_BASE}/sports/"
        f"{SPORT_KEY}/events"
    )

    payload, _ = api_get(
        url,
        {
            "apiKey": api_key,
            "dateFormat": "iso",
        },
    )

    if not isinstance(
        payload,
        list,
    ):
        raise RuntimeError(
            "Unexpected events API response."
        )

    return payload


def schedule_matchups(
    starters: pd.DataFrame,
) -> set[frozenset[str]]:
    output = set()

    for row in starters.itertuples(
        index=False
    ):
        output.add(
            frozenset(
                {
                    str(row.team),
                    str(row.opponent),
                }
            )
        )

    return output


def match_events_to_slate(
    events: list[dict[str, Any]],
    starters: pd.DataFrame,
) -> list[dict[str, Any]]:
    wanted = schedule_matchups(
        starters
    )

    matched = []

    for event in events:
        home = normalize_team_name(
            event.get(
                "home_team"
            )
        )

        away = normalize_team_name(
            event.get(
                "away_team"
            )
        )

        if not home or not away:
            continue

        if (
            frozenset(
                {
                    home,
                    away,
                }
            )
            not in wanted
        ):
            continue

        copied = dict(
            event
        )

        copied[
            "_home_abbr"
        ] = home

        copied[
            "_away_abbr"
        ] = away

        matched.append(
            copied
        )

    return matched


def fetch_event_odds(
    api_key: str,
    event_id: str,
    date: str,
    region: str,
    cache_minutes: int,
    force: bool,
    bookmakers: str | None,
) -> tuple[
    dict[str, Any],
    dict[str, str],
    bool,
]:
    path = cache_path(
        date,
        event_id,
    )

    if (
        not force
        and cache_is_fresh(
            path,
            cache_minutes,
        )
    ):
        with path.open(
            "r",
            encoding="utf-8",
        ) as handle:
            return (
                json.load(
                    handle
                ),
                {},
                True,
            )

    url = (
        f"{API_BASE}/sports/"
        f"{SPORT_KEY}/events/"
        f"{event_id}/odds"
    )

    params: dict[str, Any] = {
        "apiKey": api_key,
        "markets": ",".join(
            SUPPORTED_MARKETS
        ),
        "oddsFormat": "american",
        "dateFormat": "iso",
    }

    if bookmakers:
        params[
            "bookmakers"
        ] = bookmakers
    else:
        params[
            "regions"
        ] = region

    payload, headers = api_get(
        url,
        params,
    )

    if not isinstance(
        payload,
        dict,
    ):
        raise RuntimeError(
            "Unexpected event odds response."
        )

    with path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            payload,
            handle,
            indent=2,
        )

    return (
        payload,
        headers,
        False,
    )


def starters_for_event(
    starters: pd.DataFrame,
    home: str,
    away: str,
) -> pd.DataFrame:
    return starters.loc[
        starters["team"].isin(
            [
                home,
                away,
            ]
        )
    ].copy()


def match_pitcher(
    player_name: str,
    candidates: pd.DataFrame,
) -> pd.Series | None:
    exact_matches = []

    for index, row in candidates.iterrows():
        if names_match(
            player_name,
            row[
                "pitcher_name"
            ],
        ):
            exact_matches.append(
                index
            )

    if len(
        exact_matches
    ) == 1:
        return candidates.loc[
            exact_matches[0]
        ]

    if len(
        exact_matches
    ) > 1:
        return candidates.loc[
            exact_matches[0]
        ]

    api_tokens = (
        next(
            iter(
                name_aliases(
                    player_name
                )
            ),
            "",
        )
        .split()
    )

    if not api_tokens:
        return None

    api_last = (
        api_tokens[-1]
    )

    last_name_matches = []

    for index, row in candidates.iterrows():
        aliases = name_aliases(
            row[
                "pitcher_name"
            ]
        )

        for alias in aliases:
            tokens = alias.split()

            if (
                tokens
                and tokens[-1]
                == api_last
            ):
                last_name_matches.append(
                    index
                )
                break

    if len(
        last_name_matches
    ) == 1:
        return candidates.loc[
            last_name_matches[0]
        ]

    return None


def extract_market_rows(
    event: dict[str, Any],
    starters: pd.DataFrame,
    captured_at: str,
) -> tuple[
    list[dict[str, Any]],
    list[str],
]:
    rows = []
    unmatched_players = []

    home = (
        event.get(
            "_home_abbr"
        )
        or normalize_team_name(
            event.get(
                "home_team"
            )
        )
    )

    away = (
        event.get(
            "_away_abbr"
        )
        or normalize_team_name(
            event.get(
                "away_team"
            )
        )
    )

    candidates = starters_for_event(
        starters,
        home,
        away,
    )

    event_id = str(
        event.get(
            "id",
            "",
        )
    )

    commence_time = event.get(
        "commence_time"
    )

    for bookmaker in (
        event.get(
            "bookmakers"
        )
        or []
    ):
        sportsbook = (
            bookmaker.get(
                "title"
            )
            or bookmaker.get(
                "key"
            )
            or "UNKNOWN"
        )

        bookmaker_update = (
            bookmaker.get(
                "last_update"
            )
        )

        for market in (
            bookmaker.get(
                "markets"
            )
            or []
        ):
            market_key = market.get(
                "key"
            )

            if (
                market_key
                not in MARKET_TARGET_MAP
            ):
                continue

            target = (
                MARKET_TARGET_MAP[
                    market_key
                ]
            )

            market_update = market.get(
                "last_update"
            )

            paired: dict[
                tuple[
                    str,
                    float,
                ],
                dict[str, Any],
            ] = {}

            for outcome in (
                market.get(
                    "outcomes"
                )
                or []
            ):
                side = normalize_text(
                    outcome.get(
                        "name"
                    )
                )

                if side not in {
                    "over",
                    "under",
                }:
                    continue

                player_name = outcome.get(
                    "description"
                )

                point = outcome.get(
                    "point"
                )

                price = outcome.get(
                    "price"
                )

                if (
                    not player_name
                    or point is None
                    or price is None
                ):
                    continue

                matched = match_pitcher(
                    str(player_name),
                    candidates,
                )

                if matched is None:
                    unmatched_players.append(
                        str(
                            player_name
                        )
                    )
                    continue

                try:
                    line = float(
                        point
                    )
                    american_price = int(
                        round(
                            float(
                                price
                            )
                        )
                    )
                except (
                    TypeError,
                    ValueError,
                ):
                    continue

                key = (
                    str(
                        matched[
                            "pitcher_name"
                        ]
                    ),
                    line,
                )

                if key not in paired:
                    pitcher_id = matched.get(
                        "pitcher_mlbam_id",
                        pd.NA,
                    )

                    paired[
                        key
                    ] = {
                        "pitcher_mlbam_id":
                            pitcher_id,
                        "pitcher_name":
                            matched[
                                "pitcher_name"
                            ],
                        "team":
                            matched[
                                "team"
                            ],
                        "opponent":
                            matched[
                                "opponent"
                            ],
                        "event_id":
                            event_id,
                        "commence_time":
                            commence_time,
                        "odds_captured_at":
                            captured_at,
                        "bookmaker_last_update":
                            bookmaker_update,
                        "market_last_update":
                            market_update,
                        "target":
                            target,
                        "line":
                            line,
                        "over_odds":
                            None,
                        "under_odds":
                            None,
                        "sportsbook":
                            sportsbook,
                    }

                if side == "over":
                    paired[
                        key
                    ][
                        "over_odds"
                    ] = american_price

                elif side == "under":
                    paired[
                        key
                    ][
                        "under_odds"
                    ] = american_price

            for row in paired.values():
                if (
                    row[
                        "over_odds"
                    ]
                    is None
                    or row[
                        "under_odds"
                    ]
                    is None
                ):
                    continue

                rows.append(
                    row
                )

    return (
        rows,
        unmatched_players,
    )


def save_history(
    board: pd.DataFrame,
    date: str,
) -> Path:
    directory = (
        HISTORY_ROOT
        / date
    )

    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    stamp = (
        datetime.now(
            timezone.utc
        )
        .strftime(
            "%Y%m%dT%H%M%S%fZ"
        )
    )

    path = (
        directory
        / f"{stamp}.csv"
    )

    board.to_csv(
        path,
        index=False,
    )

    return path


def collect_odds(
    date: str,
    region: str,
    cache_minutes: int,
    force: bool,
    bookmakers: str | None,
) -> pd.DataFrame:
    api_key = require_api_key()

    starters = load_starters(
        date
    )

    print()
    print("=" * 80)
    print(
        "AUTOMATIC MLB PITCHER PROP ODDS"
    )
    print("=" * 80)
    print()

    print(
        f"Slate date: {date}"
    )
    print(
        f"Pregame starters: "
        f"{len(starters)}"
    )

    if bookmakers:
        print(
            f"Bookmakers: {bookmakers}"
        )
    else:
        print(
            f"Region: {region}"
        )

    print(
        "Markets: "
        "pitcher_strikeouts, pitcher_outs"
    )
    print(
        f"Cache window: "
        f"{cache_minutes} minute(s)"
    )
    print()

    events = fetch_events(
        api_key
    )

    matching_events = (
        match_events_to_slate(
            events,
            starters,
        )
    )

    print(
        "Upcoming MLB events returned: "
        f"{len(events)}"
    )
    print(
        "Events matched to slate: "
        f"{len(matching_events)}"
    )

    if not matching_events:
        raise RuntimeError(
            "No The Odds API events matched "
            "today's MLB starter slate."
        )

    captured_at = (
        datetime.now(
            timezone.utc
        )
        .isoformat()
    )

    all_rows = []
    unmatched = []

    api_calls = 0
    cached_events = 0

    last_remaining = None
    last_used = None
    last_cost = None

    for number, event in enumerate(
        matching_events,
        start=1,
    ):
        event_id = str(
            event.get(
                "id"
            )
        )

        away = event.get(
            "_away_abbr"
        )
        home = event.get(
            "_home_abbr"
        )

        print(
            f"[{number}/"
            f"{len(matching_events)}] "
            f"{away} @ {home}"
        )

        payload, headers, cached = (
            fetch_event_odds(
                api_key=api_key,
                event_id=event_id,
                date=date,
                region=region,
                cache_minutes=(
                    cache_minutes
                ),
                force=force,
                bookmakers=bookmakers,
            )
        )

        payload[
            "_home_abbr"
        ] = home
        payload[
            "_away_abbr"
        ] = away

        if cached:
            cached_events += 1
            print(
                "    using local cache"
            )
        else:
            api_calls += 1
            print(
                "    live API request"
            )

            last_remaining = (
                headers.get(
                    "x-requests-remaining"
                )
            )

            last_used = (
                headers.get(
                    "x-requests-used"
                )
            )

            last_cost = (
                headers.get(
                    "x-requests-last"
                )
            )

        rows, missed = (
            extract_market_rows(
                payload,
                starters,
                captured_at,
            )
        )

        all_rows.extend(
            rows
        )
        unmatched.extend(
            missed
        )

        print(
            "    complete markets: "
            f"{len(rows)}"
        )

    columns = [
        "pitcher_mlbam_id",
        "pitcher_name",
        "team",
        "opponent",
        "event_id",
        "commence_time",
        "odds_captured_at",
        "bookmaker_last_update",
        "market_last_update",
        "target",
        "line",
        "over_odds",
        "under_odds",
        "sportsbook",
    ]

    board = pd.DataFrame(
        all_rows,
        columns=columns,
    )

    if board.empty:
        raise RuntimeError(
            "The Odds API returned no complete "
            "pitcher strikeout/outs markets."
        )

    board = (
        board
        .drop_duplicates()
        .sort_values(
            [
                "pitcher_name",
                "target",
                "line",
                "sportsbook",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        REPORTS_DIR
        / f"market_api_{date}.csv"
    )

    board.to_csv(
        output_path,
        index=False,
    )

    history_path = save_history(
        board,
        date,
    )

    print()
    print("=" * 80)
    print(
        "ODDS COLLECTION COMPLETE"
    )
    print("=" * 80)
    print()

    print(
        "Complete market rows: "
        f"{len(board)}"
    )
    print(
        "Pitchers covered: "
        f"{board['pitcher_name'].nunique()}"
    )
    print(
        "Sportsbooks covered: "
        f"{board['sportsbook'].nunique()}"
    )
    print(
        "Strikeout rows: "
        f"{(board['target'] == 'strikeouts').sum()}"
    )
    print(
        "Outs rows: "
        f"{(board['target'] == 'outs_recorded').sum()}"
    )
    print(
        "Live event requests: "
        f"{api_calls}"
    )
    print(
        "Cached event responses: "
        f"{cached_events}"
    )

    if last_cost is not None:
        print(
            "Last request cost: "
            f"{last_cost}"
        )

    if last_used is not None:
        print(
            "API credits used: "
            f"{last_used}"
        )

    if last_remaining is not None:
        print(
            "API credits remaining: "
            f"{last_remaining}"
        )

    print()
    print(
        f"Odds history: {history_path}"
    )
    print()
    print(
        f"Output: {output_path}"
    )
    print()

    print(
        "SAMPLE AUTOMATED MARKETS"
    )
    print()

    print(
        board.head(
            20
        ).to_string(
            index=False
        )
    )

    if unmatched:
        unique_unmatched = sorted(
            set(
                unmatched
            )
        )

        print()
        print(
            "Unmatched sportsbook pitcher "
            f"names: {len(unique_unmatched)}"
        )

        for name in (
            unique_unmatched[:20]
        ):
            print(
                f"  - {name}"
            )

    return board


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--date",
        required=True,
        help="Slate date YYYY-MM-DD.",
    )

    parser.add_argument(
        "--region",
        default="us",
        help="The Odds API region.",
    )

    parser.add_argument(
        "--cache-minutes",
        type=int,
        default=10,
        help=(
            "Reuse event odds cached within "
            "this many minutes."
        ),
    )

    parser.add_argument(
        "--bookmakers",
        default=None,
        help=(
            "Optional comma-separated "
            "bookmaker keys."
        ),
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignore local odds cache.",
    )

    args = parser.parse_args()

    collect_odds(
        date=args.date,
        region=args.region,
        cache_minutes=args.cache_minutes,
        force=args.force,
        bookmakers=args.bookmakers,
    )


if __name__ == "__main__":
    main()
