from datetime import date

import pandas as pd

from mlb_quant.data.statcast import (
    StatcastIngestor,
    build_starting_pitcher_games,
    make_date_chunks,
    validate_statcast_frame,
)


def statcast_frame():
    return pd.DataFrame(
        [
            {
                "game_date": "2024-04-01",
                "game_pk": 1,
                "pitcher": 101,
                "batter": 201,
                "at_bat_number": 1,
                "inning": 1,
                "inning_topbot": "top",
                "pitch_number": 1,
                "events": "strikeout",
                "description": "called_strike",
                "stand": "R",
                "p_throws": "R",
                "pitch_type": "FF",
                "player_name": "Starter One",
                "away_team": "AWY",
                "home_team": "HME",
                "post_away_score": 0,
                "post_home_score": 0,
            },
            {
                "game_date": "2024-04-01",
                "game_pk": 1,
                "pitcher": 101,
                "batter": 202,
                "at_bat_number": 2,
                "inning": 1,
                "inning_topbot": "top",
                "pitch_number": 1,
                "events": "single",
                "description": "hit_into_play",
                "stand": "L",
                "p_throws": "R",
                "pitch_type": "SL",
                "player_name": "Starter One",
                "away_team": "AWY",
                "home_team": "HME",
                "post_away_score": 0,
                "post_home_score": 0,
            },
            {
                "game_date": "2024-04-01",
                "game_pk": 1,
                "pitcher": 102,
                "batter": 203,
                "at_bat_number": 3,
                "inning": 1,
                "inning_topbot": "bottom",
                "pitch_number": 1,
                "events": "walk",
                "description": "ball",
                "stand": "R",
                "p_throws": "L",
                "pitch_type": "CH",
                "player_name": "Starter Two",
                "away_team": "AWY",
                "home_team": "HME",
                "post_away_score": 0,
                "post_home_score": 0,
            },
            {
                "game_date": "2024-04-01",
                "game_pk": 1,
                "pitcher": 102,
                "batter": 204,
                "at_bat_number": 4,
                "inning": 1,
                "inning_topbot": "bottom",
                "pitch_number": 1,
                "events": "field_out",
                "description": "hit_into_play",
                "stand": "R",
                "p_throws": "L",
                "pitch_type": "FF",
                "player_name": "Starter Two",
                "away_team": "AWY",
                "home_team": "HME",
                "post_away_score": 0,
                "post_home_score": 0,
            },
        ]
    )


def test_date_chunks_are_inclusive_and_bounded():
    chunks = make_date_chunks(date(2024, 4, 1), date(2024, 4, 10), chunk_days=7)
    assert [(chunk.start.isoformat(), chunk.end.isoformat()) for chunk in chunks] == [
        ("2024-04-01", "2024-04-07"),
        ("2024-04-08", "2024-04-10"),
    ]


def test_ingestor_resumes_completed_chunk(tmp_path):
    calls = []

    def fetcher(start, end):
        calls.append((start, end))
        return statcast_frame()

    ingestor = StatcastIngestor(tmp_path, fetcher=fetcher, retry_delay_seconds=0)
    first = ingestor.download(date(2024, 4, 1), date(2024, 4, 1), chunk_days=7)
    second = ingestor.download(date(2024, 4, 1), date(2024, 4, 1), chunk_days=7)
    assert first == second
    assert len(calls) == 1
    assert first[0].exists()


def test_validation_reports_duplicate_and_invalid_values():
    frame = statcast_frame()
    frame.loc[0, "inning_topbot"] = "invalid"
    frame.loc[1, "p_throws"] = "X"
    frame = pd.concat([frame, frame.iloc[[2]]], ignore_index=True)
    report = validate_statcast_frame(frame)
    assert report.duplicate_rows == 1
    assert report.invalid_sides == 1
    assert report.invalid_handedness["p_throws"] == 1


def test_starting_pitcher_metrics_use_first_pitcher_by_side():
    result = build_starting_pitcher_games(statcast_frame())
    away = result[result["home_away"] == "away"].iloc[0]
    home = result[result["home_away"] == "home"].iloc[0]
    assert home["pitcher_mlbam_id"] == 101
    assert home["pitches_thrown"] == 2
    assert home["batters_faced"] == 2
    assert home["strikeouts"] == 1
    assert home["outs_recorded"] == 1
    assert home["hits"] == 1
    assert away["pitcher_mlbam_id"] == 102
    assert away["team"] == "AWY"
    assert away["opponent"] == "HME"
    assert away["home_away"] == "away"
    assert away["walks"] == 1
    assert away["outs_recorded"] == 1


def test_statcast_edge_events_reconcile_official_style_counts():
    frame = statcast_frame()
    frame.loc[0:1, "inning_topbot"] = "Top"
    frame.loc[0, "events"] = "sac_fly"
    frame.loc[0, "description"] = "hit_into_play"
    frame.loc[0, "pitch_type"] = None
    frame.loc[0, "pitch_number"] = 2
    frame.loc[1, "description"] = "automatic_ball"
    result = build_starting_pitcher_games(frame)
    away = result[result["pitcher_mlbam_id"] == 101].iloc[0]
    assert away["outs_recorded"] == 1
    assert away["pitches_thrown"] == 1


def test_truncated_plate_appearance_is_not_a_batter_faced():
    frame = statcast_frame()
    frame.loc[1, "events"] = "truncated_pa"
    result = build_starting_pitcher_games(frame)
    home = result[result["pitcher_mlbam_id"] == 101].iloc[0]
    assert home["batters_faced"] == 1
