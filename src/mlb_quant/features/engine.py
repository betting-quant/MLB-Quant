"""Leakage-controlled MLB pitcher and opponent feature engineering."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

TARGET_COLUMNS = ["strikeouts", "outs_recorded", "pitches_thrown", "batters_faced"]
PITCH_TYPES = (
    "CH",
    "CU",
    "EP",
    "FC",
    "FF",
    "FO",
    "FS",
    "KC",
    "KN",
    "PO",
    "SC",
    "SI",
    "SL",
    "ST",
    "SV",
)
SWING_DESCRIPTIONS = {
    "foul",
    "foul_bunt",
    "foul_tip",
    "hit_into_play",
    "missed_bunt",
    "swinging_strike",
    "swinging_strike_blocked",
}
WHIFF_DESCRIPTIONS = {"swinging_strike", "swinging_strike_blocked"}
CONTACT_DESCRIPTIONS = {"foul", "foul_bunt", "foul_tip", "hit_into_play", "missed_bunt"}
STRIKE_DESCRIPTIONS = SWING_DESCRIPTIONS | {"called_strike"}
TERMINAL_EVENTS = {
    "single",
    "double",
    "triple",
    "home_run",
    "walk",
    "intent_walk",
    "strikeout",
    "strikeout_double_play",
    "field_out",
    "force_out",
    "grounded_into_double_play",
    "double_play",
    "triple_play",
    "fielders_choice_out",
    "sac_fly",
    "sac_fly_double_play",
    "hit_by_pitch",
    "catcher_interf",
    "field_error",
    "fielders_choice",
    "other_out",
    "truncated_pa",
}
PITCH_USE_COLUMNS = [
    "pitch_count",
    "velocity_sum",
    "velocity_count",
    "whiff_count",
    "strike_count",
    "csw_count",
]


def _rate(numerator: pd.Series | float, denominator: pd.Series | float) -> pd.Series | float:
    return numerator / denominator.replace(0, np.nan) if isinstance(denominator, pd.Series) else numerator / denominator if denominator else np.nan


def _pitch_group_metrics(group: pd.DataFrame) -> dict[str, float]:
    descriptions = group["description"].astype("string")
    events = group["events"].astype("string")
    pitch_mask = group["pitch_number"].notna() & descriptions.ne("automatic_ball")
    pitches = group.loc[pitch_mask]
    desc = descriptions.loc[pitch_mask]
    zone = pd.to_numeric(group.loc[pitch_mask, "zone"], errors="coerce")
    zone_mask = zone.between(1, 9, inclusive="both")
    swings = desc.isin(SWING_DESCRIPTIONS)
    contacts = desc.isin(CONTACT_DESCRIPTIONS)
    whiffs = desc.isin(WHIFF_DESCRIPTIONS)
    called = desc.eq("called_strike")
    terminal = events.isin(TERMINAL_EVENTS)
    completed_at_bats = group.loc[~events.eq("truncated_pa"), "at_bat_number"].nunique()
    first_pitch = pitches.sort_values(["at_bat_number", "pitch_number"]).groupby("at_bat_number", sort=False).first()
    first_pitch_strikes = first_pitch["description"].isin(STRIKE_DESCRIPTIONS).sum() if not first_pitch.empty else 0
    out_zone = ~zone_mask
    out_zone_pitches = int(out_zone.sum())
    out_zone_swings = int((out_zone & swings).sum())
    zone_swings = int((zone_mask & swings).sum())
    zone_contacts = int((zone_mask & contacts).sum())
    velocity = pd.to_numeric(pitches["release_speed"], errors="coerce")
    return {
        "pitches": int(len(pitches)),
        "bf_metric": int(completed_at_bats),
        "k_count": int(events.eq("strikeout").sum() + events.eq("strikeout_double_play").sum()),
        "bb_count": int(events.isin({"walk", "intent_walk"}).sum()),
        "hits_count": int(events.isin({"single", "double", "triple", "home_run"}).sum()),
        "strike_count": int(strikes := desc.isin(STRIKE_DESCRIPTIONS).sum()),
        "called_strike_count": int(called.sum()),
        "whiff_count": int(whiffs.sum()),
        "csw_count": int((called | whiffs).sum()),
        "swing_count": int(swings.sum()),
        "contact_count": int(contacts.sum()),
        "zone_pitch_count": int(zone_mask.sum()),
        "zone_swing_count": zone_swings,
        "zone_contact_count": zone_contacts,
        "chase_swing_count": out_zone_swings,
        "chase_pitch_count": out_zone_pitches,
        "first_pitch_strike_count": int(first_pitch_strikes),
        "first_pitch_pa_count": int(len(first_pitch)),
        "velocity_sum": float(velocity.sum()),
        "velocity_count": int(velocity.notna().sum()),
        "terminal_pa_count": int(terminal.sum()),
    }


def _team_group_metrics(group: pd.DataFrame) -> dict[str, float]:
    metrics = _pitch_group_metrics(group)
    metrics["team_bf"] = metrics.pop("bf_metric")
    return metrics


def build_raw_game_metrics(raw_paths: Iterable[Path], appearances: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read raw chunks and return pitcher-game, pitch-type, and team-game metrics."""
    starter_keys = appearances[["game_id", "pitcher_mlbam_id"]].drop_duplicates()
    usecols = [
        "game_date", "game_pk", "pitcher", "batter", "events", "description", "p_throws",
        "home_team", "away_team", "inning_topbot", "zone", "release_speed", "pitch_type",
        "at_bat_number", "pitch_number",
    ]
    frames = [pd.read_parquet(path, columns=usecols) for path in raw_paths]
    frame = pd.concat(frames, ignore_index=True)
    frame = frame.rename(columns={"game_pk": "game_id", "pitcher": "pitcher_mlbam_id"})
    frame["game_id"] = pd.to_numeric(frame["game_id"], errors="coerce")
    frame["pitcher_mlbam_id"] = pd.to_numeric(frame["pitcher_mlbam_id"], errors="coerce")
    frame["game_date"] = pd.to_datetime(frame["game_date"])
    starter_frame = frame.merge(starter_keys, on=["game_id", "pitcher_mlbam_id"], how="inner")
    def add_flags(data: pd.DataFrame) -> pd.DataFrame:
        descriptions = data["description"].astype("string")
        events = data["events"].astype("string")
        pitch = data["pitch_number"].notna() & descriptions.ne("automatic_ball")
        zone = pd.to_numeric(data["zone"], errors="coerce").between(1, 9)
        data = data.copy()
        data["pitches"] = pitch.astype(int)
        data["bf_metric"] = (~events.eq("truncated_pa")).fillna(False).astype(int)
        flag = lambda value: value.fillna(False).astype(int)
        data["k_count"] = flag(events.isin({"strikeout", "strikeout_double_play"}))
        data["bb_count"] = flag(events.isin({"walk", "intent_walk"}))
        data["hits_count"] = flag(events.isin({"single", "double", "triple", "home_run"}))
        data["strike_count"] = flag(pitch & descriptions.isin(STRIKE_DESCRIPTIONS))
        data["called_strike_count"] = flag(pitch & descriptions.eq("called_strike"))
        data["whiff_count"] = flag(pitch & descriptions.isin(WHIFF_DESCRIPTIONS))
        data["csw_count"] = flag(pitch & descriptions.isin({"called_strike", *WHIFF_DESCRIPTIONS}))
        data["swing_count"] = flag(pitch & descriptions.isin(SWING_DESCRIPTIONS))
        data["contact_count"] = flag(pitch & descriptions.isin(CONTACT_DESCRIPTIONS))
        data["zone_pitch_count"] = flag(pitch & zone)
        data["zone_swing_count"] = flag(pitch & zone & descriptions.isin(SWING_DESCRIPTIONS))
        data["zone_contact_count"] = flag(pitch & zone & descriptions.isin(CONTACT_DESCRIPTIONS))
        data["chase_swing_count"] = flag(pitch & ~zone & descriptions.isin(SWING_DESCRIPTIONS))
        data["chase_pitch_count"] = flag(pitch & ~zone)
        data["velocity_sum"] = pd.to_numeric(data["release_speed"], errors="coerce").where(pitch, 0).fillna(0)
        data["velocity_count"] = flag(pitch & data["release_speed"].notna())
        first_pitch = data["pitch_number"].eq(
            data.groupby(["game_id", "pitcher_mlbam_id", "at_bat_number"])["pitch_number"].transform("min")
        )
        data["first_pitch_strike_count"] = flag(first_pitch & descriptions.isin(STRIKE_DESCRIPTIONS))
        data["first_pitch_pa_count"] = flag(first_pitch & data["pitch_number"].notna())
        return data

    flagged = add_flags(starter_frame)
    metrics = ["pitches", "bf_metric", "k_count", "bb_count", "hits_count", "strike_count", "called_strike_count", "whiff_count", "csw_count", "swing_count", "contact_count", "zone_pitch_count", "zone_swing_count", "zone_contact_count", "chase_swing_count", "chase_pitch_count", "first_pitch_strike_count", "first_pitch_pa_count", "velocity_sum", "velocity_count"]
    keys = ["game_id", "game_date", "pitcher_mlbam_id"]
    pitcher_records = flagged.groupby(keys, as_index=False)[metrics].sum()
    pitcher_records["season"] = pitcher_records["game_date"].dt.year
    pitcher_records = pitcher_records.merge(appearances[["game_id", "pitcher_mlbam_id", "outs_recorded"]], on=["game_id", "pitcher_mlbam_id"], how="left")
    type_grouped = flagged[flagged["pitch_type"].notna()].groupby(keys + ["pitch_type"], as_index=False)[metrics].sum()
    type_records = type_grouped.rename(columns={"pitches": "pitch_count"})[[*keys, "pitch_type", "pitch_count", "velocity_sum", "velocity_count", "whiff_count", "strike_count", "csw_count"]]
    type_records["season"] = type_records["game_date"].dt.year
    frame["batting_team"] = np.where(frame["inning_topbot"].astype("string").str.lower().eq("top"), frame["away_team"], frame["home_team"])
    flagged_all = add_flags(frame)
    team_keys = ["game_id", "game_date", "batting_team", "p_throws"]
    team_records = flagged_all.groupby(team_keys, as_index=False)[metrics].sum().rename(columns={"batting_team": "team", "p_throws": "opponent_hand"})
    team_records["team_bf"] = team_records.pop("bf_metric")
    team_records["season"] = team_records["game_date"].dt.year
    return pd.DataFrame(pitcher_records), pd.DataFrame(type_records), pd.DataFrame(team_records)


def _aggregate_prior(group: pd.DataFrame, date_value: pd.Timestamp, days: int | None, metrics: list[str]) -> tuple[dict[str, float], int]:
    dates = pd.to_datetime(group["game_date"])
    mask = dates < date_value
    if days is not None:
        mask &= dates >= date_value - pd.Timedelta(days=days)
    history = group.loc[mask]
    return {metric: pd.to_numeric(history[metric], errors="coerce").sum() for metric in metrics}, int(history["game_id"].nunique())


def _metric_features(prefix: str, totals: dict[str, float], starts: int, minimum_starts: int) -> dict[str, float]:
    insufficient = starts < minimum_starts
    values = {
        f"{prefix}_k_pct": _rate(totals.get("k_count", 0), totals.get("bf_metric", totals.get("team_bf", 0))),
        f"{prefix}_k_per_start": _rate(totals.get("k_count", 0), starts),
        f"{prefix}_pitches_per_start": _rate(totals.get("pitches", 0), starts),
        f"{prefix}_bf_per_start": _rate(totals.get("bf_metric", totals.get("team_bf", 0)), starts),
        f"{prefix}_outs_per_start": _rate(totals.get("outs_recorded", 0), starts),
        f"{prefix}_bb_pct": _rate(totals.get("bb_count", 0), totals.get("bf_metric", totals.get("team_bf", 0))),
        f"{prefix}_hits_per_bf": _rate(totals.get("hits_count", 0), totals.get("bf_metric", totals.get("team_bf", 0))),
        f"{prefix}_swstr_pct": _rate(totals.get("whiff_count", 0), totals.get("pitches", 0)),
        f"{prefix}_csw_pct": _rate(totals.get("csw_count", 0), totals.get("pitches", 0)),
        f"{prefix}_called_strike_pct": _rate(totals.get("called_strike_count", 0), totals.get("pitches", 0)),
        f"{prefix}_chase_rate": _rate(totals.get("chase_swing_count", 0), totals.get("chase_pitch_count", 0)),
        f"{prefix}_contact_rate": _rate(totals.get("contact_count", 0), totals.get("swing_count", 0)),
        f"{prefix}_zone_contact_rate": _rate(totals.get("zone_contact_count", 0), totals.get("zone_swing_count", 0)),
        f"{prefix}_first_pitch_strike_rate": _rate(totals.get("first_pitch_strike_count", 0), totals.get("first_pitch_pa_count", 0)),
        f"{prefix}_strike_pct": _rate(totals.get("strike_count", 0), totals.get("pitches", 0)),
        f"{prefix}_history_starts": starts,
        f"{prefix}_insufficient_history": int(insufficient),
    }
    if insufficient:
        for key in list(values):
            if key not in {f"{prefix}_history_starts", f"{prefix}_insufficient_history"}:
                values[key] = np.nan
    return values


def _pitcher_features(appearances: pd.DataFrame, pitcher_metrics: pd.DataFrame, minimum_starts: int) -> pd.DataFrame:
    frame = appearances.copy()
    frame["game_date"] = pd.to_datetime(frame["game_date"])
    metric_inputs = pitcher_metrics.rename(columns={"outs_recorded": "outs_history"}).copy()
    metric_inputs = metric_inputs.copy()
    metric_inputs["game_date"] = pd.to_datetime(metric_inputs["game_date"])
    merged = frame.merge(metric_inputs, on=["season", "game_date", "game_id", "pitcher_mlbam_id"], how="left", suffixes=("", "_metric"))
    numeric_metric_columns = [column for column in metric_inputs.columns if column not in {"season", "game_date", "game_id", "pitcher_mlbam_id"}]
    merged[numeric_metric_columns] = merged[numeric_metric_columns].fillna(0)
    merged = merged.sort_values(["pitcher_mlbam_id", "season", "game_date", "game_id"]).reset_index(drop=True)
    group_columns = ["pitcher_mlbam_id", "season"]
    aggregate_metrics = [column for column in numeric_metric_columns if column not in {"velocity_sum", "velocity_count"}]
    aggregate_metrics.append("outs_history")
    output = merged.copy()
    grouped = merged.groupby(group_columns, sort=False)

    def add_window(prefix: str, prior_totals: dict[str, pd.Series], starts: pd.Series) -> None:
        totals = {key: value for key, value in prior_totals.items()}
        totals["outs_recorded"] = totals.pop("outs_history", pd.Series(0.0, index=merged.index))
        denominator = totals.get("bf_metric", totals.get("team_bf", pd.Series(0.0, index=merged.index)))
        pitches = totals.get("pitches", pd.Series(0.0, index=merged.index))
        values = {
            f"{prefix}_k_pct": _rate(totals.get("k_count", 0), denominator),
            f"{prefix}_k_per_start": _rate(totals.get("k_count", 0), starts),
            f"{prefix}_pitches_per_start": _rate(pitches, starts),
            f"{prefix}_bf_per_start": _rate(denominator, starts),
            f"{prefix}_outs_per_start": _rate(totals["outs_recorded"], starts),
            f"{prefix}_bb_pct": _rate(totals.get("bb_count", 0), denominator),
            f"{prefix}_hits_per_bf": _rate(totals.get("hits_count", 0), denominator),
            f"{prefix}_swstr_pct": _rate(totals.get("whiff_count", 0), pitches),
            f"{prefix}_csw_pct": _rate(totals.get("csw_count", 0), pitches),
            f"{prefix}_called_strike_pct": _rate(totals.get("called_strike_count", 0), pitches),
            f"{prefix}_chase_rate": _rate(totals.get("chase_swing_count", 0), totals.get("chase_pitch_count", 0)),
            f"{prefix}_contact_rate": _rate(totals.get("contact_count", 0), totals.get("swing_count", 0)),
            f"{prefix}_zone_contact_rate": _rate(totals.get("zone_contact_count", 0), totals.get("zone_swing_count", 0)),
            f"{prefix}_first_pitch_strike_rate": _rate(totals.get("first_pitch_strike_count", 0), totals.get("first_pitch_pa_count", 0)),
            f"{prefix}_strike_pct": _rate(totals.get("strike_count", 0), pitches),
            f"{prefix}_history_starts": starts,
            f"{prefix}_insufficient_history": (starts < minimum_starts).astype(int),
        }
        output[list(values)] = pd.DataFrame(values, index=merged.index)
        insufficient = starts < minimum_starts
        for column in values:
            if column not in {f"{prefix}_history_starts", f"{prefix}_insufficient_history"}:
                output.loc[insufficient, column] = np.nan

    for window_name, size in [("season", None), ("last3", 3), ("last5", 5), ("last10", 10)]:
        if size is None:
            totals = {metric: grouped[metric].transform(lambda values: values.shift(1).cumsum()) for metric in aggregate_metrics}
            starts = grouped["game_id"].cumcount()
        else:
            totals = {metric: grouped[metric].transform(lambda values: values.shift(1).rolling(size, min_periods=1).sum()) for metric in aggregate_metrics}
            starts = grouped["game_id"].transform(lambda values: values.shift(1).rolling(size, min_periods=1).count())
        add_window(window_name, totals, starts)

    output["pitches_median_before_game"] = grouped["pitches_thrown"].transform(
        lambda values: values.shift(1).expanding().median()
    )

    for days in (30, 60):
        prior_totals = {metric: pd.Series(np.nan, index=merged.index) for metric in aggregate_metrics}
        starts = pd.Series(np.nan, index=merged.index)
        for _, group in merged.groupby(group_columns, sort=False):
            dates = group["game_date"]
            for metric in aggregate_metrics:
                values = group.set_index("game_date")[metric].rolling(f"{days}D", closed="left").sum()
                prior_totals[metric].loc[group.index] = values.to_numpy()
            starts.loc[group.index] = group.set_index("game_date")["game_id"].rolling(f"{days}D", closed="left").count().to_numpy()
        add_window(f"last{days}d", prior_totals, starts)
    for column in numeric_metric_columns:
        output.drop(columns=column, inplace=True)
    return output


def _arsenal_features(frame: pd.DataFrame, type_metrics: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    if type_metrics.empty:
        return output
    key = ["season", "game_date", "game_id", "pitcher_mlbam_id"]
    type_metrics = type_metrics.copy()
    type_metrics["game_date"] = pd.to_datetime(type_metrics["game_date"])
    type_metrics = type_metrics.sort_values(["pitcher_mlbam_id", "season", "pitch_type", "game_date", "game_id"])
    grouped = type_metrics.groupby(["pitcher_mlbam_id", "season", "pitch_type"], sort=False)
    for column in PITCH_USE_COLUMNS:
        type_metrics[f"prior_{column}"] = grouped[column].transform(lambda values: values.cumsum().shift(1))
        type_metrics[f"prior3_{column}"] = grouped[column].transform(
            lambda values: values.shift(1).rolling(3, min_periods=1).sum()
        )
    pitcher_totals = frame.sort_values(["pitcher_mlbam_id", "season", "game_date", "game_id"]).copy()
    pitcher_totals["prior_pitches"] = pitcher_totals.groupby(["pitcher_mlbam_id", "season"])["pitches_thrown"].transform(lambda values: values.cumsum().shift(1))
    pitcher_totals["prior3_pitches"] = pitcher_totals.groupby(["pitcher_mlbam_id", "season"])["pitches_thrown"].transform(lambda values: values.shift(1).rolling(3, min_periods=1).sum())
    prior_pitches = pitcher_totals[key + ["prior_pitches", "prior3_pitches"]]
    type_wide = type_metrics.set_index(key + ["pitch_type"])[[f"prior_{c}" for c in PITCH_USE_COLUMNS] + [f"prior3_{c}" for c in PITCH_USE_COLUMNS]].unstack("pitch_type")
    type_wide.columns = [f"{metric}_{pitch_type}" for metric, pitch_type in type_wide.columns]
    type_wide = type_wide.reset_index()
    merged = output.merge(type_wide, on=key, how="left")
    merged = merged.merge(prior_pitches, on=key, how="left")
    additions = {}
    for pitch_type in PITCH_TYPES:
        totals = {column: merged[f"prior_{column}_{pitch_type}"].fillna(0) if f"prior_{column}_{pitch_type}" in merged else pd.Series(0, index=merged.index) for column in PITCH_USE_COLUMNS}
        additions[f"pitch_{pitch_type}_usage"] = _rate(totals["pitch_count"], merged["prior_pitches"])
        additions[f"pitch_{pitch_type}_velocity"] = _rate(totals["velocity_sum"], totals["velocity_count"])
        additions[f"pitch_{pitch_type}_whiff_rate"] = _rate(totals["whiff_count"], totals["pitch_count"])
        additions[f"pitch_{pitch_type}_strike_rate"] = _rate(totals["strike_count"], totals["pitch_count"])
        additions[f"pitch_{pitch_type}_csw_rate"] = _rate(totals["csw_count"], totals["pitch_count"])
        recent_pitch_count = merged.get(f"prior3_pitch_count_{pitch_type}", pd.Series(0.0, index=merged.index)).fillna(0)
        recent_velocity = _rate(
            merged.get(f"prior3_velocity_sum_{pitch_type}", pd.Series(0.0, index=merged.index)),
            merged.get(f"prior3_velocity_count_{pitch_type}", pd.Series(0.0, index=merged.index)),
        )
        season_velocity = additions[f"pitch_{pitch_type}_velocity"]
        recent_usage = _rate(recent_pitch_count, merged["prior3_pitches"].replace(0, np.nan))
        additions[f"pitch_{pitch_type}_usage_change_vs_season"] = recent_usage - additions[f"pitch_{pitch_type}_usage"]
        additions[f"pitch_{pitch_type}_velocity_change_vs_season"] = recent_velocity - season_velocity
        additions[f"pitch_{pitch_type}_insufficient_history"] = (totals["pitch_count"] < 20).astype(int)
    return pd.concat([output.reset_index(drop=True), pd.DataFrame(additions)], axis=1)


def _opponent_features(frame: pd.DataFrame, team_metrics: pd.DataFrame, minimum_games: int) -> pd.DataFrame:
    output = frame.copy()
    if team_metrics.empty:
        return output
    team_metrics = team_metrics.copy()
    team_metrics["game_date"] = pd.to_datetime(team_metrics["game_date"])
    metrics = ["team_bf", "k_count", "bb_count", "pitches", "whiff_count", "swing_count", "contact_count", "chase_swing_count", "chase_pitch_count"]
    base = team_metrics.groupby(["team", "season", "game_date", "game_id"], as_index=False)[metrics].sum()
    base = base.sort_values(["team", "season", "game_date", "game_id"])
    grouped = base.groupby(["team", "season"], sort=False)
    for metric in metrics:
        base[f"prior_{metric}"] = grouped[metric].transform(lambda values: values.cumsum().shift(1))
    base["prior_games"] = grouped["game_id"].cumcount()
    hand = team_metrics.groupby(["team", "season", "game_date", "game_id", "opponent_hand"], as_index=False)[metrics].sum()
    hand = hand.sort_values(["team", "season", "opponent_hand", "game_date", "game_id"])
    hand_grouped = hand.groupby(["team", "season", "opponent_hand"], sort=False)
    for metric in metrics:
        hand[f"prior_{metric}"] = hand_grouped[metric].transform(lambda values: values.cumsum().shift(1))
    hand["prior_games"] = hand_grouped["game_id"].cumcount()
    merge_keys = ["season", "game_date", "game_id", "team"]
    opponent = output[["season", "game_date", "game_id", "opponent"]].rename(columns={"opponent": "team"})
    merged = opponent.merge(base[merge_keys + [f"prior_{m}" for m in metrics] + ["prior_games"]], on=merge_keys, how="left")
    hand_wide = hand.set_index(merge_keys + ["opponent_hand"])[[f"prior_{m}" for m in metrics] + ["prior_games"]].unstack("opponent_hand")
    hand_wide.columns = [f"{metric}_{hand_name}" for metric, hand_name in hand_wide.columns]
    merged = merged.merge(hand_wide.reset_index(), on=merge_keys, how="left")
    additions = {}
    denom = lambda name: merged[f"prior_{name}"].fillna(0)
    additions["opponent_team_k_pct"] = _rate(denom("k_count"), denom("team_bf"))
    additions["opponent_team_bb_pct"] = _rate(denom("bb_count"), denom("team_bf"))
    additions["opponent_team_swstr_pct"] = _rate(denom("whiff_count"), denom("pitches"))
    additions["opponent_team_contact_pct"] = _rate(denom("contact_count"), denom("swing_count"))
    additions["opponent_team_chase_pct"] = _rate(denom("chase_swing_count"), denom("chase_pitch_count"))
    additions["opponent_team_history_games"] = merged["prior_games"]
    additions["opponent_team_insufficient_history"] = (merged["prior_games"].fillna(0) < minimum_games).astype(int)
    for hand in ("R", "L"):
        hand_denom = lambda name: merged.get(f"prior_{name}_{hand}", pd.Series(np.nan, index=merged.index))
        additions[f"opponent_k_pct_vs_{hand}HP"] = _rate(hand_denom("k_count"), hand_denom("team_bf"))
        additions[f"opponent_contact_pct_vs_{hand}HP"] = _rate(hand_denom("contact_count"), hand_denom("swing_count"))
        additions[f"opponent_{hand}_history_games"] = merged.get(f"prior_games_{hand}", pd.Series(np.nan, index=merged.index))
    return pd.concat([output.reset_index(drop=True), pd.DataFrame(additions)], axis=1)


def build_point_in_time_features(
    appearances: pd.DataFrame,
    pitcher_metrics: pd.DataFrame,
    type_metrics: pd.DataFrame | None = None,
    team_metrics: pd.DataFrame | None = None,
    minimum_starts: int = 3,
) -> pd.DataFrame:
    """Build one pre-game feature row per starting-pitcher appearance."""
    required = {"season", "game_date", "game_id", "pitcher_mlbam_id", "opponent", "home_away", "pitcher_handedness", *TARGET_COLUMNS}
    missing = required - set(appearances.columns)
    if missing:
        raise ValueError(f"Missing appearance columns: {sorted(missing)}")
    frame = appearances.copy()
    frame["game_date"] = pd.to_datetime(frame["game_date"])
    frame = _pitcher_features(frame, pitcher_metrics, minimum_starts)
    frame["pitcher_days_rest"] = frame.groupby("pitcher_mlbam_id")["game_date"].diff().dt.days
    frame["pitcher_starts_before_game"] = frame.groupby(["pitcher_mlbam_id", "season"]).cumcount()
    frame["month"] = frame["game_date"].dt.month
    frame["home_indicator"] = frame["home_away"].eq("home").astype(int)
    previous_pitches = frame.groupby("pitcher_mlbam_id")["pitches_thrown"].shift(1)
    frame["pitches_previous_start"] = previous_pitches
    frame["starts_reaching_90_before_game"] = frame.groupby(["pitcher_mlbam_id", "season"])["pitches_thrown"].transform(lambda values: values.shift().ge(90).cumsum())
    frame["starts_reaching_100_before_game"] = frame.groupby(["pitcher_mlbam_id", "season"])["pitches_thrown"].transform(lambda values: values.shift().ge(100).cumsum())
    frame["workload_trend_last3_vs_prior3"] = frame["last3_pitches_per_start"] - frame.groupby("pitcher_mlbam_id")["last3_pitches_per_start"].shift(3)
    if type_metrics is not None:
        frame = _arsenal_features(frame, type_metrics)
    if team_metrics is not None:
        frame = _opponent_features(frame, team_metrics, minimum_starts)
    frame = frame.sort_values(["game_date", "game_id", "pitcher_mlbam_id"]).reset_index(drop=True)
    leaked_current_columns = {"hits", "walks", "runs", "source_rows", "outs_method"}
    feature_columns = [column for column in frame.columns if column not in TARGET_COLUMNS and column not in leaked_current_columns]
    frame["feature_row_id"] = np.arange(len(frame))
    return frame[feature_columns + TARGET_COLUMNS]


def feature_dictionary(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a machine-readable feature dictionary for the generated table."""
    rows = []
    for column in frame.columns:
        role = "target" if column in TARGET_COLUMNS else "identifier/context" if column in {"season", "game_date", "game_id", "pitcher_mlbam_id", "opponent", "home_away", "pitcher_handedness"} else "feature"
        rows.append({"column": column, "role": role, "dtype": str(frame[column].dtype), "missing_count": int(frame[column].isna().sum()), "description": _describe_column(column)})
    return pd.DataFrame(rows)


def _describe_column(column: str) -> str:
    if column in TARGET_COLUMNS:
        return "Post-game target; never used as an input feature."
    if "insufficient_history" in column:
        return "Flag that the minimum historical sample was not available before this game."
    if column.endswith("_pct") or "rate" in column:
        return "Historical rate calculated using observations strictly before the game date."
    if "last" in column or "season" in column or "previous" in column:
        return "Pre-game historical aggregate, shifted to exclude the current appearance."
    return "Pre-game context or historical feature."


def load_scaled_raw_paths(raw_root: Path, seasons: Iterable[int] = range(2021, 2026)) -> list[Path]:
    return [path for season in seasons for path in sorted((Path(raw_root) / f"season={season}").glob("*.parquet"))]


def generate_feature_table(
    appearance_path: Path = Path("data/processed/starting_pitcher_games_2021_2025.parquet"),
    raw_root: Path = Path("data/raw"),
    output_path: Path = Path("data/processed/model_features_2021_2025.parquet"),
    dictionary_path: Path = Path("reports/feature_dictionary_2021_2025.csv"),
) -> pd.DataFrame:
    appearances = pd.read_parquet(appearance_path)
    raw_paths = load_scaled_raw_paths(raw_root)
    pitcher_metrics, type_metrics, team_metrics = build_raw_game_metrics(raw_paths, appearances)
    features = build_point_in_time_features(appearances, pitcher_metrics, type_metrics, team_metrics)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(output_path, index=False)
    dictionary_path.parent.mkdir(parents=True, exist_ok=True)
    feature_dictionary(features).to_csv(dictionary_path, index=False)
    return features
