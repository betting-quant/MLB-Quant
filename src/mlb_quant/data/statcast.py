"""Chunked, resumable Statcast ingestion and starter-game processing."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

import pandas as pd

LOGGER = logging.getLogger(__name__)

REQUIRED_COLUMNS = {
    "game_date",
    "game_pk",
    "pitcher",
    "batter",
    "at_bat_number",
    "inning",
    "inning_topbot",
    "pitch_number",
    "events",
    "description",
    "stand",
    "p_throws",
    "pitch_type",
}
IDENTIFIER_COLUMNS = ["game_pk", "pitcher", "batter"]
DUPLICATE_KEY_COLUMNS = ["game_pk", "at_bat_number", "pitch_number", "pitcher", "batter"]
VALID_SIDES = {"top", "bottom"}
VALID_HANDS = {"L", "R", "S"}
OUT_EVENTS = {
    "field_out": 1,
    "force_out": 1,
    "strikeout": 1,
    "strikeout_double_play": 2,
    "grounded_into_double_play": 2,
    "double_play": 2,
    "triple_play": 3,
    "fielders_choice_out": 1,
    "other_out": 1,
    "sac_fly_double_play": 2,
    "sac_fly": 1,
    "caught_stealing_2b": 1,
    "caught_stealing_3b": 1,
    "caught_stealing_home": 1,
    "pickoff_1b": 1,
    "pickoff_2b": 1,
    "pickoff_3b": 1,
    "pickoff_caught_stealing_2b": 1,
    "pickoff_caught_stealing_3b": 1,
    "pickoff_caught_stealing_home": 1,
}


@dataclass(frozen=True)
class DateChunk:
    start: date
    end: date

    @property
    def key(self) -> str:
        return f"{self.start.isoformat()}_{self.end.isoformat()}"


@dataclass
class ValidationReport:
    rows: int
    columns_missing: list[str]
    missing_values: dict[str, int]
    duplicate_rows: int
    invalid_dates: int
    invalid_identifiers: dict[str, int]
    invalid_sides: int
    invalid_handedness: dict[str, int]
    invalid_innings: int
    invalid_pitch_numbers: int
    invalid_pitch_types: int

    @property
    def problems(self) -> dict[str, object]:
        return {
            name: value
            for name, value in asdict(self).items()
            if value not in (0, [], {}) and name != "rows"
        }


def make_date_chunks(start: date, end: date, chunk_days: int = 7) -> list[DateChunk]:
    """Return inclusive chunks, with no chunk extending beyond ``end``."""
    if start > end:
        raise ValueError("start date must not be after end date")
    if chunk_days < 1:
        raise ValueError("chunk_days must be positive")
    chunks = []
    cursor = start
    while cursor <= end:
        chunk_end = min(cursor + timedelta(days=chunk_days - 1), end)
        chunks.append(DateChunk(cursor, chunk_end))
        cursor = chunk_end + timedelta(days=1)
    return chunks


def validate_statcast_frame(frame: pd.DataFrame) -> ValidationReport:
    """Report schema and value-quality issues without altering the source frame."""
    missing_columns = sorted(REQUIRED_COLUMNS - set(frame.columns))
    missing_values = {
        column: int(frame[column].isna().sum())
        for column in REQUIRED_COLUMNS & set(frame.columns)
        if frame[column].isna().any()
    }
    duplicate_rows = (
        int(frame.duplicated(subset=DUPLICATE_KEY_COLUMNS).sum())
        if set(DUPLICATE_KEY_COLUMNS) <= set(frame.columns)
        else 0
    )
    dates = pd.to_datetime(frame.get("game_date", pd.Series(dtype="object")), errors="coerce")
    invalid_dates = int(dates.isna().sum()) if len(frame) else 0
    invalid_identifiers = {
        column: int(pd.to_numeric(frame[column], errors="coerce").isna().sum())
        for column in IDENTIFIER_COLUMNS
        if column in frame
    }
    sides = frame.get("inning_topbot", pd.Series(dtype="object"))
    normalized_sides = sides.astype("string").str.lower().replace({"bot": "bottom"})
    invalid_sides = int((~normalized_sides.isin(VALID_SIDES) & sides.notna()).sum())
    invalid_handedness = {
        column: int((~frame[column].isin(VALID_HANDS) & frame[column].notna()).sum())
        for column in ("stand", "p_throws")
        if column in frame
    }
    innings = pd.to_numeric(frame.get("inning", pd.Series(dtype="object")), errors="coerce")
    invalid_innings = int(((innings < 1) | (innings > 30)).sum())
    pitch_numbers = pd.to_numeric(
        frame.get("pitch_number", pd.Series(dtype="object")), errors="coerce"
    )
    invalid_pitch_numbers = int(((pitch_numbers < 1) | (pitch_numbers > 200)).sum())
    pitch_types = frame.get("pitch_type", pd.Series(dtype="object"))
    invalid_pitch_types = int((pitch_types.notna() & (pitch_types.astype(str).str.len() != 2)).sum())
    return ValidationReport(
        rows=len(frame),
        columns_missing=missing_columns,
        missing_values=missing_values,
        duplicate_rows=duplicate_rows,
        invalid_dates=invalid_dates,
        invalid_identifiers=invalid_identifiers,
        invalid_sides=invalid_sides,
        invalid_handedness=invalid_handedness,
        invalid_innings=invalid_innings,
        invalid_pitch_numbers=invalid_pitch_numbers,
        invalid_pitch_types=invalid_pitch_types,
    )


def _default_fetcher(start: date, end: date) -> pd.DataFrame:
    from pybaseball import statcast

    return statcast(start.isoformat(), end.isoformat())


class StatcastIngestor:
    """Download immutable weekly Statcast Parquet chunks with resume support."""

    def __init__(
        self,
        raw_dir: Path,
        *,
        retries: int = 3,
        retry_delay_seconds: float = 5.0,
        fetcher: Callable[[date, date], pd.DataFrame] = _default_fetcher,
    ) -> None:
        self.raw_dir = Path(raw_dir)
        self.retries = retries
        self.retry_delay_seconds = retry_delay_seconds
        self.fetcher = fetcher
        self.manifest_path = self.raw_dir / "statcast_manifest.json"

    def _load_manifest(self) -> dict[str, object]:
        if not self.manifest_path.exists():
            return {"source": "pybaseball.statcast", "chunks": {}}
        return json.loads(self.manifest_path.read_text())

    def _save_manifest(self, manifest: dict[str, object]) -> None:
        temporary = self.manifest_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True))
        temporary.replace(self.manifest_path)

    def download(self, start: date, end: date, chunk_days: int = 7) -> list[Path]:
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        manifest = self._load_manifest()
        chunks = manifest.setdefault("chunks", {})
        downloaded_paths = []
        for chunk in make_date_chunks(start, end, chunk_days):
            filename = f"statcast_{chunk.key}.parquet"
            path = self.raw_dir / filename
            record = chunks.get(chunk.key, {})
            if path.exists() and record.get("status") == "complete":
                LOGGER.info("Skipping cached chunk %s", chunk.key)
                cached_report = validate_statcast_frame(pd.read_parquet(path))
                record["validation"] = asdict(cached_report)
                record["rows"] = cached_report.rows
                chunks[chunk.key] = record
                self._save_manifest(manifest)
                downloaded_paths.append(path)
                continue
            if path.exists():
                raise FileExistsError(
                    f"Raw chunk exists without a completed manifest record: {path}"
                )
            last_error: Exception | None = None
            for attempt in range(1, self.retries + 1):
                try:
                    LOGGER.info("Downloading Statcast chunk %s (attempt %d/%d)", chunk.key, attempt, self.retries)
                    frame = self.fetcher(chunk.start, chunk.end)
                    if not isinstance(frame, pd.DataFrame):
                        raise TypeError("Statcast fetcher must return a pandas DataFrame")
                    temporary = path.with_suffix(".parquet.tmp")
                    frame.to_parquet(temporary, index=False)
                    temporary.replace(path)
                    report = validate_statcast_frame(frame)
                    chunks[chunk.key] = {
                        "status": "complete",
                        "filename": filename,
                        "start": chunk.start.isoformat(),
                        "end": chunk.end.isoformat(),
                        "rows": len(frame),
                        "validation": asdict(report),
                        "ingested_at_utc": datetime.now(timezone.utc).isoformat(),
                    }
                    self._save_manifest(manifest)
                    downloaded_paths.append(path)
                    break
                except Exception as error:  # noqa: BLE001 - retry boundary
                    last_error = error
                    LOGGER.warning("Chunk %s failed: %s", chunk.key, error)
                    if attempt < self.retries:
                        time.sleep(self.retry_delay_seconds)
            else:
                chunks[chunk.key] = {"status": "failed", "error": str(last_error)}
                self._save_manifest(manifest)
                raise RuntimeError(f"Unable to download Statcast chunk {chunk.key}") from last_error
        return downloaded_paths


def load_raw_chunks(paths: list[Path]) -> pd.DataFrame:
    """Read raw chunks for processing without changing their on-disk contents."""
    if not paths:
        return pd.DataFrame()
    return pd.concat((pd.read_parquet(path) for path in paths), ignore_index=True)


def _event_outs(event: object) -> int:
    return OUT_EVENTS.get(str(event), 0)


def _outs_recorded(group: pd.DataFrame) -> int:
    """Count event outs and reconcile completed innings with missing runner plays."""
    events = group.get("events", pd.Series(index=group.index))
    explicit_outs = int(events.map(_event_outs).fillna(0).sum())
    if "inning" not in group or "outs_when_up" not in group:
        return explicit_outs
    innings = sorted(group["inning"].dropna().unique())
    inferred_outs = 0
    for inning in innings[:-1]:
        inning_rows = group[group["inning"] == inning]
        inning_outs = int(inning_rows["events"].map(_event_outs).fillna(0).sum())
        if inning_outs < 3 and inning_rows["outs_when_up"].min() == 0:
            inferred_outs += 3 - inning_outs
    return explicit_outs + inferred_outs


def _batters_faced(group: pd.DataFrame) -> int:
    """Count completed at-bats, excluding pitch rows marked truncated."""
    events = group.get("events", pd.Series(index=group.index))
    truncated_at_bats = set(group.loc[events.eq("truncated_pa"), "at_bat_number"])
    return int(group.loc[~group["at_bat_number"].isin(truncated_at_bats), "at_bat_number"].nunique())


def build_starting_pitcher_games(frame: pd.DataFrame) -> pd.DataFrame:
    """Build one row per actual first pitcher for each game/team side.

    The first pitcher is determined from the first chronological pitch for each
    game and inning side, not from a roster assumption. Metrics are calculated
    only from rows attributed to that pitcher.
    """
    required = {"game_pk", "game_date", "pitcher", "inning_topbot", "at_bat_number"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Cannot build starter table; missing columns: {sorted(missing)}")
    if frame.empty:
        return pd.DataFrame()
    working = frame.copy()
    working["_row_order"] = range(len(working))
    sort_columns = ["game_pk", "inning_topbot", "at_bat_number", "pitch_number", "_row_order"]
    working = working.sort_values([column for column in sort_columns if column in working.columns])
    starter_keys = (
        working.groupby(["game_pk", "inning_topbot"], dropna=False, sort=False)["pitcher"]
        .first()
        .rename("starter_pitcher")
        .reset_index()
    )
    working = working.merge(starter_keys, on=["game_pk", "inning_topbot"], how="left")
    working["_side"] = working["inning_topbot"].astype("string").str.lower().replace({"bot": "bottom"})
    starter_rows = working[working["pitcher"] == working["starter_pitcher"]].copy()
    if starter_rows.empty:
        return pd.DataFrame()
    starter_rows["_outs"] = starter_rows.get("events", pd.Series(index=starter_rows.index)).map(_event_outs).fillna(0)
    starter_rows["_strikeout"] = starter_rows.get("events", pd.Series(index=starter_rows.index)).isin(
        {"strikeout", "strikeout_double_play"}
    )
    records = []
    for (game_id, side, pitcher_id), group in starter_rows.groupby(
        ["game_pk", "inning_topbot", "pitcher"], dropna=False, sort=False
    ):
        first = group.iloc[0]
        home_team = first.get("home_team")
        away_team = first.get("away_team")
        side = str(side).lower().replace("bot", "bottom")
        # Top is the away team's batting half, so the pitcher is home; bottom
        # is the home team's batting half, so the pitcher is away.
        team = home_team if side == "top" else away_team
        opponent = away_team if side == "top" else home_team
        record = {
            "game_date": pd.to_datetime(first.get("game_date"), errors="coerce").date(),
            "game_id": game_id,
            "pitcher_mlbam_id": pitcher_id,
            "pitcher_name": first.get("player_name"),
            "team": team,
            "opponent": opponent,
            "home_away": "home" if side == "top" else "away",
            "pitcher_handedness": first.get("p_throws"),
            "pitches_thrown": int(
                (group["pitch_number"].notna() & group["description"].ne("automatic_ball")).sum()
            ) if "pitch_number" in group and "description" in group else len(group),
            "batters_faced": _batters_faced(group),
            "strikeouts": int(group["_strikeout"].sum()),
            "outs_recorded": _outs_recorded(group),
            "hits": int(group.get("events", pd.Series(index=group.index)).isin({"single", "double", "triple", "home_run"}).sum()),
            "walks": int(group.get("events", pd.Series(index=group.index)).isin({"walk", "intent_walk"}).sum()),
            "runs": _runs_from_score_columns(group, side),
            "source_rows": int(len(group)),
            "outs_method": "statcast_events_with_completed_inning_reconciliation",
        }
        records.append(record)
    result = pd.DataFrame(records)
    if not result.empty:
        result = result.sort_values(["game_date", "game_id", "home_away"]).reset_index(drop=True)
    return result


def _runs_from_score_columns(group: pd.DataFrame, side: str) -> int | None:
    score_column = "post_away_score" if side == "top" else "post_home_score"
    if score_column not in group:
        return None
    scores = pd.to_numeric(group[score_column], errors="coerce").dropna()
    if scores.empty:
        return None
    # Starters begin an appearance before any score in this dataset; retain the
    # result as a derived score delta rather than claiming official earned runs.
    return max(0, int(scores.max() - scores.iloc[0]))