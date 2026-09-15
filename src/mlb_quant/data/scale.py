"""Season-by-season historical Statcast scaling orchestration."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

import pandas as pd

from .statcast import (
    REQUIRED_COLUMNS,
    StatcastIngestor,
    build_starting_pitcher_games,
    load_raw_chunks,
    validate_statcast_frame,
)

LOGGER = logging.getLogger(__name__)

REGULAR_SEASON_RANGES = {
    2021: (date(2021, 4, 1), date(2021, 10, 3)),
    2022: (date(2022, 4, 7), date(2022, 10, 5)),
    2023: (date(2023, 3, 30), date(2023, 10, 1)),
    2024: (date(2024, 3, 28), date(2024, 9, 29)),
    2025: (date(2025, 3, 27), date(2025, 9, 28)),
}

CRITICAL_COLUMNS = {
    "game_date",
    "game_pk",
    "pitcher",
    "batter",
    "inning",
    "inning_topbot",
    "at_bat_number",
    "pitch_number",
}
CRITICAL_PROBLEM_FIELDS = {
    "columns_missing",
    "duplicate_rows",
    "invalid_dates",
    "invalid_identifiers",
    "invalid_sides",
    "invalid_innings",
    "invalid_pitch_numbers",
    "invalid_pitch_types",
}


@dataclass
class SeasonReport:
    season: int
    start_date: str
    end_date: str
    raw_pitch_rows: int
    games: int
    starting_pitcher_appearances: int
    unique_starting_pitchers: int
    missing_critical_fields: dict[str, int]
    missing_noncritical_fields: dict[str, int]
    duplicate_records: int
    failed_validation_checks: list[str]
    normal_completed_games: int
    unusual_or_incomplete_games: list[int]
    processed_output: str

    @property
    def critical_failure(self) -> bool:
        return bool(self.failed_validation_checks)


def _missing_values(frame: pd.DataFrame, columns: set[str]) -> dict[str, int]:
    return {
        column: int(frame[column].isna().sum())
        for column in sorted(columns & set(frame.columns))
        if frame[column].isna().any()
    }


def _failed_checks(validation, missing_critical: dict[str, int]) -> list[str]:
    failures = []
    if validation.columns_missing:
        failures.append("missing_required_columns")
    if validation.duplicate_rows:
        failures.append("duplicate_pitch_records")
    if validation.invalid_dates:
        failures.append("invalid_dates")
    if any(validation.invalid_identifiers.values()):
        failures.append("invalid_identifiers")
    if validation.invalid_sides:
        failures.append("invalid_inning_sides")
    if validation.invalid_innings:
        failures.append("invalid_innings")
    if validation.invalid_pitch_numbers:
        failures.append("invalid_pitch_numbers")
    if validation.invalid_pitch_types:
        failures.append("invalid_pitch_types")
    if missing_critical:
        failures.append("missing_critical_values")
    return failures


def process_season(
    season: int,
    *,
    raw_root: Path = Path("data/raw"),
    processed_root: Path = Path("data/processed"),
    chunk_days: int = 7,
    retries: int = 3,
) -> SeasonReport:
    """Download, validate, and process one regular season.

    A season's raw files are isolated under ``data/raw/season=YYYY``. A critical
    validation failure raises before the caller can continue to the next season.
    """
    if season not in REGULAR_SEASON_RANGES:
        raise ValueError(f"Only regular seasons 2021-2025 are supported, got {season}")
    start, end = REGULAR_SEASON_RANGES[season]
    season_raw_dir = Path(raw_root) / f"season={season}"
    season_processed_dir = Path(processed_root)
    ingestor = StatcastIngestor(season_raw_dir, retries=retries)
    paths = ingestor.download(start, end, chunk_days)
    raw = load_raw_chunks(paths)
    validation = validate_statcast_frame(raw)
    missing_critical = _missing_values(raw, CRITICAL_COLUMNS)
    missing_noncritical = _missing_values(raw, set(REQUIRED_COLUMNS) - CRITICAL_COLUMNS)
    failed_checks = _failed_checks(validation, missing_critical)
    if failed_checks:
        raise RuntimeError(
            f"Season {season} failed critical validation: {', '.join(failed_checks)}"
        )

    starters = build_starting_pitcher_games(raw)
    starters.insert(0, "season", season)
    game_counts = starters.groupby("game_id").size()
    unusual_games = sorted(int(game_id) for game_id, count in game_counts.items() if count != 2)
    output_path = season_processed_dir / f"starting_pitcher_games_{season}.parquet"
    season_processed_dir.mkdir(parents=True, exist_ok=True)
    starters.to_parquet(output_path, index=False)
    report = SeasonReport(
        season=season,
        start_date=start.isoformat(),
        end_date=end.isoformat(),
        raw_pitch_rows=len(raw),
        games=int(raw["game_pk"].nunique()) if "game_pk" in raw else 0,
        starting_pitcher_appearances=len(starters),
        unique_starting_pitchers=int(starters["pitcher_mlbam_id"].nunique()) if not starters.empty else 0,
        missing_critical_fields=missing_critical,
        missing_noncritical_fields=missing_noncritical,
        duplicate_records=validation.duplicate_rows,
        failed_validation_checks=failed_checks,
        normal_completed_games=int((game_counts == 2).sum()),
        unusual_or_incomplete_games=unusual_games,
        processed_output=str(output_path),
    )
    LOGGER.info("Season %s complete: %s", season, json.dumps(asdict(report), default=str))
    return report


def run_historical_scale(
    *,
    seasons: tuple[int, ...] = tuple(REGULAR_SEASON_RANGES),
    raw_root: Path = Path("data/raw"),
    processed_root: Path = Path("data/processed"),
    chunk_days: int = 7,
    retries: int = 3,
) -> list[SeasonReport]:
    """Process seasons in order and stop immediately on a critical failure."""
    reports = []
    for season in seasons:
        report = process_season(
            season,
            raw_root=raw_root,
            processed_root=processed_root,
            chunk_days=chunk_days,
            retries=retries,
        )
        reports.append(report)
        if report.unusual_or_incomplete_games:
            LOGGER.warning(
                "Season %s has unusual/incomplete games: %s",
                season,
                report.unusual_or_incomplete_games,
            )
    season_tables = [pd.read_parquet(report.processed_output) for report in reports]
    consolidated = pd.concat(season_tables, ignore_index=True) if season_tables else pd.DataFrame()
    output_path = Path(processed_root) / "starting_pitcher_games_2021_2025.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    consolidated.to_parquet(output_path, index=False)
    summary_path = Path(processed_root) / "starting_pitcher_scale_report_2021_2025.json"
    summary_path.write_text(json.dumps([asdict(report) for report in reports], indent=2))
    LOGGER.info("Consolidated %d appearances to %s", len(consolidated), output_path)
    return reports


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Scale regular-season Statcast data for 2021-2025.")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--chunk-days", type=int, default=7)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s %(levelname)s %(message)s")
    reports = run_historical_scale(
        raw_root=args.raw_dir,
        processed_root=args.processed_dir,
        chunk_days=args.chunk_days,
        retries=args.retries,
    )
    print(json.dumps([asdict(report) for report in reports], indent=2))


if __name__ == "__main__":
    main()
