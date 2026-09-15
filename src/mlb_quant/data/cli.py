"""Command-line entry point for bounded Statcast ingestion."""

from __future__ import annotations

import argparse
import json
import logging
from datetime import date
from pathlib import Path

from .statcast import StatcastIngestor, build_starting_pitcher_games, load_raw_chunks, validate_statcast_frame


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download and process bounded Statcast data.")
    parser.add_argument("--start", required=True, type=date.fromisoformat, help="Inclusive YYYY-MM-DD start date")
    parser.add_argument("--end", required=True, type=date.fromisoformat, help="Inclusive YYYY-MM-DD end date")
    parser.add_argument("--chunk-days", type=int, default=7)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s %(levelname)s %(message)s")
    ingestor = StatcastIngestor(args.raw_dir, retries=args.retries)
    paths = ingestor.download(args.start, args.end, args.chunk_days)
    raw = load_raw_chunks(paths)
    report = validate_statcast_frame(raw)
    args.processed_dir.mkdir(parents=True, exist_ok=True)
    starters = build_starting_pitcher_games(raw)
    output_path = args.processed_dir / f"starting_pitcher_games_{args.start}_{args.end}.parquet"
    starters.to_parquet(output_path, index=False)
    summary = {
        "date_range": {"start": args.start.isoformat(), "end": args.end.isoformat()},
        "raw_pitch_rows": len(raw),
        "games": int(raw["game_pk"].nunique()) if "game_pk" in raw else 0,
        "starting_pitcher_appearances": len(starters),
        "validation_problems": report.problems,
        "processed_output": str(output_path),
        "sample_starting_pitcher_rows": starters.head(5).to_dict(orient="records"),
    }
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()