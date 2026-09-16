from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


RAW_DIR = Path(
    "data/raw/season=2026"
)

PROCESSED_DIR = Path(
    "data/processed/live_2026"
)

MANIFEST_PATH = (
    RAW_DIR
    / "statcast_manifest.json"
)


def date_paths(
    target_date: str,
) -> tuple[Path, Path, str]:
    key = (
        f"{target_date}_"
        f"{target_date}"
    )

    raw_path = (
        RAW_DIR
        / (
            f"statcast_"
            f"{key}.parquet"
        )
    )

    processed_path = (
        PROCESSED_DIR
        / (
            f"starting_pitcher_games_"
            f"{key}.parquet"
        )
    )

    return (
        raw_path,
        processed_path,
        key,
    )


def parquet_rows(
    path: Path,
) -> int | None:
    if not path.exists():
        return None

    try:
        return len(
            pd.read_parquet(
                path
            )
        )

    except Exception as exc:
        raise RuntimeError(
            f"Could not read {path}: {exc}"
        ) from exc


def remove_zero_manifest_entry(
    key: str,
) -> None:
    if not MANIFEST_PATH.exists():
        return

    data = json.loads(
        MANIFEST_PATH.read_text()
    )

    chunks = data.get(
        "chunks",
        {},
    )

    entry = chunks.get(
        key
    )

    if entry is None:
        return

    rows = entry.get(
        "rows"
    )

    if rows != 0:
        return

    del chunks[key]

    MANIFEST_PATH.write_text(
        json.dumps(
            data,
            indent=2,
            sort_keys=True,
        )
    )


def clean_zero_artifacts(
    target_date: str,
) -> None:
    (
        raw_path,
        processed_path,
        key,
    ) = date_paths(
        target_date
    )

    raw_rows = parquet_rows(
        raw_path
    )

    processed_rows = parquet_rows(
        processed_path
    )

    if raw_rows == 0:
        raw_path.unlink()

    if processed_rows == 0:
        processed_path.unlink()

    remove_zero_manifest_entry(
        key
    )


def completed_date_available(
    target_date: str,
) -> bool:
    (
        raw_path,
        processed_path,
        _,
    ) = date_paths(
        target_date
    )

    raw_rows = parquet_rows(
        raw_path
    )

    processed_rows = parquet_rows(
        processed_path
    )

    return bool(
        raw_rows
        and raw_rows > 0
        and processed_rows
        and processed_rows > 0
    )


def run_ingestion(
    target_date: str,
) -> None:
    command = [
        sys.executable,
        "-u",
        "-m",
        "mlb_quant.data.cli",
        "--start",
        target_date,
        "--end",
        target_date,
        "--chunk-days",
        "1",
        "--raw-dir",
        str(
            RAW_DIR
        ),
        "--processed-dir",
        str(
            PROCESSED_DIR
        ),
    ]

    print()
    print(
        "=" * 80
    )
    print(
        "STATCAST REFRESH"
    )
    print(
        "=" * 80
    )
    print()
    print(
        f"Attempting completed date: "
        f"{target_date}"
    )
    print()

    subprocess.run(
        command,
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--date",
        required=True,
        help=(
            "Prediction/slate date "
            "YYYY-MM-DD."
        ),
    )

    args = parser.parse_args()

    prediction_date = pd.Timestamp(
        args.date
    ).normalize()

    target_date = (
        prediction_date
        - pd.Timedelta(
            days=1
        )
    )

    target_text = (
        target_date
        .date()
        .isoformat()
    )

    # ---------------------------------------------
    # ALREADY AVAILABLE
    # ---------------------------------------------

    if completed_date_available(
        target_text
    ):
        print()
        print(
            "STATCAST DATA ALREADY AVAILABLE"
        )
        print(
            f"Completed date: {target_text}"
        )

        return

    # ---------------------------------------------
    # CLEAR ANY PREVIOUS ZERO-ROW CACHE
    # ---------------------------------------------

    clean_zero_artifacts(
        target_text
    )

    # ---------------------------------------------
    # TRY FRESH INGESTION
    # ---------------------------------------------

    run_ingestion(
        target_text
    )

    # ---------------------------------------------
    # VALIDATE RESULT
    # ---------------------------------------------

    if completed_date_available(
        target_text
    ):
        (
            raw_path,
            processed_path,
            _,
        ) = date_paths(
            target_text
        )

        raw_rows = parquet_rows(
            raw_path
        )

        starter_rows = parquet_rows(
            processed_path
        )

        print()
        print(
            "=" * 80
        )
        print(
            "STATCAST REFRESH COMPLETE"
        )
        print(
            "=" * 80
        )
        print()
        print(
            f"Completed date: "
            f"{target_text}"
        )
        print(
            f"Raw pitches: "
            f"{raw_rows}"
        )
        print(
            f"Starter appearances: "
            f"{starter_rows}"
        )

        return

    # ---------------------------------------------
    # SOURCE NOT READY
    #
    # Clean the bogus zero-row cache again so it
    # cannot later masquerade as completed data.
    # ---------------------------------------------

    clean_zero_artifacts(
        target_text
    )

    print()
    print(
        "=" * 80
    )
    print(
        "STATCAST DATE NOT AVAILABLE YET"
    )
    print(
        "=" * 80
    )
    print()
    print(
        f"{target_text} returned no usable data."
    )
    print(
        "Zero-row artifacts were removed."
    )
    print(
        "The live model can continue using the "
        "latest earlier valid completed date."
    )


if __name__ == "__main__":
    main()