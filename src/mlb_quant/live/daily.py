from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd


LIVE_DIR = Path(
    "data/processed/live_2026"
)

REPORTS_DIR = Path(
    "reports"
)


# =========================================================
# COMMAND RUNNER
# =========================================================


def run_module(
    module: str,
    args: list[str],
) -> None:
    command = [
        sys.executable,
        "-u",
        "-m",
        module,
        *args,
    ]

    print()
    print("=" * 80)
    print(
        "RUNNING:",
        " ".join(command),
    )
    print("=" * 80)
    print()

    subprocess.run(
        command,
        check=True,
    )


# =========================================================
# COMPLETED-DATA CUTOFF
# =========================================================


def latest_completed_date(
    prediction_date: pd.Timestamp,
) -> pd.Timestamp:
    """
    Determine the newest valid completed MLB date available
    in processed 2026 starter data.

    Empty files are ignored.
    Prediction-date/future rows are excluded.
    """

    latest = None

    for path in sorted(
        LIVE_DIR.glob(
            "starting_pitcher_games_*.parquet"
        )
    ):
        try:
            frame = pd.read_parquet(
                path,
                columns=[
                    "game_date",
                ],
            )

        except Exception:
            continue

        if frame.empty:
            continue

        dates = pd.to_datetime(
            frame["game_date"]
        ).dt.normalize()

        dates = dates.loc[
            dates < prediction_date
        ]

        if dates.empty:
            continue

        candidate = dates.max()

        if (
            latest is None
            or candidate > latest
        ):
            latest = candidate

    if latest is None:
        raise RuntimeError(
            "Could not determine the latest "
            "completed-data date."
        )

    return latest


# =========================================================
# MARKET FILE
# =========================================================


def market_file_has_entered_lines(
    market_path: Path,
) -> bool:
    """
    Return True when at least one sportsbook market row
    contains line + over odds + under odds.

    This prevents daily preparation from overwriting
    sportsbook prices that have already been entered.
    """

    if not market_path.exists():
        return False

    try:
        frame = pd.read_csv(
            market_path
        )

    except Exception as exc:
        raise RuntimeError(
            f"Could not safely inspect existing "
            f"market file {market_path}: {exc}"
        ) from exc

    required_columns = {
        "line",
        "over_odds",
        "under_odds",
    }

    if not required_columns.issubset(
        frame.columns
    ):
        return False

    if frame.empty:
        return False

    completed_rows = (
        frame[
            [
                "line",
                "over_odds",
                "under_odds",
            ]
        ]
        .notna()
        .all(
            axis=1
        )
    )

    return bool(
        completed_rows.any()
    )


# =========================================================
# ANALYZE SPORTSBOOK BOARD
# =========================================================


def analyze_slate(
    date: str,
    market_file: str | None = None,
) -> None:
    prediction_date = pd.Timestamp(
        date
    ).normalize()

    if market_file is None:
        market_path = (
            REPORTS_DIR
            / f"market_input_{date}.csv"
        )

    else:
        market_path = Path(
            market_file
        )

    if not market_path.exists():
        raise FileNotFoundError(
            f"Market file not found: "
            f"{market_path}"
        )

    if not market_file_has_entered_lines(
        market_path
    ):
        raise RuntimeError(
            "Market file exists, but no complete "
            "sportsbook line/price rows were found."
        )

    latest = latest_completed_date(
        prediction_date
    )

    as_of = (
        latest.date().isoformat()
    )

    data_age_days = int(
        (
            prediction_date
            - latest
        ).days
    )

    print()
    print("=" * 80)
    print(
        f"ANALYZING MLB SLATE: {date}"
    )
    print("=" * 80)
    print()

    print(
        f"Latest completed data: {as_of}"
    )

    print(
        f"Data age: {data_age_days} day(s)"
    )

    print(
        f"Market file: {market_path}"
    )

    # -----------------------------------------------------
    # MARKET PRICING
    # -----------------------------------------------------

    run_module(
        "mlb_quant.live.market",
        [
            "--date",
            date,
            "--market-file",
            str(
                market_path
            ),
        ],
    )

    # -----------------------------------------------------
    # ROBUST DECISION LAYER
    # -----------------------------------------------------

    run_module(
        "mlb_quant.live.decision",
        [
            "--date",
            date,
            "--as-of",
            as_of,
        ],
    )

    print()
    print("=" * 80)
    print(
        "LIVE ANALYSIS COMPLETE"
    )
    print("=" * 80)
    print()

    print(
        "Market analysis:"
    )

    print(
        REPORTS_DIR
        / (
            f"live_market_analysis_"
            f"{date}.csv"
        )
    )

    print()

    print(
        "Decision analysis:"
    )

    print(
        REPORTS_DIR
        / (
            f"live_decisions_"
            f"{date}.csv"
        )
    )


# =========================================================
# PREPARE DAILY SLATE
# =========================================================


def prepare_slate(
    date: str,
) -> None:
    prediction_date = pd.Timestamp(
        date
    ).normalize()

    print()
    print("=" * 80)
    print(
        f"PREPARING MLB SLATE: {date}"
    )
    print("=" * 80)

    # -----------------------------------------------------
    # 1. REFRESH YESTERDAY'S STATCAST DATA
    # -----------------------------------------------------

    run_module(
        "mlb_quant.live.refresh",
        [
            "--date",
            date,
        ],
    )

    # -----------------------------------------------------
    # 2. FETCH CURRENT PROBABLE STARTERS
    # -----------------------------------------------------

    run_module(
        "mlb_quant.live.schedule",
        [
            "--date",
            date,
        ],
    )

    # -----------------------------------------------------
    # 3. BUILD POINT-IN-TIME FEATURES
    # -----------------------------------------------------

    run_module(
        "mlb_quant.live.features",
        [
            "--date",
            date,
        ],
    )

    # -----------------------------------------------------
    # 4. RUN PRODUCTION MODELS + DISTRIBUTIONS
    # -----------------------------------------------------

    run_module(
        "mlb_quant.live.predict",
        [
            "--date",
            date,
        ],
    )

    # -----------------------------------------------------
    # 5. DETERMINE CURRENT DATA CUTOFF
    # -----------------------------------------------------

    latest = latest_completed_date(
        prediction_date
    )

    data_age_days = int(
        (
            prediction_date
            - latest
        ).days
    )

    print()
    print(
        f"Latest valid completed data: "
        f"{latest.date()}"
    )

    print(
        f"Data age: "
        f"{data_age_days} day(s)"
    )

    # -----------------------------------------------------
    # 6. CREATE / PROTECT MARKET INPUT
    # -----------------------------------------------------

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    market_path = (
        REPORTS_DIR
        / f"market_input_{date}.csv"
    )

    has_lines = (
        market_file_has_entered_lines(
            market_path
        )
    )

    if has_lines:
        print()
        print("=" * 80)
        print(
            "EXISTING SPORTSBOOK LINES DETECTED"
        )
        print("=" * 80)
        print()

        print(
            "The existing market input file "
            "will NOT be overwritten:"
        )

        print()
        print(
            market_path
        )

    else:
        run_module(
            "mlb_quant.live.market_template",
            [
                "--date",
                date,
            ],
        )

        # Recheck after template creation.
        has_lines = (
            market_file_has_entered_lines(
                market_path
            )
        )

    # -----------------------------------------------------
    # 7. PREPARATION COMPLETE
    # -----------------------------------------------------

    print()
    print("=" * 80)
    print(
        "SLATE PREPARATION COMPLETE"
    )
    print("=" * 80)
    print()

    print(
        "Market input:"
    )

    print(
        market_path
    )

    # -----------------------------------------------------
    # 8. AUTO-ANALYZE IF ODDS EXIST
    # -----------------------------------------------------

    if market_file_has_entered_lines(
        market_path
    ):
        print()
        print(
            "Sportsbook lines are available."
        )

        print(
            "Running market pricing + "
            "decision analysis automatically..."
        )

        analyze_slate(
            date=date,
            market_file=str(
                market_path
            ),
        )

    else:
        print()
        print("=" * 80)
        print(
            "WAITING FOR SPORTSBOOK LINES"
        )
        print("=" * 80)
        print()

        print(
            "No completed sportsbook prices "
            "are available yet."
        )

        print()
        print(
            "The slate, features, projections, "
            "and probability distributions are ready."
        )

        print()
        print(
            "Once sportsbook lines are entered, "
            "rerun the SAME command:"
        )

        print()

        print(
            f"{sys.executable} -u "
            f"-m mlb_quant.live.daily "
            f"--date {date}"
        )


# =========================================================
# CLI
# =========================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Daily MLB pitcher-prop production pipeline."
        )
    )

    parser.add_argument(
        "--date",
        required=True,
        help=(
            "Slate date in YYYY-MM-DD format."
        ),
    )

    parser.add_argument(
        "--analyze",
        action="store_true",
        help=(
            "Skip slate preparation and run "
            "market + decision analysis directly."
        ),
    )

    parser.add_argument(
        "--market-file",
        default=None,
        help=(
            "Optional sportsbook market CSV. "
            "Defaults to "
            "reports/market_input_DATE.csv."
        ),
    )

    args = parser.parse_args()

    # -----------------------------------------------------
    # DATE VALIDATION
    # -----------------------------------------------------

    try:
        parsed_date = pd.Timestamp(
            args.date
        ).normalize()

    except Exception as exc:
        raise ValueError(
            "--date must use YYYY-MM-DD format."
        ) from exc

    if (
        parsed_date.strftime(
            "%Y-%m-%d"
        )
        != args.date
    ):
        raise ValueError(
            "--date must use YYYY-MM-DD format."
        )

    # -----------------------------------------------------
    # MODE
    # -----------------------------------------------------

    if args.analyze:
        analyze_slate(
            date=args.date,
            market_file=args.market_file,
        )

    else:
        prepare_slate(
            date=args.date,
        )


if __name__ == "__main__":
    main()