"""Final calibration audit for MLB Quant v0.1."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PREDICTION_PATH = Path(
    "data/processed/distribution_predictions_v0_1.parquet"
)

REPORT_PATH = Path(
    "reports/final_calibration_v0_1.md"
)


SPORTSBOOK_LINES = {
    "strikeouts": [
        3.5,
        4.5,
        5.5,
        6.5,
        7.5,
    ],
    "outs_recorded": [
        14.5,
        15.5,
        16.5,
        17.5,
        18.5,
    ],
}


def validate_input(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    required = {
        "target",
        "line",
        "probability_over",
        "actual_over",
    }

    missing = required - set(
        frame.columns
    )

    if missing:
        raise ValueError(
            "Distribution prediction file "
            "is missing required columns: "
            f"{sorted(missing)}"
        )

    data = frame.copy()

    if "season" in data.columns:
        data = data[
            pd.to_numeric(
                data["season"],
                errors="coerce",
            ).eq(2025)
        ].copy()

    if data.empty:
        raise ValueError(
            "No 2025 holdout rows found."
        )

    return data


def sportsbook_line_table(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for target in (
        "strikeouts",
        "outs_recorded",
    ):
        target_frame = frame[
            frame["target"].eq(target)
        ]

        for line in SPORTSBOOK_LINES[
            target
        ]:
            group = target_frame[
                np.isclose(
                    pd.to_numeric(
                        target_frame["line"],
                        errors="coerce",
                    ),
                    float(line),
                )
            ]

            if group.empty:
                continue

            predicted = float(
                pd.to_numeric(
                    group[
                        "probability_over"
                    ],
                    errors="coerce",
                ).mean()
            )

            observed = float(
                pd.to_numeric(
                    group[
                        "actual_over"
                    ],
                    errors="coerce",
                ).mean()
            )

            rows.append(
                {
                    "target": target,
                    "line": float(line),
                    "n": int(len(group)),
                    "predicted": predicted,
                    "observed": observed,
                    "calibration_error": (
                        predicted
                        - observed
                    ),
                }
            )

    return pd.DataFrame(rows)


def betting_probability_table(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Calibration of raw Over probabilities in betting ranges."""

    data = frame[
        frame["target"].isin(
            [
                "strikeouts",
                "outs_recorded",
            ]
        )
    ].copy()

    data[
        "betting_probability"
    ] = pd.to_numeric(
        data["probability_over"],
        errors="coerce",
    )

    data[
        "betting_observed"
    ] = pd.to_numeric(
        data["actual_over"],
        errors="coerce",
    )

    data = data.dropna(
        subset=[
            "betting_probability",
            "betting_observed",
        ]
    )

    data["bucket"] = pd.cut(
        data[
            "betting_probability"
        ],
        bins=[
            0.55,
            0.60,
            0.65,
            0.70,
            1.0000001,
        ],
        labels=[
            "55-60%",
            "60-65%",
            "65-70%",
            "70%+",
        ],
        right=False,
    )

    data = data[
        data["bucket"].notna()
    ].copy()

    rows = []

    for target in (
        "outs_recorded",
        "strikeouts",
    ):
        target_frame = data[
            data["target"].eq(target)
        ]

        for bucket in (
            "55-60%",
            "60-65%",
            "65-70%",
            "70%+",
        ):
            group = target_frame[
                target_frame[
                    "bucket"
                ].astype(str).eq(
                    bucket
                )
            ]

            if group.empty:
                continue

            predicted = float(
                group[
                    "betting_probability"
                ].mean()
            )

            observed = float(
                group[
                    "betting_observed"
                ].mean()
            )

            rows.append(
                {
                    "target": target,
                    "bucket": bucket,
                    "n": int(len(group)),
                    "predicted": predicted,
                    "observed": observed,
                    "calibration_error": (
                        predicted
                        - observed
                    ),
                }
            )

    return pd.DataFrame(rows)


def main() -> None:
    if not PREDICTION_PATH.exists():
        raise FileNotFoundError(
            "Distribution predictions "
            f"not found: {PREDICTION_PATH}"
        )

    frame = pd.read_parquet(
        PREDICTION_PATH
    )

    frame = validate_input(
        frame
    )

    line_table = sportsbook_line_table(
        frame
    )

    bucket_table = (
        betting_probability_table(
            frame
        )
    )

    if line_table.empty:
        raise ValueError(
            "No sportsbook calibration "
            "rows were produced."
        )

    if bucket_table.empty:
        raise ValueError(
            "No probability-bucket "
            "rows were produced."
        )

    report = [
        "# MLB Final Calibration Audit v0.1",
        "",
        "Untouched 2025 holdout only.",
        "",
        "## Sportsbook-relevant lines",
        "",
        line_table.to_markdown(
            index=False,
            floatfmt=".4f",
        ),
        "",
        "## Betting probability buckets",
        "",
        bucket_table.to_markdown(
            index=False,
            floatfmt=".4f",
        ),
        "",
    ]

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_PATH.write_text(
        "\n".join(report),
        encoding="utf-8",
    )

    print(
        f"Report: {REPORT_PATH}"
    )


if __name__ == "__main__":
    main()
