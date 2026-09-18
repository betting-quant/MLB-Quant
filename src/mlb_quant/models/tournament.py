"""Chronological tournament for MLB pitcher count targets."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import PoissonRegressor, Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_poisson_deviance,
    mean_squared_error,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


TARGETS = (
    "strikeouts",
    "outs_recorded",
    "batters_faced",
    "pitches_thrown",
)

IDENTIFIER_COLUMNS = {
    "season",
    "game_date",
    "game_id",
    "pitcher_mlbam_id",
    "pitcher_name",
    "team",
    "opponent",
    "home_away",
    "pitcher_handedness",
    "feature_row_id",
}

TARGET_COLUMNS = set(TARGETS)

VALIDATION_FOLDS = (
    (2021, 2022),
    (2021, 2022, 2023),
)

HOLDOUT_SEASON = 2025


@dataclass
class Split:
    train_seasons: tuple[int, ...]
    validation_season: int


@dataclass
class ModelResult:
    target: str
    model: str
    split: str
    mae: float
    rmse: float
    bias: float
    correlation: float
    poisson_deviance: float
    n: int


def chronological_splits(frame: pd.DataFrame) -> list[Split]:
    seasons = sorted(
        int(season)
        for season in frame["season"].unique()
    )

    return [
        Split(
            tuple(seasons[:index]),
            seasons[index],
        )
        for index in range(1, len(seasons))
        if seasons[index] <= 2024
    ]


WORKLOAD_ONLY_COLUMNS = {
    "season_pitches_per_start",
    "season_bf_per_start",
    "season_outs_per_start",
    "pitches_median_before_game",
    "pitcher_days_rest",
    "pitcher_starts_before_game",
    "pitches_previous_start",
    "starts_reaching_90_before_game",
    "starts_reaching_100_before_game",
    "workload_trend_last3_vs_prior3",
}

ROLLING_PREFIXES = (
    "last3_",
    "last5_",
    "last10_",
    "last30d_",
    "last60d_",
)


def _is_season_ability(column: str) -> bool:
    return (
        column.startswith("season_")
        and column not in WORKLOAD_ONLY_COLUMNS
    )


def model_feature_columns(
    frame: pd.DataFrame,
    family: str = "full",
) -> list[str]:
    """Return model input columns, cumulative by feature family.

    Families are additive per the ablation protocol:
    pitcher_history is a subset of workload,
    workload is a subset of opponent,
    opponent is a subset of recent,
    recent is a subset of full.

    Full additionally includes arsenal/pitch-mix features.
    """

    excluded = IDENTIFIER_COLUMNS | TARGET_COLUMNS

    numeric = [
        column
        for column in frame.select_dtypes(
            include=["number"]
        ).columns
        if column not in excluded
    ]

    if family == "full":
        return numeric

    ability = {
        column
        for column in numeric
        if _is_season_ability(column)
    }

    ability |= (
        {"month", "home_indicator"}
        & set(numeric)
    )

    workload_extra = {
        column
        for column in numeric
        if column in WORKLOAD_ONLY_COLUMNS
    }

    opponent_extra = {
        column
        for column in numeric
        if column.startswith("opponent_")
    }

    recent_extra = {
        column
        for column in numeric
        if column.startswith(ROLLING_PREFIXES)
    }

    cumulative = {
        "pitcher_history": ability,
        "workload": (
            ability
            | workload_extra
        ),
        "opponent": (
            ability
            | workload_extra
            | opponent_extra
        ),
        "recent": (
            ability
            | workload_extra
            | opponent_extra
            | recent_extra
        ),
    }

    if family not in cumulative:
        raise ValueError(
            f"Unknown feature family: {family}"
        )

    selected = cumulative[family]

    return [
        column
        for column in numeric
        if column in selected
    ]


def _fold_feature_columns(
    train: pd.DataFrame,
    columns: list[str],
) -> list[str]:
    """Drop columns with zero observed values in this training fold.

    Fitting an imputer on an all-NaN column both warns and
    silently contributes nothing.

    Dropping it using only the training fold keeps train and
    validation columns consistent without touching future data.
    """

    if not columns:
        return columns

    has_values = train[columns].notna().any()

    return [
        column
        for column in columns
        if has_values[column]
    ]


def make_estimator(
    model_name: str,
    random_state: int = 42,
):
    imputer = SimpleImputer(
        strategy="median",
        add_indicator=True,
    )

    if model_name == "ridge":
        return Pipeline(
            [
                (
                    "imputer",
                    imputer,
                ),
                (
                    "scale",
                    StandardScaler(),
                ),
                (
                    "model",
                    Ridge(alpha=10.0),
                ),
            ]
        )

    if model_name == "poisson":
        # Scaling is required for lbfgs to converge
        # reliably within a bounded number of iterations.
        return Pipeline(
            [
                (
                    "imputer",
                    imputer,
                ),
                (
                    "scale",
                    StandardScaler(),
                ),
                (
                    "model",
                    PoissonRegressor(
                        alpha=0.1,
                        max_iter=3000,
                    ),
                ),
            ]
        )

    if model_name == "hist_gradient_boosting":
        return Pipeline(
            [
                (
                    "imputer",
                    imputer,
                ),
                (
                    "model",
                    HistGradientBoostingRegressor(
                        max_iter=180,
                        learning_rate=0.05,
                        max_leaf_nodes=15,
                        l2_regularization=1.0,
                        random_state=random_state,
                    ),
                ),
            ]
        )

    raise ValueError(
        f"Unknown model: {model_name}"
    )


def _safe_correlation(
    actual: pd.Series,
    predicted: np.ndarray,
) -> float:
    if (
        actual.nunique() < 2
        or np.nanstd(predicted) == 0
    ):
        return 0.0

    return float(
        np.corrcoef(
            actual.to_numpy(),
            predicted,
        )[0, 1]
    )


def score_predictions(
    actual: pd.Series,
    predicted: np.ndarray,
) -> dict[str, float]:
    predicted = np.maximum(
        np.asarray(
            predicted,
            dtype=float,
        ),
        0.0,
    )

    actual_values = actual.to_numpy(
        dtype=float
    )

    error = (
        predicted
        - actual_values
    )

    # Poisson deviance requires predicted means
    # to be strictly positive.
    #
    # Observed zero counts remain valid and unchanged.
    safe_predicted = np.maximum(
        predicted,
        1e-9,
    )

    poisson_deviance = float(
        mean_poisson_deviance(
            actual_values,
            safe_predicted,
        )
    )

    return {
        "mae": float(
            mean_absolute_error(
                actual,
                predicted,
            )
        ),
        "rmse": float(
            np.sqrt(
                mean_squared_error(
                    actual,
                    predicted,
                )
            )
        ),
        "bias": float(
            error.mean()
        ),
        "correlation": _safe_correlation(
            actual,
            predicted,
        ),
        "poisson_deviance": poisson_deviance,
        "n": int(
            len(actual)
        ),
    }


def naive_predictions(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target: str,
) -> np.ndarray:
    feature_map = {
        "strikeouts": "season_k_per_start",
        "outs_recorded": "season_outs_per_start",
        "batters_faced": "season_bf_per_start",
        "pitches_thrown": "season_pitches_per_start",
    }

    feature = feature_map[target]

    fallback = float(
        train[target].mean()
    )

    return (
        test[feature]
        .fillna(fallback)
        .to_numpy(dtype=float)
    )


def fit_and_predict(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target: str,
    model_name: str,
    columns: list[str],
) -> tuple[object, np.ndarray]:
    estimator = make_estimator(
        model_name
    )

    estimator.fit(
        train[columns],
        train[target],
    )

    prediction = np.maximum(
        estimator.predict(
            test[columns]
        ),
        0.0,
    )

    return (
        estimator,
        prediction,
    )


def _fit_decomposed_strikeouts(
    train: pd.DataFrame,
    test: pd.DataFrame,
    columns: list[str],
) -> tuple[dict[str, object], np.ndarray]:
    bf_columns = columns

    k_rate = (
        train["strikeouts"]
        / train["batters_faced"].replace(
            0,
            np.nan,
        )
    )

    bf_model = make_estimator(
        "ridge"
    )

    rate_model = make_estimator(
        "ridge"
    )

    bf_model.fit(
        train[bf_columns],
        train["batters_faced"],
    )

    rate_model.fit(
        train[bf_columns],
        k_rate.fillna(
            k_rate.mean()
        ),
    )

    bf_prediction = np.maximum(
        bf_model.predict(
            test[bf_columns]
        ),
        0,
    )

    rate_prediction = np.clip(
        rate_model.predict(
            test[bf_columns]
        ),
        0,
        1,
    )

    prediction = (
        bf_prediction
        * rate_prediction
    )

    return (
        {
            "bf_model": bf_model,
            "rate_model": rate_model,
        },
        prediction,
    )


def _fit_opportunity_outs(
    train: pd.DataFrame,
    test: pd.DataFrame,
    columns: list[str],
) -> tuple[dict[str, object], np.ndarray]:
    bf_model = make_estimator(
        "ridge"
    )

    outs_rate_model = make_estimator(
        "ridge"
    )

    bf_model.fit(
        train[columns],
        train["batters_faced"],
    )

    outs_rate = (
        train["outs_recorded"]
        / train["batters_faced"].replace(
            0,
            np.nan,
        )
    )

    outs_rate_model.fit(
        train[columns],
        outs_rate.fillna(
            outs_rate.mean()
        ),
    )

    bf_prediction = np.maximum(
        bf_model.predict(
            test[columns]
        ),
        0,
    )

    rate_prediction = np.clip(
        outs_rate_model.predict(
            test[columns]
        ),
        0,
        3,
    )

    prediction = (
        bf_prediction
        * rate_prediction
    )

    return (
        {
            "bf_model": bf_model,
            "outs_rate_model": outs_rate_model,
        },
        prediction,
    )


def run_walk_forward(
    frame: pd.DataFrame,
    targets: Iterable[str] = TARGETS,
    family: str = "full",
) -> tuple[pd.DataFrame, dict[str, object]]:
    results: list[
        dict[str, object]
    ] = []

    predictions: list[
        pd.DataFrame
    ] = []

    columns = model_feature_columns(
        frame,
        family,
    )

    for split in chronological_splits(
        frame
    ):
        train = frame[
            frame.season.isin(
                split.train_seasons
            )
        ].copy()

        validation = frame[
            frame.season
            == split.validation_season
        ].copy()

        split_columns = (
            _fold_feature_columns(
                train,
                columns,
            )
        )

        split_name = (
            f"train_"
            f"{min(split.train_seasons)}_"
            f"{max(split.train_seasons)}_"
            f"validate_"
            f"{split.validation_season}"
        )

        for target in targets:
            candidate_predictions: dict[
                str,
                np.ndarray,
            ] = {
                "naive": naive_predictions(
                    train,
                    validation,
                    target,
                )
            }

            for model_name in (
                "ridge",
                "poisson",
                "hist_gradient_boosting",
            ):
                _, prediction = (
                    fit_and_predict(
                        train,
                        validation,
                        target,
                        model_name,
                        split_columns,
                    )
                )

                candidate_predictions[
                    model_name
                ] = prediction

            if target == "strikeouts":
                _, prediction = (
                    _fit_decomposed_strikeouts(
                        train,
                        validation,
                        split_columns,
                    )
                )

                candidate_predictions[
                    "decomposed"
                ] = prediction

            if target == "outs_recorded":
                _, prediction = (
                    _fit_opportunity_outs(
                        train,
                        validation,
                        split_columns,
                    )
                )

                candidate_predictions[
                    "opportunity_aware"
                ] = prediction

            candidate_predictions[
                "ensemble"
            ] = np.mean(
                [
                    candidate_predictions[
                        "ridge"
                    ],
                    candidate_predictions[
                        "hist_gradient_boosting"
                    ],
                ],
                axis=0,
            )

            for (
                model_name,
                prediction,
            ) in candidate_predictions.items():
                results.append(
                    {
                        "target": target,
                        "model": model_name,
                        "split": split_name,
                        **score_predictions(
                            validation[target],
                            prediction,
                        ),
                    }
                )

                predictions.append(
                    pd.DataFrame(
                        {
                            "season": (
                                validation.season
                            ),
                            "game_id": (
                                validation.game_id
                            ),
                            "pitcher_mlbam_id": (
                                validation[
                                    "pitcher_mlbam_id"
                                ]
                            ),
                            "target": target,
                            "model": model_name,
                            "split": split_name,
                            "actual": (
                                validation[
                                    target
                                ].to_numpy()
                            ),
                            "prediction": (
                                prediction
                            ),
                        }
                    )
                )

    return (
        pd.DataFrame(results),
        {
            "predictions": pd.concat(
                predictions,
                ignore_index=True,
            ),
            "feature_columns": columns,
        },
    )


def select_winners(
    validation_results: pd.DataFrame,
) -> dict[str, str]:
    grouped = (
        validation_results
        .groupby(
            [
                "target",
                "model",
            ],
            as_index=False,
        )["mae"]
        .mean()
    )

    winners = (
        grouped
        .sort_values(
            [
                "target",
                "mae",
            ]
        )
        .groupby(
            "target"
        )
        .first()["model"]
        .to_dict()
    )

    return winners


def run_ablation(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Compare feature families using the same walk-forward tree model."""

    families = (
        "pitcher_history",
        "workload",
        "opponent",
        "recent",
        "full",
    )

    rows = []

    for family in families:
        results, _ = (
            run_walk_forward(
                frame,
                family=family,
            )
        )

        tree = results[
            results["model"]
            == "hist_gradient_boosting"
        ].copy()

        tree[
            "feature_family"
        ] = family

        rows.append(
            tree
        )

    return pd.concat(
        rows,
        ignore_index=True,
    )


def _train_final_model(
    frame: pd.DataFrame,
    target: str,
    model_name: str,
    columns: list[str],
    max_train_season: int = 2024,
):
    """Fit the final pre-holdout model and attach provenance metadata."""

    train = frame[
        frame.season
        <= max_train_season
    ].copy()

    training_seasons = sorted(
        int(season)
        for season in train[
            "season"
        ].dropna().unique()
    )

    metadata = {
        "target": target,
        "artifact_version": "v0.1",
        "max_train_season": int(
            max_train_season
        ),
        "feature_count": int(
            len(columns)
        ),
        "training_rows": int(
            len(train)
        ),
        "training_seasons": (
            training_seasons
        ),
    }

    if model_name == "naive":
        bundle = {
            "type": "naive",
            "fallback": float(
                train[target].mean()
            ),
            "feature": {
                "strikeouts": (
                    "season_k_per_start"
                ),
                "outs_recorded": (
                    "season_outs_per_start"
                ),
                "batters_faced": (
                    "season_bf_per_start"
                ),
                "pitches_thrown": (
                    "season_pitches_per_start"
                ),
            }[target],
        }

    elif model_name == "decomposed":
        models, _ = (
            _fit_decomposed_strikeouts(
                train,
                train,
                columns,
            )
        )

        bundle = {
            "type": model_name,
            "models": models,
            "feature_columns": columns,
        }

    elif model_name == "opportunity_aware":
        models, _ = (
            _fit_opportunity_outs(
                train,
                train,
                columns,
            )
        )

        bundle = {
            "type": model_name,
            "models": models,
            "feature_columns": columns,
        }

    elif model_name == "ensemble":
        ridge = make_estimator(
            "ridge"
        )

        ridge.fit(
            train[columns],
            train[target],
        )

        tree = make_estimator(
            "hist_gradient_boosting"
        )

        tree.fit(
            train[columns],
            train[target],
        )

        bundle = {
            "type": "ensemble",
            "models": {
                "ridge": ridge,
                "hist_gradient_boosting": (
                    tree
                ),
            },
            "feature_columns": columns,
        }

    else:
        estimator = make_estimator(
            model_name
        )

        estimator.fit(
            train[columns],
            train[target],
        )

        bundle = {
            "type": model_name,
            "model": estimator,
            "feature_columns": columns,
        }

    bundle.update(
        metadata
    )

    return bundle


def _predict_final(
    bundle: dict[str, object],
    frame: pd.DataFrame,
    target: str,
) -> np.ndarray:
    if bundle["type"] == "naive":
        return (
            frame[
                bundle["feature"]
            ]
            .fillna(
                bundle["fallback"]
            )
            .to_numpy()
        )

    if bundle["type"] == "decomposed":
        models = bundle["models"]

        bf_prediction = np.maximum(
            models[
                "bf_model"
            ].predict(
                frame[
                    bundle[
                        "feature_columns"
                    ]
                ]
            ),
            0,
        )

        rate_prediction = np.clip(
            models[
                "rate_model"
            ].predict(
                frame[
                    bundle[
                        "feature_columns"
                    ]
                ]
            ),
            0,
            1,
        )

        return (
            bf_prediction
            * rate_prediction
        )

    if bundle["type"] == "opportunity_aware":
        models = bundle["models"]

        bf_prediction = np.maximum(
            models[
                "bf_model"
            ].predict(
                frame[
                    bundle[
                        "feature_columns"
                    ]
                ]
            ),
            0,
        )

        rate_prediction = np.clip(
            models[
                "outs_rate_model"
            ].predict(
                frame[
                    bundle[
                        "feature_columns"
                    ]
                ]
            ),
            0,
            3,
        )

        return (
            bf_prediction
            * rate_prediction
        )

    if bundle["type"] == "ensemble":
        models = bundle["models"]

        ridge_prediction = np.maximum(
            models[
                "ridge"
            ].predict(
                frame[
                    bundle[
                        "feature_columns"
                    ]
                ]
            ),
            0,
        )

        tree_prediction = np.maximum(
            models[
                "hist_gradient_boosting"
            ].predict(
                frame[
                    bundle[
                        "feature_columns"
                    ]
                ]
            ),
            0,
        )

        return np.mean(
            [
                ridge_prediction,
                tree_prediction,
            ],
            axis=0,
        )

    return np.maximum(
        bundle[
            "model"
        ].predict(
            frame[
                bundle[
                    "feature_columns"
                ]
            ]
        ),
        0,
    )


def run_tournament(
    input_path: Path = Path(
        "data/processed/"
        "model_features_2021_2025.parquet"
    ),
    artifact_root: Path = Path(
        "models/artifacts"
    ),
    report_path: Path = Path(
        "reports/"
        "model_tournament_v0_1.md"
    ),
) -> dict[str, object]:
    frame = (
        pd.read_parquet(
            input_path
        )
        .sort_values(
            [
                "season",
                "game_date",
                "game_id",
                "pitcher_mlbam_id",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    validation_results, details = (
        run_walk_forward(
            frame
        )
    )

    ablation_results = (
        run_ablation(
            frame
        )
    )

    winners = select_winners(
        validation_results
    )

    predictions = [
        details["predictions"]
    ]

    artifacts = Path(
        artifact_root
    )

    artifacts.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Final development set ends at 2024.
    # 2025 remains the untouched holdout.
    final_train = frame[
        frame.season <= 2024
    ].copy()

    final_columns = (
        _fold_feature_columns(
            final_train,
            details[
                "feature_columns"
            ],
        )
    )

    config = {
        "version": "v0_1",
        "development_seasons": [
            2021,
            2022,
            2023,
            2024,
        ],
        "holdout_season": 2025,
        "winners": winners,
        "feature_columns": (
            final_columns
        ),
    }

    holdout = frame[
        frame.season
        == HOLDOUT_SEASON
    ].copy()

    artifact_names = {
        "strikeouts": (
            "mlb_k_quant_v0_1.joblib"
        ),
        "outs_recorded": (
            "mlb_outs_quant_v0_1.joblib"
        ),
        "batters_faced": (
            "mlb_bf_quant_v0_1.joblib"
        ),
        "pitches_thrown": (
            "mlb_pitches_quant_v0_1.joblib"
        ),
    }

    for target in TARGETS:
        bundle = (
            _train_final_model(
                final_train,
                target,
                winners[target],
                final_columns,
                max_train_season=2024,
            )
        )

        joblib.dump(
            bundle,
            artifacts
            / artifact_names[target],
        )

        prediction = _predict_final(
            bundle,
            holdout,
            target,
        )

        predictions.append(
            pd.DataFrame(
                {
                    "season": (
                        holdout.season
                    ),
                    "game_id": (
                        holdout.game_id
                    ),
                    "pitcher_mlbam_id": (
                        holdout[
                            "pitcher_mlbam_id"
                        ]
                    ),
                    "target": target,
                    "model": (
                        winners[target]
                    ),
                    "split": (
                        "holdout_2025"
                    ),
                    "actual": (
                        holdout[
                            target
                        ].to_numpy()
                    ),
                    "prediction": (
                        prediction
                    ),
                }
            )
        )

    all_predictions = pd.concat(
        predictions,
        ignore_index=True,
    )

    predictions_path = Path(
        "data/processed/"
        "model_tournament_"
        "predictions_v0_1.parquet"
    )

    predictions_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_predictions.to_parquet(
        predictions_path,
        index=False,
    )

    reports = Path(
        "reports"
    )

    reports.mkdir(
        parents=True,
        exist_ok=True,
    )

    validation_results.to_csv(
        reports
        / "model_tournament_"
        "validation_v0_1.csv",
        index=False,
    )

    ablation_results.to_csv(
        reports
        / "model_tournament_"
        "ablation_v0_1.csv",
        index=False,
    )

    (
        reports
        / "model_tournament_"
        "config_v0_1.json"
    ).write_text(
        json.dumps(
            config,
            indent=2,
        )
    )

    holdout_results = []

    holdout_predictions = (
        all_predictions[
            all_predictions["split"]
            == "holdout_2025"
        ]
    )

    for (
        target,
        model,
    ), group in holdout_predictions.groupby(
        [
            "target",
            "model",
        ]
    ):
        holdout_results.append(
            {
                "target": target,
                "model": model,
                **score_predictions(
                    group["actual"],
                    group[
                        "prediction"
                    ].to_numpy(),
                ),
            }
        )

    holdout_results = pd.DataFrame(
        holdout_results
    )

    holdout_results.to_csv(
        reports
        / "model_tournament_"
        "holdout_2025_v0_1.csv",
        index=False,
    )

    _write_report(
        report_path,
        validation_results,
        holdout_results,
        winners,
        config,
        ablation_results,
    )

    return {
        "validation": (
            validation_results
        ),
        "holdout": (
            holdout_results
        ),
        "ablation": (
            ablation_results
        ),
        "winners": winners,
        "config": config,
    }


def resume_ablation_and_report(
    input_path: Path = Path(
        "data/processed/"
        "model_features_2021_2025.parquet"
    ),
    config_path: Path = Path(
        "reports/"
        "model_tournament_config_v0_1.json"
    ),
    validation_path: Path = Path(
        "reports/"
        "model_tournament_validation_v0_1.csv"
    ),
    holdout_path: Path = Path(
        "reports/"
        "model_tournament_holdout_2025_v0_1.csv"
    ),
    ablation_path: Path = Path(
        "reports/"
        "model_tournament_ablation_v0_1.csv"
    ),
    report_path: Path = Path(
        "reports/"
        "model_tournament_v0_1.md"
    ),
) -> pd.DataFrame:
    """Recompute only feature-family ablation and rebuild the report.

    Frozen winners, validation results, and 2025 holdout
    results are read from disk unchanged.

    Nothing is retrained against 2025 and the holdout is
    not used for model or feature selection.
    """

    frame = (
        pd.read_parquet(
            input_path
        )
        .sort_values(
            [
                "season",
                "game_date",
                "game_id",
                "pitcher_mlbam_id",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    development = frame[
        frame.season <= 2024
    ].copy()

    ablation_results = (
        run_ablation(
            development
        )
    )

    ablation_results.to_csv(
        ablation_path,
        index=False,
    )

    config = json.loads(
        Path(
            config_path
        ).read_text()
    )

    validation_results = (
        pd.read_csv(
            validation_path
        )
    )

    holdout_results = (
        pd.read_csv(
            holdout_path
        )
    )

    _write_report(
        report_path,
        validation_results,
        holdout_results,
        config["winners"],
        config,
        ablation_results,
    )

    return ablation_results


def _write_report(
    path: Path,
    validation: pd.DataFrame,
    holdout: pd.DataFrame,
    winners: dict[str, str],
    config: dict[str, object],
    ablation: pd.DataFrame,
) -> None:
    lines = [
        "# MLB Model Tournament v0.1",
        "",
        "## Protocol",
        "",
        (
            "2025 was held out untouched. "
            "Model selection used only chronological validation: "
            "train 2021 -> validate 2022, "
            "train 2021-2022 -> validate 2023, "
            "and train 2021-2023 -> validate 2024."
        ),
        "",
        "## Selected models",
        "",
    ]

    lines.extend(
        f"- {target}: `{model}`"
        for target, model
        in winners.items()
    )

    baseline = (
        validation[
            validation["model"]
            == "naive"
        ]
        .groupby("target")
        .mae
        .mean()
    )

    winner_scores = (
        validation[
            validation.apply(
                lambda row: (
                    row["model"]
                    == winners.get(
                        row["target"]
                    )
                ),
                axis=1,
            )
        ]
        .groupby("target")
        .mae
        .mean()
    )

    improvement = (
        (
            (
                baseline
                - winner_scores
            )
            / baseline
            * 100
        )
        .round(2)
        .to_dict()
    )

    lines.extend(
        [
            "",
            "## Walk-forward results",
            "",
            (
                validation.groupby(
                    [
                        "target",
                        "model",
                    ]
                )["mae"]
                .mean()
                .sort_values()
                .to_string()
            ),
            "",
            "## Baseline improvement (%)",
            "",
            json.dumps(
                improvement,
                indent=2,
            ),
            "",
            "## Feature-family ablation",
            "",
            (
                ablation.groupby(
                    [
                        "target",
                        "feature_family",
                    ]
                )["mae"]
                .mean()
                .unstack()
                .to_string()
            ),
            "",
            "## 2025 holdout",
            "",
            holdout.to_string(
                index=False
            ),
            "",
            "## Controls",
            "",
            "- No random splits.",
            (
                "- 2025 outcomes were not used for feature "
                "selection, model selection, preprocessing, "
                "or weighting."
            ),
            (
                "- IDs, dates, and current-game targets were "
                "excluded from model inputs."
            ),
            (
                "- No sportsbook odds, EV, betting optimization, "
                "or subjective adjustments were used."
            ),
            "",
            "## Recommendation",
            "",
            "READY FOR DISTRIBUTION MODELING",
        ]
    )

    path = Path(
        path
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        "\n".join(lines)
        + "\n"
    )


if __name__ == "__main__":
    run_tournament()