from pathlib import Path
import json
import math

import joblib
import numpy as np
import pandas as pd

from mlb_quant.models.distribution_types import EmpiricalResidualModel

from scipy.stats import poisson, nbinom
from sklearn.metrics import brier_score_loss, log_loss


INPUT_PATH = Path("data/processed/model_tournament_predictions_v0_1.parquet")
REPORT_PATH = Path("reports/distribution_model_v0_1.md")
OUTPUT_PATH = Path("data/processed/distribution_predictions_v0_1.parquet")
ARTIFACT_PATH = Path("models/artifacts/mlb_distribution_v0_1.joblib")


TARGET_CONFIG = {
    "strikeouts": {
        "model": "ensemble",
        "lines": np.arange(2.5, 10.6, 1.0),
        "min_value": 0,
        "max_value": 20,
    },
    "outs_recorded": {
        "model": "hist_gradient_boosting",
        "lines": np.arange(11.5, 21.6, 1.0),
        "min_value": 0,
        "max_value": 27,
    },
}


def safe_probability(p):
    return np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)


def poisson_probability(mu, line):
    mu = np.maximum(np.asarray(mu, dtype=float), 1e-6)
    threshold = math.floor(line)
    return poisson.sf(threshold, mu)


def estimate_nb_alpha(actual, prediction):
    actual = np.asarray(actual, dtype=float)
    mu = np.maximum(np.asarray(prediction, dtype=float), 1e-6)

    values = ((actual - mu) ** 2 - mu) / (mu ** 2)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return 1e-6

    return max(float(np.mean(values)), 1e-6)


def negative_binomial_probability(mu, line, alpha):
    mu = np.maximum(np.asarray(mu, dtype=float), 1e-6)

    if alpha <= 1e-6:
        return poisson_probability(mu, line)

    n = 1.0 / alpha
    p = n / (n + mu)

    return nbinom.sf(math.floor(line), n, p)



def probability_dataset(frame, method_name, fitted_object, config):
    rows = []

    for line in config["lines"]:
        actual_over = (frame["actual"].to_numpy() > line).astype(int)
        prediction = frame["prediction"].to_numpy()

        if method_name == "poisson":
            probabilities = poisson_probability(prediction, line)

        elif method_name == "negative_binomial":
            probabilities = negative_binomial_probability(
                prediction,
                line,
                fitted_object["alpha"],
            )

        elif method_name in {"empirical", "conditional_empirical"}:
            probabilities = fitted_object.probability_over(
                prediction,
                line,
                config["min_value"],
                config["max_value"],
            )

        else:
            raise ValueError(method_name)

        probabilities = safe_probability(probabilities)

        part = frame[
            [
                "season",
                "game_id",
                "pitcher_mlbam_id",
                "actual",
                "prediction",
            ]
        ].copy()

        part["line"] = line
        part["actual_over"] = actual_over
        part["probability_over"] = probabilities
        part["method"] = method_name

        rows.append(part)

    return pd.concat(rows, ignore_index=True)


def fit_distribution(method_name, training_frame):
    if method_name == "poisson":
        return {}

    if method_name == "negative_binomial":
        alpha = estimate_nb_alpha(
            training_frame["actual"],
            training_frame["prediction"],
        )

        return {"alpha": alpha}

    if method_name == "empirical":
        model = EmpiricalResidualModel(conditional=False)

        return model.fit(
            training_frame["prediction"],
            training_frame["actual"],
        )

    if method_name == "conditional_empirical":
        model = EmpiricalResidualModel(
            conditional=True,
            n_bins=5,
            min_bin_size=250,
        )

        return model.fit(
            training_frame["prediction"],
            training_frame["actual"],
        )

    raise ValueError(method_name)


def score_probability_frame(frame):
    y = frame["actual_over"].to_numpy()
    p = safe_probability(frame["probability_over"].to_numpy())

    return {
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "n": int(len(frame)),
    }


def calibration_table(frame):
    data = frame.copy()

    data["probability_bucket"] = pd.cut(
        data["probability_over"],
        bins=np.linspace(0, 1, 11),
        include_lowest=True,
    )

    result = (
        data.groupby("probability_bucket", observed=False)
        .agg(
            n=("actual_over", "size"),
            predicted_probability=("probability_over", "mean"),
            observed_rate=("actual_over", "mean"),
        )
        .reset_index()
    )

    result["calibration_error"] = (
        result["predicted_probability"] - result["observed_rate"]
    )

    return result


def select_development_rows(df, target, model):
    return df[
        (df["target"] == target)
        & (df["model"] == model)
        & (df["split"] != "holdout_2025")
    ].copy()


def select_holdout_rows(df, target, model):
    return df[
        (df["target"] == target)
        & (df["model"] == model)
        & (df["split"] == "holdout_2025")
    ].copy()


def run_target(target, df):
    config = TARGET_CONFIG[target]
    model_name = config["model"]

    development = select_development_rows(df, target, model_name)
    holdout = select_holdout_rows(df, target, model_name)

    methods = [
        "poisson",
        "negative_binomial",
        "empirical",
        "conditional_empirical",
    ]

    meta_splits = [
        ([2022], 2023),
        ([2022, 2023], 2024),
    ]

    validation_results = []
    validation_predictions = []

    for training_seasons, validation_season in meta_splits:
        train = development[
            development["season"].isin(training_seasons)
        ].copy()

        valid = development[
            development["season"] == validation_season
        ].copy()

        for method_name in methods:
            fitted = fit_distribution(method_name, train)

            probability_frame = probability_dataset(
                valid,
                method_name,
                fitted,
                config,
            )

            score = score_probability_frame(probability_frame)

            validation_results.append(
                {
                    "target": target,
                    "method": method_name,
                    "training_seasons": ",".join(map(str, training_seasons)),
                    "validation_season": validation_season,
                    **score,
                }
            )

            probability_frame["target"] = target
            probability_frame["validation_season"] = validation_season

            validation_predictions.append(probability_frame)

    validation_results = pd.DataFrame(validation_results)

    summary = (
        validation_results.groupby(["target", "method"])
        .agg(
            brier=("brier", "mean"),
            log_loss=("log_loss", "mean"),
        )
        .reset_index()
        .sort_values(["brier", "log_loss"])
    )

    winner = summary.iloc[0]["method"]

    final_distribution = fit_distribution(
        winner,
        development,
    )

    holdout_probabilities = probability_dataset(
        holdout,
        winner,
        final_distribution,
        config,
    )

    holdout_probabilities["target"] = target
    holdout_probabilities["validation_season"] = 2025

    holdout_score = score_probability_frame(holdout_probabilities)
    holdout_calibration = calibration_table(holdout_probabilities)

    return {
        "target": target,
        "winner": winner,
        "validation_results": validation_results,
        "summary": summary,
        "validation_predictions": pd.concat(
            validation_predictions,
            ignore_index=True,
        ),
        "holdout_predictions": holdout_probabilities,
        "holdout_score": holdout_score,
        "holdout_calibration": holdout_calibration,
        "artifact": final_distribution,
    }


def main():
    df = pd.read_parquet(INPUT_PATH)

    required = {
        "season",
        "game_id",
        "pitcher_mlbam_id",
        "target",
        "model",
        "split",
        "actual",
        "prediction",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    results = {}

    all_predictions = []
    report_sections = []

    for target in ["strikeouts", "outs_recorded"]:
        result = run_target(target, df)
        results[target] = result

        all_predictions.append(result["validation_predictions"])
        all_predictions.append(result["holdout_predictions"])

        report_sections.append(f"# {target}")
        report_sections.append("")
        report_sections.append(
            f"Selected distribution: `{result['winner']}`"
        )
        report_sections.append("")

        report_sections.append("## Development results")
        report_sections.append("")
        report_sections.append(
            result["summary"].to_markdown(index=False)
        )
        report_sections.append("")

        report_sections.append("## Untouched 2025 holdout")
        report_sections.append("")
        report_sections.append(
            f"- Brier score: {result['holdout_score']['brier']:.6f}"
        )
        report_sections.append(
            f"- Log loss: {result['holdout_score']['log_loss']:.6f}"
        )
        report_sections.append(
            f"- Binary line observations: {result['holdout_score']['n']}"
        )
        report_sections.append("")

        report_sections.append("## 2025 calibration")
        report_sections.append("")
        report_sections.append(
            result["holdout_calibration"].to_markdown(index=False)
        )
        report_sections.append("")

    predictions = pd.concat(all_predictions, ignore_index=True)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    predictions.to_parquet(OUTPUT_PATH, index=False)

    artifact = {
        target: {
            "method": result["winner"],
            "model": TARGET_CONFIG[target]["model"],
            "distribution": result["artifact"],
            "config": TARGET_CONFIG[target],
        }
        for target, result in results.items()
    }

    joblib.dump(artifact, ARTIFACT_PATH)

    header = [
        "# MLB Distribution Model v0.1",
        "",
        "Distribution selection used only chronological OOF predictions.",
        "",
        "Meta-validation:",
        "- 2022 residual history -> evaluate 2023",
        "- 2022-2023 residual history -> evaluate 2024",
        "- Freeze selected distribution",
        "- Evaluate exactly once on untouched 2025 holdout",
        "",
    ]

    REPORT_PATH.write_text(
        "\n".join(header + report_sections),
        encoding="utf-8",
    )

    print("Distribution modeling complete.")
    print("")
    print("Winners:")

    for target, result in results.items():
        print(f"  {target}: {result['winner']}")

    print("")
    print(f"Predictions: {OUTPUT_PATH}")
    print(f"Artifact: {ARTIFACT_PATH}")
    print(f"Report: {REPORT_PATH}")


if __name__ == "__main__":
    main()