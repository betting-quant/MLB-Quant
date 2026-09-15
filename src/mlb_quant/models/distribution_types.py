import numpy as np


class EmpiricalResidualModel:
    def __init__(self, conditional=False, n_bins=5, min_bin_size=250):
        self.conditional = conditional
        self.n_bins = n_bins
        self.min_bin_size = min_bin_size

        self.global_residuals = None
        self.bin_edges = None
        self.bin_residuals = {}

    def fit(self, prediction, actual):
        prediction = np.asarray(prediction, dtype=float)
        actual = np.asarray(actual, dtype=float)

        residuals = actual - prediction
        mask = np.isfinite(residuals) & np.isfinite(prediction)

        prediction = prediction[mask]
        residuals = residuals[mask]

        self.global_residuals = residuals

        if not self.conditional:
            return self

        quantiles = np.linspace(0, 1, self.n_bins + 1)
        edges = np.quantile(prediction, quantiles)
        edges = np.unique(edges)

        if len(edges) < 3:
            self.conditional = False
            return self

        edges[0] = -np.inf
        edges[-1] = np.inf
        self.bin_edges = edges

        bin_ids = np.digitize(
            prediction,
            edges[1:-1],
            right=False,
        )

        for bin_id in np.unique(bin_ids):
            r = residuals[bin_ids == bin_id]

            if len(r) >= self.min_bin_size:
                self.bin_residuals[int(bin_id)] = r

        return self

    def residuals_for_prediction(self, prediction):
        if not self.conditional or self.bin_edges is None:
            return self.global_residuals

        bin_id = int(
            np.digitize(
                [prediction],
                self.bin_edges[1:-1],
                right=False,
            )[0]
        )

        return self.bin_residuals.get(
            bin_id,
            self.global_residuals,
        )

    def probability_over(
        self,
        predictions,
        line,
        min_value,
        max_value,
    ):
        probabilities = []

        for pred in np.asarray(predictions, dtype=float):
            residuals = self.residuals_for_prediction(pred)

            simulated = pred + residuals
            simulated = np.clip(
                simulated,
                min_value,
                max_value,
            )

            probabilities.append(
                float(np.mean(simulated > line))
            )

        return np.asarray(probabilities)