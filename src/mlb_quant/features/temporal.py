"""Time-aware feature construction contracts."""

from datetime import datetime


def require_prediction_cutoff(prediction_time: datetime) -> datetime:
    """Validate the cutoff used to construct pre-game features."""
    if prediction_time.tzinfo is None:
        raise ValueError("prediction_time must be timezone-aware")
    return prediction_time


# Future feature builders should filter every source row to observation_time < cutoff.
