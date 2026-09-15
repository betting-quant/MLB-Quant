from datetime import datetime, timezone

import pytest

from mlb_quant.data.ingestion import validate_as_of
from mlb_quant.features.temporal import require_prediction_cutoff


def test_temporal_guards_require_timezone_aware_cutoffs():
    with pytest.raises(ValueError):
        validate_as_of(datetime(2024, 4, 1))

    with pytest.raises(ValueError):
        require_prediction_cutoff(datetime(2024, 4, 1))


def test_temporal_guards_accept_aware_cutoffs():
    cutoff = datetime(2024, 4, 1, tzinfo=timezone.utc)
    assert validate_as_of(cutoff) == cutoff
    assert require_prediction_cutoff(cutoff) == cutoff
