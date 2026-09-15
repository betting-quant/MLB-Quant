from datetime import date

import pytest

from mlb_quant.data.scale import REGULAR_SEASON_RANGES, process_season


def test_scale_is_limited_to_2021_through_2025():
    assert set(REGULAR_SEASON_RANGES) == {2021, 2022, 2023, 2024, 2025}
    assert REGULAR_SEASON_RANGES[2021] == (date(2021, 4, 1), date(2021, 10, 3))
    assert REGULAR_SEASON_RANGES[2025] == (date(2025, 3, 27), date(2025, 9, 28))


def test_scale_rejects_2026_before_network_access(tmp_path):
    with pytest.raises(ValueError, match="2021-2025"):
        process_season(2026, raw_root=tmp_path / "raw", processed_root=tmp_path / "processed")