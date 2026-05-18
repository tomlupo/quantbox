"""Tests for the Frequency value object (issue #20 / PR B)."""

from __future__ import annotations

import pandas as pd
import pytest

from quantbox.frequency import Frequency

# ---------------------------------------------------------------------------
# parse()
# ---------------------------------------------------------------------------


def test_parse_returns_frequency_instance_unchanged():
    f = Frequency(pd.Timedelta("1h"), "NYSE")
    assert Frequency.parse(f) is f


def test_parse_string_defaults_calendar_to_24_7():
    f = Frequency.parse("1d")
    assert f.bar_size == pd.Timedelta("1d")
    assert f.calendar == "24/7"


def test_parse_dict_full_spec():
    f = Frequency.parse({"bar_size": "4h", "calendar": "NYSE"})
    assert f.bar_size == pd.Timedelta("4h")
    assert f.calendar == "NYSE"


def test_parse_dict_partial_uses_defaults():
    f = Frequency.parse({"bar_size": "1h"})
    assert f.calendar == "24/7"


def test_parse_monthly_M_suffix_approximated_as_30_days():
    f = Frequency.parse("1M")
    assert f.bar_size == pd.Timedelta(days=30)


def test_parse_rejects_int():
    with pytest.raises(TypeError):
        Frequency.parse(123)


def test_parse_rejects_garbage_string():
    with pytest.raises(ValueError, match="cannot parse"):
        Frequency.parse("garbage")


# ---------------------------------------------------------------------------
# bars_per_year()
# ---------------------------------------------------------------------------


def test_bars_per_year_24_7_daily_is_365():
    assert Frequency.parse("1d").bars_per_year() == 365.0


def test_bars_per_year_24_7_hourly_is_8760():
    f = Frequency.parse({"bar_size": "1h", "calendar": "24/7"})
    assert f.bars_per_year() == 8760.0


def test_bars_per_year_24_7_4h_is_2190():
    f = Frequency.parse({"bar_size": "4h", "calendar": "24/7"})
    assert f.bars_per_year() == 2190.0


def test_bars_per_year_nyse_daily_near_250():
    """NYSE 2023 had 250 trading days. Accept ±5 for calendar variation."""
    f = Frequency.parse({"bar_size": "1d", "calendar": "NYSE"})
    assert 245 <= f.bars_per_year() <= 255


def test_bars_per_year_nyse_hourly_near_1625():
    """NYSE 6.5h session × ~250 days = ~1625. Accept ±25 for half-days."""
    f = Frequency.parse({"bar_size": "1h", "calendar": "NYSE"})
    assert 1590 <= f.bars_per_year() <= 1660


def test_bars_per_year_xwar_daily_polish_calendar():
    """Warsaw stock exchange — relevant for Polish funds."""
    f = Frequency.parse({"bar_size": "1d", "calendar": "XWAR"})
    assert 245 <= f.bars_per_year() <= 255


# ---------------------------------------------------------------------------
# valid_dates / to_pandas_offset / round-trip
# ---------------------------------------------------------------------------


def test_valid_dates_24_7_returns_continuous_range():
    f = Frequency.parse("1d")
    dates = f.valid_dates("2024-01-01", "2024-01-07")
    assert len(dates) == 7  # 24/7 calendar has all 7 days


def test_valid_dates_nyse_skips_weekends_and_holidays():
    f = Frequency.parse({"bar_size": "1d", "calendar": "NYSE"})
    # 2024-01-01 was a Monday holiday (New Year's Day). 2024-01-02 is the first trading day.
    dates = f.valid_dates("2024-01-01", "2024-01-07")
    # NYSE excludes Sat 2024-01-06, Sun 2024-01-07, and 2024-01-01 holiday → 4 trading days
    assert len(dates) == 4


def test_to_pandas_offset_round_trip():
    f = Frequency.parse("1d")
    offset = f.to_pandas_offset()
    assert offset == pd.Timedelta("1d")


def test_to_dict_round_trips():
    original = Frequency.parse({"bar_size": "4h", "calendar": "NYSE"})
    roundtripped = Frequency.parse(original.to_dict())
    assert roundtripped == original


def test_frozen_dataclass_is_hashable():
    """Should be usable as dict key / cache key."""
    a = Frequency(pd.Timedelta("1d"), "NYSE")
    b = Frequency(pd.Timedelta("1d"), "NYSE")
    assert hash(a) == hash(b)
    assert {a: "value"}[b] == "value"
