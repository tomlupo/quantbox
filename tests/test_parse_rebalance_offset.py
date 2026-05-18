"""Tests for parse_rebalance_offset — strict rebalance-frequency parser (issue #20 follow-up).

The legacy `get_rebalancing_dates` parser treated lowercase `"1m"` as MONTHS
while ccxt and pandas both treat it as MINUTES. This parser closes that hole
by rejecting the ambiguous lowercase form with a clear suggestion.
"""

from __future__ import annotations

import pandas as pd
import pytest

from quantbox.frequency import parse_rebalance_offset

# ---------------------------------------------------------------------------
# Accepts canonical pandas-offset strings
# ---------------------------------------------------------------------------


def test_accepts_uppercase_D_W_M_Y():
    assert isinstance(parse_rebalance_offset("1D"), pd.DateOffset)
    assert isinstance(parse_rebalance_offset("1W"), pd.DateOffset)
    assert isinstance(parse_rebalance_offset("1M"), pd.DateOffset)
    assert isinstance(parse_rebalance_offset("1Y"), pd.DateOffset)


def test_accepts_multi_unit_pandas_offsets():
    assert isinstance(parse_rebalance_offset("3D"), pd.DateOffset)
    assert isinstance(parse_rebalance_offset("2W"), pd.DateOffset)
    assert isinstance(parse_rebalance_offset("6M"), pd.DateOffset)


def test_accepts_explicit_minute_spellings():
    """min and ms are unambiguous (3+ char suffix)."""
    assert isinstance(parse_rebalance_offset("1min"), pd.DateOffset)
    assert isinstance(parse_rebalance_offset("30min"), pd.DateOffset)
    assert isinstance(parse_rebalance_offset("100ms"), pd.DateOffset)


def test_accepts_dateoffset_passthrough():
    do = pd.DateOffset(days=5)
    assert parse_rebalance_offset(do) is do


# ---------------------------------------------------------------------------
# Rejects the ambiguous "1m" / "5m" / etc.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("spec", ["1m", "5m", "15m", "30m"])
def test_rejects_lowercase_m_with_helpful_message(spec):
    with pytest.raises(ValueError) as exc:
        parse_rebalance_offset(spec)
    msg = str(exc.value)
    assert "ambiguous" in msg
    # Should suggest both unambiguous alternatives
    assert spec[:-1] + "M" in msg  # months
    assert spec[:-1] + "min" in msg  # minutes


def test_rejects_non_string_non_dateoffset():
    with pytest.raises(TypeError):
        parse_rebalance_offset(123)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        parse_rebalance_offset(None)  # type: ignore[arg-type]


def test_rejects_garbage_string():
    with pytest.raises(ValueError):
        parse_rebalance_offset("not-an-offset")


# ---------------------------------------------------------------------------
# Integration with get_rebalancing_dates (the original consumer)
# ---------------------------------------------------------------------------


def test_get_rebalancing_dates_str_paths_route_through_strict_parser():
    """get_rebalancing_dates should reject ambiguous "1m" too."""
    from quantbox.plugins.backtesting.vectorbt_engine import get_rebalancing_dates

    dates = pd.date_range("2024-01-01", periods=100, freq="D")

    # Valid uppercase still works
    weekly = get_rebalancing_dates(dates, "1W")
    assert len(weekly) > 10

    monthly = get_rebalancing_dates(dates, "1M")
    assert 2 <= len(monthly) <= 5  # ~3 months in 100 days

    # Ambiguous "1m" rejected
    with pytest.raises(ValueError, match="ambiguous"):
        get_rebalancing_dates(dates, "1m")


def test_get_rebalancing_dates_accepts_dateoffset_directly():
    from quantbox.plugins.backtesting.vectorbt_engine import get_rebalancing_dates

    dates = pd.date_range("2024-01-01", periods=100, freq="D")
    result = get_rebalancing_dates(dates, pd.DateOffset(weeks=2))
    assert len(result) >= 5  # ~7 fortnights in 100 days


def test_get_rebalancing_dates_int_form_unchanged():
    """Backwards compatibility: int form still means "every N bars"."""
    from quantbox.plugins.backtesting.vectorbt_engine import get_rebalancing_dates

    dates = pd.date_range("2024-01-01", periods=100, freq="D")
    assert len(get_rebalancing_dates(dates, 1)) == 100
    assert len(get_rebalancing_dates(dates, 5)) == 20
    assert len(get_rebalancing_dates(dates, 21)) == 5


def test_get_rebalancing_dates_none_means_buy_and_hold():
    from quantbox.plugins.backtesting.vectorbt_engine import get_rebalancing_dates

    dates = pd.date_range("2024-01-01", periods=100, freq="D")
    result = get_rebalancing_dates(dates, None)
    assert len(result) == 1
    assert result[0] == dates[0]


def test_get_rebalancing_dates_list_form_unchanged():
    from quantbox.plugins.backtesting.vectorbt_engine import get_rebalancing_dates

    dates = pd.date_range("2024-01-01", periods=100, freq="D")
    explicit = [dates[10], dates[50], dates[90]]
    result = get_rebalancing_dates(dates, explicit)
    assert len(result) == 3
    assert list(result) == explicit
