"""pandas_offset_alias: the old offset spellings keep their meaning on pandas 2.2+ and 3.

pandas 3 removed "M", "Y", "H", "T", ... ("Invalid frequency"). quantbox configs and
users still write them. Each old spelling must parse to the SAME offset it meant on
pandas 2, and every other string must pass through untouched. Runs on whichever
pandas is installed: the lock (3.x) in CI, and pandas 2.3.3 for consumers.
"""

from __future__ import annotations

import time

import pandas as pd
import pytest

from quantbox.frequency import pandas_offset_alias, parse_rebalance_offset

to_offset = pd.tseries.frequencies.to_offset

OLD_SPELLINGS = [
    ("1M", pd.offsets.MonthEnd(1)),
    ("6M", pd.offsets.MonthEnd(6)),
    ("M", pd.offsets.MonthEnd(1)),
    ("+2M", pd.offsets.MonthEnd(2)),
    ("1BM", pd.offsets.BusinessMonthEnd(1)),
    ("1SM", pd.offsets.SemiMonthEnd(1)),
    ("1Q", pd.offsets.QuarterEnd(startingMonth=12)),
    ("Q-DEC", pd.offsets.QuarterEnd(startingMonth=12)),
    ("BQ", pd.offsets.BQuarterEnd(startingMonth=12)),
    ("1Y", pd.offsets.YearEnd(1)),
    ("1A", pd.offsets.YearEnd(1)),
    ("A-JUN", pd.offsets.YearEnd(month=6)),
    ("BY", pd.offsets.BYearEnd(1)),
    ("BA", pd.offsets.BYearEnd(1)),
    ("AS", pd.offsets.YearBegin(1)),
    ("AS-JAN", pd.offsets.YearBegin(month=1)),
    ("BAS", pd.offsets.BYearBegin(1)),
    ("4H", pd.offsets.Hour(4)),
    ("1BH", pd.offsets.BusinessHour(1)),
    ("5T", pd.offsets.Minute(5)),
    ("30S", pd.offsets.Second(30)),
    ("100L", pd.offsets.Milli(100)),
    ("10U", pd.offsets.Micro(10)),
    ("1N", pd.offsets.Nano(1)),
    # compound, fractional and numeric-anchor spellings (codex on #331)
    ("1H30T", pd.Timedelta(minutes=90)),
    ("1D1H", pd.Timedelta(hours=25)),
    ("1.5H", pd.Timedelta(minutes=90)),
]

UNCHANGED = ["1D", "1W", "W-FRI", "2W", "ME", "6ME", "MS", "QS", "YS", "BMS", "1min", "1h", "100ms", "not-an-offset"]


@pytest.mark.parametrize(("old", "offset"), OLD_SPELLINGS)
def test_an_old_spelling_keeps_its_meaning(old, offset):
    """A Timedelta expectation (compound, fractional) is compared as the tick offset's span."""
    got = to_offset(pandas_offset_alias(old))
    if isinstance(offset, pd.Timedelta):
        assert pd.Timedelta(got) == offset
    else:
        assert got == offset


@pytest.mark.parametrize("spec", UNCHANGED)
def test_every_other_string_passes_through(spec):
    assert pandas_offset_alias(spec) == spec


@pytest.mark.parametrize(
    "bad",
    ["A" * 5000 + "!", "-A" * 5000 + "!", "W" + "-1D" * 5000 + "!", "1a" * 5000 + "!", "1" * 5000 + "!"],
    ids=["letters", "minus-letter", "anchor-signed", "digit-letter", "digits"],
)
def test_a_long_bad_string_is_answered_at_once(bad):
    """Guard (review rounds 3, 4): whole-string regexes took 13 s on 26 letters, 2 s on "-A"*32."""
    start = time.perf_counter()
    assert pandas_offset_alias(bad) == bad
    assert time.perf_counter() - start < 0.5


@pytest.mark.parametrize("spec", ["1m", "5m"])
def test_the_ambiguous_lowercase_m_is_still_refused(spec):
    with pytest.raises(ValueError, match="ambiguous"):
        parse_rebalance_offset(spec)
