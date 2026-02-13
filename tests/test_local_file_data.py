"""Tests for LocalFileDataPlugin — local_file_data."""

import numpy as np
import pandas as pd
import pytest

from quantbox.plugins.datasources.local_file_data import (
    DUCKDB_AVAILABLE,
    LocalFileDataPlugin,
    _pivot_long_to_wide,
    _read_file,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_wide_prices(symbols: list[str], dates: pd.DatetimeIndex, rng: np.random.RandomState) -> pd.DataFrame:
    """Create a wide-format prices DataFrame (DatetimeIndex x symbol columns)."""
    data = rng.uniform(50.0, 200.0, size=(len(dates), len(symbols)))
    return pd.DataFrame(data, index=dates, columns=symbols)


def _make_long_prices(symbols: list[str], dates: pd.DatetimeIndex, rng: np.random.RandomState) -> pd.DataFrame:
    """Create a long-format prices DataFrame with date, symbol, close columns."""
    rows = []
    for d in dates:
        for s in symbols:
            rows.append({"date": d, "symbol": s, "close": rng.uniform(50.0, 200.0)})
    return pd.DataFrame(rows)


def _write_wide_parquet(path, symbols, dates, rng):
    """Write a wide-format parquet and return the DataFrame used."""
    df = _make_wide_prices(symbols, dates, rng)
    # Store date as a column so the reader can detect it
    df.index.name = "date"
    df.to_parquet(str(path))
    return df


def _write_long_parquet(path, symbols, dates, rng):
    """Write a long-format parquet and return the DataFrame used."""
    df = _make_long_prices(symbols, dates, rng)
    df.to_parquet(str(path), index=False)
    return df


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestLocalFileDataPlugin:
    """Tests for LocalFileDataPlugin covering wide/long loading, date filtering,
    optional keys, missing files, empty files, type preservation, and DuckDB path."""

    # -- Fixtures ----------------------------------------------------------

    @pytest.fixture
    def rng(self):
        return np.random.RandomState(42)

    @pytest.fixture
    def symbols(self):
        return ["BTC", "ETH", "SOL"]

    @pytest.fixture
    def dates(self):
        return pd.date_range("2026-01-01", periods=30, freq="D")

    @pytest.fixture
    def wide_data_dir(self, tmp_path, symbols, dates, rng):
        """Create a temp directory with prices.parquet and volume.parquet in wide format."""
        _write_wide_parquet(tmp_path / "prices.parquet", symbols, dates, rng)
        _write_wide_parquet(tmp_path / "volume.parquet", symbols, dates, rng)
        _write_wide_parquet(tmp_path / "market_cap.parquet", symbols, dates, rng)
        return tmp_path

    @pytest.fixture
    def plugin_with_paths(self, wide_data_dir):
        """Plugin initialized with paths pointing to test parquet files."""
        return LocalFileDataPlugin(
            prices_path=str(wide_data_dir / "prices.parquet"),
            volume_path=str(wide_data_dir / "volume.parquet"),
            market_cap_path=str(wide_data_dir / "market_cap.parquet"),
        )

    @pytest.fixture
    def universe(self, symbols):
        return pd.DataFrame({"symbol": symbols})

    # -- 1. Wide-format parquet loading ------------------------------------

    def test_load_wide_format_returns_dict(self, plugin_with_paths, universe):
        """load_market_data returns Dict[str, DataFrame] when given wide-format parquet."""
        result = plugin_with_paths.load_market_data(universe, "2026-01-30", {})
        assert isinstance(result, dict)
        assert isinstance(result["prices"], pd.DataFrame)
        assert not result["prices"].empty

    def test_load_wide_format_correct_symbols(self, plugin_with_paths, universe, symbols):
        """Wide-format output has the expected symbol columns."""
        result = plugin_with_paths.load_market_data(universe, "2026-01-30", {})
        prices = result["prices"]
        for sym in symbols:
            assert sym in prices.columns, f"Expected column {sym} in prices DataFrame"

    # -- 2. Date filtering (asof) ------------------------------------------

    def test_date_filtering_via_asof(self, plugin_with_paths, universe):
        """Data should be filtered up to the asof date."""
        asof = "2026-01-15"
        result = plugin_with_paths.load_market_data(universe, asof, {})
        prices = result["prices"]
        assert not prices.empty
        assert prices.index.max() <= pd.Timestamp(asof)

    def test_date_filtering_start_and_end_via_read_file(self, tmp_path, symbols, dates, rng):
        """_read_file with asof parameter limits rows to <= asof."""
        path = tmp_path / "prices.parquet"
        _write_wide_parquet(path, symbols, dates, rng)
        df = _read_file(str(path), asof="2026-01-10")
        assert not df.empty
        assert df.index.max() <= pd.Timestamp("2026-01-10")

    # -- 3. prices key always present --------------------------------------

    def test_prices_key_always_present(self, tmp_path, rng, symbols, dates):
        """Even with minimal config, 'prices' key is in the output dict."""
        path = tmp_path / "prices.parquet"
        _write_wide_parquet(path, symbols, dates, rng)
        plugin = LocalFileDataPlugin(prices_path=str(path))
        universe = pd.DataFrame({"symbol": symbols})
        result = plugin.load_market_data(universe, "2026-01-30", {})
        assert "prices" in result
        assert not result["prices"].empty

    def test_prices_key_present_even_without_path(self):
        """When no prices_path is set, 'prices' is still in result (as empty DF)."""
        plugin = LocalFileDataPlugin()
        universe = pd.DataFrame({"symbol": ["X"]})
        result = plugin.load_market_data(universe, "2026-01-30", {})
        assert "prices" in result
        assert result["prices"].empty

    # -- 4. Optional keys: volume, market_cap ------------------------------

    def test_optional_volume_loaded(self, plugin_with_paths, universe):
        """Volume DataFrame is loaded when volume_path is provided."""
        result = plugin_with_paths.load_market_data(universe, "2026-01-30", {})
        assert "volume" in result
        assert not result["volume"].empty

    def test_optional_market_cap_loaded(self, plugin_with_paths, universe):
        """Market cap DataFrame is loaded when market_cap_path is provided."""
        result = plugin_with_paths.load_market_data(universe, "2026-01-30", {})
        assert "market_cap" in result
        assert not result["market_cap"].empty

    def test_funding_rates_loaded_when_path_set(self, tmp_path, symbols, dates, rng):
        """Funding rates are loaded when funding_rates_path is set."""
        path = tmp_path / "funding.parquet"
        _write_wide_parquet(path, symbols, dates, rng)
        prices_path = tmp_path / "prices.parquet"
        _write_wide_parquet(prices_path, symbols, dates, rng)
        plugin = LocalFileDataPlugin(
            prices_path=str(prices_path),
            funding_rates_path=str(path),
        )
        universe = pd.DataFrame({"symbol": symbols})
        result = plugin.load_market_data(universe, "2026-01-30", {})
        assert "funding_rates" in result
        assert not result["funding_rates"].empty

    # -- 5. Missing files: graceful handling -------------------------------

    def test_missing_volume_returns_empty_df(self, tmp_path, symbols, dates, rng):
        """When volume.parquet does not exist, result['volume'] is an empty DataFrame."""
        prices_path = tmp_path / "prices.parquet"
        _write_wide_parquet(prices_path, symbols, dates, rng)
        plugin = LocalFileDataPlugin(
            prices_path=str(prices_path),
            volume_path=str(tmp_path / "nonexistent_volume.parquet"),
        )
        universe = pd.DataFrame({"symbol": symbols})
        result = plugin.load_market_data(universe, "2026-01-30", {})
        assert "volume" in result
        assert result["volume"].empty

    def test_missing_file_read_file_returns_empty(self, tmp_path):
        """_read_file returns empty DataFrame for non-existent path."""
        df = _read_file(str(tmp_path / "does_not_exist.parquet"))
        assert isinstance(df, pd.DataFrame)
        assert df.empty

    # -- 6. Long-format parquet loading ------------------------------------

    def test_long_format_pivoted_to_wide(self, tmp_path, symbols, dates, rng):
        """Long-format data (date, symbol, close) is auto-pivoted to wide format."""
        path = tmp_path / "prices_long.parquet"
        _write_long_parquet(path, symbols, dates, rng)
        df = _read_file(str(path))
        # After pivot, columns should be symbols, index should be DatetimeIndex
        assert isinstance(df.index, pd.DatetimeIndex)
        for sym in symbols:
            assert sym in df.columns

    def test_long_format_via_plugin(self, tmp_path, symbols, dates, rng):
        """Plugin correctly loads long-format parquet and returns wide dict."""
        path = tmp_path / "prices_long.parquet"
        _write_long_parquet(path, symbols, dates, rng)
        plugin = LocalFileDataPlugin(prices_path=str(path))
        universe = pd.DataFrame({"symbol": symbols})
        result = plugin.load_market_data(universe, "2026-01-30", {})
        prices = result["prices"]
        assert isinstance(prices.index, pd.DatetimeIndex)
        assert set(symbols).issubset(set(prices.columns))

    def test_pivot_long_to_wide_function(self, symbols, dates, rng):
        """_pivot_long_to_wide correctly pivots a long DataFrame."""
        long_df = _make_long_prices(symbols, dates, rng)
        wide = _pivot_long_to_wide(long_df)
        assert "date" in wide.columns
        for sym in symbols:
            assert sym in wide.columns

    # -- 7. Empty parquet file handling ------------------------------------

    def test_empty_parquet_returns_empty_df(self, tmp_path):
        """An empty parquet file results in an empty DataFrame without errors."""
        path = tmp_path / "empty.parquet"
        pd.DataFrame().to_parquet(str(path))
        df = _read_file(str(path))
        assert isinstance(df, pd.DataFrame)
        assert df.empty

    def test_empty_prices_in_plugin(self, tmp_path):
        """Plugin handles an empty prices parquet gracefully."""
        path = tmp_path / "empty_prices.parquet"
        pd.DataFrame().to_parquet(str(path))
        plugin = LocalFileDataPlugin(prices_path=str(path))
        universe = pd.DataFrame({"symbol": ["BTC"]})
        result = plugin.load_market_data(universe, "2026-01-30", {})
        assert result["prices"].empty

    # -- 8. Column type preservation (DatetimeIndex, float values) ---------

    def test_datetime_index_type(self, plugin_with_paths, universe):
        """Prices DataFrame index is a DatetimeIndex."""
        result = plugin_with_paths.load_market_data(universe, "2026-01-30", {})
        prices = result["prices"]
        assert isinstance(prices.index, pd.DatetimeIndex)

    def test_float_values_preserved(self, plugin_with_paths, universe, symbols):
        """Price values remain as float dtype."""
        result = plugin_with_paths.load_market_data(universe, "2026-01-30", {})
        prices = result["prices"]
        for sym in symbols:
            assert prices[sym].dtype in (np.float64, np.float32, float), (
                f"Expected float dtype for {sym}, got {prices[sym].dtype}"
            )

    # -- 9. Multiple symbols in wide format --------------------------------

    def test_many_symbols_wide(self, tmp_path, dates, rng):
        """Loading wide-format with many symbols preserves all columns."""
        many_symbols = [f"TOKEN_{i}" for i in range(20)]
        path = tmp_path / "prices.parquet"
        _write_wide_parquet(path, many_symbols, dates, rng)
        plugin = LocalFileDataPlugin(prices_path=str(path))
        universe = pd.DataFrame({"symbol": many_symbols})
        result = plugin.load_market_data(universe, "2026-01-30", {})
        prices = result["prices"]
        assert len(prices.columns) == 20
        for sym in many_symbols:
            assert sym in prices.columns

    def test_multiple_symbols_row_count(self, tmp_path, dates, rng):
        """Each symbol column has the same number of rows equal to the date range."""
        syms = ["A", "B", "C", "D", "E"]
        path = tmp_path / "prices.parquet"
        _write_wide_parquet(path, syms, dates, rng)
        df = _read_file(str(path))
        assert len(df) == len(dates)
        for s in syms:
            assert df[s].notna().sum() == len(dates)

    # -- 10. DuckDB-based loading path -------------------------------------

    @pytest.mark.skipif(not DUCKDB_AVAILABLE, reason="duckdb not installed")
    def test_duckdb_read_path_used(self, tmp_path, symbols, dates, rng):
        """When duckdb is available, _read_file uses the DuckDB path successfully."""
        path = tmp_path / "prices.parquet"
        _write_wide_parquet(path, symbols, dates, rng)
        # This exercises _read_via_duckdb under the hood
        df = _read_file(str(path), asof="2026-01-20")
        assert not df.empty
        assert isinstance(df.index, pd.DatetimeIndex)
        assert df.index.max() <= pd.Timestamp("2026-01-20")

    @pytest.mark.skipif(not DUCKDB_AVAILABLE, reason="duckdb not installed")
    def test_duckdb_long_format_with_symbol_filter(self, tmp_path, symbols, dates, rng):
        """DuckDB path handles long-format data with symbol filtering."""
        path = tmp_path / "prices_long.parquet"
        _write_long_parquet(path, symbols, dates, rng)
        df = _read_file(str(path), asof="2026-01-30", symbols=["BTC", "ETH"])
        assert not df.empty
        # After pivot, only filtered symbols should remain
        assert "BTC" in df.columns
        assert "ETH" in df.columns
        assert "SOL" not in df.columns

    @pytest.mark.skipif(not DUCKDB_AVAILABLE, reason="duckdb not installed")
    def test_duckdb_csv_loading(self, tmp_path, symbols, dates, rng):
        """DuckDB path can also read CSV files."""
        path = tmp_path / "prices.csv"
        df_src = _make_wide_prices(symbols, dates, rng)
        df_src.index.name = "date"
        df_src.to_csv(str(path))
        df = _read_file(str(path))
        assert not df.empty
        assert isinstance(df.index, pd.DatetimeIndex)
        for sym in symbols:
            assert sym in df.columns

    # -- Bonus: params override paths on plugin ----------------------------

    def test_params_override_prices_path(self, tmp_path, symbols, dates, rng):
        """params['prices_path'] overrides the instance prices_path."""
        # Write two different files
        path_a = tmp_path / "prices_a.parquet"
        path_b = tmp_path / "prices_b.parquet"
        df_a = _make_wide_prices(symbols, dates[:10], rng)
        df_a.index.name = "date"
        df_a.to_parquet(str(path_a))
        df_b = _make_wide_prices(symbols, dates, rng)
        df_b.index.name = "date"
        df_b.to_parquet(str(path_b))

        plugin = LocalFileDataPlugin(prices_path=str(path_a))
        universe = pd.DataFrame({"symbol": symbols})

        # Without override: uses path_a (10 rows)
        result_a = plugin.load_market_data(universe, "2026-01-30", {})
        assert len(result_a["prices"]) == 10

        # With override: uses path_b (30 rows)
        result_b = plugin.load_market_data(universe, "2026-01-30", {"prices_path": str(path_b)})
        assert len(result_b["prices"]) == 30

    # -- Bonus: load_universe ----------------------------------------------

    def test_load_universe_from_prices(self, tmp_path, symbols, dates, rng):
        """load_universe extracts symbols from prices file when no universe_path set."""
        path = tmp_path / "prices.parquet"
        _write_wide_parquet(path, symbols, dates, rng)
        plugin = LocalFileDataPlugin(prices_path=str(path))
        univ = plugin.load_universe({})
        assert set(univ["symbol"].tolist()) == set(symbols)

    def test_load_universe_from_explicit_symbols(self):
        """load_universe returns explicit symbols from params."""
        plugin = LocalFileDataPlugin()
        univ = plugin.load_universe({"symbols": ["X", "Y", "Z"]})
        assert univ["symbol"].tolist() == ["X", "Y", "Z"]

    # -- Bonus: PluginMeta -------------------------------------------------

    def test_meta_name(self):
        assert LocalFileDataPlugin.meta.name == "local_file_data"

    def test_meta_kind(self):
        assert LocalFileDataPlugin.meta.kind == "data"

    def test_meta_outputs(self):
        assert "prices" in LocalFileDataPlugin.meta.outputs
        assert "volume" in LocalFileDataPlugin.meta.outputs
        assert "market_cap" in LocalFileDataPlugin.meta.outputs
        assert "funding_rates" in LocalFileDataPlugin.meta.outputs
