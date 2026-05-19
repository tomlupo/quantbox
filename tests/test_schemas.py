"""Tests for schema validation."""

import pandas as pd

from quantbox.schemas import (
    AGGREGATED_WEIGHTS_SCHEMA,
    FILLS_SCHEMA,
    OHLCV_SCHEMA,
    ORDERS_SCHEMA,
    PORTFOLIO_DAILY_SCHEMA,
    SCHEMA_REGISTRY,
    STRATEGY_WEIGHTS_SCHEMA,
    DataFrameSchema,
    SchemaField,
    validate_wide_prices,
)


class TestSchemaField:
    def test_valid_number(self):
        f = SchemaField("price", "number", min_value=0)
        assert f.validate(pd.Series([1.0, 2.0, 3.0])) == []

    def test_wrong_dtype(self):
        f = SchemaField("price", "number")
        errors = f.validate(pd.Series(["a", "b"]))
        assert len(errors) == 1
        assert "expected dtype" in errors[0]

    def test_nulls_not_allowed(self):
        f = SchemaField("price", "number", nullable=False)
        errors = f.validate(pd.Series([1.0, float("nan"), 3.0]))
        assert len(errors) == 1
        assert "null" in errors[0]

    def test_nulls_allowed(self):
        f = SchemaField("price", "number", nullable=True)
        assert f.validate(pd.Series([1.0, float("nan")])) == []

    def test_min_value(self):
        f = SchemaField("price", "number", min_value=0)
        errors = f.validate(pd.Series([1.0, -0.5, 2.0]))
        assert len(errors) == 1
        assert "below min_value" in errors[0]

    def test_max_value(self):
        f = SchemaField("price", "number", max_value=10)
        errors = f.validate(pd.Series([5.0, 15.0]))
        assert len(errors) == 1
        assert "above max_value" in errors[0]

    def test_string_field(self):
        f = SchemaField("name", "string")
        assert f.validate(pd.Series(["BTC", "ETH"])) == []

    def test_datetime_field(self):
        f = SchemaField("date", "datetime")
        s = pd.Series(pd.to_datetime(["2026-01-01", "2026-01-02"]))
        assert f.validate(s) == []


class TestDataFrameSchema:
    def test_valid_df(self):
        schema = DataFrameSchema(
            "test",
            fields=(
                SchemaField("symbol", "string"),
                SchemaField("weight", "number"),
            ),
        )
        df = pd.DataFrame({"symbol": ["BTC"], "weight": [0.5]})
        assert schema.validate(df) == []

    def test_missing_required(self):
        schema = DataFrameSchema(
            "test",
            fields=(SchemaField("symbol", "string", required=True),),
        )
        df = pd.DataFrame({"other": [1]})
        errors = schema.validate(df)
        assert any("Missing required" in e for e in errors)

    def test_optional_missing(self):
        schema = DataFrameSchema(
            "test",
            fields=(SchemaField("symbol", "string", required=False),),
        )
        df = pd.DataFrame({"other": [1]})
        assert schema.validate(df) == []

    def test_extra_columns_allowed(self):
        schema = DataFrameSchema(
            "test",
            fields=(SchemaField("symbol", "string"),),
            allow_extra_columns=True,
        )
        df = pd.DataFrame({"symbol": ["BTC"], "extra": [1]})
        assert schema.validate(df) == []

    def test_extra_columns_rejected(self):
        schema = DataFrameSchema(
            "test",
            fields=(SchemaField("symbol", "string"),),
            allow_extra_columns=False,
        )
        df = pd.DataFrame({"symbol": ["BTC"], "extra": [1]})
        errors = schema.validate(df)
        assert any("Unexpected columns" in e for e in errors)


class TestValidateWidePrices:
    def test_valid(self):
        dates = pd.date_range("2026-01-01", periods=5)
        df = pd.DataFrame(
            {"BTC": [100, 101, 102, 103, 104], "ETH": [50, 51, 52, 53, 54]},
            index=dates,
        )
        assert validate_wide_prices(df) == []

    def test_no_datetime_index(self):
        df = pd.DataFrame({"BTC": [100, 101]}, index=[0, 1])
        errors = validate_wide_prices(df)
        assert any("DatetimeIndex" in e for e in errors)

    def test_non_numeric_column(self):
        dates = pd.date_range("2026-01-01", periods=2)
        df = pd.DataFrame({"BTC": ["a", "b"]}, index=dates)
        errors = validate_wide_prices(df)
        assert any("not numeric" in e for e in errors)

    def test_negative_values(self):
        dates = pd.date_range("2026-01-01", periods=2)
        df = pd.DataFrame({"BTC": [100, -1]}, index=dates)
        errors = validate_wide_prices(df)
        assert any("negative" in e for e in errors)


class TestBuiltinSchemas:
    def test_ohlcv_valid(self):
        df = pd.DataFrame(
            {
                "date": pd.to_datetime(["2026-01-01"]),
                "open": [100.0],
                "high": [105.0],
                "low": [95.0],
                "close": [102.0],
                "volume": [1000.0],
            }
        )
        assert OHLCV_SCHEMA.validate(df) == []

    def test_strategy_weights_valid(self):
        df = pd.DataFrame(
            {
                "strategy": ["momentum"],
                "symbol": ["BTC"],
                "weight": [0.5],
            }
        )
        assert STRATEGY_WEIGHTS_SCHEMA.validate(df) == []

    def test_aggregated_weights_valid(self):
        df = pd.DataFrame({"symbol": ["BTC", "ETH"], "weight": [0.6, 0.4]})
        assert AGGREGATED_WEIGHTS_SCHEMA.validate(df) == []

    def test_orders_valid(self):
        df = pd.DataFrame(
            {
                "symbol": ["BTC"],
                "side": ["buy"],
                "qty": [0.1],
                "order_type": ["market"],
            }
        )
        assert ORDERS_SCHEMA.validate(df) == []

    def test_fills_valid(self):
        df = pd.DataFrame(
            {
                "symbol": ["BTC"],
                "side": ["buy"],
                "qty": [0.1],
                "price": [50000.0],
                "fee": [5.0],
            }
        )
        assert FILLS_SCHEMA.validate(df) == []

    def test_portfolio_daily_valid(self):
        df = pd.DataFrame(
            {
                "date": pd.to_datetime(["2026-01-01"]),
                "value": [100000.0],
                "cash": [10000.0],
            }
        )
        assert PORTFOLIO_DAILY_SCHEMA.validate(df) == []

    def test_registry_has_all(self):
        assert set(SCHEMA_REGISTRY.keys()) == {
            "ohlcv",
            "strategy_weights",
            "aggregated_weights",
            "orders",
            "fills",
            "portfolio_daily",
        }
