"""Lightweight pandas-native schema validation for data contracts.

Coexists with existing ``schemas/*.schema.json`` files and ``validate_ohlcv()``
in the datasources utils — this module provides reusable Python definitions.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class SchemaField:
    """Validates a single DataFrame column (Series)."""

    name: str
    dtype: str  # "number", "string", "datetime", "bool"
    required: bool = True
    nullable: bool = False
    min_value: float | None = None
    max_value: float | None = None
    description: str = ""

    _DTYPE_CHECKS = {
        "number": lambda s: pd.api.types.is_numeric_dtype(s),
        "string": lambda s: pd.api.types.is_string_dtype(s) or pd.api.types.is_object_dtype(s),
        "datetime": lambda s: pd.api.types.is_datetime64_any_dtype(s),
        "bool": lambda s: pd.api.types.is_bool_dtype(s),
    }

    def validate(self, series: pd.Series) -> list[str]:
        errors: list[str] = []
        check = self._DTYPE_CHECKS.get(self.dtype)
        if check and not check(series):
            errors.append(f"Column '{self.name}': expected dtype '{self.dtype}', got '{series.dtype}'")
            return errors  # skip further checks if type is wrong
        if not self.nullable and series.isna().any():
            n_null = int(series.isna().sum())
            errors.append(f"Column '{self.name}': {n_null} null values (nullable=False)")
        if self.min_value is not None and pd.api.types.is_numeric_dtype(series):
            below = series.dropna() < self.min_value
            if below.any():
                errors.append(f"Column '{self.name}': {int(below.sum())} values below min_value={self.min_value}")
        if self.max_value is not None and pd.api.types.is_numeric_dtype(series):
            above = series.dropna() > self.max_value
            if above.any():
                errors.append(f"Column '{self.name}': {int(above.sum())} values above max_value={self.max_value}")
        return errors


@dataclass(frozen=True)
class DataFrameSchema:
    """Validates a DataFrame against a set of field definitions."""

    name: str
    fields: tuple  # tuple of SchemaField
    allow_extra_columns: bool = True

    def validate(self, df: pd.DataFrame) -> list[str]:
        errors: list[str] = []
        columns = set(df.columns)
        for f in self.fields:
            if f.name not in columns:
                if f.required:
                    errors.append(f"Missing required column: '{f.name}'")
                continue
            errors.extend(f.validate(df[f.name]))
        if not self.allow_extra_columns:
            expected = {f.name for f in self.fields}
            extra = columns - expected
            if extra:
                errors.append(f"Unexpected columns: {sorted(extra)}")
        return errors


def validate_wide_prices(df: pd.DataFrame) -> list[str]:
    """Validate a wide-format prices DataFrame (DatetimeIndex x symbol columns)."""
    errors: list[str] = []
    if not isinstance(df.index, pd.DatetimeIndex):
        errors.append(f"Expected DatetimeIndex, got {type(df.index).__name__}")
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            errors.append(f"Column '{col}' is not numeric (dtype={df[col].dtype})")
        else:
            neg = (df[col].dropna() < 0).sum()
            if neg > 0:
                errors.append(f"Column '{col}' has {neg} negative values")
    return errors


# ── Built-in Schema Instances ───────────────────────────────

OHLCV_SCHEMA = DataFrameSchema(
    name="ohlcv",
    fields=(
        SchemaField("date", "datetime", required=True),
        SchemaField("open", "number", required=True, min_value=0),
        SchemaField("high", "number", required=True, min_value=0),
        SchemaField("low", "number", required=True, min_value=0),
        SchemaField("close", "number", required=True, min_value=0),
        SchemaField("volume", "number", required=True, min_value=0),
    ),
)

STRATEGY_WEIGHTS_SCHEMA = DataFrameSchema(
    name="strategy_weights",
    fields=(
        SchemaField("strategy", "string", required=True),
        SchemaField("symbol", "string", required=True),
        SchemaField("weight", "number", required=True),
    ),
)

AGGREGATED_WEIGHTS_SCHEMA = DataFrameSchema(
    name="aggregated_weights",
    fields=(
        SchemaField("symbol", "string", required=True),
        SchemaField("weight", "number", required=True),
    ),
)

ORDERS_SCHEMA = DataFrameSchema(
    name="orders",
    fields=(
        SchemaField("symbol", "string", required=True),
        SchemaField("side", "string", required=True),
        SchemaField("qty", "number", required=True, min_value=0),
        SchemaField("order_type", "string", required=True),
    ),
)

FILLS_SCHEMA = DataFrameSchema(
    name="fills",
    fields=(
        SchemaField("symbol", "string", required=True),
        SchemaField("side", "string", required=True),
        SchemaField("qty", "number", required=True, min_value=0),
        SchemaField("price", "number", required=True, min_value=0),
        SchemaField("fee", "number", required=True, min_value=0),
    ),
)

PORTFOLIO_DAILY_SCHEMA = DataFrameSchema(
    name="portfolio_daily",
    fields=(
        SchemaField("date", "datetime", required=True),
        SchemaField("value", "number", required=True),
        SchemaField("cash", "number", required=True),
    ),
)

SCHEMA_REGISTRY: dict[str, DataFrameSchema] = {
    "ohlcv": OHLCV_SCHEMA,
    "strategy_weights": STRATEGY_WEIGHTS_SCHEMA,
    "aggregated_weights": AGGREGATED_WEIGHTS_SCHEMA,
    "orders": ORDERS_SCHEMA,
    "fills": FILLS_SCHEMA,
    "portfolio_daily": PORTFOLIO_DAILY_SCHEMA,
}
