"""PyArrow-based warehouse schemas for structured data validation.

Provides ``WarehouseSchema`` base class and domain schemas for prices,
trades, fills, positions, and portfolio snapshots.

Adapted from quantlabnew/datalayer schemas.
"""

from __future__ import annotations

from dataclasses import dataclass

import pyarrow as pa

# ── Schema field & base class ────────────────────────────────


@dataclass
class SchemaField:
    """Definition of a single column in a warehouse schema."""

    name: str
    dtype: pa.DataType
    nullable: bool = True
    description: str = ""


class WarehouseSchema:
    """Base class for warehouse schema definitions.

    Subclasses define fields as class-level ``SchemaField`` attributes.

    Example::

        class MySchema(WarehouseSchema):
            id = SchemaField("id", pa.string(), nullable=False)
            value = SchemaField("value", pa.float64())

        errors = MySchema.validate(table)
        table = MySchema.cast(table)
    """

    @classmethod
    def fields(cls) -> list[SchemaField]:
        return [getattr(cls, name) for name in dir(cls) if isinstance(getattr(cls, name), SchemaField)]

    @classmethod
    def field_names(cls) -> list[str]:
        return [f.name for f in cls.fields()]

    @classmethod
    def to_arrow_schema(cls) -> pa.Schema:
        return pa.schema([pa.field(f.name, f.dtype, nullable=f.nullable) for f in cls.fields()])

    @classmethod
    def validate(cls, table: pa.Table) -> list[str]:
        """Validate a PyArrow Table against this schema.

        Returns a list of error strings (empty if valid).
        """
        errors: list[str] = []
        for f in cls.fields():
            if f.name not in table.column_names:
                if not f.nullable:
                    errors.append(f"Missing required column: {f.name}")
                continue
            actual = table.schema.field(f.name).type
            if not actual.equals(f.dtype) and not _types_compatible(actual, f.dtype):
                errors.append(f"Type mismatch for {f.name}: expected {f.dtype}, got {actual}")
        return errors

    @classmethod
    def cast(cls, table: pa.Table) -> pa.Table:
        """Cast table columns to match schema types."""
        schema = cls.to_arrow_schema()
        arrays = []
        for arrow_field in schema:
            if arrow_field.name in table.column_names:
                col = table.column(arrow_field.name)
                if not col.type.equals(arrow_field.type):
                    col = col.cast(arrow_field.type)
                arrays.append(col)
            elif arrow_field.nullable:
                arrays.append(pa.nulls(len(table), type=arrow_field.type))
            else:
                raise ValueError(f"Missing required column: {arrow_field.name}")
        return pa.table(dict(zip(schema.names, arrays, strict=False)))

    @classmethod
    def empty_table(cls) -> pa.Table:
        return pa.table({f.name: pa.array([], type=f.dtype) for f in cls.fields()})


def _types_compatible(actual: pa.DataType, expected: pa.DataType) -> bool:
    if pa.types.is_integer(actual) and pa.types.is_floating(expected):
        return True
    if pa.types.is_integer(actual) and pa.types.is_integer(expected):
        return True
    if (pa.types.is_date(actual) or pa.types.is_timestamp(actual)) and (
        pa.types.is_date(expected) or pa.types.is_timestamp(expected)
    ):
        return True
    # string / large_string / utf8 are interchangeable
    if pa.types.is_string(actual) and pa.types.is_string(expected):
        return True
    if pa.types.is_large_string(actual) and pa.types.is_string(expected):
        return True
    return pa.types.is_string(actual) and pa.types.is_large_string(expected)


# ── Domain schemas ────────────────────────────────────────────


class PriceBarSchema(WarehouseSchema):
    """OHLCV price bars (long format, one row per symbol per date)."""

    symbol = SchemaField("symbol", pa.string(), nullable=False, description="Trading symbol")
    date = SchemaField("date", pa.date32(), nullable=False, description="Bar date (partition key)")
    open = SchemaField("open", pa.float64(), nullable=False, description="Opening price")
    high = SchemaField("high", pa.float64(), nullable=False, description="High price")
    low = SchemaField("low", pa.float64(), nullable=False, description="Low price")
    close = SchemaField("close", pa.float64(), nullable=False, description="Closing price")
    volume = SchemaField("volume", pa.float64(), description="Trading volume")


class TradeSchema(WarehouseSchema):
    """Executed trade records."""

    trade_id = SchemaField("trade_id", pa.string(), nullable=False, description="Unique trade ID")
    symbol = SchemaField("symbol", pa.string(), nullable=False, description="Trading symbol")
    side = SchemaField("side", pa.string(), nullable=False, description="buy or sell")
    quantity = SchemaField("quantity", pa.float64(), nullable=False, description="Trade quantity")
    price = SchemaField("price", pa.float64(), nullable=False, description="Execution price")
    commission = SchemaField("commission", pa.float64(), description="Commission/fee")
    slippage = SchemaField("slippage", pa.float64(), description="Slippage vs reference")
    trade_time = SchemaField("trade_time", pa.timestamp("us"), nullable=False, description="Execution timestamp")
    strategy = SchemaField("strategy", pa.string(), description="Strategy identifier")
    date = SchemaField("date", pa.date32(), nullable=False, description="Trade date (partition key)")


class FillSchema(WarehouseSchema):
    """Individual fill/execution records."""

    fill_id = SchemaField("fill_id", pa.string(), nullable=False, description="Unique fill ID")
    order_id = SchemaField("order_id", pa.string(), nullable=False, description="Parent order ID")
    symbol = SchemaField("symbol", pa.string(), nullable=False, description="Trading symbol")
    side = SchemaField("side", pa.string(), nullable=False, description="Fill side")
    quantity = SchemaField("quantity", pa.float64(), nullable=False, description="Fill quantity")
    price = SchemaField("price", pa.float64(), nullable=False, description="Fill price")
    fill_time = SchemaField("fill_time", pa.timestamp("us"), nullable=False, description="Fill timestamp")
    date = SchemaField("date", pa.date32(), nullable=False, description="Fill date (partition key)")


class PositionSchema(WarehouseSchema):
    """Point-in-time position snapshots."""

    symbol = SchemaField("symbol", pa.string(), nullable=False, description="Trading symbol")
    quantity = SchemaField("quantity", pa.float64(), nullable=False, description="Position qty (+ long, - short)")
    avg_cost = SchemaField("avg_cost", pa.float64(), description="Average cost basis")
    market_value = SchemaField("market_value", pa.float64(), description="Current market value")
    unrealized_pnl = SchemaField("unrealized_pnl", pa.float64(), description="Unrealized P&L")
    weight = SchemaField("weight", pa.float64(), description="Portfolio weight")
    as_of = SchemaField("as_of", pa.timestamp("us"), nullable=False, description="Snapshot timestamp")
    date = SchemaField("date", pa.date32(), nullable=False, description="Position date (partition key)")


class PortfolioSnapshotSchema(WarehouseSchema):
    """Portfolio-level aggregate snapshots."""

    date = SchemaField("date", pa.date32(), nullable=False, description="Snapshot date")
    nav = SchemaField("nav", pa.float64(), nullable=False, description="Net asset value")
    cash = SchemaField("cash", pa.float64(), description="Cash balance")
    gross_exposure = SchemaField("gross_exposure", pa.float64(), description="|long| + |short|")
    net_exposure = SchemaField("net_exposure", pa.float64(), description="long - short")
    num_positions = SchemaField("num_positions", pa.int64(), description="Number of positions")
    daily_pnl = SchemaField("daily_pnl", pa.float64(), description="Daily P&L")
    daily_returns = SchemaField("daily_returns", pa.float64(), description="Daily returns")
