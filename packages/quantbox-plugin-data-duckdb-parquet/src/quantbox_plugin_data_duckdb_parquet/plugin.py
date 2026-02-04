from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional
import pandas as pd
import duckdb

from quantbox.contracts import PluginMeta

@dataclass
class DuckDBParquetData:
    prices_path: str = "./data/curated/prices.parquet"
    fx_path: str | None = None
    meta = PluginMeta(
        name="eod.duckdb_parquet.v1",
        kind="data",
        version="0.1.0",
        core_compat=">=0.1,<0.2",
        description="DuckDB over Parquet EOD prices",
        tags=("duckdb","parquet","eod"),
        capabilities=("prices",),
        schema_version="v1",
        params_schema={
            "type":"object",
            "properties":{
                "universe": {"type":"object","properties":{"symbols":{"type":"array","items":{"type":"string"}}},"required":["symbols"]},
                "prices": {"type":"object","properties":{"lookback_days":{"type":"integer","minimum":30}}}
            }
        },
        examples=(
            "plugins:\n  data:\n    name: eod.duckdb_parquet.v1\n    params_init:\n      prices_path: ./data/curated/prices.parquet",
        ),
    )

    def load_universe(self, params: Dict[str, Any]) -> pd.DataFrame:
        symbols = params.get("symbols")
        if not symbols:
            raise ValueError("Universe params must include 'symbols' list.")
        return pd.DataFrame({"symbol": list(symbols)})

    def load_prices(self, universe: pd.DataFrame, asof: str, params: Dict[str, Any]) -> pd.DataFrame:
        lookback_days = int(params.get("lookback_days", 365*3))
        con = duckdb.connect(database=":memory:")
        con.execute("SET enable_progress_bar=false;")
        con.execute("CREATE VIEW prices AS SELECT * FROM read_parquet(?)", [self.prices_path])
        con.register("universe", universe)
        q = """
        SELECT date, symbol, close
        FROM prices
        WHERE symbol IN (SELECT symbol FROM universe)
          AND CAST(date AS DATE) <= CAST(? AS DATE)
          AND CAST(date AS DATE) >= date_add(CAST(? AS DATE), -?)
        ORDER BY symbol, date
        """
        df = con.execute(q, [asof, asof, lookback_days]).fetchdf()
        return df

    def load_fx(self, asof: str, params: Dict[str, Any]) -> Optional[pd.DataFrame]:
        # Optional FX rates parquet with columns: date, pair, rate
        if not self.fx_path:
            return None
        con = duckdb.connect(database=':memory:')
        con.execute('SET enable_progress_bar=false;')
        con.execute('CREATE VIEW fx AS SELECT * FROM read_parquet(?)', [self.fx_path])
        q = """
        SELECT date, pair, rate
        FROM fx
        WHERE CAST(date AS DATE) <= CAST(? AS DATE)
        ORDER BY CAST(date AS DATE) DESC
        LIMIT 2000
        """
        df = con.execute(q, [asof]).fetchdf()
        return df if len(df) else None
