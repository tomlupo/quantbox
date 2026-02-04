from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import date

OUT = Path("data/curated/prices.parquet")
OUT.parent.mkdir(parents=True, exist_ok=True)

symbols = ["SPY","QQQ","IWM","EEM","TLT","GLD"]
start = date(2023, 1, 2)
end = date(2026, 1, 31)

dates = pd.bdate_range(start, end).date
rng = np.random.default_rng(42)

rows = []
for sym in symbols:
    drift = rng.normal(0.00015, 0.00005)
    vol = abs(rng.normal(0.012, 0.003))
    price = 100.0 + rng.normal(0, 1)
    for d in dates:
        r = drift + rng.normal(0, vol)
        price = max(1.0, price * (1.0 + r))
        rows.append((str(d), sym, float(price)))

df = pd.DataFrame(rows, columns=["date","symbol","close"])
df.to_parquet(OUT, index=False)
print("Wrote", OUT, "rows:", len(df))

# Optional FX sample: EURUSD
FX = Path("data/curated/fx.parquet")
FX.parent.mkdir(parents=True, exist_ok=True)
fx_dates = pd.bdate_range(start, end).date
eurusd = 1.05 + np.cumsum(rng.normal(0, 0.001, size=len(fx_dates)))
fx_df = pd.DataFrame({"date":[str(d) for d in fx_dates], "pair":["EURUSD"]*len(fx_dates), "rate":eurusd.astype(float)})
fx_df.to_parquet(FX, index=False)
print("Wrote", FX, "rows:", len(fx_df))

