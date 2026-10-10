#!/usr/bin/env bash
# The CLIENT SMOKE: prove a base `quantbox` install (no extras) is the client
# install ADR-0010 promises (TOM-1451; it grew out of TOM-1334's no-vectorbt
# check). Builds a CLEAN venv from this checkout — no lockfile, no dev group,
# no extras — then:
#   1. asserts no library an extra declares is installed (vectorbt, numba,
#      ccxt, duckdb, httpx, plotly, requests, ...; otherwise nothing is
#      proven) and the base libraries are;
#   2. imports every quantbox module: every CORE module imports; every other
#      module imports or raises MissingExtraError naming a declared extra,
#      never a bare ImportError; the vectorbt engine modules name [vectorbt];
#   3. runs `quantbox plugins list`, `plugins schema --json` and `plugins doctor`;
#   4. runs the canonical momentum backtest with engine: rsims via `quantbox run`;
#   5. asserts `engine: vectorbt` fails naming the [vectorbt] extra;
#   6. asserts a research-only command (`quantbox warehouse`) fails naming [research];
#   7. installs the client pack fixture (tests/fixtures/client_pack: one
#      quantbox.strategies and one quantbox.data entry point, core imports
#      only, in-memory data), checks its plugins are listed and its import
#      loads no upper-layer module, and runs one rsims backtest on it.
# Used by ci.yml job `no-vectorbt`; runnable locally from the repo root.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

uv venv -q -p "${PYTHON_VERSION:-3.12}" "$WORK/venv"
PY="$WORK/venv/bin/python"
QB="$WORK/venv/bin/quantbox"
uv pip install -q -p "$PY" "$ROOT"

# The layer of each module (ADR-0010) comes from its one owner, the import-linter
# contracts in pyproject.toml, read by scripts/layer_map.py (stdlib only). A
# core module must import on a base install.
export QB_ROOT="$ROOT"

echo "== 1. extras absent, base present"
"$PY" - <<'EOF'
import importlib.util, sys
# The same list as tests/test_base_install.py NOT_IN_BASE: every library an extra
# declares, by import name. requests too: [data] and [trade] declare it, so a base
# dependency that pulls it in transitively would hide a missing-extra path.
ABSENT = (
    "vectorbt", "numba", "ccxt", "duckdb", "httpx", "plotly", "requests", "pycoingecko",
    "binance", "ib_insync", "sklearn", "arch", "matplotlib", "seaborn",
)
BASE = ("pandas", "numpy", "scipy", "pyarrow", "yaml", "jsonschema", "statsmodels", "pandas_market_calendars", "typer")
present = [m for m in ABSENT if importlib.util.find_spec(m)]
if present:
    sys.exit(f"FAIL: base install pulled {present} — an extra leaked into the base dependencies")
missing = [m for m in BASE if not importlib.util.find_spec(m)]
if missing:
    sys.exit(f"FAIL: base install lacks {missing} — a core import is not a declared base dependency")
print(f"ok: absent {', '.join(ABSENT)}; present {', '.join(BASE)}")
EOF

echo "== 2. every module imports, or names its extra"
"$PY" - <<'EOF'
import importlib, importlib.metadata, os, sys
from pathlib import Path
import quantbox
from quantbox._removed import REMOVED
from quantbox.exceptions import MissingExtraError

sys.path.insert(0, os.path.join(os.environ["QB_ROOT"], "scripts"))
from layer_map import layer_of, read_layers

layers = read_layers(Path(os.environ["QB_ROOT"]) / "pyproject.toml")

def layer(mod):
    return layer_of(mod, layers)

extras = set(importlib.metadata.metadata("quantbox").get_all("Provides-Extra") or [])
root = Path(quantbox.__file__).parent
names = []
for p in sorted(root.rglob("*.py")):
    parts = list(p.relative_to(root.parent).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    names.append(".".join(parts))
ENGINE = {"quantbox.adapters.vectorbt", "quantbox.plugins.backtesting.vectorbt_engine"}
ok, n_core, refused, bad, tombstones = 0, 0, {}, {}, []
for name in names:
    if name in REMOVED:
        # A path removed in 0.13.0 (TOM-1457): its tombstone must refuse, naming the replacement.
        try:
            importlib.import_module(name)
            bad[name] = "a removed path imported"
        except ImportError as exc:
            if "removed in quantbox" in str(exc) and not isinstance(exc, MissingExtraError):
                tombstones.append(name)
            else:
                bad[name] = f"tombstone raised the wrong error: {exc!r}"
        continue
    is_core = layer(name) == "core"
    n_core += is_core
    try:
        importlib.import_module(name)
        ok += 1
        if name in ENGINE:
            bad[name] = "imported without vectorbt"
    except MissingExtraError as exc:
        if is_core:
            bad[name] = f"core module needs an extra: {exc}"
        elif exc.extra not in extras:
            bad[name] = f"names [{exc.extra}], which pyproject.toml does not declare ({sorted(extras)})"
        elif name in ENGINE and exc.extra != "vectorbt":
            bad[name] = f"engine module names [{exc.extra}], not [vectorbt]"
        else:
            refused[name] = exc.extra
    except BaseException as exc:
        bad[name] = repr(exc)
if ok < 150 or n_core < 60:
    bad["<walk>"] = f"only {ok} modules imported, {n_core} core — the walk did not look"
if not tombstones:
    bad["<tombstones>"] = "the walk met no tombstone of a removed path"
if not ENGINE <= set(refused):
    bad["<engine>"] = f"engine modules did not refuse: {sorted(ENGINE - set(refused))}"
if bad:
    sys.exit("FAIL:\n" + "\n".join(f"  {k}: {v}" for k, v in sorted(bad.items())))
print(f"ok: {len(names)} modules; {ok} imported ({n_core} core, all of them); {len(tombstones)} removed paths refuse;")
for name, extra in sorted(refused.items()):
    print(f"    {name} -> MissingExtraError [{extra}]")
EOF

echo "== 3. CLI: plugins list, schema, doctor"
"$QB" plugins list >/dev/null
"$QB" plugins schema --json >"$WORK/schema.json"
"$PY" - "$WORK/schema.json" <<'EOF'
import json, sys
rows = json.load(open(sys.argv[1]))["plugins"]
refused = {r["id"]: r["missing_extra"] for r in rows if r.get("missing_extra")}
if len(rows) < 50:
    sys.exit(f"FAIL: plugins schema listed only {len(rows)} plugins")
print(f"ok: plugins list; plugins schema lists {len(rows)} plugins, {len(refused)} name their extra: {refused}")
EOF
(cd "$ROOT" && "$QB" plugins doctor >"$WORK/doctor.log" 2>&1) || {
    cat "$WORK/doctor.log"
    echo "FAIL: quantbox plugins doctor failed on a base install"
    exit 1
}
echo "ok: plugins doctor"

# The canonical momentum config: self-contained (bundled fixture parquet, no
# network). Engine swapped to rsims; artifacts redirected into the scratch dir.
# Fixture paths are repo-relative, so both runs execute from $ROOT.
SRC="$ROOT/cookbook/canonical/configs/momentum.yaml"
sed "s|root: ./artifacts|root: $WORK/artifacts|" "$SRC" >"$WORK/vbt.yaml"
sed 's/engine: vectorbt/engine: rsims/' "$WORK/vbt.yaml" >"$WORK/rsims.yaml"
grep -q "engine: vectorbt" "$WORK/vbt.yaml"
grep -q "engine: rsims" "$WORK/rsims.yaml"
grep -q "root: $WORK/artifacts" "$WORK/rsims.yaml"

echo "== 4. quantbox run, engine: rsims"
(cd "$ROOT" && "$QB" run -c "$WORK/rsims.yaml" >"$WORK/rsims.log" 2>&1) || {
    cat "$WORK/rsims.log"
    echo "FAIL: rsims backtest did not run on a base install"
    exit 1
}
ls "$WORK"/artifacts/*/metrics.json >/dev/null || {
    echo "FAIL: rsims run wrote no metrics.json"
    exit 1
}
echo "ok: quantbox run (engine: rsims)"

echo "== 5. engine: vectorbt names the extra"
if (cd "$ROOT" && NO_COLOR=1 COLUMNS=500 "$QB" run -c "$WORK/vbt.yaml" >"$WORK/vbt.log" 2>&1); then
    echo "FAIL: engine: vectorbt succeeded without vectorbt installed"
    exit 1
fi
if ! grep -q "quantbox\[vectorbt\]" "$WORK/vbt.log"; then
    tail -20 "$WORK/vbt.log"
    echo "FAIL: engine: vectorbt failed without naming the [vectorbt] extra"
    exit 1
fi
echo "ok: engine: vectorbt refuses, naming quantbox[vectorbt]"

echo "== 6. a research-only command names the extra"
if (cd "$WORK" && NO_COLOR=1 COLUMNS=500 "$QB" warehouse init -r "$WORK/warehouse" >"$WORK/wh.log" 2>&1); then
    echo "FAIL: quantbox warehouse succeeded without the [research] extra"
    exit 1
fi
if ! grep -q "quantbox\[research\]" "$WORK/wh.log"; then
    cat "$WORK/wh.log"
    echo "FAIL: quantbox warehouse failed without naming the [research] extra"
    exit 1
fi
if grep -q "Traceback" "$WORK/wh.log"; then
    cat "$WORK/wh.log"
    echo "FAIL: quantbox warehouse printed a traceback, not one line naming the extra"
    exit 1
fi
echo "ok: quantbox warehouse refuses, naming quantbox[research]"

echo "== 7. client pack: entry points, core-only imports, one rsims backtest"
PACK="$ROOT/tests/fixtures/client_pack"
uv pip install -q -p "$PY" "$ROOT" "$PACK"
"$PY" - <<'EOF'
import importlib.util, os, sys
from pathlib import Path
import qb_client_pack.plugins  # noqa: F401

sys.path.insert(0, os.path.join(os.environ["QB_ROOT"], "scripts"))
from layer_map import layer_of, read_layers

layers = read_layers(Path(os.environ["QB_ROOT"]) / "pyproject.toml")

def layer(mod):
    return layer_of(mod, layers)

loaded = sorted(m for m in sys.modules if m == "quantbox" or m.startswith("quantbox."))
above = [m for m in loaded if layer(m) != "core"]
if above:
    sys.exit(f"FAIL: importing the client pack loaded upper-layer modules: {above}")
leaked = [m for m in ("vectorbt", "numba", "ccxt", "duckdb", "httpx", "plotly") if importlib.util.find_spec(m)]
if leaked:
    sys.exit(f"FAIL: installing the client pack pulled {leaked}")
print(f"ok: the pack imports {len(loaded)} quantbox modules, all core")
EOF
"$QB" plugins list --json >"$WORK/plugins.json"
for id in client_pack.top_momentum.v1 client_pack.synthetic_data.v1; do
    grep -q "\"$id\"" "$WORK/plugins.json" || {
        echo "FAIL: quantbox plugins list does not show the pack's $id"
        exit 1
    }
done
echo "ok: plugins list shows client_pack.top_momentum.v1 and client_pack.synthetic_data.v1"
sed "s|root: ./artifacts|root: $WORK/pack-artifacts|" "$PACK/run.yaml" >"$WORK/pack.yaml"
grep -q "root: $WORK/pack-artifacts" "$WORK/pack.yaml"
grep -q "engine: rsims" "$WORK/pack.yaml"
(cd "$WORK" && "$QB" run -c "$WORK/pack.yaml" >"$WORK/pack.log" 2>&1) || {
    cat "$WORK/pack.log"
    echo "FAIL: the client pack's rsims backtest did not run on a base install"
    exit 1
}
ls "$WORK"/pack-artifacts/*/metrics.json >/dev/null || {
    echo "FAIL: the client pack's run wrote no metrics.json"
    exit 1
}
echo "ok: quantbox run on the client pack (engine: rsims)"
echo "PASS: base quantbox is the client install (no library an extra declares)"
