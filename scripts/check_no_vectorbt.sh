#!/usr/bin/env bash
# Prove a base `quantbox` install (no extras) imports and runs without vectorbt
# (TOM-1334). Builds a CLEAN venv from this checkout — no lockfile, no dev group,
# no extras — then:
#   1. asserts vectorbt and numba are NOT installed (otherwise nothing is proven);
#   2. imports every quantbox module except the two engine modules;
#   3. runs `quantbox plugins list` and an rsims backtest via `quantbox run`;
#   4. asserts `engine: vectorbt` fails naming the [vectorbt] extra.
# Used by ci.yml job `no-vectorbt`; runnable locally from the repo root.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

uv venv -q -p "${PYTHON_VERSION:-3.12}" "$WORK/venv"
PY="$WORK/venv/bin/python"
QB="$WORK/venv/bin/quantbox"
uv pip install -q -p "$PY" "$ROOT"

echo "== 1. extra absent"
"$PY" - <<'EOF'
import importlib.util, sys
present = [m for m in ("vectorbt", "numba") if importlib.util.find_spec(m)]
if present:
    sys.exit(f"FAIL: base install pulled {present} — the [vectorbt] extra leaked into base deps")
assert importlib.util.find_spec("plotly"), "FAIL: plotly must be a declared base dependency"
print("ok: vectorbt/numba absent, plotly present")
EOF

echo "== 2. every non-engine module imports"
"$PY" - <<'EOF'
import importlib, pkgutil, sys
import quantbox
from quantbox.exceptions import MissingExtraError
ENGINE = {"quantbox.adapters.vectorbt", "quantbox.plugins.backtesting.vectorbt_engine"}
ok, bad = 0, {}
for info in pkgutil.walk_packages(quantbox.__path__, "quantbox."):
    try:
        importlib.import_module(info.name)
        ok += 1
    except MissingExtraError as exc:
        if info.name not in ENGINE:
            bad[info.name] = repr(exc)
    except BaseException as exc:
        bad[info.name] = repr(exc)
for m in ENGINE:
    try:
        importlib.import_module(m)
        bad[m] = "imported without vectorbt"
    except MissingExtraError as exc:
        assert "quantbox[vectorbt]" in str(exc), exc
if ok < 100:
    bad["<walk>"] = f"only {ok} modules imported — the walk did not look"
if bad:
    sys.exit(f"FAIL: {bad}")
print(f"ok: {ok} modules imported, engine modules refuse with the extra named")
EOF

echo "== 3. CLI"
"$QB" plugins list >/dev/null
echo "ok: quantbox plugins list"

# The canonical momentum config: self-contained (bundled fixture parquet, no
# network). Engine swapped to rsims; artifacts redirected into the scratch dir.
# Fixture paths are repo-relative, so both runs execute from $ROOT.
SRC="$ROOT/cookbook/canonical/configs/momentum.yaml"
sed "s|root: ./artifacts|root: $WORK/artifacts|" "$SRC" >"$WORK/vbt.yaml"
sed 's/engine: vectorbt/engine: rsims/' "$WORK/vbt.yaml" >"$WORK/rsims.yaml"
grep -q "engine: vectorbt" "$WORK/vbt.yaml"
grep -q "engine: rsims" "$WORK/rsims.yaml"
grep -q "root: $WORK/artifacts" "$WORK/rsims.yaml"
(cd "$ROOT" && "$QB" run -c "$WORK/rsims.yaml" >"$WORK/rsims.log" 2>&1) || {
    cat "$WORK/rsims.log"
    echo "FAIL: rsims backtest did not run without vectorbt"
    exit 1
}
ls "$WORK"/artifacts/*/metrics.json >/dev/null || {
    echo "FAIL: rsims run wrote no metrics.json"
    exit 1
}
echo "ok: quantbox run (engine: rsims)"

echo "== 4. engine: vectorbt names the extra"
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
echo "PASS: base quantbox runs without vectorbt"
