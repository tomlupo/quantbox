"""A base install (no extras) imports, lists plugins and names every missing extra (TOM-1451).

The dev env carries every extra, so each check runs in a SUBPROCESS whose
import system refuses the libraries the base install leaves out (the harness of
``test_without_vectorbt``, which first proves the block is live). The clean-venv
CI job (``ci.yml`` -> ``no-vectorbt``, ``scripts/check_no_vectorbt.sh``) proves
the real install.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from test_without_vectorbt import _run

# Every library pyproject.toml keeps out of the base install, by import name.
NOT_IN_BASE = (
    "vectorbt",
    "numba",
    "ccxt",
    "duckdb",
    "httpx",
    "plotly",
    "requests",
    "pycoingecko",
    "binance",
    "ib_insync",
    "sklearn",
    "arch",
    "matplotlib",
    "seaborn",
)

# The modules that refuse to import on a base install, and the extra each names.
# Exact on purpose: a module joining this list (above all a core one) must be a
# deliberate change to this table, not a silent one.
REFUSES_ON_BASE = {
    "quantbox.adapters.vectorbt": "vectorbt",
    "quantbox.plugins.backtesting.vectorbt_engine": "vectorbt",
    "quantbox.plugins.datasources.binance_data": "data",
    "quantbox.plugins.datasources.binance_data_plugin": "data",
    "quantbox.plugins.datasources.hyperliquid_cached_data_plugin": "data",
    "quantbox.plugins.datasources.hyperliquid_data_plugin": "data",
    "quantbox.plugins.publisher.telegram": "trade",
}


def _last_json(proc) -> dict:
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_every_module_imports_or_names_a_declared_extra():
    result = _last_json(
        _run(
            """
            import importlib, importlib.metadata, json
            from pathlib import Path
            import quantbox
            from quantbox._removed import REMOVED
            from quantbox.exceptions import MissingExtraError

            root = Path(quantbox.__file__).parent
            names = []
            for p in sorted(root.rglob("*.py")):
                parts = list(p.relative_to(root.parent).with_suffix("").parts)
                if parts[-1] == "__init__":
                    parts = parts[:-1]
                names.append(".".join(parts))
            ok, refused, failed, tombstones = [], {}, {}, []
            for name in names:
                try:
                    importlib.import_module(name)
                    ok.append(name)
                except MissingExtraError as exc:
                    refused[name] = exc.extra
                except ImportError as exc:
                    # A tombstone of a path removed in 0.13.0 (TOM-1457) refuses by design.
                    if name in REMOVED and "removed in quantbox" in str(exc):
                        tombstones.append(name)
                    else:
                        failed[name] = f"{type(exc).__name__}: {exc}"
                except BaseException as exc:  # noqa: BLE001 - report every other failure
                    failed[name] = f"{type(exc).__name__}: {exc}"
            extras = importlib.metadata.metadata("quantbox").get_all("Provides-Extra")
            print(json.dumps({"ok": ok, "refused": refused, "failed": failed, "tombstones": tombstones, "extras": extras}))
            """,
            blocked=NOT_IN_BASE,
        )
    )
    assert result["failed"] == {}, "a bare import failure on a base install (never a MissingExtraError)"
    assert result["tombstones"], "the walk met no tombstone of a removed path"
    assert result["refused"] == REFUSES_ON_BASE
    assert set(result["refused"].values()) <= set(result["extras"])
    assert len(result["ok"]) >= 150, "the walk did not look"


def test_plugins_list_schema_and_doctor_work_on_a_base_install(tmp_path):
    result = _last_json(
        _run(
            f"""
            import contextlib, io, json, os, sys
            from quantbox.cli import main

            def cli(*argv):
                out = io.StringIO()
                sys.argv = ["quantbox", *argv]
                code = 0
                with contextlib.redirect_stdout(out):
                    try:
                        main()
                    except SystemExit as exc:
                        code = exc.code or 0
                return code, out.getvalue()

            os.chdir({str(tmp_path)!r})
            listed = cli("plugins", "list", "--json")
            schema = cli("plugins", "schema", "--json")
            doctor = cli("plugins", "doctor", "--json")
            print(json.dumps({{"list": listed, "schema": schema, "doctor": doctor}}))
            """,
            blocked=NOT_IN_BASE,
        )
    )
    for cmd in ("list", "schema", "doctor"):
        assert result[cmd][0] == 0, (cmd, result[cmd])
    listed = json.loads(result["list"][1])
    assert "binance.live_data.v1" in listed["data"] and "telegram.publisher.v1" in listed["publishers"]

    rows = json.loads(result["schema"][1])["plugins"]
    assert {r["id"]: r["missing_extra"] for r in rows if "missing_extra" in r} == {
        "binance.live_data.v1": "data",
        "hyperliquid.data.cached.v1": "data",
        "hyperliquid.data.v1": "data",
        "telegram.publisher.v1": "trade",
    }
    assert any(r["id"] == "data.synthetic.v1" and r["params_schema"] for r in rows)

    doctor = json.loads(result["doctor"][1])
    rows = doctor["results"] if isinstance(doctor, dict) else doctor
    warned = {r["name"]: r["message"] for r in rows if r["message"].startswith("missing_extra")}
    assert warned == {
        "binance.live_data.v1": "missing_extra: install quantbox[data] (httpx)",
        "hyperliquid.data.cached.v1": "missing_extra: install quantbox[data] (requests)",
        "hyperliquid.data.v1": "missing_extra: install quantbox[data] (requests)",
        "telegram.publisher.v1": "missing_extra: install quantbox[trade] (httpx)",
    }


def test_a_research_command_fails_on_one_line_naming_the_extra(tmp_path):
    proc = _run(
        f"""
        import sys
        from quantbox.cli import main

        sys.argv = ["quantbox", "warehouse", "init", "-r", {str(tmp_path / "wh")!r}]
        main()
        """,
        blocked=NOT_IN_BASE,
    )
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "quantbox[research]" in proc.stderr
    assert "Traceback" not in proc.stderr, proc.stderr
    assert "Exception ignored" not in proc.stderr, proc.stderr


def test_the_client_pack_imports_on_a_base_install_and_loads_only_core():
    """robo's quantbox surface, strategy_cache included, loads with no extra and stays in core (TOM-1451)."""
    pack_src = Path(__file__).parent / "fixtures" / "client_pack" / "src"
    layer_map_dir = Path(__file__).resolve().parents[1] / "scripts"
    result = _last_json(
        _run(
            f"""
            import json, sys
            from pathlib import Path
            sys.path.insert(0, {str(pack_src)!r})
            sys.path.insert(0, {str(layer_map_dir)!r})
            import qb_client_pack.plugins
            from layer_map import layer_of, read_layers

            layers = read_layers(Path({str(layer_map_dir)!r}).parent / "pyproject.toml")
            loaded = sorted(m for m in sys.modules if m == "quantbox" or m.startswith("quantbox."))
            print(json.dumps({{"loaded": loaded, "above": [m for m in loaded if layer_of(m, layers) != "core"]}}))
            """,
            blocked=NOT_IN_BASE,
        )
    )
    assert "quantbox.cache.strategy_cache" in result["loaded"]
    assert result["above"] == []


def test_lazy_packages_resolve_without_the_extra_and_refuse_with_it_named():
    result = _last_json(
        _run(
            """
            import json
            from quantbox.exceptions import MissingExtraError
            import quantbox.plugins.datasources as ds
            import quantbox.plugins.broker as br
            import quantbox.plugins.publisher as pub

            out = {"synthetic": ds.SyntheticDataPlugin.meta.name, "sim": br.SimPaperBroker.meta.name}
            for label, mod, name in (("binance", ds, "BinanceDataPlugin"), ("telegram", pub, "TelegramPublisher")):
                try:
                    getattr(mod, name)
                    out[label] = "imported"
                except MissingExtraError as exc:
                    out[label] = exc.extra
            try:
                ds.NoSuchThing
            except AttributeError:
                out["unknown"] = "AttributeError"
            out["alias"] = ds.DuckDBParquetData is ds.LocalFileDataPlugin
            print(json.dumps(out))
            """,
            blocked=NOT_IN_BASE,
        )
    )
    assert result == {
        "synthetic": "data.synthetic.v1",
        "sim": "sim.paper.v1",
        "binance": "data",
        "telegram": "trade",
        "unknown": "AttributeError",
        "alias": True,
    }


def test_the_package_names_are_the_same_objects_as_before():
    import quantbox.plugins.broker as br
    import quantbox.plugins.datasources as ds
    import quantbox.plugins.publisher as pub
    from quantbox.plugins.broker.binance_live import BinanceLiveBroker
    from quantbox.plugins.broker.ibkr_stub import PaperBrokerStub
    from quantbox.plugins.datasources.binance_data import MarketDataSnapshot
    from quantbox.plugins.datasources.local_file_data import LocalFileDataPlugin
    from quantbox.plugins.publisher.telegram import TelegramPublisher

    assert set(dir(ds)) >= set(ds.__all__) and set(dir(br)) >= set(br.__all__)
    assert ds.MarketDataSnapshot is MarketDataSnapshot
    assert ds.DuckDBParquetData is LocalFileDataPlugin
    assert br.IBKRPaperBrokerStub is PaperBrokerStub
    assert br.BinanceLiveBroker is BinanceLiveBroker
    assert pub.TelegramPublisher is TelegramPublisher
    for mod in (ds, br, pub):
        for name in mod.__all__:
            assert getattr(mod, name) is not None, f"{mod.__name__}.{name}"


def test_market_cap_live_fetch_names_the_data_extra(tmp_path):
    result = _last_json(
        _run(
            f"""
            import json
            from quantbox.exceptions import MissingExtraError
            from quantbox.market_cap import MarketCapProvider, load_pit_market_cap, map_symbol

            out = {{"map": map_symbol("kPEPE", {{"PEPE"}})}}
            for source in ("coingecko", "coinmarketcap"):
                try:
                    MarketCapProvider(cache_dir={str(tmp_path)!r}, source=source).fetch_rankings()
                    out[source] = "fetched"
                except MissingExtraError as exc:
                    out[source] = [exc.extra, exc.name]
            print(json.dumps(out))
            """,
            blocked=NOT_IN_BASE,
        )
    )
    assert result == {
        "map": ["PEPE", 1000.0],
        "coingecko": ["data", "pycoingecko"],
        "coinmarketcap": ["data", "httpx"],
    }


_REPO = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("ask", ["full_report", "full_report_variants", "auto_ingest"])
def test_an_opt_in_research_feature_on_a_base_install_raises_naming_the_extra(tmp_path, ask):
    """A best-effort ``except Exception`` must not swallow a missing extra (ADR-0010).

    full_report needs plotly and warehouse.auto_ingest needs duckdb, both [research].
    Each used to log a warning and exit 0 with a hollow or missing output. The
    control run (``none``) proves the same config runs on a base install, so the
    raise comes from the asked-for feature and nothing else.
    """
    result = _last_json(
        _run(
            f"""
            import json, os
            import yaml
            from quantbox.exceptions import MissingExtraError
            from quantbox.registry import PluginRegistry
            from quantbox.runner import run_from_config

            os.chdir({str(_REPO)!r})
            ask = {ask!r}

            def config(feature):
                with open("cookbook/canonical/configs/momentum.yaml") as fh:
                    cfg = yaml.safe_load(fh)
                cfg["artifacts"]["root"] = {str(tmp_path)!r} + "/" + feature
                params = cfg["plugins"]["pipeline"]["params"]
                params["engine"] = "rsims"
                if feature.startswith("full_report"):
                    params["full_report"] = True
                if feature == "full_report_variants":
                    strat = cfg["plugins"]["strategies"][0]
                    params["variants"] = [
                        {{"name": "a", "strategy": {{"name": strat["name"], "params": {{}}}}}},
                        {{"name": "b", "strategy": {{"name": strat["name"], "params": {{}}}}}},
                    ]
                if feature == "auto_ingest":
                    cfg["warehouse"] = {{"auto_ingest": True, "root": {str(tmp_path / "wh")!r}}}
                return cfg

            registry = PluginRegistry.discover()
            out = {{}}
            for feature in ("none", ask):
                try:
                    run_from_config(config(feature), registry)
                    out[feature] = "ran"
                except MissingExtraError as exc:
                    out[feature] = [exc.extra, exc.name]
            print(json.dumps(out))
            """,
            blocked=NOT_IN_BASE,
        )
    )
    missing = "duckdb" if ask == "auto_ingest" else "plotly"
    assert result == {"none": "ran", ask: ["research", missing]}


def test_is_transient_without_httpx_keeps_every_non_httpx_answer():
    result = _last_json(
        _run(
            """
            import json, sys
            from quantbox.retry import is_transient

            class RateLimitExceeded(Exception): ...
            class AuthenticationError(ConnectionError): ...
            class Coded(Exception):
                code = -1003

            cases = {
                "connection": ConnectionError(), "timeout": TimeoutError(), "os": OSError(),
                "ratelimit": RateLimitExceeded(), "auth": AuthenticationError(), "coded": Coded(),
                "value": ValueError(),
            }
            out = {k: is_transient(v) for k, v in cases.items()}
            out["httpx_loaded"] = "httpx" in sys.modules
            print(json.dumps(out))
            """,
            blocked=NOT_IN_BASE,
        )
    )
    assert result == {
        "connection": True,
        "timeout": True,
        "os": True,
        "ratelimit": True,
        "auth": False,
        "coded": True,
        "value": False,
        "httpx_loaded": False,
    }


def test_is_transient_still_reads_httpx_errors_when_httpx_is_installed():
    httpx = pytest.importorskip("httpx")
    from quantbox.retry import is_transient

    request = httpx.Request("GET", "https://example.invalid")

    def status(code: int) -> httpx.HTTPStatusError:
        return httpx.HTTPStatusError("x", request=request, response=httpx.Response(code, request=request))

    assert is_transient(httpx.ConnectTimeout("x", request=request))
    assert is_transient(httpx.ReadError("x", request=request))
    assert is_transient(status(429)) and is_transient(status(503))
    assert not is_transient(status(404)) and not is_transient(status(401))
