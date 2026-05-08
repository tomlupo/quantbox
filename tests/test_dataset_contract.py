from dataclasses import is_dataclass

from quantbox.dataset import (
    CoverageReport,
    DatasetManifest,
    DatasetPlugin,
)


def test_dataset_manifest_is_dataclass_with_required_fields():
    assert is_dataclass(DatasetManifest)
    fields = {f for f in DatasetManifest.__dataclass_fields__}
    assert {"name", "version", "date_range", "symbols_count", "data_fields"} <= fields


def test_coverage_report_is_dataclass_with_required_fields():
    assert is_dataclass(CoverageReport)
    fields = {f for f in CoverageReport.__dataclass_fields__}
    assert {"per_symbol", "per_field", "overall"} <= fields


def test_dataset_plugin_is_a_protocol():
    # Duck-typed: any object with the right attrs is a DatasetPlugin
    class Fake:
        meta = None
        dataset_id = "x"
        dataset_version = "1"
        capabilities = ()

        def load_prices(self):
            pass

        def load_universe(self):
            pass

        def load_fx(self):
            pass

        def manifest(self):
            pass

        def manifest_hash(self):
            pass

        def coverage_report(self):
            pass

    assert isinstance(Fake(), DatasetPlugin)
