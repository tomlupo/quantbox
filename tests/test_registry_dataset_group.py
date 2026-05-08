from quantbox.registry import ENTRYPOINT_GROUPS, PluginRegistry


def test_dataset_and_capability_groups_exist():
    assert ENTRYPOINT_GROUPS["dataset"] == "quantbox.datasets"
    assert ENTRYPOINT_GROUPS["capability"] == "quantbox.capabilities"


def test_registry_has_datasets_and_capabilities_attrs():
    reg = PluginRegistry.discover()
    assert hasattr(reg, "datasets")
    assert isinstance(reg.datasets, dict)
    assert hasattr(reg, "capabilities")
    assert isinstance(reg.capabilities, dict)
