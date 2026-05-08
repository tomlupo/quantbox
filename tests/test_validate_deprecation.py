import warnings

from quantbox.validate import validate_config


def test_legacy_dataset_root_emits_deprecation_warning():
    cfg = {
        "plugins": {
            "pipeline": {"name": "any"},
            "data": {
                "name": "dataset.curated.v1",
                "params_init": {"dataset_root": "./datasets", "dataset": "crypto-spot-daily"},
            },
        }
    }
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        validate_config(cfg)
        assert any("dataset_root" in str(x.message) for x in w)
