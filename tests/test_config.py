import pytest

from quant_forecast_lab.config import DEFAULT_MODEL_VERSION, DEFAULT_USE_RETURNS, validate_model_options


def test_default_configuration_is_compatible():
    validate_model_options(DEFAULT_MODEL_VERSION, DEFAULT_USE_RETURNS)


@pytest.mark.parametrize("model", ["v3", "v5", "v6", "v7", "v8"])
def test_return_target_rejects_incompatible_models(model):
    with pytest.raises(ValueError):
        validate_model_options(model, True)
