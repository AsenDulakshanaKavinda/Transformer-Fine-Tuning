import pytest
import os
from unittest.mock import patch, MagicMock
from finetune.src.utils.text_classification.model_loader import Model_Loader

@pytest.fixture
def setup_env(tmp_path):
    """Temporary environment for the test"""
    os.environ["ENV"] = "local"
    os.environ["MODEL_TYPE"] = "encoder_only"
    yield
    os.environ.pop("ENV", None)
    os.environ.pop("MODEL_TYPE", None)


def test_model_loader_init(setup_env):
    """Ensure Model_Loader initializes and loads config without crashing."""
    try:
        loader = Model_Loader()
        assert loader.config is not None, "Config should be loaded"
    except Exception as e:
        pytest.fail(f"Model_Loader failed to initialize: {e}")


@patch("finetune.src.loader.BertForSequenceClassification.from_pretrained")
def test_model_loader_load_llm(mock_from_pretrained, setup_env):
    """
    Test load_llm() without downloading actual models.
    Mocks Hugging Face model loading.
    """
    # Simulate a successful model load
    fake_model = MagicMock(name="FakeBertModel")
    mock_from_pretrained.return_value = fake_model

    loader = Model_Loader()
    model = loader.load_llm()

    mock_from_pretrained.assert_called_once()
    assert model == fake_model, "Model_Loader should return mocked model"


def test_model_loader_invalid_model_type(setup_env):
    """Test handling when MODEL_TYPE does not exist in config."""
    os.environ["MODEL_TYPE"] = "invalid_type"

    loader = Model_Loader()

    with pytest.raises(Exception):
        loader.load_llm()
