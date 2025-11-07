import pytest
from transformers import BertForSequenceClassification
from src.utils.text_classification.model_loader import Model_Loader


@pytest.fixture
def model_loader():
    """Fixture to initialize the Model_Loader once."""
    return Model_Loader()


def test_model_loader_init(model_loader):
    """Test that Model_Loader initializes without errors."""
    assert model_loader is not None, "Model_Loader instance should not be None"
    assert hasattr(model_loader, "config"), "Model_Loader should have a config attribute"


def test_model_loading(model_loader):
    """Test that the model loads successfully."""
    model = model_loader.load_llm()
    assert isinstance(model, BertForSequenceClassification), \
        f"Expected BertForSequenceClassification, got {type(model)}"
