import pytest
from unittest.mock import MagicMock, patch
import os
import shutil

from src import inference_engine
from src import config
from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer
from peft import PeftModel, LoraConfig # LoraConfig is not directly used in inference, but it's part of peft

# --- Fixtures for PIIObfuscator tests ---

@pytest.fixture
def mock_ner_callable():
    """
    Mock the NER pipeline callable to return predictable results.
    """
    mock = MagicMock()
    mock.return_value = [
        {'entity_group': 'PER', 'score': 0.99, 'word': 'Иван Иванов', 'start': 8, 'end': 19},
        {'entity_group': 'PER', 'score': 0.98, 'word': 'Марией Петровой', 'start': 35, 'end': 50},
        {'entity_group': 'LOC', 'score': 0.97, 'word': 'Берлине', 'start': 65, 'end': 72},
    ]
    return mock

@pytest.fixture
def pii_obfuscator(mock_ner_callable):
    """
    Fixture for PIIObfuscator instance with a mocked NER callable.
    """
    return inference_engine.PIIObfuscator(mock_ner_callable)

# --- Fixtures for create_inference_pipeline tests ---

@pytest.fixture
def mock_transformers_auto(monkeypatch):
    """
    Fixture to mock AutoModelForTokenClassification and AutoTokenizer
    to prevent actual model downloads during testing.
    """
    mock_model = MagicMock(spec=AutoModelForTokenClassification)
    mock_tokenizer = MagicMock(spec=AutoTokenizer)
    
    # Mock return values for common methods
    mock_model.from_pretrained.return_value = mock_model
    mock_tokenizer.from_pretrained.return_value = mock_tokenizer

    monkeypatch.setattr(inference_engine, 'AutoModelForTokenClassification', mock_model)
    monkeypatch.setattr(inference_engine, 'AutoTokenizer', mock_tokenizer)
    monkeypatch.setattr(inference_engine, 'pipeline', MagicMock()) # Mock pipeline creation
    return mock_model, mock_tokenizer

@pytest.fixture
def mock_peft_model(mock_transformers_auto, monkeypatch):
    """
    Fixture to mock PeftModel for LoRA specific tests.
    """
    mock_model_from_tf_auto, mock_tokenizer_from_tf_auto = mock_transformers_auto
    
    mock_peft_model_instance = MagicMock()
    mock_peft_model_instance.from_pretrained.return_value = mock_peft_model_instance
    # Ensure merge_and_unload returns something that pipeline can take, e.g., the mocked transformer model
    mock_peft_model_instance.merge_and_unload.return_value = mock_model_from_tf_auto

    monkeypatch.setattr(inference_engine, 'PeftModel', mock_peft_model_instance)
    # Return the peft_model instance, and also the underlying transformers mocks for assertions
    return mock_peft_model_instance, mock_model_from_tf_auto, mock_tokenizer_from_tf_auto


# --- Helper for creating dummy saved models ---
@pytest.fixture
def create_dummy_saved_model(tmp_path):
    def _create_dummy(path):
        os.makedirs(path, exist_ok=True)
        # Create dummy config.json and tokenizer files
        with open(os.path.join(path, "config.json"), "w") as f:
            f.write('{"_name_or_path": "test-model"}')
        with open(os.path.join(path, "tokenizer.json"), "w") as f:
            f.write('{}')
        with open(os.path.join(path, "special_tokens_map.json"), "w") as f:
            f.write('{}')
        with open(os.path.join(path, "tokenizer_config.json"), "w") as f:
            f.write('{}')
        with open(os.path.join(path, "vocab.txt"), "w") as f:
            f.write('dummy vocab')
    return _create_dummy


# --- PIIObfuscator class methods tests ---

def test_pii_obfuscator_init(pii_obfuscator, mock_ner_callable):
    """Test PIIObfuscator initialization."""
    assert pii_obfuscator.ner == mock_ner_callable
    assert pii_obfuscator.entity_map == {}
    assert pii_obfuscator.counters == {"person": 0, "org": 0, "location": 0, "misc": 0}

def test_pii_obfuscator_next_id(pii_obfuscator):
    """Test _next_id method."""
    assert pii_obfuscator._next_id("person") == "person-1"
    assert pii_obfuscator.counters["person"] == 1
    assert pii_obfuscator._next_id("person") == "person-2"
    assert pii_obfuscator.counters["person"] == 2
    assert pii_obfuscator._next_id("org") == "org-1"
    assert pii_obfuscator.counters["org"] == 1

def test_pii_obfuscator_normalize(pii_obfuscator):
    """Test _normalize method."""
    assert pii_obfuscator._normalize("  Test  String  ") == "Test String"
    assert pii_obfuscator._normalize("SingleWord") == "SingleWord"
    assert pii_obfuscator._normalize("") == ""
    assert pii_obfuscator._normalize("  ") == ""

def test_pii_obfuscator_extract_entities(pii_obfuscator, mock_ner_callable):
    """Test _extract_entities method."""
    text = "Test text with Иван Иванов and Марией Петровой."
    entities = pii_obfuscator._extract_entities(text)
    mock_ner_callable.assert_called_with(text)
    
    expected_results_from_mock = [
        {'entity_group': 'PER', 'score': 0.99, 'word': 'Иван Иванов', 'start': 8, 'end': 19},
        {'entity_group': 'PER', 'score': 0.98, 'word': 'Марией Петровой', 'start': 35, 'end': 50},
        {'entity_group': 'LOC', 'score': 0.97, 'word': 'Берлине', 'start': 65, 'end': 72},
    ]
    # Simulate filtering logic inside _extract_entities
    filtered_entities = []
    for r in expected_results_from_mock:
        raw_label = r.get("entity_group") or r.get("entity") or "MISC"
        label = pii_obfuscator.label_map.get(raw_label, "misc")
        if label in pii_obfuscator.allowed_types:
            word = pii_obfuscator._normalize(r.get("word") or r.get("text") or "")
            if word:
                filtered_entities.append((word, label))
    
    assert entities == filtered_entities

def test_pii_obfuscator_obfuscate_text(pii_obfuscator, mock_ner_callable):
    """Test obfuscate_text method."""
    text = "Дорогой Иван Иванов, ваша встреча с Марией Петровой состоится в Берлине."
    obfuscated_text = pii_obfuscator.obfuscate_text(text)

    mock_ner_callable.assert_called_with(text)
    
    # Expected replacements based on mock_ner_callable's return value
    # Иван Иванов -> person-1
    # Марией Петровой -> person-2
    # Берлине -> location-1
    assert "person-1" in obfuscated_text
    assert "person-2" in obfuscated_text
    assert "location-1" in obfuscated_text
    assert "Иван Иванов" not in obfuscated_text
    assert "Марией Петровой" not in obfuscated_text
    assert "Берлине" not in obfuscated_text
    assert pii_obfuscator.entity_map == {
        'Иван Иванов': 'person-1',
        'Марией Петровой': 'person-2',
        'Берлине': 'location-1'
    }

# --- create_inference_pipeline function tests ---

def test_create_inference_pipeline_base_model(mock_transformers_auto, monkeypatch):
    """Test create_inference_pipeline with base model (no fine-tuning)."""
    mock_model, mock_tokenizer = mock_transformers_auto
    
    # Set config to use base model
    monkeypatch.setattr(config, 'USE_FINETUNED_MODEL', False)
    monkeypatch.setattr(config, 'MODEL_NAME', "test/base-model")

    obfuscator = inference_engine.create_inference_pipeline()

    # Verify pipeline was called with the base model name and tokenizer
    # In src/inference_engine.py, pipeline is called with model_to_use (which is base_model_name)
    # The mock_model.from_pretrained is not called because `pipeline` handles model loading internally
    mock_model.from_pretrained.assert_not_called() 
    # mock_tokenizer.from_pretrained is called for the base tokenizer
    mock_tokenizer.from_pretrained.assert_called_with("test/base-model")
    
    inference_engine.pipeline.assert_called_with("token-classification", 
                                                model="test/base-model", 
                                                tokenizer=mock_tokenizer, 
                                                aggregation_strategy="simple")
    assert isinstance(obfuscator, inference_engine.PIIObfuscator)


def test_create_inference_pipeline_full_finetuned_model(mock_transformers_auto, monkeypatch, create_dummy_saved_model, mock_project_root):
    """Test create_inference_pipeline with a fully fine-tuned model."""
    mock_model, mock_tokenizer = mock_transformers_auto

    # Set config for full fine-tuning
    monkeypatch.setattr(config, 'USE_FINETUNED_MODEL', True)
    monkeypatch.setattr(config, 'FINETUNING_STRATEGY', "full")
    monkeypatch.setattr(config, 'MODEL_NAME', "test/base-model") # Base model name
    
    # Create dummy saved fine-tuned model (directory structure with config/tokenizer files)
    finetuned_path = os.path.join(mock_project_root, "models", "my_finetuned_ner_model")
    create_dummy_saved_model(finetuned_path)
    monkeypatch.setattr(config, 'FINETUNED_MODEL_PATH', finetuned_path)

    obfuscator = inference_engine.create_inference_pipeline()
    
    # Verify pipeline was called with the fine-tuned model path and its tokenizer loaded from there
    # For full fine-tuning, `pipeline` is directly called with `model_path_to_load`
    mock_model.from_pretrained.assert_not_called() 
    mock_tokenizer.from_pretrained.assert_called_with(finetuned_path) # tokenizer should be loaded from finetuned path

    inference_engine.pipeline.assert_called_with("token-classification",
                                                model=finetuned_path,
                                                tokenizer=mock_tokenizer,
                                                aggregation_strategy="simple")
    assert isinstance(obfuscator, inference_engine.PIIObfuscator)


def test_create_inference_pipeline_lora_finetuned_model(mock_transformers_auto, mock_peft_model, monkeypatch, create_dummy_saved_model, mock_project_root):
    """Test create_inference_pipeline with a LoRA fine-tuned model."""
    mock_peft_instance, mock_model_from_tf_auto, mock_tokenizer_from_tf_auto = mock_peft_model

    # Set config for LoRA fine-tuning
    monkeypatch.setattr(config, 'USE_FINETUNED_MODEL', True)
    monkeypatch.setattr(config, 'FINETUNING_STRATEGY', "lora")
    monkeypatch.setattr(config, 'MODEL_NAME', "test/base-model") # Base model name
    
    # Create dummy saved LoRA adapters
    lora_path = os.path.join(mock_project_root, "models", "my_lora_adapters")
    create_dummy_saved_model(lora_path) # LoRA adapters are also directories
    monkeypatch.setattr(config, 'LORA_ADAPTERS_PATH', lora_path)

    obfuscator = inference_engine.create_inference_pipeline()

    # Verify AutoModelForTokenClassification.from_pretrained was called for the base model
    mock_model_from_tf_auto.from_pretrained.assert_called_with("test/base-model")
    # Verify PeftModel.from_pretrained was called for the adapters
    mock_peft_instance.from_pretrained.assert_called_with(mock_model_from_tf_auto.from_pretrained.return_value, lora_path)
    mock_peft_instance.merge_and_unload.assert_called_once()
    
    # Verify tokenizer was tried from lora path, then base model name
    # The tokenizer used in the pipeline should be the one potentially loaded from lora_path (if successful)
    mock_tokenizer_from_tf_auto.from_pretrained.assert_any_call(lora_path) # It tries lora_path first
    mock_tokenizer_from_tf_auto.from_pretrained.assert_any_call("test/base-model") # Also checks the fallback
    
    # The pipeline should be called with the merged model
    inference_engine.pipeline.assert_called_with("token-classification",
                                                model=mock_peft_instance.merge_and_unload.return_value,
                                                tokenizer=mock_tokenizer_from_tf_auto,
                                                aggregation_strategy="simple")
    assert isinstance(obfuscator, inference_engine.PIIObfuscator)