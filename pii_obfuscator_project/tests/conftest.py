import pytest
import os
import json
from unittest.mock import patch

# Import the main config file
from src import config

@pytest.fixture
def mock_project_root(tmp_path):
    """
    Fixture to mock the _PROJECT_ROOT in config.py to a temporary directory.
    This ensures tests are isolated and don't write to actual project paths.
    """
    original_project_root = config._PROJECT_ROOT
    original_output_dir = config.OUTPUT_DIR
    original_finetuned_model_path = config.FINETUNED_MODEL_PATH
    original_lora_adapters_path = config.LORA_ADAPTERS_PATH
    original_training_data_path = config.TRAINING_DATA_PATH


    with patch.object(config, '_PROJECT_ROOT', str(tmp_path)):
        # Re-evaluate paths in config that depend on _PROJECT_ROOT
        config.OUTPUT_DIR = os.path.join(config._PROJECT_ROOT, "models", "results_ner_finetune")
        config.FINETUNED_MODEL_PATH = os.path.join(config._PROJECT_ROOT, "models", "my_finetuned_ner_model")
        config.LORA_ADAPTERS_PATH = os.path.join(config._PROJECT_ROOT, "models", "my_lora_adapters")
        config.TRAINING_DATA_PATH = os.path.join(config._PROJECT_ROOT, "data", "training_data.json")
        yield tmp_path
    # Restore original _PROJECT_ROOT and dependent paths after tests
    config._PROJECT_ROOT = original_project_root
    config.OUTPUT_DIR = original_output_dir
    config.FINETUNED_MODEL_PATH = original_finetuned_model_path
    config.LORA_ADAPTERS_PATH = original_lora_adapters_path
    config.TRAINING_DATA_PATH = original_training_data_path


@pytest.fixture
def sample_training_data_path(mock_project_root):
    """
    Fixture to create a sample training_data.json file in a temporary directory
    and return its path.
    """
    data_content = [
        {
            "words": ["Иван", "Иванов", "встретился", "с", "Марией", "Петровой", "в", "Берлине", "."],
            "labels": ["B-PER", "I-PER", "O", "O", "B-PER", "I-PER", "O", "B-LOC", "O"]
        },
        {
            "words": ["Петр", "Сергеевич", "работает", "в", "компании", "Газпром", "в", "Москве", "."],
            "labels": ["B-PER", "I-PER", "O", "O", "O", "B-ORG", "O", "B-LOC", "O"]
        },
        {
            "words": ["Hugging", "Face", "это", "крупная", "компания", ",", "расположенная", "в", "США", "."],
            "labels": ["B-ORG", "I-ORG", "O", "O", "O", "O", "O", "O", "B-LOC", "O"]
        },
        {
            "words": ["Мой", "друг", "Сергей", "живет", "в", "Санкт-Петербурге", "."],
            "labels": ["O", "O", "B-PER", "O", "O", "B-LOC", "I-LOC", "O"]
        }
    ]
    
    # Create the data directory inside the mocked project root
    data_dir = mock_project_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Write the sample data to the mocked path
    file_path = data_dir / "training_data.json"
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data_content, f, ensure_ascii=False, indent=4)
        
    return file_path
