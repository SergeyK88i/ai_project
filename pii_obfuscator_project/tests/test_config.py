# pii_obfuscator_project/tests/test_config.py
import os
import pytest
from src import config

def test_project_root_is_defined():
    """Test that _PROJECT_ROOT is defined and is a string."""
    assert hasattr(config, '_PROJECT_ROOT')
    assert isinstance(config._PROJECT_ROOT, str)
    assert os.path.isdir(config._PROJECT_ROOT) # It should point to a valid directory

def test_model_name_is_defined():
    """Test that MODEL_NAME is defined."""
    assert hasattr(config, 'MODEL_NAME')
    assert isinstance(config.MODEL_NAME, str)
    assert len(config.MODEL_NAME) > 0

def test_labels_list_is_defined():
    """Test that LABELS_LIST is defined and is a list of strings."""
    assert hasattr(config, 'LABELS_LIST')
    assert isinstance(config.LABELS_LIST, list)
    assert len(config.LABELS_LIST) > 0
    assert all(isinstance(label, str) for label in config.LABELS_LIST)

def test_paths_are_absolute_and_correct(mock_project_root):
    """
    Test that output paths are absolute and correctly formed relative to _PROJECT_ROOT.
    Uses mock_project_root fixture to ensure paths are based on a test temp directory.
    """
    # config._PROJECT_ROOT is already mocked by the fixture
    
    assert os.path.isabs(config.OUTPUT_DIR)
    assert config.OUTPUT_DIR.startswith(str(mock_project_root))
    assert "models/results_ner_finetune" in config.OUTPUT_DIR

    assert os.path.isabs(config.FINETUNED_MODEL_PATH)
    assert config.FINETUNED_MODEL_PATH.startswith(str(mock_project_root))
    assert "models/my_finetuned_ner_model" in config.FINETUNED_MODEL_PATH
    
    assert os.path.isabs(config.LORA_ADAPTERS_PATH)
    assert config.LORA_ADAPTERS_PATH.startswith(str(mock_project_root))
    assert "models/my_lora_adapters" in config.LORA_ADAPTERS_PATH

    assert os.path.isabs(config.TRAINING_DATA_PATH)
    assert config.TRAINING_DATA_PATH.startswith(str(mock_project_root))
    assert "data/training_data.json" in config.TRAINING_DATA_PATH

def test_data_settings_are_defined():
    """Test that data-related settings are defined and have correct types."""
    assert hasattr(config, 'RANDOM_STATE')
    assert isinstance(config.RANDOM_STATE, int)

    assert hasattr(config, 'TEST_SIZE')
    assert isinstance(config.TEST_SIZE, float)
    assert 0.0 <= config.TEST_SIZE <= 1.0

def test_inference_settings_are_defined():
    """Test that inference settings are defined and have correct types."""
    assert hasattr(config, 'USE_FINETUNED_MODEL')
    assert isinstance(config.USE_FINETUNED_MODEL, bool)

def test_training_settings_are_defined():
    """Test that training settings are defined and have correct types."""
    assert hasattr(config, 'FINETUNING_STRATEGY')
    assert isinstance(config.FINETUNING_STRATEGY, str)
    assert config.FINETUNING_STRATEGY in ["full", "lora"]

    assert hasattr(config, 'NUM_TRAIN_EPOCHS')
    assert isinstance(config.NUM_TRAIN_EPOCHS, int)
    assert config.NUM_TRAIN_EPOCHS > 0

    assert hasattr(config, 'PER_DEVICE_TRAIN_BATCH_SIZE')
    assert isinstance(config.PER_DEVICE_TRAIN_BATCH_SIZE, int)
    assert config.PER_DEVICE_TRAIN_BATCH_SIZE > 0
    
    assert hasattr(config, 'WARMUP_STEPS')
    assert isinstance(config.WARMUP_STEPS, int)
    assert config.WARMUP_STEPS >= 0

    assert hasattr(config, 'WEIGHT_DECAY')
    assert isinstance(config.WEIGHT_DECAY, float)
    assert config.WEIGHT_DECAY >= 0.0

    assert hasattr(config, 'LOGGING_DIR')
    assert isinstance(config.LOGGING_DIR, str)

    assert hasattr(config, 'LOGGING_STEPS')
    assert isinstance(config.LOGGING_STEPS, int)
    assert config.LOGGING_STEPS > 0

    assert hasattr(config, 'EVAL_STRATEGY')
    assert isinstance(config.EVAL_STRATEGY, str)
    assert config.EVAL_STRATEGY in ["no", "steps", "epoch"]

    assert hasattr(config, 'SAVE_STRATEGY')
    assert isinstance(config.SAVE_STRATEGY, str)
    assert config.SAVE_STRATEGY in ["no", "steps", "epoch"]
    
    assert hasattr(config, 'LOAD_BEST_MODEL_AT_END')
    assert isinstance(config.LOAD_BEST_MODEL_AT_END, bool)

    assert hasattr(config, 'METRIC_FOR_BEST_MODEL')
    assert isinstance(config.METRIC_FOR_BEST_MODEL, str)

    assert hasattr(config, 'PUSH_TO_HUB')
    assert isinstance(config.PUSH_TO_HUB, bool)

    assert hasattr(config, 'REPORT_TO')
    assert isinstance(config.REPORT_TO, str)

def test_lora_parameters_are_defined():
    """Test that LoRA-specific parameters are defined and have correct types."""
    assert hasattr(config, 'LORA_R')
    assert isinstance(config.LORA_R, int)
    assert config.LORA_R > 0

    assert hasattr(config, 'LORA_ALPHA')
    assert isinstance(config.LORA_ALPHA, int)
    assert config.LORA_ALPHA > 0

    assert hasattr(config, 'LORA_DROPOUT')
    assert isinstance(config.LORA_DROPOUT, float)
    assert 0.0 <= config.LORA_DROPOUT <= 1.0

    assert hasattr(config, 'LORA_TARGET_MODULES')
    assert isinstance(config.LORA_TARGET_MODULES, list)
    assert all(isinstance(module, str) for module in config.LORA_TARGET_MODULES)

