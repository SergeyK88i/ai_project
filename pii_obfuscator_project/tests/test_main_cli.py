# pii_obfuscator_project/tests/test_main_cli.py
import pytest
from unittest.mock import patch, MagicMock
import sys # Для мокирования sys.argv

# Assuming main_cli is executable
from src import main_cli
from src import config

# Mocking train_model and create_inference_pipeline functions
@pytest.fixture
def mock_train_model():
    with patch('src.main_cli.train_model') as mock:
        yield mock

@pytest.fixture
def mock_create_inference_pipeline():
    mock_obfuscator = MagicMock()
    mock_obfuscator.obfuscate_text.return_value = "Obfuscated Text"
    mock_obfuscator.entity_map = {"Original": "Replacement"}
    with patch('src.main_cli.create_inference_pipeline', return_value=mock_obfuscator) as mock:
        yield mock

# Helper to run main_cli.main with custom args
def run_main_cli(monkeypatch, args):
    monkeypatch.setattr(sys, 'argv', ['main_cli.py'] + args)
    main_cli.main()

def test_train_command_default_strategy(monkeypatch, mock_train_model):
    """Test 'train' command with default strategy."""
    original_strategy = config.FINETUNING_STRATEGY
    try:
        run_main_cli(monkeypatch, ['train'])
        mock_train_model.assert_called_once()
        assert config.FINETUNING_STRATEGY == original_strategy # Should remain default
    finally:
        config.FINETUNING_STRATEGY = original_strategy

def test_train_command_full_strategy(monkeypatch, mock_train_model):
    """Test 'train' command with --strategy full."""
    original_strategy = config.FINETUNING_STRATEGY
    try:
        run_main_cli(monkeypatch, ['train', '--strategy', 'full'])
        mock_train_model.assert_called_once()
        assert config.FINETUNING_STRATEGY == "full"
    finally:
        config.FINETUNING_STRATEGY = original_strategy

def test_train_command_lora_strategy(monkeypatch, mock_train_model):
    """Test 'train' command with --strategy lora."""
    original_strategy = config.FINETUNING_STRATEGY
    try:
        run_main_cli(monkeypatch, ['train', '--strategy', 'lora'])
        mock_train_model.assert_called_once()
        assert config.FINETUNING_STRATEGY == "lora"
    finally:
        config.FINETUNING_STRATEGY = original_strategy

def test_obfuscate_command_default_finetuned_full_strategy(monkeypatch, mock_create_inference_pipeline, capsys):
    """Test 'obfuscate' command with default finetuned and full strategy."""
    original_use_finetuned = config.USE_FINETUNED_MODEL
    original_strategy = config.FINETUNING_STRATEGY
    try:
        run_main_cli(monkeypatch, ['obfuscate', '--text', 'Some text'])
        mock_create_inference_pipeline.assert_called_once()
        assert config.USE_FINETUNED_MODEL == True # Default
        assert config.FINETUNING_STRATEGY == "full" # Default
        
        captured = capsys.readouterr()
        assert "Obfuscated Text" in captured.out
        assert "Original -> Replacement" in captured.out
    finally:
        config.USE_FINETUNED_MODEL = original_use_finetuned
        config.FINETUNING_STRATEGY = original_strategy


def test_obfuscate_command_with_explicit_args(monkeypatch, mock_create_inference_pipeline, capsys):
    """Test 'obfuscate' command with explicit --use_finetuned and --strategy."""
    original_use_finetuned = config.USE_FINETUNED_MODEL
    original_strategy = config.FINETUNING_STRATEGY
    try:
        run_main_cli(monkeypatch, ['obfuscate', '--text', 'Another text', '--use_finetuned', 'true', '--strategy', 'lora'])
        mock_create_inference_pipeline.assert_called_once()
        assert config.USE_FINETUNED_MODEL == True
        assert config.FINETUNING_STRATEGY == "lora"
        
        captured = capsys.readouterr()
        assert "Obfuscated Text" in captured.out
    finally:
        config.USE_FINETUNED_MODEL = original_use_finetuned
        config.FINETUNING_STRATEGY = original_strategy


def test_obfuscate_command_with_base_model(monkeypatch, mock_create_inference_pipeline, capsys):
    """Test 'obfuscate' command with --use_finetuned false (should use base model)."""
    original_use_finetuned = config.USE_FINETUNED_MODEL
    original_strategy = config.FINETUNING_STRATEGY
    try:
        run_main_cli(monkeypatch, ['obfuscate', '--text', 'Base model text', '--use_finetuned', 'false'])
        mock_create_inference_pipeline.assert_called_once()
        assert config.USE_FINETUNED_MODEL == False
        assert config.FINETUNING_STRATEGY == original_strategy # Strategy should remain default
        
        captured = capsys.readouterr()
        assert "Obfuscated Text" in captured.out
    finally:
        config.USE_FINETUNED_MODEL = original_use_finetuned
        config.FINETUNING_STRATEGY = original_strategy

def test_no_command_shows_help(monkeypatch, capsys):
    """Test that running main_cli without a command prints help."""
    run_main_cli(monkeypatch, [])
    captured = capsys.readouterr()
    assert "usage: main_cli.py" in captured.out
    assert "Available commands" in captured.out
