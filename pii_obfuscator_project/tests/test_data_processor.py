# pii_obfuscator_project/tests/test_data_processor.py
import pytest
from src import config
from src import data_processor
from datasets import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizerFast

# Fixture for a dummy tokenizer (using the same model as in config)
@pytest.fixture(scope="module")
def dummy_tokenizer():
    return AutoTokenizer.from_pretrained(config.MODEL_NAME)

def test_tokenizer_loaded_correctly(dummy_tokenizer):
    """Test that the tokenizer is loaded and is an instance of PreTrainedTokenizerFast."""
    assert isinstance(data_processor.tokenizer, PreTrainedTokenizerFast)
    assert data_processor.tokenizer.name_or_path == config.MODEL_NAME

def test_label_id_mappings_created_correctly():
    """Test that label_to_id and id_to_label mappings are consistent and cover all labels."""
    assert isinstance(data_processor.label_to_id, dict)
    assert isinstance(data_processor.id_to_label, dict)
    assert len(data_processor.label_to_id) == len(config.LABELS_LIST)
    assert len(data_processor.id_to_label) == len(config.LABELS_LIST)
    
    for label in config.LABELS_LIST:
        assert label in data_processor.label_to_id
        assert data_processor.id_to_label[data_processor.label_to_id[label]] == label

def test_tokenize_and_align_labels(dummy_tokenizer):
    """Test the tokenization and label alignment logic."""
    words = ["Иван", "Иванов", "работает", "в", "Газпром", "."]
    word_labels = ["B-PER", "I-PER", "O", "O", "B-ORG", "O"]

    tokenized_inputs = data_processor.tokenize_and_align_labels(words, word_labels)

    assert "input_ids" in tokenized_inputs
    assert "labels" in tokenized_inputs
    assert len(tokenized_inputs["input_ids"]) == len(tokenized_inputs["labels"])

    # Example checks: CLS/SEP tokens should have -100 label, actual words should have correct label IDs
    # This is a simplified check, more robust tests would map back to labels
    # Using the tokenizer to decode for better human readability in assertions
    decoded_tokens = dummy_tokenizer.convert_ids_to_tokens(tokenized_inputs["input_ids"])
    # A basic check for alignment and special tokens
    assert tokenized_inputs["labels"][0] == -100  # CLS token
    assert tokenized_inputs["labels"][-1] == -100 # SEP token

    # Find where 'Иван' is tokenized and check its label
    ivan_idx = -1
    for i, token_id in enumerate(tokenized_inputs["input_ids"]):
        if dummy_tokenizer.decode([token_id]) == "иван": # Tokenizer might lowercase
            ivan_idx = i
            break
    
    if ivan_idx != -1:
        assert data_processor.id_to_label[tokenized_inputs["labels"][ivan_idx]] == "B-PER"
    
    # Check "Иванов" which is I-PER
    ivanov_idx = -1
    for i, token_id in enumerate(tokenized_inputs["input_ids"]):
        if dummy_tokenizer.decode([token_id]).startswith("иванов"): # Tokenizer might split
            ivanov_idx = i
            break
    if ivanov_idx != -1:
        assert data_processor.id_to_label[tokenized_inputs["labels"][ivanov_idx]] == "I-PER"
        
    # Check 'Газпром' which is B-ORG
    gazprom_idx = -1
    for i, token_id in enumerate(tokenized_inputs["input_ids"]):
        if dummy_tokenizer.decode([token_id]).startswith("газпром"):
            gazprom_idx = i
            break
    if gazprom_idx != -1:
        assert data_processor.id_to_label[tokenized_inputs["labels"][gazprom_idx]] == "B-ORG"


def test_load_and_process_data_creates_datasets(mock_project_root, sample_training_data_path):
    """
    Test that load_and_process_data correctly loads, splits, and processes data
    into Hugging Face Dataset objects.
    """
    # Ensure config.TRAINING_DATA_PATH points to our sample data
    # This is handled by mock_project_root fixture which updates config.TRAINING_DATA_PATH

    train_dataset, eval_dataset, id_to_label_map = data_processor.load_and_process_data()

    assert isinstance(train_dataset, Dataset)
    assert isinstance(eval_dataset, Dataset)
    assert len(train_dataset) > 0
    assert len(eval_dataset) > 0
    
    # Check that 'labels' column exists and contains valid IDs
    assert "labels" in train_dataset.column_names
    assert all(isinstance(label_id, int) for sample in train_dataset for label_id in sample["labels"])
    
    # Check that 'offset_mapping' is removed
    assert "offset_mapping" not in train_dataset.column_names
    assert "offset_mapping" not in eval_dataset.column_names