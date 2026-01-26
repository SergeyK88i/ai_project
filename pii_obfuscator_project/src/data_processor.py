import json
import torch
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

from . import config # Импортируем наш конфиг

# Загружаем токенизатор (используем имя модели из конфига)
tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)

# 2. Подготовка меток
# Модель работает с числовыми ID меток. Нам нужно сопоставить текстовые метки с ID.
# Эти ID можно найти в model.config.id2label или создать свои, если модель новая.
# Для wikineural-multilingual-ner, стандартные метки такие (должны совпадать с теми, что использует модель):
label_to_id = {label: i for i, label in enumerate(config.LABELS_LIST)}
id_to_label = {i: label for i, label in enumerate(config.LABELS_LIST)}

# 3. Функция для токенизации и выравнивания меток
def tokenize_and_align_labels(words, word_labels):
    tokenized_inputs = tokenizer(words, truncation=True, is_split_into_words=True, return_offsets_mapping=True)
    
    word_ids = tokenized_inputs.word_ids(batch_index=0)
    
    label_ids = []
    previous_word_idx = None
    
    for word_idx in word_ids:
        if word_idx is None:
            label_ids.append(-100)
        elif word_idx != previous_word_idx:
            label_ids.append(label_to_id[word_labels[word_idx]])
        else:
            current_word_label = word_labels[word_idx]
            if current_word_label.startswith("B-"):
                label_ids.append(label_to_id[f"I-{current_word_label[2:]}"])
            else:
                label_ids.append(label_to_id[current_word_label])
        previous_word_idx = word_idx

    tokenized_inputs["labels"] = label_ids
    return tokenized_inputs

# --- Функция для загрузки и обработки данных ---
def load_and_process_data():
    # Загружаем данные из JSON файла
    with open(config.TRAINING_DATA_PATH, 'r', encoding='utf-8') as f:
        training_data_from_file = json.load(f)

    # Используем train_test_split для разделения нашего TRAINING_DATA
    train_raw_data, eval_raw_data = train_test_split(training_data_from_file, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE)

    train_processed_data = []
    for item in train_raw_data:
        train_processed_data.append(tokenize_and_align_labels(item["words"], item["labels"]))

    eval_processed_data = []
    for item in eval_raw_data:
        eval_processed_data.append(tokenize_and_align_labels(item["words"], item["labels"]))

    train_processed_data = [{k: v for k, v in item.items() if k != 'offset_mapping'} for item in train_processed_data]
    eval_processed_data = [{k: v for k, v in item.items() if k != 'offset_mapping'} for item in eval_processed_data]

    train_dataset = Dataset.from_list(train_processed_data)
    eval_dataset = Dataset.from_list(eval_processed_data)

    return train_dataset, eval_dataset, id_to_label
