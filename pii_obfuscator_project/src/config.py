import os

# --- Общие настройки модели ---
MODEL_NAME = "Babelscape/wikineural-multilingual-ner"

# --- Настройки меток NER ---
# Список всех меток, которые модель может распознавать (включая O)
LABELS_LIST = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']

# --- Пути сохранения ---
# Определяем корневую директорию проекта (два уровня вверх от config.py)
_PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

# Директория для сохранения промежуточных результатов обучения (чекпоинтов)
OUTPUT_DIR = os.path.join(_PROJECT_ROOT, "models", "results_ner_finetune")
# Путь для сохранения финальной дообученной модели (для полного дообучения)
FINETUNED_MODEL_PATH = os.path.join(_PROJECT_ROOT, "models", "my_finetuned_ner_model")
# Путь для сохранения LoRA адаптеров
LORA_ADAPTERS_PATH = os.path.join(_PROJECT_ROOT, "models", "my_lora_adapters")


# --- Настройки данных ---
TRAINING_DATA_PATH = os.path.join(_PROJECT_ROOT, "data", "training_data.json")
RANDOM_STATE = 42 # Для воспроизводимости разделения данных
TEST_SIZE = 0.25 # Доля данных для валидации

# --- Настройки инференса ---
USE_FINETUNED_MODEL = True # Установите False, чтобы использовать базовую модель из Hugging Face Hub

# --- Настройки обучения ---
# Стратегия дообучения: "full" для полного дообучения, "lora" для LoRA
FINETUNING_STRATEGY = "full" # Измените на "lora" для использования LoRA

# Параметры TrainingArguments
NUM_TRAIN_EPOCHS = 5
PER_DEVICE_TRAIN_BATCH_SIZE = 2
WARMUP_STEPS = 50
WEIGHT_DECAY = 0.01
LOGGING_DIR = "./logs"
LOGGING_STEPS = 10
EVAL_STRATEGY = "epoch"
SAVE_STRATEGY = "epoch"
LOAD_BEST_MODEL_AT_END = True
METRIC_FOR_BEST_MODEL = "f1"
PUSH_TO_HUB = False
REPORT_TO = "none"

# LoRA специфичные параметры
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.1
# Для BertForTokenClassification обычно это 'query' и 'value' слои.
LORA_TARGET_MODULES = ["query", "value"]