import re
from typing import Dict, List, Tuple

from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer
from peft import PeftModel # Импортируем PeftModel

from . import config

class PIIObfuscator: # Переименовали класс для общности
    def __init__(self, ner_callable):
        self.ner = ner_callable
        self.entity_map: Dict[str, str] = {}
        self.counters: Dict[str, int] = {
            "person": 0,
            "org": 0,
            "location": 0,
            "misc": 0,
        }
        self.label_map = {
            "PER": "person", 
            "PERSON": "person",
            "ORG": "org", 
            "ORGANIZATION": "org",
            "LOC": "location", 
            "LOCATION": "location", 
            "GPE": "location",
            "MISC": "misc",
        }
        self.allowed_types = {"person", "org", "location", "misc"}

    def _next_id(self, typ: str) -> str:
        self.counters[typ] += 1
        return f"{typ}-{self.counters[typ]}"

    def _normalize(self, s: str) -> str:
        return re.sub(r"\s+", " ", s).strip()

    def _extract_entities(self, text: str) -> List[Tuple[str, str]]:
        if not text:
            return []
        results = self.ner(text) # Вызов нейросети!
        # print(f"DEBUG: NER raw results: {results}") # DEBUG LINE (Убрали, так как это теперь часть инференса)
        found: List[Tuple[str, str]] = []
        for r in results:
            raw_label = r.get("entity_group") or r.get("entity") or "MISC"
            label = self.label_map.get(raw_label, "misc")
            if label not in self.allowed_types:
                continue
            word = self._normalize(r.get("word") or r.get("text") or "")
            if word:
                found.append((word, label))
            elif r.get("start") is not None and r.get("end") is not None:
                original_span_text = text[r["start"]:r["end"]]
                found.append((self._normalize(original_span_text), label))
        return found
 
    def obfuscate_text(self, text: str) -> str:
        if not text:
            return text
        entities = self._extract_entities(text)
        if not entities:
            return text

        unique_words: Dict[str, str] = {}
        for word, label in entities:
            if word not in self.entity_map:
                replacement = self._next_id(label)
                self.entity_map[word] = replacement
            unique_words[word] = self.entity_map[word]
     
        sorted_pairs = sorted(unique_words.items(), key=lambda x: len(x[0]), reverse=True)

        obfuscated = text
        for old, new in sorted_pairs:
            obfuscated = re.sub(r'\b' + re.escape(old) + r'\b', new, obfuscated)

        return obfuscated

# Функция для создания пайплайна инференса
def create_inference_pipeline():
    print("Загружаем модель для инференса... (может занять некоторое время)")
    
    base_model_name = config.MODEL_NAME
    
    # Загружаем базовый токенизатор
    base_tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    
    model_to_use = None
    tokenizer_to_use = base_tokenizer

    if config.USE_FINETUNED_MODEL:
        if config.FINETUNING_STRATEGY == "lora":
            # Загружаем базовую модель
            base_model = AutoModelForTokenClassification.from_pretrained(base_model_name)
            # Загружаем LoRA адаптеры и объединяем их с базовой моделью
            # PeftModel.from_pretrained автоматически загружает адаптеры поверх базовой модели.
            model_to_use = PeftModel.from_pretrained(base_model, config.LORA_ADAPTERS_PATH)
            # Объединяем адаптеры с базовой моделью для инференса.
            # Это создает одну модель, которую можно передать в pipeline.
            model_to_use = model_to_use.merge_and_unload()
            # Токенизатор обычно не меняется при LoRA, но если LoRA-адаптеры были сохранены с токенизатором,
            # то можно попытаться загрузить его оттуда. Иначе используем базовый.
            try:
                tokenizer_to_use = AutoTokenizer.from_pretrained(config.LORA_ADAPTERS_PATH)
            except Exception:
                tokenizer_to_use = base_tokenizer
            
        else: # Full fine-tuning
            model_to_use = config.FINETUNED_MODEL_PATH
            # При полном дообучении токенизатор тоже сохраняется вместе с моделью
            tokenizer_to_use = AutoTokenizer.from_pretrained(config.FINETUNED_MODEL_PATH)
    else: # Use base model (no fine-tuning)
        model_to_use = base_model_name
        tokenizer_to_use = base_tokenizer


    ner_pipeline = pipeline("token-classification", model=model_to_use, tokenizer=tokenizer_to_use, aggregation_strategy="simple")
    print("Модель загружена.")
    return PIIObfuscator(ner_pipeline)
