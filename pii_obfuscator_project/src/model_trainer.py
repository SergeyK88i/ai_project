import os
import numpy as np
import evaluate
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer, DataCollatorForTokenClassification
from peft import LoraConfig, get_peft_model # Импортируем LoRA

from . import config
from .data_processor import load_and_process_data, tokenizer, id_to_label

def train_model():
    # Загружаем и обрабатываем данные
    train_dataset, eval_dataset, id_to_label_map = load_and_process_data()

    print("Данные для обучения подготовлены:")
    print(f"  Размер обучающего датасета: {len(train_dataset)}")
    print(f"  Размер валидационного датасета: {len(eval_dataset)}")
    print(train_dataset)
    print(eval_dataset)

    # --- 5. Определяем метрики для оценки ---
    metric = evaluate.load("seqeval")

    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        true_labels = [[id_to_label_map[l] for l in label if l != -100] for label in labels]
        true_predictions = [
            [id_to_label_map[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        results = metric.compute(predictions=true_predictions, references=true_labels)
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }

    # Обновим конфигурацию модели, чтобы она знала наши метки
    model_config = AutoModelForTokenClassification.from_pretrained(config.MODEL_NAME).config
    model_config.id2label = id_to_label_map
    model_config.label2id = {label:id for id, label in id_to_label_map.items()}

    # --- 6. Загружаем модель для дообучения ---
    model = AutoModelForTokenClassification.from_pretrained(
        config.MODEL_NAME,
        config=model_config # Передаем нашу обновленную конфигурацию с метками
    )

    # --- Применяем LoRA, если выбрана стратегия LoRA ---
    if config.FINETUNING_STRATEGY == "lora":
        print(f"Применяем LoRA с параметрами: r={config.LORA_R}, alpha={config.LORA_ALPHA}, dropout={config.LORA_DROPOUT}, target_modules={config.LORA_TARGET_MODULES}")
        lora_config = LoraConfig(
            r=config.LORA_R,
            lora_alpha=config.LORA_ALPHA,
            target_modules=config.LORA_TARGET_MODULES,
            lora_dropout=config.LORA_DROPOUT,
            bias="none", # или "lora_only", в зависимости от предпочтений
            task_type="TOKEN_CLS", # Для задач классификации токенов
        )
        model = get_peft_model(model, lora_config)
        print("Количество обучаемых параметров модели (LoRA):")
        model.print_trainable_parameters()
    else:
        print("Используется полное дообучение.")


    # --- 7. Настраиваем аргументы для обучения (из config.py) ---
    training_args = TrainingArguments(
        output_dir=config.OUTPUT_DIR,
        num_train_epochs=config.NUM_TRAIN_EPOCHS,
        per_device_train_batch_size=config.PER_DEVICE_TRAIN_BATCH_SIZE,
        warmup_steps=config.WARMUP_STEPS,
        weight_decay=config.WEIGHT_DECAY,
        logging_dir=config.LOGGING_DIR,
        logging_steps=config.LOGGING_STEPS,
        eval_strategy=config.EVAL_STRATEGY,
        save_strategy=config.SAVE_STRATEGY,
        load_best_model_at_end=config.LOAD_BEST_MODEL_AT_END,
        metric_for_best_model=config.METRIC_FOR_BEST_MODEL,
        push_to_hub=config.PUSH_TO_HUB,
        report_to=config.REPORT_TO,
    )

    # --- 8. Создаем объект Trainer ---
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # --- 9. Запускаем обучение ---
    print("\nНачинаем дообучение модели...")
    trainer.train()
    print("\nДообучение завершено!")

    # --- 10. Оцениваем финальную модель ---
    print("\n--- Оценка дообученной модели на валидационном датасете ---")
    final_eval_results = trainer.evaluate(eval_dataset)
    print(f"Финальные метрики на валидационном датасете: {final_eval_results}")

    # --- 11. Сохраняем дообученную модель ---
    if config.FINETUNING_STRATEGY == "lora":
        lora_adapters_path = config.LORA_ADAPTERS_PATH
        # Создаем директорию, если она не существует
        os.makedirs(lora_adapters_path, exist_ok=True)
        trainer.model.save_pretrained(lora_adapters_path)
        tokenizer.save_pretrained(lora_adapters_path) # Сохраняем токенизатор вместе с адаптерами
        print(f"\nLoRA адаптеры сохранены в: {lora_adapters_path}")
    else:
        finetuned_model_path = config.FINETUNED_MODEL_PATH
        trainer.save_model(finetuned_model_path)
        tokenizer.save_pretrained(finetuned_model_path) # Сохраняем токенизатор вместе с моделью
        print(f"\nДообученная модель сохранена в: {finetuned_model_path}")
