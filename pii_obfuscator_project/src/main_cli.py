# pii_obfuscator_project/src/main_cli.py

import argparse
from . import config # Import config to access defaults

from .model_trainer import train_model
from .inference_engine import create_inference_pipeline

def main():
    parser = argparse.ArgumentParser(description="PII Obfuscator CLI for training and inference.")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Subparser для команды 'train'
    train_parser = subparsers.add_parser("train", help="Train the PII obfuscation model.")
    train_parser.add_argument("--strategy", type=str, default=config.FINETUNING_STRATEGY,
                              choices=["full", "lora"],
                              help=f"Fine-tuning strategy: 'full' (default: {config.FINETUNING_STRATEGY}) or 'lora'.")

    # Subparser для команды 'obfuscate'
    obfuscate_parser = subparsers.add_parser("obfuscate", help="Obfuscate PII in a given text.")
    obfuscate_parser.add_argument("--text", type=str, required=True, help="Text to obfuscate.")
    obfuscate_parser.add_argument("--use_finetuned", type=lambda x: x.lower() == 'true', default=str(config.USE_FINETUNED_MODEL).lower(),
                                  help=f"Use a fine-tuned model for obfuscation (default: {config.USE_FINETUNED_MODEL}).")
    obfuscate_parser.add_argument("--strategy", type=str, default=config.FINETUNING_STRATEGY,
                                  choices=["full", "lora"],
                                  help=f"Finetuning strategy used for the model to be loaded (default: {config.FINETUNING_STRATEGY}). Only applicable if --use_finetuned is True.")


    args = parser.parse_args()

    # Dynamically update config based on CLI arguments
    if args.command == "train":
        config.FINETUNING_STRATEGY = args.strategy
        print("Запуск обучения модели PII obfuscation...")
        train_model()
    elif args.command == "obfuscate":
        config.USE_FINETUNED_MODEL = args.use_finetuned
        config.FINETUNING_STRATEGY = args.strategy
        print("Запуск обфускации текста...")
        obfuscator = create_inference_pipeline()
        
        # Пример использования
        print(f"\n--- Оригинальный текст ---")
        print(args.text)

        obfuscated_text = obfuscator.obfuscate_text(args.text)

        print(f"\n--- Обфусцированный текст ---")
        print(obfuscated_text)

        print("\n--- Найденные замены ---")
        for original, replacement in obfuscator.entity_map.items():
            print(f"{original} -> {replacement}")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
