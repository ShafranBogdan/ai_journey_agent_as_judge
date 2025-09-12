#!/usr/bin/env python3
"""
Agent-as-Judge Model Training Script
====================================

Этот скрипт реализует обучение модели-судьи для оценки ответов LLM моделей.
Поддерживает различные стратегии обучения и эксперименты.
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import torch
import wandb
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
import argparse
from pathlib import Path
import json
from datetime import datetime

# Unsloth imports
from unsloth import FastModel, is_bf16_supported
from unsloth.chat_templates import get_chat_template
from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForSeq2Seq
import warnings
warnings.filterwarnings("ignore")


class AgentJudgeConfig:
    """Конфигурация для обучения Agent-as-Judge модели"""
    
    def __init__(self):
        # Параметры модели
        self.model_name = "Qwen/Qwen3-0.6B"
        self.max_seq_length = 2048  # Увеличиваем для длинных промптов
        self.load_in_4bit = True    # Для экономии памяти
        self.load_in_8bit = False
        
        # Параметры обучения
        self.per_device_train_batch_size = 4
        self.per_device_eval_batch_size = 8
        self.gradient_accumulation_steps = 4
        self.warmup_ratio = 0.05
        self.num_train_epochs = 3
        self.learning_rate = 2e-4
        self.weight_decay = 0.01
        self.lr_scheduler_type = "cosine"
        self.logging_steps = 25
        self.eval_strategy = "steps"
        self.eval_steps = 200
        self.save_strategy = "steps" 
        self.save_steps = 500
        self.save_total_limit = 2
        
        # Параметры LoRA (если используется)
        self.use_lora = True
        self.lora_r = 16
        self.lora_alpha = 32
        self.lora_dropout = 0.1
        self.target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", 
                              "gate_proj", "up_proj", "down_proj"]
        
        # Прочие параметры
        self.seed = 42
        self.fp16 = not is_bf16_supported()
        self.bf16 = is_bf16_supported()
        self.gradient_checkpointing = True
        self.dataloader_num_workers = 2


class DataPreprocessor:
    """Класс для предобработки данных"""
    
    def __init__(self, tokenizer, config: AgentJudgeConfig):
        self.tokenizer = tokenizer
        self.config = config
        
    def format_prompt(self, example: Dict) -> str:
        """Форматирует промпт в нужном формате для обучения"""
        
        # Извлекаем промпт из данных
        input_prompt = example['prompt']
        score = example['score']
        
        # Создаем инструкцию для модели
        system_prompt = """Ты - модель-судья, которая оценивает качество ответов других языковых моделей. Проанализируй предоставленную информацию и выставь оценку согласно указанному критерию и шкале оценивания. Ответь только числом от 0 до 3 (или -1, если критерий не применим)."""
        
        # Формируем диалог
        conversation = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": input_prompt},
            {"role": "assistant", "content": str(score)}
        ]
        
        # Применяем chat template
        formatted = self.tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=False
        )
        
        return formatted
    
    def prepare_dataset(self, df: pd.DataFrame) -> Dataset:
        """Подготавливает датасет для обучения"""
        
        # Преобразуем данные
        formatted_data = []
        for _, row in df.iterrows():
            formatted_text = self.format_prompt(row.to_dict())
            formatted_data.append({"text": formatted_text})
        
        return Dataset.from_list(formatted_data)
    
    def split_data(self, df: pd.DataFrame, test_size: float = 0.15, 
                   stratify: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Разбивает данные на обучающую и валидационную выборки"""
        
        if stratify:
            # Стратифицированное разбиение по классам
            train_df, val_df = train_test_split(
                df, test_size=test_size, 
                stratify=df['score'], 
                random_state=self.config.seed
            )
        else:
            train_df, val_df = train_test_split(
                df, test_size=test_size, 
                random_state=self.config.seed
            )
        
        return train_df, val_df
    
    def balance_classes(self, df: pd.DataFrame, method: str = "undersample") -> pd.DataFrame:
        """Балансирует классы в данных"""
        
        if method == "undersample":
            # Подвыборка до размера наименьшего класса
            min_count = df['score'].value_counts().min()
            balanced_dfs = []
            
            for score in df['score'].unique():
                score_df = df[df['score'] == score]
                if len(score_df) > min_count:
                    score_df = score_df.sample(n=min_count, random_state=self.config.seed)
                balanced_dfs.append(score_df)
            
            return pd.concat(balanced_dfs, ignore_index=True).sample(
                frac=1, random_state=self.config.seed
            )
        
        elif method == "oversample":
            # Дублирование до размера наибольшего класса
            max_count = df['score'].value_counts().max()
            balanced_dfs = []
            
            for score in df['score'].unique():
                score_df = df[df['score'] == score]
                if len(score_df) < max_count:
                    # Дублируем с заменой
                    additional_samples = max_count - len(score_df)
                    extra_df = score_df.sample(
                        n=additional_samples, 
                        replace=True, 
                        random_state=self.config.seed
                    )
                    score_df = pd.concat([score_df, extra_df], ignore_index=True)
                balanced_dfs.append(score_df)
            
            return pd.concat(balanced_dfs, ignore_index=True).sample(
                frac=1, random_state=self.config.seed
            )
        
        else:
            return df


class AgentJudgeTrainer:
    """Основной класс для обучения модели-судьи"""
    
    def __init__(self, config: AgentJudgeConfig, experiment_name: str = "agent_judge_v1"):
        self.config = config
        self.experiment_name = experiment_name
        self.model = None
        self.tokenizer = None
        self.trainer = None
        
        # Установка seed для воспроизводимости
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        
    def setup_model_and_tokenizer(self):
        """Инициализирует модель и токенизатор"""
        
        print(f"🤖 Загружаем модель {self.config.model_name}...")
        
        if self.config.use_lora:
            # LoRA обучение
            self.model, self.tokenizer = FastModel.from_pretrained(
                model_name=self.config.model_name,
                max_seq_length=self.config.max_seq_length,
                load_in_4bit=self.config.load_in_4bit,
                load_in_8bit=self.config.load_in_8bit,
                dtype=torch.bfloat16 if self.config.bf16 else torch.float16,
            )
            
            # Добавляем LoRA адаптеры
            self.model = FastModel.get_peft_model(
                self.model,
                r=self.config.lora_r,
                target_modules=self.config.target_modules,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                bias="none",
                use_gradient_checkpointing="unsloth",
                random_state=self.config.seed,
            )
            
        else:
            # Полное файн-тюнинг
            self.model, self.tokenizer = FastModel.from_pretrained(
                model_name=self.config.model_name,
                max_seq_length=self.config.max_seq_length,
                load_in_4bit=self.config.load_in_4bit,
                load_in_8bit=self.config.load_in_8bit,
                dtype=torch.bfloat16 if self.config.bf16 else torch.float16,
                full_finetuning=True,
            )
        
        # Настраиваем chat template
        self.tokenizer = get_chat_template(
            self.tokenizer,
            chat_template="qwen-3",
        )
        
        print(f"✅ Модель загружена! Параметры: {self.model.get_model_size():,}")
        
    def prepare_training_data(self, train_path: str, balance_classes: bool = False, 
                            balance_method: str = "undersample") -> DatasetDict:
        """Подготавливает данные для обучения"""
        
        print("📊 Загружаем и подготавливаем данные...")
        
        # Загружаем данные
        df = pd.read_csv(train_path)
        print(f"Загружено {len(df)} примеров")
        print("Распределение классов:")
        print(df['score'].value_counts().sort_index())
        
        # Балансируем классы если нужно
        if balance_classes:
            print(f"⚖️ Балансируем классы методом '{balance_method}'...")
            df = DataPreprocessor(self.tokenizer, self.config).balance_classes(
                df, method=balance_method
            )
            print(f"После балансировки: {len(df)} примеров")
            print(df['score'].value_counts().sort_index())
        
        # Разбиваем на train/val
        preprocessor = DataPreprocessor(self.tokenizer, self.config)
        train_df, val_df = preprocessor.split_data(df)
        
        print(f"Тренировочная выборка: {len(train_df)} примеров")
        print(f"Валидационная выборка: {len(val_df)} примеров")
        
        # Подготавливаем датасеты
        train_dataset = preprocessor.prepare_dataset(train_df)
        val_dataset = preprocessor.prepare_dataset(val_df)
        
        return DatasetDict({
            "train": train_dataset,
            "validation": val_dataset
        })
    
    def setup_trainer(self, dataset: DatasetDict, output_dir: str):
        """Настраивает trainer для обучения"""
        
        training_args = TrainingArguments(
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_eval_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            warmup_ratio=self.config.warmup_ratio,
            num_train_epochs=self.config.num_train_epochs,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            lr_scheduler_type=self.config.lr_scheduler_type,
            logging_steps=self.config.logging_steps,
            eval_strategy=self.config.eval_strategy,
            eval_steps=self.config.eval_steps,
            save_strategy=self.config.save_strategy,
            save_steps=self.config.save_steps,
            save_total_limit=self.config.save_total_limit,
            fp16=self.config.fp16,
            bf16=self.config.bf16,
            gradient_checkpointing=self.config.gradient_checkpointing,
            dataloader_num_workers=self.config.dataloader_num_workers,
            output_dir=output_dir,
            report_to=["wandb"] if "WANDB_PROJECT" in os.environ else [],
            run_name=self.experiment_name,
        )
        
        self.trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=dataset["train"],
            eval_dataset=dataset["validation"],
            dataset_text_field="text",
            max_seq_length=self.config.max_seq_length,
            data_collator=DataCollatorForSeq2Seq(
                tokenizer=self.tokenizer, 
                return_tensors="pt", 
                padding=True
            ),
            args=training_args,
        )
    
    def train(self, train_path: str, output_dir: str, **kwargs):
        """Запускает процесс обучения"""
        
        # Создаем директорию для сохранения
        os.makedirs(output_dir, exist_ok=True)
        
        # Инициализируем wandb если доступен
        if "WANDB_PROJECT" in os.environ:
            wandb.init(
                project=os.environ["WANDB_PROJECT"],
                name=self.experiment_name,
                config=self.config.__dict__
            )
        
        # Настраиваем модель
        self.setup_model_and_tokenizer()
        
        # Подготавливаем данные
        dataset = self.prepare_training_data(train_path, **kwargs)
        
        # Настраиваем trainer
        self.setup_trainer(dataset, output_dir)
        
        # Сохраняем конфигурацию
        config_path = os.path.join(output_dir, "training_config.json")
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(self.config.__dict__, f, indent=2, ensure_ascii=False)
        
        print("🚀 Начинаем обучение...")
        
        # Запускаем обучение
        self.trainer.train()
        
        # Сохраняем финальную модель
        print("💾 Сохраняем модель...")
        self.trainer.save_model(output_dir)
        
        # Сохраняем в формате unsloth
        self.model.save_pretrained(os.path.join(output_dir, "unsloth_model"))
        self.tokenizer.save_pretrained(os.path.join(output_dir, "unsloth_model"))
        
        print(f"✅ Обучение завершено! Модель сохранена в {output_dir}")
        
        return self.trainer.state.log_history


def run_experiment_1_baseline():
    """Эксперимент 1: Базовое обучение с LoRA"""
    
    print("\n" + "="*60)
    print("🧪 ЭКСПЕРИМЕНТ 1: Baseline LoRA Fine-tuning")
    print("="*60)
    
    config = AgentJudgeConfig()
    config.use_lora = True
    config.lora_r = 16
    config.learning_rate = 2e-4
    config.num_train_epochs = 3
    
    trainer = AgentJudgeTrainer(config, "exp1_baseline_lora")
    
    return trainer.train(
        train_path="aij_judge_task_1_train.csv",
        output_dir="models/exp1_baseline_lora",
        balance_classes=False
    )


def run_experiment_2_balanced():
    """Эксперимент 2: Обучение на сбалансированных данных"""
    
    print("\n" + "="*60)
    print("🧪 ЭКСПЕРИМЕНТ 2: Balanced Classes Training")
    print("="*60)
    
    config = AgentJudgeConfig()
    config.use_lora = True
    config.lora_r = 32  # Увеличиваем LoRA rank
    config.lora_alpha = 64
    config.learning_rate = 1.5e-4
    config.num_train_epochs = 4
    
    trainer = AgentJudgeTrainer(config, "exp2_balanced_lora")
    
    return trainer.train(
        train_path="aij_judge_task_1_train.csv",
        output_dir="models/exp2_balanced_lora", 
        balance_classes=True,
        balance_method="undersample"
    )


def run_experiment_3_full_finetuning():
    """Эксперимент 3: Полное файн-тюнинг (если хватает памяти)"""
    
    print("\n" + "="*60)
    print("🧪 ЭКСПЕРИМЕНТ 3: Full Fine-tuning")
    print("="*60)
    
    config = AgentJudgeConfig()
    config.use_lora = False  # Полное файн-тюнинг
    config.per_device_train_batch_size = 2  # Уменьшаем batch size
    config.gradient_accumulation_steps = 8  # Компенсируем accumulation
    config.learning_rate = 1e-4  # Меньший learning rate
    config.num_train_epochs = 2  # Меньше эпох для экономии времени
    config.load_in_4bit = True  # Обязательно для экономии памяти
    
    trainer = AgentJudgeTrainer(config, "exp3_full_finetuning")
    
    return trainer.train(
        train_path="aij_judge_task_1_train.csv",
        output_dir="models/exp3_full_finetuning",
        balance_classes=True,
        balance_method="oversample"
    )


def run_experiment_4_enhanced():
    """Эксперимент 4: Улучшенная конфигурация с оптимизациями"""
    
    print("\n" + "="*60)
    print("🧪 ЭКСПЕРИМЕНТ 4: Enhanced Configuration")
    print("="*60)
    
    config = AgentJudgeConfig()
    config.use_lora = True
    config.lora_r = 64  # Большой rank для лучшей выразительности
    config.lora_alpha = 128
    config.lora_dropout = 0.05  # Меньший dropout
    config.learning_rate = 3e-4  # Выше learning rate
    config.num_train_epochs = 5
    config.warmup_ratio = 0.1
    config.weight_decay = 0.005  # Меньшая регуляризация
    config.max_seq_length = 3072  # Увеличиваем для длинных промптов
    
    trainer = AgentJudgeTrainer(config, "exp4_enhanced_lora")
    
    return trainer.train(
        train_path="aij_judge_task_1_train.csv",
        output_dir="models/exp4_enhanced_lora",
        balance_classes=True,
        balance_method="undersample"
    )


def main():
    parser = argparse.ArgumentParser(description="Agent-as-Judge Model Training")
    parser.add_argument("--experiment", type=str, default="all",
                       choices=["all", "1", "2", "3", "4"],
                       help="Выбор эксперимента для запуска")
    parser.add_argument("--data_path", type=str, default="aij_judge_task_1_train.csv",
                       help="Путь к тренировочным данным")
    parser.add_argument("--wandb_project", type=str, default=None,
                       help="Название проекта в W&B для логирования")
    
    args = parser.parse_args()
    
    # Настраиваем W&B если указан
    if args.wandb_project:
        os.environ["WANDB_PROJECT"] = args.wandb_project
    
    # Запускаем эксперименты
    results = {}
    
    if args.experiment == "all" or args.experiment == "1":
        try:
            results["exp1"] = run_experiment_1_baseline()
        except Exception as e:
            print(f"❌ Ошибка в эксперименте 1: {e}")
    
    if args.experiment == "all" or args.experiment == "2":
        try:
            results["exp2"] = run_experiment_2_balanced()
        except Exception as e:
            print(f"❌ Ошибка в эксперименте 2: {e}")
    
    if args.experiment == "all" or args.experiment == "3":
        try:
            results["exp3"] = run_experiment_3_full_finetuning()
        except Exception as e:
            print(f"❌ Ошибка в эксперименте 3: {e}")
            print("💡 Попробуйте уменьшить batch_size или использовать LoRA")
    
    if args.experiment == "all" or args.experiment == "4":
        try:
            results["exp4"] = run_experiment_4_enhanced()
        except Exception as e:
            print(f"❌ Ошибка в эксперименте 4: {e}")
    
    print("\n" + "="*60)
    print("🎯 ВСЕ ЭКСПЕРИМЕНТЫ ЗАВЕРШЕНЫ!")
    print("="*60)
    print("Результаты сохранены в папке models/")
    print("Для инференса используйте соответствующий run.py")


if __name__ == "__main__":
    main()
