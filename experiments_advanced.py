#!/usr/bin/env python3
"""
Advanced Experiments for Agent-as-Judge
=======================================

Дополнительные эксперименты для улучшения качества модели:
- Data augmentation
- Multi-task learning 
- Knowledge distillation
- Advanced regularization
"""

import os
import pandas as pd
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
import random
import re
from datasets import Dataset
from transformers import AutoTokenizer
from unsloth import FastModel
import warnings
warnings.filterwarnings("ignore")


class DataAugmenter:
    """Класс для аугментации данных"""
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
    
    def paraphrase_prompts(self, df: pd.DataFrame, augment_ratio: float = 0.3) -> pd.DataFrame:
        """Перефразирует промпты для аугментации данных"""
        
        paraphrase_templates = {
            "задание для оценки": [
                "Задача для оценивания",
                "Пример для анализа", 
                "Случай для рассмотрения",
                "Материал для оценки"
            ],
            "эталонный ответ": [
                "Правильный ответ",
                "Верный ответ",
                "Образцовый ответ",
                "Корректный ответ"
            ],
            "ответ для оценки": [
                "Проверяемый ответ",
                "Анализируемый ответ",
                "Оцениваемый ответ",
                "Исследуемый ответ"
            ],
            "критерий оценки": [
                "Критерий оценивания",
                "Параметр оценки",
                "Показатель качества",
                "Мерило оценки"
            ]
        }
        
        augmented_data = []
        num_to_augment = int(len(df) * augment_ratio)
        
        # Выбираем случайные примеры для аугментации
        indices_to_augment = random.sample(range(len(df)), num_to_augment)
        
        for idx in indices_to_augment:
            row = df.iloc[idx].copy()
            prompt = row['prompt']
            
            # Применяем перефразирование
            for original, variants in paraphrase_templates.items():
                if original in prompt.lower():
                    replacement = random.choice(variants)
                    # Заменяем с сохранением регистра
                    prompt = re.sub(
                        re.escape(original), 
                        replacement, 
                        prompt, 
                        flags=re.IGNORECASE
                    )
            
            # Добавляем небольшие вариации в формулировки
            variations = [
                ("Выполни", "Выполните"),
                ("Найди", "Найдите"), 
                ("Реши", "Решите"),
                ("Определи", "Определите")
            ]
            
            for old, new in variations:
                if random.random() < 0.3:  # 30% вероятность замены
                    prompt = prompt.replace(old, new)
            
            row['prompt'] = prompt
            augmented_data.append(row)
        
        # Объединяем с оригинальными данными
        augmented_df = pd.concat([df, pd.DataFrame(augmented_data)], ignore_index=True)
        return augmented_df.sample(frac=1, random_state=self.seed)  # Перемешиваем
    
    def add_noise_to_scores(self, df: pd.DataFrame, noise_ratio: float = 0.1) -> pd.DataFrame:
        """Добавляет шум к оценкам для регуляризации"""
        
        df_noisy = df.copy()
        num_to_noise = int(len(df) * noise_ratio)
        noise_indices = random.sample(range(len(df)), num_to_noise)
        
        for idx in noise_indices:
            current_score = df_noisy.iloc[idx]['score']
            
            # Добавляем шум ±1 с ограничениями
            noise = random.choice([-1, 1])
            new_score = current_score + noise
            
            # Ограничиваем диапазон
            new_score = max(-1, min(3, new_score))
            df_noisy.iloc[idx, df_noisy.columns.get_loc('score')] = new_score
        
        return df_noisy
    
    def create_hard_negatives(self, df: pd.DataFrame) -> pd.DataFrame:
        """Создает сложные отрицательные примеры"""
        
        hard_negatives = []
        
        # Находим примеры с высокими оценками
        high_score_examples = df[df['score'] >= 2].copy()
        
        for _, row in high_score_examples.iterrows():
            # Создаем версию с понижением оценки
            hard_neg = row.copy()
            
            # Изменяем промпт, добавляя проблемы
            prompt = row['prompt']
            
            # Добавляем типичные ошибки в "ответ для оценки"
            if "### Ответ для оценки:" in prompt:
                parts = prompt.split("### Ответ для оценки:")
                if len(parts) == 2:
                    answer_part = parts[1]
                    
                    # Добавляем ошибки форматирования
                    error_modifications = [
                        lambda x: x + " (дополнительный текст)",
                        lambda x: "Ответ: " + x,
                        lambda x: x + "\n\nДополнительные пояснения...",
                        lambda x: x.upper(),  # Неправильный регистр
                        lambda x: x + " и еще немного текста"
                    ]
                    
                    modification = random.choice(error_modifications)
                    modified_answer = modification(answer_part.strip())
                    
                    hard_neg['prompt'] = parts[0] + "### Ответ для оценки:\n" + modified_answer
                    hard_neg['score'] = max(0, row['score'] - 1)  # Понижаем оценку
                    
                    hard_negatives.append(hard_neg)
        
        if hard_negatives:
            hard_neg_df = pd.DataFrame(hard_negatives)
            return pd.concat([df, hard_neg_df], ignore_index=True)
        
        return df


class AdvancedTrainingConfig:
    """Расширенная конфигурация для продвинутых экспериментов"""
    
    def __init__(self):
        # Базовые параметры
        self.model_name = "Qwen/Qwen3-0.6B"
        self.max_seq_length = 3072
        self.load_in_4bit = True
        
        # Расширенные параметры обучения
        self.per_device_train_batch_size = 2
        self.gradient_accumulation_steps = 8
        self.learning_rate = 5e-5
        self.num_train_epochs = 6
        self.warmup_ratio = 0.15
        self.weight_decay = 0.01
        
        # Расширенные параметры LoRA
        self.use_lora = True
        self.lora_r = 128  # Большой rank для лучшей выразительности
        self.lora_alpha = 256
        self.lora_dropout = 0.05
        self.target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
            "lm_head"  # Добавляем голову языковой модели
        ]
        
        # Regularization
        self.gradient_clipping = 1.0
        self.label_smoothing = 0.1
        self.drop_path_rate = 0.1
        
        # Curriculum learning
        self.curriculum_learning = True
        self.easy_examples_ratio = 0.3
        
        # Scheduling
        self.use_cosine_schedule = True
        self.min_lr_ratio = 0.1


def run_experiment_5_data_augmentation():
    """Эксперимент 5: Обучение с аугментацией данных"""
    
    print("\n" + "="*60)
    print("🧪 ЭКСПЕРИМЕНТ 5: Data Augmentation Training")
    print("="*60)
    
    # Загружаем и аугментируем данные
    df = pd.read_csv("aij_judge_task_1_train.csv")
    print(f"Исходные данные: {len(df)} примеров")
    
    augmenter = DataAugmenter(seed=42)
    
    # Применяем аугментацию
    df_augmented = augmenter.paraphrase_prompts(df, augment_ratio=0.4)
    print(f"После перефразирования: {len(df_augmented)} примеров")
    
    df_augmented = augmenter.create_hard_negatives(df_augmented)
    print(f"После добавления hard negatives: {len(df_augmented)} примеров")
    
    # Сохраняем аугментированные данные
    df_augmented.to_csv("aij_judge_augmented.csv", index=False)
    
    # Обучаем с расширенной конфигурацией
    config = AdvancedTrainingConfig()
    
    from train_agent_judge import AgentJudgeTrainer
    trainer = AgentJudgeTrainer(config, "exp5_augmented_data")
    
    return trainer.train(
        train_path="aij_judge_augmented.csv",
        output_dir="models/exp5_augmented_data",
        balance_classes=True,
        balance_method="undersample"
    )


def run_experiment_6_curriculum_learning():
    """Эксперимент 6: Curriculum Learning"""
    
    print("\n" + "="*60)
    print("🧪 ЭКСПЕРИМЕНТ 6: Curriculum Learning")
    print("="*60)
    
    # Загружаем данные
    df = pd.read_csv("aij_judge_task_1_train.csv")
    
    # Определяем сложность примеров
    def calculate_difficulty(prompt):
        """Простая эвристика для определения сложности"""
        difficulty_score = 0
        
        # Длина промпта
        difficulty_score += len(prompt) / 1000
        
        # Наличие сложных элементов
        if "функци" in prompt.lower():
            difficulty_score += 2
        if "код" in prompt.lower():
            difficulty_score += 1.5
        if "математик" in prompt.lower():
            difficulty_score += 1
        if len(re.findall(r'\d+', prompt)) > 5:
            difficulty_score += 1
        
        return difficulty_score
    
    # Рассчитываем сложность
    df['difficulty'] = df['prompt'].apply(calculate_difficulty)
    
    # Сортируем по сложности
    df_sorted = df.sort_values('difficulty')
    
    # Разбиваем на этапы curriculum
    total_size = len(df_sorted)
    stage1_size = int(total_size * 0.3)  # 30% легких
    stage2_size = int(total_size * 0.5)  # 50% средних
    
    stage1_data = df_sorted[:stage1_size]
    stage2_data = df_sorted[:stage1_size + stage2_size]
    stage3_data = df_sorted  # Все данные
    
    print(f"Stage 1 (легкие): {len(stage1_data)} примеров")
    print(f"Stage 2 (средние): {len(stage2_data)} примеров")
    print(f"Stage 3 (все): {len(stage3_data)} примеров")
    
    # Сохраняем этапы
    stage1_data.to_csv("curriculum_stage1.csv", index=False)
    stage2_data.to_csv("curriculum_stage2.csv", index=False)
    stage3_data.to_csv("curriculum_stage3.csv", index=False)
    
    # TODO: Реализовать поэтапное обучение
    # Здесь нужно модифицировать trainer для поддержки curriculum learning
    print("💡 Curriculum learning требует модификации trainer'а")
    print("Для демо обучаем на полных данных с curriculum config")
    
    config = AdvancedTrainingConfig()
    config.curriculum_learning = True
    
    from train_agent_judge import AgentJudgeTrainer
    trainer = AgentJudgeTrainer(config, "exp6_curriculum")
    
    return trainer.train(
        train_path="curriculum_stage3.csv",
        output_dir="models/exp6_curriculum",
        balance_classes=False
    )


def run_experiment_7_ensemble():
    """Эксперимент 7: Ensemble из разных моделей"""
    
    print("\n" + "="*60)  
    print("🧪 ЭКСПЕРИМЕНТ 7: Model Ensemble")
    print("="*60)
    
    # Обучаем несколько моделей с разными конфигурациями
    configs = []
    
    # Конфиг 1: Консервативный
    config1 = AdvancedTrainingConfig()
    config1.learning_rate = 1e-4
    config1.lora_r = 32
    config1.num_train_epochs = 4
    
    # Конфиг 2: Агрессивный  
    config2 = AdvancedTrainingConfig()
    config2.learning_rate = 5e-4
    config2.lora_r = 256
    config2.num_train_epochs = 3
    
    # Конфиг 3: Сбалансированный
    config3 = AdvancedTrainingConfig()
    config3.learning_rate = 2e-4
    config3.lora_r = 64
    config3.num_train_epochs = 5
    
    configs = [
        (config1, "exp7_ensemble_conservative"),
        (config2, "exp7_ensemble_aggressive"), 
        (config3, "exp7_ensemble_balanced")
    ]
    
    results = {}
    
    for config, exp_name in configs:
        try:
            print(f"\n🔄 Обучаем модель: {exp_name}")
            
            from train_agent_judge import AgentJudgeTrainer
            trainer = AgentJudgeTrainer(config, exp_name)
            
            result = trainer.train(
                train_path="aij_judge_task_1_train.csv",
                output_dir=f"models/{exp_name}",
                balance_classes=True,
                balance_method="undersample"
            )
            
            results[exp_name] = result
            
        except Exception as e:
            print(f"❌ Ошибка в {exp_name}: {e}")
    
    print(f"\n✅ Обучено {len(results)} моделей для ансамбля")
    return results


def create_ensemble_inference():
    """Создает скрипт для ensemble inference"""
    
    ensemble_script = '''#!/usr/bin/env python3
"""
Ensemble Inference for Agent-as-Judge
====================================
"""

import pandas as pd
import numpy as np
from collections import Counter
import os

def ensemble_predict(test_path: str, pred_path: str):
    """Ансамблевое предсказание"""
    
    # Пути к моделям ансамбля
    ensemble_models = [
        "models/exp7_ensemble_conservative/unsloth_model",
        "models/exp7_ensemble_aggressive/unsloth_model", 
        "models/exp7_ensemble_balanced/unsloth_model"
    ]
    
    # Получаем предсказания от каждой модели
    all_predictions = []
    
    for model_path in ensemble_models:
        if os.path.exists(model_path):
            print(f"🔮 Получаем предсказания от {model_path}")
            
            # Здесь должен быть код инференса для каждой модели
            # predictions = get_predictions(test_path, model_path)
            # all_predictions.append(predictions)
    
    # Агрегируем предсказания (голосование по большинству)
    if all_predictions:
        final_predictions = []
        
        for i in range(len(all_predictions[0])):
            votes = [pred[i] for pred in all_predictions]
            majority_vote = Counter(votes).most_common(1)[0][0]
            final_predictions.append(majority_vote)
        
        # Сохраняем результат
        test_df = pd.read_csv(test_path)
        result_df = pd.DataFrame({
            'id': test_df['id'],
            'score': final_predictions
        })
        
        result_df.to_csv(pred_path, index=False)
        print(f"💾 Ensemble результаты сохранены: {pred_path}")

if __name__ == "__main__":
    ensemble_predict("test.csv", "ensemble_predictions.csv")
'''
    
    with open("ensemble_inference.py", "w", encoding="utf-8") as f:
        f.write(ensemble_script)
    
    print("📝 Создан скрипт ensemble_inference.py")


def main():
    """Запуск продвинутых экспериментов"""
    
    print("🚀 ЗАПУСК ПРОДВИНУТЫХ ЭКСПЕРИМЕНТОВ")
    print("="*60)
    
    experiments = [
        ("5", "Data Augmentation", run_experiment_5_data_augmentation),
        ("6", "Curriculum Learning", run_experiment_6_curriculum_learning), 
        ("7", "Model Ensemble", run_experiment_7_ensemble),
    ]
    
    results = {}
    
    for exp_id, exp_name, exp_func in experiments:
        try:
            print(f"\n▶️ Запускаем эксперимент {exp_id}: {exp_name}")
            result = exp_func()
            results[exp_id] = result
            print(f"✅ Эксперимент {exp_id} завершен")
            
        except Exception as e:
            print(f"❌ Ошибка в эксперименте {exp_id}: {e}")
            import traceback
            traceback.print_exc()
    
    # Создаем ensemble инференс
    create_ensemble_inference()
    
    print("\n🎯 Все продвинутые эксперименты завершены!")
    print(f"Успешно выполнено: {len(results)} из {len(experiments)}")


if __name__ == "__main__":
    main()
