#!/usr/bin/env python3
"""
Simple Advanced Experiments for Agent-as-Judge
==============================================

Упрощенные эксперименты для улучшения качества модели:
- Data augmentation (простое)
- Balanced classes (простое)
- Different LoRA configurations (простое)
- Basic ensemble (простое)

Убраны все сложности - только работающие идеи.
Основан на стиле Agent_as_judge_finetune.ipynb
"""

import pandas as pd
import numpy as np
import torch
import random
import re
from datasets import Dataset
from unsloth import FastModel
from unsloth.chat_templates import get_chat_template, train_on_responses_only
from trl import SFTTrainer, SFTConfig
import warnings
warnings.filterwarnings("ignore")

# Продвинутые импорты
try:
    import optuna
    OPTUNA_AVAILABLE = True
    print("✅ Optuna доступен для hyperparameter optimization")
except ImportError:
    OPTUNA_AVAILABLE = False
    print("⚠️ Optuna недоступен - установите: pip install optuna")

from collections import Counter
import json
import os


def set_seed(seed=42):
    """Устанавливает seed для воспроизводимости"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def simple_augment_data(df, augment_ratio=0.3):
    """Простая аугментация данных"""
    print(f"🔄 Применяем аугментацию данных (ratio={augment_ratio})...")
    
    replacements = {
        "задание для оценки": ["задача для анализа", "пример для оценки"],
        "эталонный ответ": ["правильный ответ", "корректный ответ"],
        "ответ для оценки": ["проверяемый ответ", "анализируемый ответ"],
        "критерий оценки": ["параметр оценки", "показатель качества"]
    }
    
    augmented_rows = []
    num_to_augment = int(len(df) * augment_ratio)
    sample_indices = random.sample(range(len(df)), min(num_to_augment, len(df)))
    
    for idx in sample_indices:
        row = df.iloc[idx].copy()
        prompt = row['prompt']
        
        # Применяем замены
        for original, variants in replacements.items():
            if original.lower() in prompt.lower():
                new_phrase = random.choice(variants)
                prompt = prompt.lower().replace(original.lower(), new_phrase.lower())
        
        # Небольшие вариации
        if random.random() < 0.5:
            prompt = prompt.replace("выполни", "выполните")
            prompt = prompt.replace("найди", "найдите")
        
        row['prompt'] = prompt
        augmented_rows.append(row)
    
    augmented_df = pd.concat([df, pd.DataFrame(augmented_rows)], ignore_index=True)
    print(f"После аугментации: {len(augmented_df)} примеров (+{len(augmented_rows)} новых)")
    
    return augmented_df.sample(frac=1, random_state=42)


def balance_classes(df, method="undersample"):
    """Простая балансировка классов"""
    print(f"⚖️ Балансируем классы методом '{method}'...")
    
    score_counts = df['score'].value_counts()
    print("До балансировки:", score_counts.sort_index().to_dict())
    
    if method == "undersample":
        min_count = score_counts.min()
        balanced_dfs = []
        
        for score in df['score'].unique():
            score_df = df[df['score'] == score]
            if len(score_df) > min_count:
                score_df = score_df.sample(n=min_count, random_state=42)
            balanced_dfs.append(score_df)
        
        balanced_df = pd.concat(balanced_dfs, ignore_index=True)
        
    elif method == "oversample":
        max_count = score_counts.max()
        balanced_dfs = []
        
        for score in df['score'].unique():
            score_df = df[df['score'] == score]
            while len(score_df) < max_count:
                to_add = min(max_count - len(score_df), len(df[df['score'] == score]))
                extra = df[df['score'] == score].sample(n=to_add, replace=True, random_state=42)
                score_df = pd.concat([score_df, extra], ignore_index=True)
            balanced_dfs.append(score_df)
        
        balanced_df = pd.concat(balanced_dfs, ignore_index=True)
    else:
        balanced_df = df
    
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print("После балансировки:", balanced_df['score'].value_counts().sort_index().to_dict())
    return balanced_df


def setup_model_simple(model_name="Qwen/Qwen3-0.6B", max_seq_length=1024, lora_r=16):
    """Простая настройка модели"""
    print(f"🤖 Настраиваем модель {model_name} с LoRA rank={lora_r}...")
    
    model, tokenizer = FastModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
        dtype=None
    )
    
    model = FastModel.get_peft_model(
        model,
        r=lora_r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 
                       "gate_proj", "up_proj", "down_proj"],
        lora_alpha=32,
        lora_dropout=0.1,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )
    
    tokenizer = get_chat_template(tokenizer, chat_template="qwen-3")
    print("✅ Модель настроена")
    
    return model, tokenizer


def prepare_dataset_simple(df, tokenizer, test_size=0.2):
    """Простая подготовка данных"""
    print("📝 Подготавливаем данные...")
    
    dataset = Dataset.from_pandas(df[['prompt', 'score']])
    
    def format_examples(examples):
        messages = [
            [{'role': 'user', 'content': prompt}, 
             {'role': 'assistant', 'content': str(int(score))}]
            for prompt, score in zip(examples['prompt'], examples['score'])
        ]
        
        texts = [
            tokenizer.apply_chat_template(
                message, tokenize=False, add_generation_prompt=False
            ) 
            for message in messages
        ]
        
        return {"text": texts}
    
    dataset = dataset.map(format_examples, batched=True)
    dataset = dataset.train_test_split(test_size=test_size, seed=42)
    
    print(f"Обучающая выборка: {len(dataset['train'])}")
    print(f"Тестовая выборка: {len(dataset['test'])}")
    
    return dataset


def train_simple(model, tokenizer, dataset, output_dir, epochs=2, batch_size=2, learning_rate=2e-4):
    """Простое обучение"""
    print(f"🚀 Начинаем обучение на {epochs} эпох...")
    
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'],
        args=SFTConfig(
            dataset_text_field="text",
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=2,
            warmup_ratio=0.05,
            num_train_epochs=epochs,
            learning_rate=learning_rate,
            logging_steps=10,
            optim="adamw_torch",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=42,
            output_dir=output_dir,
            # Убираем evaluation_strategy - не поддерживается в новых версиях SFTConfig
            # evaluation_strategy="steps",
            # eval_steps=50,
            save_steps=100,
            save_total_limit=2,
            report_to="none",
        ),
    )
    
    # Обучение только на ответах - ИСПРАВЛЕНО с отключением мультипроцессинга
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
        num_proc=1  # Отключаем мультипроцессинг для избежания ошибок
    )
    
    trainer.train()
    trainer.save_model()
    
    print(f"✅ Обучение завершено, модель сохранена в {output_dir}")
    return trainer


def simple_inference(model, tokenizer, test_prompts, max_tokens=10):
    """Простой инференс для оценки модели"""
    results = []
    
    model.eval()
    with torch.no_grad():
        for prompt in test_prompts:
            # Форматируем как в run.py
            messages = [{"role": "user", "content": prompt}]
            formatted = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            
            # Токенизация и генерация
            inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
            outputs = model.generate(
                **inputs, max_new_tokens=max_tokens, temperature=0.0, do_sample=False
            )
            
            # Декодирование
            generated = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
            score = int(generated[0]) if generated and generated[0].isdigit() else 0
            results.append(score)
    
    return results


def optuna_objective(trial, train_df, val_df):
    """Objective function для Optuna оптимизации"""
    if not OPTUNA_AVAILABLE:
        raise ImportError("Optuna не установлен")
    
    # Предлагаемые гиперпараметры
    lora_r = trial.suggest_categorical('lora_r', [8, 16, 32, 64])
    lora_alpha = trial.suggest_categorical('lora_alpha', [16, 32, 64, 128])
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 5e-4, log=True)
    batch_size = trial.suggest_categorical('batch_size', [1, 2, 4])
    epochs = trial.suggest_int('epochs', 1, 3)
    
    try:
        # Настройка модели с предложенными параметрами
        model, tokenizer = FastModel.from_pretrained(
            model_name="Qwen/Qwen3-0.6B",
            max_seq_length=1024,
            load_in_4bit=True,
            dtype=None
        )
        
        model = FastModel.get_peft_model(
            model,
            r=lora_r,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_alpha=lora_alpha,
            lora_dropout=0.1,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=42,
        )
        
        tokenizer = get_chat_template(tokenizer, chat_template="qwen-3")
        
        # Подготовка данных
        train_dataset = prepare_dataset_simple(train_df, tokenizer, test_size=0.0)['train']
        val_dataset = prepare_dataset_simple(val_df, tokenizer, test_size=0.0)['train']
        
        # Обучение
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=None,  # Убираем eval во время оптимизации
            args=SFTConfig(
                dataset_text_field="text",
                per_device_train_batch_size=batch_size,
                gradient_accumulation_steps=2,
                warmup_ratio=0.05,
                num_train_epochs=epochs,
                learning_rate=learning_rate,
                logging_steps=100,
                optim="adamw_torch",
                weight_decay=0.01,
                lr_scheduler_type="linear",
                seed=42,
                output_dir=f"./optuna_trial_{trial.number}",
                save_strategy="no",  # Не сохраняем промежуточные результаты
                report_to="none",
            ),
        )
        
        trainer = train_on_responses_only(
            trainer,
            instruction_part="<|im_start|>user\n",
            response_part="<|im_start|>assistant\n",
        )
        
        trainer.train()
        
        # Оценка на валидационных данных
        val_predictions = simple_inference(model, tokenizer, val_df['prompt'].tolist())
        val_true = val_df['score'].tolist()
        
        # Простая метрика - accuracy
        accuracy = sum(p == t for p, t in zip(val_predictions, val_true)) / len(val_true)
        
        # Очистка памяти
        del model, tokenizer, trainer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return accuracy
        
    except Exception as e:
        print(f"Trial {trial.number} failed: {e}")
        return 0.0


def run_optuna_optimization(n_trials=20):
    """Запускает Optuna оптимизацию гиперпараметров"""
    if not OPTUNA_AVAILABLE:
        print("❌ Optuna не установлен, пропускаем оптимизацию")
        return None
    
    print("\n" + "="*60)
    print("🎯 OPTUNA HYPERPARAMETER OPTIMIZATION")
    print("="*60)
    
    # Загружаем и подготавливаем данные
    df = pd.read_csv("aij_judge_task_1_train.csv")
    sample_df = df.sample(n=500, random_state=42)  # Маленькая выборка для быстрой оптимизации
    
    # Разбиваем на train/val для оптимизации
    train_df = sample_df.iloc[:400]
    val_df = sample_df.iloc[400:]
    
    print(f"Оптимизация на {len(train_df)} train + {len(val_df)} val примерах")
    print(f"Запускаем {n_trials} trials...")
    
    # Создаем study
    study = optuna.create_study(direction='maximize', study_name="agent_judge_hp_opt")
    
    # Оптимизируем
    study.optimize(
        lambda trial: optuna_objective(trial, train_df, val_df), 
        n_trials=n_trials,
        timeout=1800,  # 30 минут максимум
        n_jobs=1  # Без параллелизма для стабильности
    )
    
    # Результаты
    print(f"\n🏆 Лучшие параметры (accuracy={study.best_value:.4f}):")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    # Сохраняем результаты
    with open("optuna_best_params.json", "w") as f:
        json.dump({
            "best_params": study.best_params,
            "best_value": study.best_value,
            "n_trials": len(study.trials)
        }, f, indent=2)
    
    return study.best_params


class SimpleEnsemble:
    """Простой ensemble из нескольких моделей"""
    
    def __init__(self):
        self.models = []
        self.tokenizers = []
        self.model_weights = []
    
    def add_model(self, model, tokenizer, weight=1.0):
        """Добавляет модель в ensemble"""
        self.models.append(model)
        self.tokenizers.append(tokenizer)
        self.model_weights.append(weight)
        print(f"✅ Добавлена модель в ensemble (всего: {len(self.models)})")
    
    def predict(self, test_prompts, max_tokens=10):
        """Ensemble предсказание"""
        if not self.models:
            raise ValueError("Нет моделей в ensemble")
        
        all_predictions = []
        
        # Получаем предсказания от каждой модели
        for i, (model, tokenizer, weight) in enumerate(zip(self.models, self.tokenizers, self.model_weights)):
            print(f"🔮 Получаем предсказания от модели {i+1}/{len(self.models)}...")
            
            model_preds = simple_inference(model, tokenizer, test_prompts, max_tokens)
            all_predictions.append(model_preds)
        
        # Агрегируем предсказания (мода/голосование по большинству)
        final_predictions = []
        
        for i in range(len(test_prompts)):
            # Собираем голоса всех моделей для этого примера
            votes = []
            for j, model_preds in enumerate(all_predictions):
                model_pred = model_preds[i]
                weight = int(self.model_weights[j])
                votes.extend([model_pred] * weight)
            
            # Находим наиболее частый ответ
            vote_counts = Counter(votes)
            majority_vote = vote_counts.most_common(1)[0][0]
            final_predictions.append(majority_vote)
        
        return final_predictions
    
    def save_ensemble(self, output_dir):
        """Сохраняет все модели ensemble"""
        os.makedirs(output_dir, exist_ok=True)
        
        for i, (model, tokenizer) in enumerate(zip(self.models, self.tokenizers)):
            model_dir = os.path.join(output_dir, f"model_{i}")
            model.save_pretrained(model_dir)
            tokenizer.save_pretrained(model_dir)
        
        # Сохраняем метаинформацию
        meta = {
            "num_models": len(self.models),
            "model_weights": self.model_weights
        }
        with open(os.path.join(output_dir, "ensemble_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        
        print(f"💾 Ensemble сохранен в {output_dir}")


def create_ensemble_models():
    """Создает ensemble из разных конфигураций"""
    print("\n" + "="*60)
    print("🎭 CREATING ENSEMBLE MODELS")
    print("="*60)
    
    # Загружаем данные
    df = pd.read_csv("aij_judge_task_1_train.csv")
    sample_df = df.sample(n=1000, random_state=42)
    
    ensemble = SimpleEnsemble()
    
    # Конфигурации для разных моделей
    configs = [
        {"lora_r": 16, "lr": 2e-4, "epochs": 2, "name": "conservative"},
        {"lora_r": 32, "lr": 1.5e-4, "epochs": 3, "name": "balanced"}, 
        {"lora_r": 64, "lr": 1e-4, "epochs": 2, "name": "expressive"}
    ]
    
    for i, config in enumerate(configs):
        print(f"\n🔄 Обучаем модель {i+1}: {config['name']}")
        set_seed(42 + i)  # Разные seeds для разнообразия
        
        try:
            # Настройка модели
            model, tokenizer = setup_model_simple(lora_r=config["lora_r"])
            
            # Подготовка данных (разные разбиения для разнообразия)
            dataset = prepare_dataset_simple(sample_df, tokenizer, test_size=0.1 + i*0.05)
            
            # Обучение
            trainer = train_simple(
                model, tokenizer, dataset, f"./ensemble_model_{i}", 
                epochs=config["epochs"], learning_rate=config["lr"]
            )
            
            # Добавляем в ensemble с весом
            weight = 1.0 if config["name"] != "balanced" else 1.5  # Больший вес сбалансированной модели
            ensemble.add_model(model, tokenizer, weight)
            
        except Exception as e:
            print(f"❌ Ошибка в модели {i+1}: {e}")
            continue
    
    # Сохраняем ensemble
    if ensemble.models:
        ensemble.save_ensemble("./final_ensemble")
        
        # Тестируем ensemble
        test_sample = sample_df.head(20)
        ensemble_preds = ensemble.predict(test_sample['prompt'].tolist())
        
        print(f"\n📊 Тест ensemble на {len(test_sample)} примерах:")
        for i, (true_score, pred_score) in enumerate(zip(test_sample['score'], ensemble_preds)):
            print(f"  {i+1}: True={true_score}, Pred={pred_score}")
        
        accuracy = sum(t == p for t, p in zip(test_sample['score'], ensemble_preds)) / len(test_sample)
        print(f"Accuracy: {accuracy:.3f}")
    
    return ensemble


def run_experiment_5_data_augmentation():
    """Эксперимент 5: Data Augmentation (упрощенный)"""
    print("\n" + "="*60)
    print("🧪 ЭКСПЕРИМЕНТ 5: Simple Data Augmentation")
    print("="*60)
    
    set_seed(42)
    
    # Загружаем данные
    df = pd.read_csv("aij_judge_task_1_train.csv")
    print(f"Исходные данные: {len(df)} примеров")
    
    # Берем выборку для быстрого эксперимента
    sample_df = df.sample(n=1500, random_state=42)
    
    # Применяем аугментацию
    augmented_df = simple_augment_data(sample_df, augment_ratio=0.4)
    
    # Балансируем
    final_df = balance_classes(augmented_df, method="undersample")
    
    # Настройка модели
    model, tokenizer = setup_model_simple(lora_r=32)
    
    # Подготовка данных
    dataset = prepare_dataset_simple(final_df, tokenizer)
    
    # Обучение
    trainer = train_simple(
        model, tokenizer, dataset, "./exp5_augmented", 
        epochs=3, learning_rate=2e-4
    )
    
    return trainer


def run_experiment_6_balanced():
    """Эксперимент 6: Balanced Classes (упрощенный)"""
    print("\n" + "="*60)
    print("🧪 ЭКСПЕРИМЕНТ 6: Simple Balanced Classes")
    print("="*60)
    
    set_seed(42)
    
    # Загружаем и балансируем данные
    df = pd.read_csv("aij_judge_task_1_train.csv")
    sample_df = df.sample(n=2000, random_state=42)
    balanced_df = balance_classes(sample_df, method="undersample")
    
    # Настройка модели
    model, tokenizer = setup_model_simple(lora_r=16)
    
    # Подготовка данных
    dataset = prepare_dataset_simple(balanced_df, tokenizer)
    
    # Обучение
    trainer = train_simple(
        model, tokenizer, dataset, "./exp6_balanced", 
        epochs=3, learning_rate=1.5e-4
    )
    
    return trainer


def run_experiment_7_ranks():
    """Эксперимент 7: Different LoRA Ranks (упрощенный)"""
    print("\n" + "="*60)
    print("🧪 ЭКСПЕРИМЕНТ 7: Simple LoRA Ranks")
    print("="*60)
    
    results = {}
    df = pd.read_csv("aij_judge_task_1_train.csv")
    sample_df = df.sample(n=800, random_state=42)
    
    for rank in [8, 16, 32]:  # Упрощаем список
        print(f"\n🔄 Тестируем LoRA rank = {rank}")
        set_seed(42)
        
        model, tokenizer = setup_model_simple(lora_r=rank)
        dataset = prepare_dataset_simple(sample_df, tokenizer)
        
        trainer = train_simple(
            model, tokenizer, dataset, f"./exp7_rank_{rank}", 
            epochs=2, batch_size=4
        )
        
        results[f"rank_{rank}"] = trainer
        
        # Очистка памяти
        del model, tokenizer, trainer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return results


def run_experiment_8_optuna():
    """Эксперимент 8: Optuna Hyperparameter Optimization"""
    if not OPTUNA_AVAILABLE:
        print("❌ Optuna недоступен, устанавливаем...")
        import subprocess
        subprocess.check_call(["pip", "install", "optuna"])
        try:
            import optuna
            print("✅ Optuna установлен успешно")
        except ImportError:
            print("❌ Не удалось установить Optuna")
            return None
    
    best_params = run_optuna_optimization(n_trials=15)
    
    if best_params:
        print("\n🔄 Обучаем финальную модель с лучшими параметрами...")
        
        # Обучаем модель с найденными параметрами
        df = pd.read_csv("aij_judge_task_1_train.csv")
        sample_df = df.sample(n=1500, random_state=42)
        
        model, tokenizer = setup_model_simple(lora_r=best_params.get('lora_r', 32))
        dataset = prepare_dataset_simple(sample_df, tokenizer)
        
        trainer = train_simple(
            model, tokenizer, dataset, "./exp8_optuna_best",
            epochs=best_params.get('epochs', 3),
            batch_size=best_params.get('batch_size', 2),
            learning_rate=best_params.get('learning_rate', 2e-4)
        )
        
        return trainer
    
    return None


def run_experiment_9_ensemble():
    """Эксперимент 9: Ensemble Models"""
    print("\n" + "="*60)
    print("🧪 ЭКСПЕРИМЕНТ 9: Ensemble Models")
    print("="*60)
    
    return create_ensemble_models()


def main():
    """Главная функция - запускает продвинутые эксперименты"""
    print("🚀 ПРОДВИНУТЫЕ ЭКСПЕРИМЕНТЫ С OPTUNA И ENSEMBLE")
    print("="*60)
    
    experiments = [
        ("5", "Data Augmentation", run_experiment_5_data_augmentation),
        ("6", "Balanced Classes", run_experiment_6_balanced), 
        ("7", "LoRA Ranks", run_experiment_7_ranks),
        ("8", "Optuna Optimization", run_experiment_8_optuna),
        ("9", "Ensemble Models", run_experiment_9_ensemble),
    ]
    
    results = {}
    
    # Интерактивный выбор эксперимента
    print("\nВыберите эксперименты для запуска:")
    for exp_id, exp_name, _ in experiments:
        print(f"  {exp_id} - {exp_name}")
    print("  all - Все эксперименты")
    print("  advanced - Только Optuna + Ensemble (8,9)")
    
    choice = input("\nВведите номер(а) через запятую или 'all'/'advanced': ").strip()
    
    if choice == "all":
        selected_experiments = experiments
    elif choice == "advanced":
        selected_experiments = [exp for exp in experiments if exp[0] in ["8", "9"]]
    else:
        selected_ids = [x.strip() for x in choice.split(",")]
        selected_experiments = [exp for exp in experiments if exp[0] in selected_ids]
    
    print(f"\n🎯 Запускаем {len(selected_experiments)} экспериментов...")
    
    for exp_id, exp_name, exp_func in selected_experiments:
        try:
            print(f"\n▶️ Запускаем эксперимент {exp_id}: {exp_name}")
            result = exp_func()
            results[exp_id] = result
            print(f"✅ Эксперимент {exp_id} завершен")
            
        except Exception as e:
            print(f"❌ Ошибка в эксперименте {exp_id}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n🎯 Все эксперименты завершены!")
    print(f"Успешно выполнено: {len(results)} из {len(selected_experiments)}")
    
    if results:
        print("\n📋 Результаты:")
        print("- Обычные модели сохранены в exp5_*, exp6_*, exp7_*")
        if "8" in results and results["8"]:
            print("- Лучшие параметры Optuna сохранены в optuna_best_params.json")
            print("- Оптимизированная модель сохранена в exp8_optuna_best/")
        if "9" in results and results["9"]:
            print("- Ensemble модели сохранены в final_ensemble/")
            print("- Используйте ensemble для лучшего качества!")
    
    print("\n💡 Рекомендации:")
    print("1. Для лучшего качества используйте ensemble или модель из exp8_optuna_best")
    print("2. Сохраните лучшую модель как aij_qwen_0.6b для использования с run.py")
    print("3. Проверьте optuna_best_params.json для оптимальных гиперпараметров")


def create_run_py_compatible_model():
    """Создает модель совместимую с run.py из лучших результатов"""
    print("🔄 Создаем модель для run.py...")
    
    # Проверяем какие модели доступны
    models_to_check = [
        ("exp8_optuna_best", "Optuna-оптимизированная модель"),
        ("exp6_balanced", "Сбалансированная модель"),
        ("exp5_augmented", "Аугментированная модель"),
    ]
    
    best_model_dir = None
    for model_dir, description in models_to_check:
        if os.path.exists(model_dir):
            best_model_dir = model_dir
            print(f"✅ Найдена {description} в {model_dir}")
            break
    
    if best_model_dir:
        print(f"📁 Копируем {best_model_dir} в aij_qwen_0.6b для run.py...")
        import shutil
        
        if os.path.exists("aij_qwen_0.6b"):
            shutil.rmtree("aij_qwen_0.6b")
        shutil.copytree(best_model_dir, "aij_qwen_0.6b")
        
        print("✅ Модель готова для использования с run.py!")
        print("Теперь можно запускать: python run.py --test_path test.csv --pred_path predictions.csv")
    else:
        print("❌ Не найдено обученных моделей. Сначала запустите эксперименты.")


if __name__ == "__main__":
    main()