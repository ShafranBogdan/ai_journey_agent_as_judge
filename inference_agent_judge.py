#!/usr/bin/env python3
"""
Agent-as-Judge Model Inference Script
=====================================

Улучшенный скрипт для инференса обученной модели-судьи.
Поддерживает различные стратегии инференса и post-processing.
"""

import os
import pandas as pd
import numpy as np
import torch
from typing import List, Dict, Optional, Union
import argparse
import re
from pathlib import Path
import json
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# Imports для работы с моделью
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams
from unsloth import FastModel
import gc


class AgentJudgeInference:
    """Класс для инференса Agent-as-Judge модели"""
    
    def __init__(self, model_path: str, device: str = "auto", 
                 use_vllm: bool = True, max_length: int = 10):
        self.model_path = model_path
        self.device = device
        self.use_vllm = use_vllm
        self.max_length = max_length
        self.model = None
        self.tokenizer = None
        
        self.load_model()
    
    def load_model(self):
        """Загружает модель и токенизатор"""
        
        print(f"🤖 Загружаем модель из {self.model_path}...")
        
        try:
            if self.use_vllm:
                # Используем vLLM для быстрого инференса
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
                self.model = LLM(
                    model=self.model_path,
                    tensor_parallel_size=1,
                    trust_remote_code=True,
                    enforce_eager=True,
                    dtype='bfloat16',
                    max_model_len=2048,
                    gpu_memory_utilization=0.8
                )
                print("✅ Модель загружена с vLLM")
                
            else:
                # Используем стандартный transformers
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.bfloat16,
                    device_map=self.device,
                    trust_remote_code=True
                )
                print("✅ Модель загружена с transformers")
                
        except Exception as e:
            print(f"❌ Ошибка загрузки модели: {e}")
            print("💡 Попробуйте использовать baseline модель")
            # Fallback к baseline модели
            self.model_path = "baseline/aij_qwen_0.6b"
            self.load_baseline_model()
    
    def load_baseline_model(self):
        """Загружает baseline модель как fallback"""
        
        print("🔄 Загружаем baseline модель...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        
        if self.use_vllm:
            self.model = LLM(
                model=self.model_path,
                tensor_parallel_size=1,
                trust_remote_code=True,
                enforce_eager=True,
                dtype='bfloat16'
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16,
                device_map=self.device,
                trust_remote_code=True
            )
    
    def format_prompt_for_inference(self, prompt: str) -> str:
        """Форматирует промпт для инференса"""
        
        # Добавляем системный промпт если его нет
        system_prompt = """Ты - модель-судья, которая оценивает качество ответов других языковых моделей. Проанализируй предоставленную информацию и выставь оценку согласно указанному критерию и шкале оценивания. Ответь только числом от 0 до 3 (или -1, если критерий не применим)."""
        
        conversation = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        formatted = self.tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
        
        return formatted
    
    def predict_batch_vllm(self, prompts: List[str]) -> List[str]:
        """Batch предсказание с vLLM"""
        
        # Форматируем промпты
        formatted_prompts = [self.format_prompt_for_inference(p) for p in prompts]
        
        # Настраиваем параметры генерации
        sampling_params = SamplingParams(
            max_tokens=self.max_length,
            temperature=0.0,  # Детерминированная генерация
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            stop_token_ids=[self.tokenizer.eos_token_id]
        )
        
        # Генерируем ответы
        outputs = self.model.generate(formatted_prompts, sampling_params)
        
        # Извлекаем тексты
        results = [output.outputs[0].text.strip() for output in outputs]
        
        return results
    
    def predict_batch_transformers(self, prompts: List[str]) -> List[str]:
        """Batch предсказание с transformers"""
        
        results = []
        
        for prompt in tqdm(prompts, desc="Генерируем ответы"):
            formatted_prompt = self.format_prompt_for_inference(prompt)
            
            # Токенизация
            inputs = self.tokenizer(
                formatted_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048
            ).to(self.model.device)
            
            # Генерация
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_length,
                    temperature=0.0,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # Декодирование
            generated_text = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            ).strip()
            
            results.append(generated_text)
            
            # Очистка памяти
            del inputs, outputs
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return results
    
    def extract_score_from_text(self, text: str) -> int:
        """Извлекает числовую оценку из сгенерированного текста"""
        
        # Убираем лишние символы и пробелы
        text = text.strip()
        
        # Ищем цифры в начале текста
        if text and text[0].isdigit():
            score = int(text[0])
            # Проверяем валидность оценки
            if score in [0, 1, 2, 3] or score == -1:
                return score
        
        # Ищем паттерны с оценками
        patterns = [
            r'^\s*(-1|[0-3])\s*$',  # Только число
            r'Оценка:\s*(-1|[0-3])',  # "Оценка: X"
            r'Балл:\s*(-1|[0-3])',    # "Балл: X"
            r'Score:\s*(-1|[0-3])',   # "Score: X"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return int(match.group(1))
        
        # Если ничего не найдено, возвращаем 0 (наиболее безопасное значение)
        return 0
    
    def predict(self, prompts: List[str], batch_size: int = 32) -> List[int]:
        """Основной метод для предсказания оценок"""
        
        all_scores = []
        
        # Обрабатываем батчами
        for i in tqdm(range(0, len(prompts), batch_size), desc="Обрабатываем батчи"):
            batch_prompts = prompts[i:i + batch_size]
            
            if self.use_vllm:
                batch_results = self.predict_batch_vllm(batch_prompts)
            else:
                batch_results = self.predict_batch_transformers(batch_prompts)
            
            # Извлекаем оценки
            batch_scores = [self.extract_score_from_text(result) for result in batch_results]
            all_scores.extend(batch_scores)
            
            # Очистка памяти
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return all_scores
    
    def evaluate_on_validation(self, val_data_path: str) -> Dict:
        """Оценивает модель на валидационных данных"""
        
        print("📊 Оценка модели на валидационных данных...")
        
        # Загружаем данные
        df = pd.read_csv(val_data_path)
        prompts = df['prompt'].tolist()
        true_scores = df['score'].tolist()
        
        # Получаем предсказания
        predicted_scores = self.predict(prompts)
        
        # Вычисляем метрики
        from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
        
        accuracy = accuracy_score(true_scores, predicted_scores)
        f1 = f1_score(true_scores, predicted_scores, average='weighted')
        
        # Создаем отчет
        report = {
            'accuracy': accuracy,
            'f1_score': f1,
            'classification_report': classification_report(
                true_scores, predicted_scores, output_dict=True
            ),
            'confusion_matrix': confusion_matrix(true_scores, predicted_scores).tolist()
        }
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1-score: {f1:.4f}")
        
        return report


def main():
    parser = argparse.ArgumentParser(description="Agent-as-Judge Model Inference")
    parser.add_argument("--test_path", type=str, required=True,
                       help="Путь к тестовому CSV файлу")
    parser.add_argument("--pred_path", type=str, required=True,
                       help="Путь для сохранения предсказаний")
    parser.add_argument("--model_path", type=str, 
                       default="models/exp1_baseline_lora/unsloth_model",
                       help="Путь к обученной модели")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Размер батча для инференса")
    parser.add_argument("--use_vllm", action="store_true", default=True,
                       help="Использовать vLLM для инференса")
    parser.add_argument("--max_length", type=int, default=10,
                       help="Максимальная длина генерации")
    parser.add_argument("--validate", type=str, default=None,
                       help="Путь к валидационным данным для оценки")
    
    args = parser.parse_args()
    
    # Проверяем существование модели
    if not os.path.exists(args.model_path):
        print(f"⚠️ Модель не найдена по пути: {args.model_path}")
        print("🔄 Используем baseline модель...")
        args.model_path = "baseline/aij_qwen_0.6b"
    
    try:
        # Инициализируем inference
        predictor = AgentJudgeInference(
            model_path=args.model_path,
            use_vllm=args.use_vllm,
            max_length=args.max_length
        )
        
        # Если нужна валидация
        if args.validate:
            report = predictor.evaluate_on_validation(args.validate)
            
            # Сохраняем отчет
            report_path = args.pred_path.replace('.csv', '_validation_report.json')
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            print(f"📋 Отчет сохранен: {report_path}")
        
        # Загружаем тестовые данные
        print(f"📂 Загружаем тестовые данные: {args.test_path}")
        test_df = pd.read_csv(args.test_path)
        
        print(f"📊 Найдено {len(test_df)} примеров для обработки")
        
        # Получаем предсказания
        print("🔮 Генерируем предсказания...")
        predictions = predictor.predict(
            test_df['prompt'].tolist(),
            batch_size=args.batch_size
        )
        
        # Сохраняем результаты
        results_df = pd.DataFrame({
            'id': test_df['id'],
            'score': predictions
        })
        
        results_df.to_csv(args.pred_path, index=False)
        print(f"💾 Результаты сохранены: {args.pred_path}")
        
        # Показываем статистику предсказаний
        print("\n📈 Статистика предсказаний:")
        print(results_df['score'].value_counts().sort_index())
        
    except Exception as e:
        print(f"❌ Ошибка во время инференса: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
