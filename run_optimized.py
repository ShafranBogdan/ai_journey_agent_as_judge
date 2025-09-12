#!/usr/bin/env python3
"""
Optimized Run Script for Agent-as-Judge Competition
===================================================

Оптимизированный скрипт для финального сабмита в соревновании.
Автоматически выбирает лучшую доступную модель и применяет
различные техники оптимизации для повышения качества.
"""

import os
import pandas as pd
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams
from argparse import ArgumentParser
import re
import json
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")


def find_best_model():
    """Находит лучшую доступную обученную модель"""
    
    # Список моделей в порядке приоритета
    model_candidates = [
        "models/exp4_enhanced_lora/unsloth_model",
        "models/exp2_balanced_lora/unsloth_model", 
        "models/exp3_full_finetuning/unsloth_model",
        "models/exp1_baseline_lora/unsloth_model",
        "baseline/aij_qwen_0.6b",  # Fallback
    ]
    
    for model_path in model_candidates:
        if os.path.exists(model_path):
            print(f"🎯 Используем модель: {model_path}")
            return model_path
    
    # Если ничего не найдено, используем baseline
    print("⚠️ Не найдено обученных моделей, используем baseline")
    return "aij_qwen_0.6b"


def enhanced_prompt_formatting(prompt: str) -> str:
    """Улучшенное форматирование промпта для лучшего понимания"""
    
    # Добавляем четкие инструкции
    system_instruction = """Ты эксперт-судья, оценивающий качество ответов языковых моделей. 

ВАЖНО: Ответь ТОЛЬКО одной цифрой от 0 до 3 (или -1 если критерий неприменим).

Оценивай согласно шкале:
- 0: Ответ неправильный и не соответствует формату
- 1: Формат правильный, но ответ неправильный  
- 2: Ответ правильный, но формат неправильный
- 3: Ответ правильный и формат правильный
- -1: Критерий неприменим

Анализируй внимательно и объективно."""
    
    # Форматируем как диалог
    formatted = f"<|system|>\n{system_instruction}<|end|>\n<|user|>\n{prompt}<|end|>\n<|assistant|>\n"
    
    return formatted


def extract_score_robust(text: str) -> int:
    """Надежное извлечение оценки из текста с множественными стратегиями"""
    
    if not text:
        return 0
    
    text = text.strip()
    
    # Стратегия 1: Первый символ - цифра
    if text and text[0].isdigit():
        first_digit = int(text[0])
        if first_digit in [-1, 0, 1, 2, 3]:
            return first_digit
    
    # Стратегия 2: Ищем -1 в начале
    if text.startswith('-1'):
        return -1
    
    # Стратегия 3: Регулярные выражения
    patterns = [
        r'^(-1|[0-3])(?:\s|$|[^\d])',  # Число в начале
        r'(?:Оценка|Балл|Score):\s*(-1|[0-3])',  # С подписью
        r'(?:^|\s)(-1|[0-3])(?:\s|$)',  # В любом месте
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return int(match.group(1))
    
    # Стратегия 4: Поиск всех цифр и выбор наиболее подходящей
    digits = re.findall(r'-?\d+', text)
    valid_scores = []
    
    for digit_str in digits:
        try:
            digit = int(digit_str)
            if digit in [-1, 0, 1, 2, 3]:
                valid_scores.append(digit)
        except:
            continue
    
    if valid_scores:
        # Если есть валидные оценки, берем первую
        return valid_scores[0]
    
    # Стратегия 5: Эвристики по содержанию
    text_lower = text.lower()
    
    if any(word in text_lower for word in ['отличн', 'превосходн', 'идеальн', 'perfect']):
        return 3
    elif any(word in text_lower for word in ['хорош', 'правильн', 'correct', 'good']):
        return 2  
    elif any(word in text_lower for word in ['частичн', 'неполн', 'partial']):
        return 1
    elif any(word in text_lower for word in ['неверн', 'неправильн', 'wrong', 'incorrect']):
        return 0
    elif any(word in text_lower for word in ['неприменим', 'не применим', 'not applicable']):
        return -1
    
    # Последний resort - возвращаем 0
    return 0


def ensemble_prediction(prompts: List[str], model, tokenizer, 
                       num_runs: int = 3, temperature: float = 0.1) -> List[int]:
    """Ансамблевое предсказание для повышения надежности"""
    
    all_predictions = []
    
    for run in range(num_runs):
        # Немного варьируем температуру для разнообразия
        temp = temperature if run == 0 else temperature + 0.05 * run
        
        # Форматируем промпты
        formatted_prompts = [enhanced_prompt_formatting(p) for p in prompts]
        
        # Генерируем ответы
        sampling_params = SamplingParams(
            max_tokens=15,  # Увеличиваем для большей надежности
            temperature=temp,
            top_p=0.9,
            frequency_penalty=0.0,
            presence_penalty=0.0,
        )
        
        outputs = model.generate(formatted_prompts, sampling_params)
        answers = [output.outputs[0].text for output in outputs]
        
        # Извлекаем оценки
        scores = [extract_score_robust(answer) for answer in answers]
        all_predictions.append(scores)
    
    # Агрегируем предсказания (мода)
    final_predictions = []
    for i in range(len(prompts)):
        run_predictions = [pred[i] for pred in all_predictions]
        
        # Находим наиболее частую оценку
        from collections import Counter
        counter = Counter(run_predictions)
        most_common = counter.most_common(1)[0][0]
        
        final_predictions.append(most_common)
    
    return final_predictions


def adaptive_batch_processing(prompts: List[str], model, tokenizer, 
                            initial_batch_size: int = 32) -> List[int]:
    """Адаптивная обработка батчами с автоматическим уменьшением размера при OOM"""
    
    batch_size = initial_batch_size
    all_scores = []
    i = 0
    
    while i < len(prompts):
        try:
            batch_prompts = prompts[i:i + batch_size]
            
            # Обычное предсказание
            formatted_prompts = [enhanced_prompt_formatting(p) for p in batch_prompts]
            
            sampling_params = SamplingParams(
                max_tokens=10,
                temperature=0.0,
                top_p=1.0,
            )
            
            outputs = model.generate(formatted_prompts, sampling_params)
            answers = [output.outputs[0].text for output in outputs]
            scores = [extract_score_robust(answer) for answer in answers]
            
            all_scores.extend(scores)
            i += batch_size
            
        except torch.cuda.OutOfMemoryError:
            # Уменьшаем размер батча при нехватке памяти
            batch_size = max(1, batch_size // 2)
            print(f"⚠️ OOM! Уменьшаем batch_size до {batch_size}")
            torch.cuda.empty_cache()
            continue
            
        except Exception as e:
            print(f"❌ Ошибка в батче {i}: {e}")
            # Пропускаем проблемный батч и присваиваем 0
            batch_scores = [0] * len(prompts[i:i + batch_size])
            all_scores.extend(batch_scores)
            i += batch_size
    
    return all_scores


def main(test_path: str, pred_path: str):
    """Основная функция с оптимизированным пайплайном"""
    
    print("🚀 Запуск оптимизированного Agent-as-Judge...")
    
    # Находим лучшую модель
    MODEL_PATH = find_best_model()
    
    # Загружаем тестовые данные
    test_df = pd.read_csv(test_path)
    print(f"📊 Загружено {len(test_df)} примеров")
    
    # Инициализируем модель
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        
        llm = LLM(
            model=MODEL_PATH,
            tensor_parallel_size=1,
            trust_remote_code=True,
            enforce_eager=True,
            dtype='bfloat16',
            max_model_len=3072,  # Увеличиваем для длинных промптов
            gpu_memory_utilization=0.85  # Оптимизируем использование GPU
        )
        
        print("✅ Модель успешно загружена")
        
    except Exception as e:
        print(f"❌ Ошибка загрузки модели: {e}")
        return
    
    # Получаем предсказания с адаптивным батчингом
    print("🔮 Генерируем предсказания...")
    
    prompts = test_df['prompt'].tolist()
    
    # Выбираем стратегию в зависимости от размера данных
    if len(prompts) < 1000:
        # Для небольших данных используем ансамбль
        print("🎭 Используем ансамблевое предсказание...")
        results = ensemble_prediction(prompts, llm, tokenizer)
    else:
        # Для больших данных используем адаптивный батчинг
        print("⚡ Используем адаптивное батчирование...")
        results = adaptive_batch_processing(prompts, llm, tokenizer)
    
    # Постобработка результатов
    print("🔧 Постобработка результатов...")
    
    # Применяем простую эвристику для корректировки
    processed_results = []
    for i, (result, prompt) in enumerate(zip(results, prompts)):
        
        # Корректировка на основе анализа промпта
        if 'не применим' in prompt.lower() or 'неприменим' in prompt.lower():
            # Если в промпте явно указано, что критерий не применим
            processed_result = -1
        elif result not in [-1, 0, 1, 2, 3]:
            # Если результат некорректный, исправляем на 0
            processed_result = 0
        else:
            processed_result = result
        
        processed_results.append(processed_result)
    
    # Сохраняем результаты
    result_df = pd.DataFrame({
        'id': test_df['id'],
        'score': processed_results
    })
    
    result_df.to_csv(pred_path, index=False)
    
    print(f"💾 Результаты сохранены в {pred_path}")
    print("\n📈 Статистика предсказаний:")
    print(result_df['score'].value_counts().sort_index())
    
    # Сохраняем дополнительную информацию
    meta_info = {
        'model_used': MODEL_PATH,
        'total_examples': len(test_df),
        'prediction_distribution': result_df['score'].value_counts().to_dict(),
        'processing_strategy': 'ensemble' if len(prompts) < 1000 else 'adaptive_batching'
    }
    
    meta_path = pred_path.replace('.csv', '_meta.json')
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta_info, f, indent=2, ensure_ascii=False)
    
    print(f"📋 Метаинформация сохранена в {meta_path}")
    print("✅ Обработка завершена успешно!")


if __name__ == "__main__":
    parser = ArgumentParser(description="Optimized Agent-as-Judge Inference")
    parser.add_argument("--test_path", type=str, required=True,
                       help="Путь к тестовому CSV файлу")
    parser.add_argument("--pred_path", type=str, required=True,
                       help="Путь для сохранения предсказаний")
    
    args = parser.parse_args()
    main(args.test_path, args.pred_path)
