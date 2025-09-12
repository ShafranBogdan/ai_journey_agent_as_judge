# Agent-as-Judge Model Training 🤖⚖️

Комплексное решение для обучения модели-судьи в рамках соревнования AI Journey.

## 📋 Описание задачи

Цель: обучить модель, которая будет оценивать ответы других LLM по различным критериям качества. Модель должна выставлять оценки от 0 до 3 (или -1 если критерий неприменим) в трех типах задач:

1. **Статический бенчмарк** - задачи с правильным ответом
2. **Function Calling** - оценка корректности вызовов функций
3. **Генеративные задачи** - открытые задачи без единственно правильного ответа

## 🏗️ Архитектура решения

### Базовая модель
- **Qwen3-0.6B** - компактная и эффективная модель от Alibaba
- **Unsloth** - библиотека для ускоренного обучения
- **LoRA/QLoRA** - эффективное файн-тюнинг с минимальными ресурсами

### Стратегии обучения
1. **Baseline LoRA** - стандартный подход
2. **Balanced Classes** - обучение на сбалансированных данных
3. **Full Fine-tuning** - полное переобучение (при наличии ресурсов)
4. **Enhanced Config** - оптимизированные гиперпараметры

## 📊 Анализ данных

**Тренировочный датасет**: 5,197 примеров
- 0 баллов: 1,445 примеров (28%)
- 1 балл: 1,911 примеров (37%)
- 2 балла: 880 примеров (17%) 
- 3 балла: 961 примеров (18%)

**Типы задач**:
- Математические: 1,531 (29%)
- Генеративные: 1,215 (23%)
- Прочие: 2,451 (47%)

## 🚀 Установка и запуск

### Требования
```bash
# Основные зависимости
pip install torch transformers datasets accelerate peft trl
pip install pandas numpy scikit-learn tqdm wandb

# Unsloth для эффективного обучения  
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# vLLM для быстрого инференса
pip install vllm

# Дополнительные оптимизации
pip install xformers bitsandbytes triton
```

### Быстрый старт

```bash
# 1. Базовое обучение
python train_agent_judge.py --experiment 1

# 2. Обучение на сбалансированных данных
python train_agent_judge.py --experiment 2

# 3. Все эксперименты сразу
python train_agent_judge.py --experiment all

# 4. С логированием в W&B
python train_agent_judge.py --experiment all --wandb_project "agent-judge-exp"
```

### Продвинутые эксперименты

```bash
# Эксперименты с аугментацией данных
python experiments_advanced.py

# Curriculum learning
python experiments_advanced.py --curriculum

# Ensemble обучение
python experiments_advanced.py --ensemble
```

## 📈 Эксперименты и результаты

### Эксперимент 1: Baseline LoRA
**Цель**: Установить базовую производительность
**Конфигурация**:
- LoRA rank: 16
- Learning rate: 2e-4
- Epochs: 3
- Batch size: 4

**Ожидаемые улучшения**: 
- Быстрое обучение
- Хорошая отправная точка
- Минимальные требования к памяти

### Эксперимент 2: Balanced Classes
**Цель**: Решить проблему дисбаланса классов
**Особенности**:
- Undersample/Oversample стратегии
- Увеличенный LoRA rank: 32
- Больше эпох: 4

**Ожидаемые улучшения**:
- Лучший F1-score
- Более равномерное качество по классам

### Эксперимент 3: Full Fine-tuning
**Цель**: Максимальная адаптация модели
**Предупреждение**: ⚠️ Требует много GPU памяти!
**Конфигурация**:
- Полное переобучение всех весов
- Уменьшенный batch size
- Quantization обязательно

### Эксперимент 4: Enhanced Configuration
**Цель**: Оптимизированные гиперпараметры
**Особенности**:
- LoRA rank: 64
- Увеличенная длина последовательности: 3072
- Продвинутое расписание learning rate

## 🔧 Инференс

### Стандартный инференс
```bash
python inference_agent_judge.py \
    --test_path test.csv \
    --pred_path predictions.csv \
    --model_path models/exp1_baseline_lora/unsloth_model
```

### Оптимизированный инференс
```bash
python run_optimized.py \
    --test_path test.csv \
    --pred_path predictions.csv
```

**Особенности оптимизированного инференса**:
- ✅ Автоматический выбор лучшей модели
- ✅ Адаптивное батчирование (избегает OOM)
- ✅ Ensemble предсказания для небольших датасетов
- ✅ Улучшенная постобработка результатов
- ✅ Robust извлечение оценок из текста

## 📝 Формат данных

### Входные данные (train)
```csv
id,prompt,score
abc123,"### Задание для оценки:\n...### Эталонный ответ:\n...### Ответ для оценки:\n...### Критерий оценки:\n...### Шкала оценивания по критерию:\n...",2
```

### Выходные данные (predictions)
```csv  
id,score
abc123,2
def456,0
```

## 🎯 Ключевые улучшения

### 1. Улучшенный промпт-инжиниринг
```python
system_prompt = """Ты - модель-судья, которая оценивает качество ответов других языковых моделей. 
Проанализируй предоставленную информацию и выставь оценку согласно указанному критерию и шкале оценивания. 
Ответь только числом от 0 до 3 (или -1, если критерий не применим)."""
```

### 2. Robust Score Extraction
Множественные стратегии извлечения оценки:
- Первая цифра в ответе
- Регулярные выражения  
- Семантический анализ
- Эвристики по ключевым словам

### 3. Adaptive Training
- Автоматическое уменьшение batch size при OOM
- Gradient checkpointing для экономии памяти
- Mixed precision training

### 4. Data Augmentation
- Перефразирование промптов
- Создание hard negatives
- Curriculum learning по сложности

## 📊 Мониторинг и логирование

### W&B Integration
```bash
export WANDB_PROJECT="agent-judge-experiments"
python train_agent_judge.py --experiment all
```

**Отслеживаемые метрики**:
- Training/Validation Loss
- Accuracy by score class
- F1-score (macro/weighted)
- Learning rate schedule
- Memory usage

### Local Logging
Все эксперименты сохраняют:
- `training_config.json` - конфигурация обучения
- `validation_report.json` - метрики на валидации  
- Model checkpoints
- Training logs

## 🔄 Pipeline для продакшена

```bash
# 1. Обучение лучшей модели
python train_agent_judge.py --experiment 4

# 2. Валидация на отложенной выборке  
python inference_agent_judge.py --validate validation.csv

# 3. Финальное предсказание
python run_optimized.py --test_path test.csv --pred_path submission.csv

# 4. Проверка результатов
python -c "
import pandas as pd
df = pd.read_csv('submission.csv')
print('Распределение оценок:', df.score.value_counts().sort_index())
print('Валидность оценок:', df.score.isin([-1,0,1,2,3]).all())
"
```

## ⚡ Советы по оптимизации

### Для ограниченных ресурсов
- Используйте LoRA с rank 16-32
- Включите gradient checkpointing
- Уменьшите batch size, увеличьте gradient accumulation
- Используйте 4-bit quantization

### Для мощных GPU
- Попробуйте full fine-tuning
- Увеличьте LoRA rank до 128+
- Используйте больший batch size
- Экспериментируйте с ensemble

### Для лучшего качества
- Балансируйте классы в данных
- Применяйте data augmentation  
- Используйте curriculum learning
- Настраивайте температуру для инференса

## 🐛 Troubleshooting

### OutOfMemoryError
```bash
# Уменьшить batch size
export CUDA_VISIBLE_DEVICES=0
python train_agent_judge.py --experiment 1 # Baseline uses smaller config
```

### Низкое качество модели
```bash
# Попробовать сбалансированные данные
python train_agent_judge.py --experiment 2

# Или продвинутую конфигурацию
python train_agent_judge.py --experiment 4
```

### Проблемы с установкой Unsloth
```bash
# Альтернативная установка
pip install --no-deps unsloth
# Или используйте Docker
docker pull unsloth/unsloth:latest
```

## 📚 Дополнительные ресурсы

- [Unsloth Documentation](https://github.com/unslothai/unsloth)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [Qwen3 Model Card](https://huggingface.co/Qwen/Qwen3-0.6B)
- [vLLM Documentation](https://docs.vllm.ai/)

## 🤝 Вклад и развитие

Предложения по улучшению:
1. **Multi-task learning** - одновременное обучение на всех типах задач
2. **Knowledge distillation** - дистилляция от более крупной модели
3. **Active learning** - итеративное улучшение с помощью сложных примеров  
4. **Cross-validation** - более надежная оценка качества
5. **Hyperparameter optimization** - автоматический поиск лучших параметров

---

**Удачи в соревновании! 🏆**
