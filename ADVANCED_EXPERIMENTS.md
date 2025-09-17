# 🚀 Advanced Experiments for Agent-as-Judge

Продвинутые эксперименты с **Optuna hyperparameter optimization** и **Ensemble методами** для максимального качества модели-судьи.

## 🎯 Новые возможности

### ✨ **Добавлено в experiments_advanced.py:**

1. **🔍 Optuna Hyperparameter Optimization**
   - Автоматический поиск лучших гиперпараметров
   - Оптимизация LoRA rank, learning rate, batch size, epochs
   - Сохранение лучших параметров в `optuna_best_params.json`

2. **🎭 Ensemble Models**
   - Обучение нескольких моделей с разными конфигурациями
   - Взвешенное голосование для финальных предсказаний
   - Автоматическое сохранение ensemble в `final_ensemble/`

3. **🔄 Smart Model Selection**
   - Автоматический выбор лучшей модели для `run.py`
   - Приоритет: Optuna > Balanced > Augmented

## 📋 Доступные эксперименты

| ID | Название | Описание |
|----|----------|----------|
| 5 | Data Augmentation | Аугментация данных + балансировка |
| 6 | Balanced Classes | Простая балансировка классов |
| 7 | LoRA Ranks | Сравнение разных LoRA ranks [8,16,32] |
| 8 | **Optuna Optimization** | 🆕 Hyperparameter optimization |
| 9 | **Ensemble Models** | 🆕 Ensemble из разных моделей |

## 🚀 Как запустить

### Вариант 1: Интерактивный выбор
```bash
python experiments_advanced.py
# Выберите эксперименты: 5,6,7,8,9 или all/advanced
```

### Вариант 2: Только продвинутые техники
```bash
python experiments_advanced.py
# Введите: advanced
# Запустит только Optuna (8) + Ensemble (9)
```

### Вариант 3: Конкретные эксперименты
```bash
python experiments_advanced.py
# Введите: 8,9
# Запустит Optuna и Ensemble
```

## 🎯 Рекомендуемая стратегия

### Для быстрого результата:
1. Запустите **эксперимент 6** (Balanced Classes) - быстро и эффективно
2. Используйте полученную модель

### Для лучшего качества:
1. Запустите **эксперимент 8** (Optuna) - найдет лучшие параметры
2. Запустите **эксперимент 9** (Ensemble) - создаст ансамбль моделей
3. Используйте ensemble или оптимизированную модель

### Для максимального результата:
1. Запустите **все эксперименты** - долго, но всесторонне
2. Сравните результаты всех подходов
3. Выберите лучшую модель или ensemble

## 🔧 Установка зависимостей

```bash
# Основные зависимости уже есть
pip install unsloth

# Для Optuna
pip install optuna

# Полный список
pip install torch transformers datasets accelerate peft trl pandas numpy scikit-learn tqdm optuna unsloth
```

## 📊 Результаты экспериментов

После выполнения вы получите:

### 📁 **Папки с моделями:**
- `exp5_augmented/` - Data Augmentation модель
- `exp6_balanced/` - Balanced Classes модель  
- `exp7_rank_X/` - Модели с разными LoRA ranks
- `exp8_optuna_best/` - 🏆 Оптимизированная Optuna моделью
- `final_ensemble/` - 🎭 Ensemble из нескольких моделей

### 📋 **Файлы результатов:**
- `optuna_best_params.json` - Лучшие найденные гиперпараметры
- `final_ensemble/ensemble_meta.json` - Метаданные ensemble

### 🎯 **Готовая модель для продакшена:**
- `aij_qwen_0.6b/` - Автоматически выбранная лучшая модель

## 💡 Оптимизация Optuna

### Параметры поиска:
- **lora_r**: [8, 16, 32, 64] - размер LoRA адаптера
- **lora_alpha**: [16, 32, 64, 128] - параметр масштабирования LoRA
- **learning_rate**: [1e-5, 5e-4] - скорость обучения (log-scale)
- **batch_size**: [1, 2, 4] - размер батча
- **epochs**: [1, 2, 3] - количество эпох

### Настройки:
- **15 trials** по умолчанию (можно изменить в коде)
- **30 минут timeout** максимум
- **Оптимизация accuracy** на валидационной выборке
- **Автоматическое сохранение** лучших параметров

## 🎭 Ensemble Strategy

### Конфигурации моделей:
1. **Conservative** (lora_r=16, lr=2e-4) - стабильная модель
2. **Balanced** (lora_r=32, lr=1.5e-4, weight=1.5) - основная модель  
3. **Expressive** (lora_r=64, lr=1e-4) - выразительная модель

### Голосование:
- **Взвешенное большинство** - Balanced модель имеет больший вес
- **Robustness** - ошибки одной модели компенсируются другими
- **Diversity** - разные seeds и data splits для разнообразия

## 🏆 Ожидаемые улучшения

### По сравнению с baseline:
- **Optuna optimization**: +5-8% accuracy
- **Ensemble methods**: +3-7% accuracy  
- **Data augmentation**: +2-5% accuracy
- **Balanced classes**: +3-6% F1-score

### Совместное использование:
- **Optuna + Ensemble**: +8-15% общего улучшения
- **Лучшая стабильность** предсказаний
- **Снижение overfitting** за счет ensemble

## 🔄 Интеграция с run.py

После эксперимента автоматически создается `aij_qwen_0.6b/` для использования:

```bash
# Стандартный инференс
python run.py --test_path test.csv --pred_path predictions.csv

# Или используйте оптимизированную версию
python run_optimized.py --test_path test.csv --pred_path predictions.csv
```

## ❓ FAQ

**Q: Optuna работает медленно?**  
A: Да, это нормально. Уменьшите n_trials в коде или используйте меньшую выборку данных.

**Q: Ensemble требует много памяти?**  
A: Да, держит несколько моделей. Для экономии памяти уменьшите количество моделей в ensemble.

**Q: Какой эксперимент выбрать для соревнования?**  
A: Для быстрого результата - эксперимент 6. Для лучшего качества - эксперимент 8 + 9.

**Q: Можно ли комбинировать подходы?**  
A: Да! Например, используйте аугментацию данных + Optuna оптимизацию + Ensemble.

---

**🎯 Цель: Достичь максимального качества Agent-as-Judge через продвинутые техники ML!** 🏆
