# Генерация и обучение моделей для интегралов

Лаконичное описание содержимого репозитория и основных команд для работы с данными и обучением LoRA-модели.

## Структура
- `data/book_of_integrals.csv` — шаблоны интегралов (колонки Section, Integral (LHS), Result (RHS), parameter).
- `data/synt/` — пример синтетического датасета: `metadata.csv` + PNG-изображения в `images/`.
- `integral_generator.py` — создание датасетов интегралов из CSV-шаблонов.
- `train_lora_simple.py` — обучение LoRA на стабиль-диффузион пайплайне по метаданным из датасета.
- `visualize_lora.py` — генерация примеров и сравнение базовой модели с обученной LoRA.
- `lora_integrals/samples/` — примеры сгенерированных изображений (grid и отдельные кадры).
- `presentation.ipynb` — ноутбук с материалами презентации/экспериментами.

## Зависимости
Python 3.10+ и пакеты: `torch`, `torchvision`, `diffusers`, `peft`, `pandas`, `numpy`, `pillow`, `matplotlib`, `tqdm`, `transformers`, `tensorboard` (для логов). Установка примером:

```bash
pip install torch torchvision diffusers peft pandas numpy pillow matplotlib tqdm transformers tensorboard
```

## Подготовка датасета
Создание собственного набора с LaTeX-интегралами (выход — `metadata.csv` и PNG в `images/`):

```bash
python integral_generator.py \
  --num 2000 \
  --out ./data/synt \
  --templates ./data/book_of_integrals.csv \
  --seed 0
```

Параметр `--no-bounds` отключает добавление случайных пределов интегрирования к шаблонам без них; `--dpi` и `--fontsize` управляют качеством отрисовки.

## Обучение LoRA
Скрипт `train_lora_simple.py` ожидает пути к датасету и каталог для чекпоинтов, которые задаются в начале `main()` (переменные `dataset_path`, `dataset_root`, `output_dir`, `model_id`). Перед запуском пропишите свои пути, затем выполните:

```bash
python train_lora_simple.py
```

Выход: чекпоинты по эпохам в `output_dir/epoch_*/` и финальная версия в `output_dir/final/`. Для логов TensorBoard укажите `use_tensorboard = True` и смотрите `tensorboard --logdir <output_dir>/runs`.

## Генерация и сравнение
После обучения загрузите LoRA и сгенерируйте изображения по метаданным:

```bash
python visualize_lora.py \
  --mode generate \
  --lora-path ./lora_integrals/final \
  --csv-path ./data/synt/metadata.csv \
  --dataset-root ./data/synt \
  --num-samples 6
```

Для сравнения базовой модели с LoRA используйте `--mode compare`. Флаги `--height/--width` задают размер, `--no-auto-wide` отключает автоматический подбор ширины по аспекту датасета.

## Быстрый старт
1. Установите зависимости.
2. Сгенерируйте или используйте готовый пример в `data/synt/`.
3. Настройте пути в `train_lora_simple.py` и запустите обучение.
4. Проверьте качество через `visualize_lora.py` и готовые примеры в `lora_integrals/samples/`.
