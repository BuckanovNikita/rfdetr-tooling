# rfdetr-tooling

Инструменты для дообучения, валидации и инференса моделей [RF-DETR](https://github.com/roboflow/rf-detr) на пользовательских датасетах. Поддерживает форматы COCO и YOLO, а также CSV-определения датасетов (формат cveta2).

## Возможности

- **Тренировка** — обёртка над rfdetr API с расширенной конфигурацией через pydantic
- **Валидация** — mAP@50/75/50:95 на val-сете с автоматическим маппингом class ID
- **Инференс** — батчированный predict с выводом в YOLO/CSV форматы и визуализацией
- **Прямоугольные разрешения** — поддержка WxH (letterbox или true resize) для train/val/predict
- **DDP** — автоматический multi-GPU через torchrun
- **Логирование** — TensorBoard, W&B, MLflow, ClearML
- **CLI** — единая точка входа `rfdetr-tool` с key=value синтаксисом

## Установка

```bash
uv sync
```

## CLI

```
rfdetr-tool <команда> [key=value ...]
```

Приоритет конфигов: CLI аргументы > YAML (`cfg=path.yaml`) > значения по умолчанию

### Команды

| Команда | Описание |
|---------|----------|
| `train` | Тренировка модели |
| `val` | Валидация на val-сете (mAP) |
| `predict` | Инференс на изображениях |
| `cfg` | Генерация дефолтного YAML-конфига |
| `test` | Встроенные smoke-тесты CLI, линтеров и mypy |

## Тренировка

```bash
# Базовый запуск
rfdetr-tool train data=./dataset variant=base epochs=100 batch_size=8

# С YAML-конфигом + CLI-переопределениями
rfdetr-tool train cfg=config.yaml data=./dataset epochs=50

# С логированием в ClearML
rfdetr-tool train data=./dataset clearml=true project=my-project run=exp-1

# С прямоугольным resolution (letterbox)
rfdetr-tool train data=./dataset resolution=960x608

# С прямоугольным resolution (true resize, без сохранения AR)
rfdetr-tool train data=./dataset resolution=960x608 resize_mode=true
```

### Варианты моделей

| Вариант | Класс | Разрешение по умолчанию |
|---------|-------|------------------------|
| `nano` | RFDETRNano | 384 |
| `small` | RFDETRSmall | 512 |
| `base` | RFDETRBase | 560 |
| `medium` | RFDETRMedium | 576 |
| `large` | RFDETRLarge | 704 |

### Параметры тренировки

| Параметр | Тип | По умолчанию | Описание |
|----------|-----|--------------|----------|
| `data` | `str` | **обязательный** | Путь к датасету (COCO или YOLO) |
| `variant` | `str` | `base` | Вариант модели |
| `epochs` | `int` | `100` | Количество эпох |
| `batch_size` | `int` | `4` | Размер батча |
| `lr` | `float` | `1e-4` | Learning rate декодера |
| `lr_encoder` | `float` | `1.5e-4` | Learning rate энкодера |
| `lr_drop` | `int` | `100` | Эпоха снижения LR |
| `weight_decay` | `float` | `1e-4` | Weight decay |
| `grad_accum_steps` | `int` | `4` | Шаги накопления градиентов |
| `warmup_epochs` | `float` | `0.0` | Warmup-эпохи |
| `resolution` | `int\|WxH` | `None` | Разрешение входа (int — квадрат, WxH — прямоугольник) |
| `resize_mode` | `str` | `auto` | Режим resize: `auto`, `letterbox`, `true` |
| `output_dir` | `str` | `output` | Директория для результатов |
| `device` | `str` | `auto` | Устройство: `auto`, `cpu`, `cuda`, `mps` |
| `resume` | `str\|None` | `None` | Чекпоинт для продолжения обучения |
| `checkpoint_interval` | `int` | `10` | Сохранение чекпоинта каждые N эпох |
| `use_ema` | `bool` | `true` | Экспоненциальное скользящее среднее |
| `ema_decay` | `float` | `0.993` | Коэффициент затухания EMA |
| `ema_tau` | `int` | `100` | Временная константа EMA |
| `early_stopping` | `bool` | `false` | Ранняя остановка |
| `early_stopping_patience` | `int` | `10` | Эпох без улучшения до остановки |
| `early_stopping_min_delta` | `float` | `0.001` | Минимальный порог улучшения |
| `multi_scale` | `bool` | `true` | Мультимасштабная аугментация |
| `drop_path` | `float` | `0.0` | Drop path rate |
| `seed` | `int\|None` | `None` | Random seed |
| `num_workers` | `int` | `2` | Воркеры DataLoader |
| `progress_bar` | `bool` | `false` | Показывать progress bar |
| `gpus` | `int` | `1` | Количество GPU (>1 — DDP через torchrun) |
| `sync_bn` | `bool` | `true` | Синхронизация BatchNorm при DDP |
| `tensorboard` | `bool` | `true` | Логирование в TensorBoard |
| `wandb` | `bool` | `false` | Логирование в W&B |
| `mlflow` | `bool` | `false` | Логирование в MLflow |
| `clearml` | `bool` | `false` | Логирование в ClearML + загрузка артефактов |
| `project` | `str\|None` | `None` | Имя проекта для трекеров |
| `run` | `str\|None` | `None` | Имя запуска для трекеров |

### DDP (multi-GPU)

```bash
rfdetr-tool train data=./dataset gpus=4 batch_size=16
```

При `gpus > 1` CLI автоматически перезапускается через `torchrun --standalone --nproc_per_node=N`. Несовместимо с `device=cpu` и `device=mps`.

## Валидация

```bash
# Базовая валидация
rfdetr-tool val weights=model.pth data=./dataset

# С кастомным порогом и resolution
rfdetr-tool val weights=model.pth data=./dataset threshold=0.3 resolution=640

# Прямоугольный resolution
rfdetr-tool val weights=model.pth data=./dataset resolution=960x608
```

Валидация вычисляет mAP@50, mAP@75 и mAP@50:95 на val-сете (ищет `valid/` или `val/` в директории датасета). Поддерживает только COCO-формат аннотаций (`_annotations.coco.json`).

Автоматический маппинг class ID: предсказания модели маппятся на GT category_id через совпадение имён классов, что корректно работает и для pretrained COCO (91 класс с пропусками), и для дообученных моделей (contiguous IDs).

### Параметры валидации

| Параметр | Тип | По умолчанию | Описание |
|----------|-----|--------------|----------|
| `data` | `str` | **обязательный** | Путь к датасету |
| `weights` | `str` | **обязательный** | Путь к весам |
| `variant` | `str` | `base` | Вариант модели |
| `threshold` | `float` | `0.5` | Порог уверенности |
| `device` | `str` | `auto` | Устройство |
| `batch_size` | `int` | `4` | Размер батча (зарезервировано) |
| `resolution` | `int\|WxH` | `None` | Разрешение входа |
| `resize_mode` | `str` | `auto` | Режим resize: `auto`, `letterbox`, `true` |

## Инференс

```bash
# Папка с изображениями → YOLO-формат
rfdetr-tool predict source=./images weights=model.pth

# YOLO-датасет → CSV с сохранением сплитов
rfdetr-tool predict source=./dataset weights=model.pth format=csv

# Несколько источников
rfdetr-tool predict source=dir1,dir2,data.yaml weights=model.pth

# С визуализацией
rfdetr-tool predict source=./images weights=model.pth visualize=true

# Настройка порогов
rfdetr-tool predict source=./images weights=model.pth conf_threshold=0.25 nms_threshold=0.5

# Прямоугольный resolution
rfdetr-tool predict source=./images weights=model.pth resolution=960x608

# True resize (без letterbox)
rfdetr-tool predict source=./images weights=model.pth resolution=960x608 resize_mode=true
```

### Форматы входных данных

| Источник | Описание |
|----------|----------|
| Папка | Рекурсивный обход файлов изображений |
| Папка с `data.yaml` | Распознаётся как YOLO-датасет, сплиты сохраняются |
| `.yaml`/`.yml` файл | YOLO dataset yaml (должен содержать ключ `names`) |
| `.csv` файл | CSV с колонкой `image_path` (формат cveta2) |
| Через запятую | Комбинация любых вышеуказанных источников |

Поддерживаемые форматы изображений: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`, `.tiff`, `.tif`, `.heif`, `.heic`, `.avif`

### Форматы выхода

**YOLO** (`format=yolo`, по умолчанию):

```
predict_output/
  dataset.yaml
  labels/
    train/
      img1.txt       # class_id xc yc w h confidence
    val/
      img2.txt
    unsplit/
      img3.txt
```

**CSV** (`format=csv`):

```
predict_output/
  predictions.csv    # колонки cveta2: image_path, instance_label, bbox_*, confidence, split...
```

### Параметры инференса

| Параметр | Тип | По умолчанию | Описание |
|----------|-----|--------------|----------|
| `source` | `str` | **обязательный** | Пути через запятую |
| `weights` | `str` | **обязательный** | Путь к весам |
| `variant` | `str` | `base` | Вариант модели |
| `conf_threshold` | `float` | `0.01` | Минимальная уверенность |
| `nms_threshold` | `float` | `0.25` | IoU порог NMS |
| `agnostic_nms` | `bool` | `false` | Class-agnostic NMS |
| `resolution` | `int\|WxH` | `None` | Разрешение входа |
| `resize_mode` | `str` | `auto` | Режим resize: `auto`, `letterbox`, `true` |
| `batch_size` | `int` | `4` | Размер батча |
| `device` | `str` | `auto` | Устройство |
| `output_dir` | `str` | `predict_output` | Директория для результатов |
| `format` | `str` | `yolo` | Формат выхода: `yolo`, `csv` |
| `visualize` | `bool` | `false` | Сохранять визуализации |
| `check_image_sizes` | `bool` | `false` | Читать размер каждого изображения |

## Генерация конфига

```bash
# Вывод в stdout
rfdetr-tool cfg variant=base

# Запись в файл
rfdetr-tool cfg variant=large output=large_config.yaml

# С кастомным resolution
rfdetr-tool cfg variant=base resolution=960x608
```

Генерирует YAML-конфиг со всеми параметрами тренировки и комментариями по секциям. Конфиг затем передаётся через `cfg=path.yaml`.

## Resolution и resize_mode

Параметр `resolution` принимает:
- **int** — квадратное разрешение (например `640`)
- **WxH** — прямоугольное разрешение (например `960x608`, ширина × высота)

Значение должно быть больше 0 и кратно 32.

Параметр `resize_mode` контролирует способ приведения изображений к целевому разрешению:

| Режим | Описание |
|-------|----------|
| `auto` | По умолчанию. Для прямоугольного resolution — letterbox, для квадратного — стандартный resize rfdetr |
| `letterbox` | Сохраняет пропорции (aspect ratio), добавляя padding серым (114) до целевого размера |
| `true` | Растягивает изображение до целевого размера без сохранения пропорций |

При прямоугольном resolution автоматически отключается `multi_scale` аугментация при тренировке.

## Smoke-тесты

```bash
# Все тесты
rfdetr-tool test

# По категориям
rfdetr-tool test category=cli          # CLI smoke-тесты
rfdetr-tool test category=linter       # ruff check + format
rfdetr-tool test category=typecheck    # mypy
```

Тесты проверяют: корректность CLI (help, ошибки валидации, неизвестные параметры), линтинг и типизацию. Результаты сохраняются в `.test_output/`.

## Форматы датасетов

### COCO

```
dataset/
  train/
    _annotations.coco.json
    img001.jpg
    ...
  valid/
    _annotations.coco.json
    img001.jpg
    ...
```

`bbox: [x, y, width, height]` — верхний левый угол + размер, пиксели.

### YOLO

```
dataset/
  data.yaml
  train/
    images/
      img001.jpg
    labels/
      img001.txt     # class_id xc yc w h (нормализованные, 0-indexed)
  valid/
    images/
    labels/
```

### CSV (cveta2)

Колонки: `image_name`, `image_width`, `image_height`, `instance_label`, `bbox_x_tl`, `bbox_y_tl`, `bbox_x_br`, `bbox_y_br`, `split`, `image_path`

## Python API

```python
from rfdetr_tooling.train import train
from rfdetr_tooling.val import val
from rfdetr_tooling.predict import predict

# Тренировка
train(
    "datasets/my_dataset",
    variant="base",
    epochs=80,
    batch_size=8,
    resolution="960x608",
    resize_mode="letterbox",
    early_stopping=True,
    wandb=True,
    project="my-detection",
    run="base-80ep",
)

# Валидация
val(
    data="datasets/my_dataset",
    weights="output/checkpoint_best_ema.pth",
    variant="base",
    threshold=0.3,
)

# Инференс
predict(
    source="./images",
    weights="output/checkpoint_best_ema.pth",
    variant="base",
    conf_threshold=0.25,
    format="csv",
    visualize=True,
)
```

## Разработка

```bash
uv sync                              # установка зависимостей
uv run ruff format .                 # форматирование
uv run ruff check .                  # линтинг
uv run ruff check --fix .            # автофикс линтинга
uv run mypy .                        # проверка типов
```
