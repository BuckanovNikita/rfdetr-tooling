# Обучение моделей RF-DETR

## Быстрый старт

```python
from rfdetr_tooling.train import train

train("path/to/dataset", epochs=50, batch_size=8)
```

Обучает модель RF-DETR Base в течение 50 эпох. Директория датасета должна быть в формате COCO или YOLO — rfdetr определяет формат автоматически.

## Функция `train()`

```python
def train(
    dataset_dir: str | Path,
    *,
    variant: str = "base",
    epochs: int = 100,
    batch_size: int = 4,
    output_dir: str | Path = "output",
    **kwargs,  # передаются в rfdetr TrainConfig
) -> None
```

### Параметры

| Параметр | Тип | По умолчанию | Описание |
|----------|-----|--------------|----------|
| `dataset_dir` | `str \| Path` | *обязательный* | Путь к датасету в формате COCO или YOLO |
| `variant` | `str` | `"base"` | Вариант модели (см. таблицу ниже) |
| `epochs` | `int` | `100` | Количество эпох обучения |
| `batch_size` | `int` | `4` | Размер батча |
| `output_dir` | `str \| Path` | `"output"` | Директория для чекпоинтов и логов |

Все дополнительные именованные аргументы передаются в `TrainConfig` rfdetr.

## Варианты моделей

| Вариант | Класс | Разрешение | Энкодер |
|---------|-------|------------|---------|
| `"nano"` | `RFDETRNano` | 384 | DINOv2-S (windowed) |
| `"small"` | `RFDETRSmall` | 512 | DINOv2-S (windowed) |
| `"base"` | `RFDETRBase` | 560 | DINOv2-S (windowed) |
| `"medium"` | `RFDETRMedium` | 576 | DINOv2-S (windowed) |
| `"large"` | `RFDETRLarge` | 704 | DINOv2-S (windowed) |

Меньшие варианты обучаются быстрее и потребляют меньше VRAM. Большие варианты дают более высокую точность.

## Формат датасета

rfdetr автоматически определяет формат (COCO или YOLO) по структуре директории.

### Формат COCO

```
dataset/
  train/
    _annotations.coco.json
    img001.jpg
    img002.jpg
    ...
  valid/
    _annotations.coco.json
    img001.jpg
    ...
```

JSON-файл следует формату аннотаций COCO: `bbox: [x, y, width, height]` (верхний левый угол + размер, в пикселях).

### Формат YOLO

```
dataset/
  data.yaml
  train/
    images/
      img001.jpg
      ...
    labels/
      img001.txt
      ...
  valid/
    images/
      img001.jpg
      ...
    labels/
      img001.txt
      ...
```

Файлы меток используют нормализованный формат YOLO: `class_id xc yc w h` (центр + размер, значения в [0, 1]).

`data.yaml` задает имена классов и пути:

```yaml
train: train/images
val: valid/images
nc: 3
names: ["cat", "dog", "bird"]
```

## Параметры rfdetr (pass-through)

Передаются через `**kwargs` в `TrainConfig` rfdetr.

### Скорость обучения

| Параметр | Тип | По умолчанию | Описание |
|----------|-----|--------------|----------|
| `lr` | `float` | `1e-4` | Скорость обучения декодера |
| `lr_encoder` | `float` | `1.5e-4` | Скорость обучения энкодера (backbone) |
| `lr_drop` | `int` | `100` | Эпоха снижения скорости обучения |
| `lr_vit_layer_decay` | `float` | `0.8` | Послойное затухание LR для ViT-энкодера |
| `lr_component_decay` | `float` | `0.7` | Покомпонентное затухание LR |

### Оптимизация

| Параметр | Тип | По умолчанию | Описание |
|----------|-----|--------------|----------|
| `weight_decay` | `float` | `1e-4` | Регуляризация весов |
| `grad_accum_steps` | `int` | `4` | Шаги накопления градиентов |
| `warmup_epochs` | `float` | `0.0` | Эпохи прогрева (0 = выключено) |

### EMA

| Параметр | Тип | По умолчанию | Описание |
|----------|-----|--------------|----------|
| `use_ema` | `bool` | `True` | Использовать экспоненциальное скользящее среднее |
| `ema_decay` | `float` | `0.993` | Коэффициент затухания EMA |
| `ema_tau` | `int` | `100` | Временная константа EMA |

### Данные

| Параметр | Тип | По умолчанию | Описание |
|----------|-----|--------------|----------|
| `num_workers` | `int` | `2` | Количество воркеров загрузчика данных |
| `multi_scale` | `bool` | `True` | Мультимасштабная аугментация |
| `resolution` | `int\|WxH` | *(зависит от варианта)* | Разрешение входа: int (квадрат) или WxH (прямоугольник, например `960x608`) |
| `resize_mode` | `str` | `auto` | Режим resize при прямоугольном resolution: `auto`, `letterbox`, `true` |

### Чекпоинты

| Параметр | Тип | По умолчанию | Описание |
|----------|-----|--------------|----------|
| `resume` | `str \| None` | `None` | Путь к чекпоинту для продолжения обучения |
| `checkpoint_interval` | `int` | `10` | Сохранять чекпоинт каждые N эпох |

### Ранняя остановка

| Параметр | Тип | По умолчанию | Описание |
|----------|-----|--------------|----------|
| `early_stopping` | `bool` | `False` | Включить раннюю остановку |
| `early_stopping_patience` | `int` | `10` | Эпох без улучшения до остановки |
| `early_stopping_min_delta` | `float` | `0.001` | Минимальный порог улучшения |

### Логирование

| Параметр | Тип | По умолчанию | Описание |
|----------|-----|--------------|----------|
| `tensorboard` | `bool` | `True` | Логирование в TensorBoard |
| `wandb` | `bool` | `False` | Логирование в Weights & Biases |
| `mlflow` | `bool` | `False` | Логирование в MLflow |
| `clearml` | `bool` | `False` | Логирование в ClearML |
| `project` | `str \| None` | `None` | Имя проекта для трекеров экспериментов |
| `run` | `str \| None` | `None` | Имя запуска для трекеров экспериментов |

### Устройство

| Параметр | Тип | По умолчанию | Описание |
|----------|-----|--------------|----------|
| `device` | `str` | `"auto"` | Устройство: `"auto"`, `"cpu"`, `"cuda"` или `"mps"` |

### DDP (multi-GPU)

| Параметр | Тип | По умолчанию | Описание |
|----------|-----|--------------|----------|
| `gpus` | `int` | `1` | Количество GPU. При >1 — автоматический перезапуск через torchrun |
| `sync_bn` | `bool` | `True` | Синхронизация BatchNorm при DDP |

При `gpus > 1` CLI перезапускается через `torchrun --standalone --nproc_per_node=N`. Несовместимо с `device=cpu` и `device=mps`.

### Продвинутые

| Параметр | Тип | По умолчанию | Описание |
|----------|-----|--------------|----------|
| `gradient_checkpointing` | `bool` | `False` | Экономия памяти за счет пересчета активаций |
| `drop_path` | `float` | `0.0` | Drop path для регуляризации |
| `seed` | `int\|None` | `None` | Random seed |
| `progress_bar` | `bool` | `False` | Показывать progress bar |

## Прямоугольный resolution

Параметр `resolution` принимает строку `WxH` (например `960x608`), которая парсится в кортеж `(H, W)`. Значение должно быть кратно 32.

Параметр `resize_mode` контролирует способ приведения изображений к целевому размеру:

- `auto` (по умолчанию) — letterbox для прямоугольного, стандартный resize для квадратного
- `letterbox` — сохраняет пропорции, дополняет padding (серый, 114)
- `true` — растягивает без сохранения пропорций

При прямоугольном resolution автоматически отключается `multi_scale`.

```bash
# CLI
rfdetr-tool train data=./dataset resolution=960x608
rfdetr-tool train data=./dataset resolution=960x608 resize_mode=true
```

```python
# Python API
train("./dataset", resolution="960x608", resize_mode="letterbox")
```

## Полный пример

```python
from rfdetr_tooling.train import train

train(
    "datasets/my_coco_dataset",
    variant="base",
    epochs=80,
    batch_size=8,
    output_dir="runs/experiment_01",
    lr=5e-5,
    lr_encoder=1e-4,
    warmup_epochs=5.0,
    grad_accum_steps=2,
    resolution="960x608",
    resize_mode="letterbox",
    early_stopping=True,
    early_stopping_patience=15,
    wandb=True,
    project="my-detection",
    run="base-80ep",
)
```
