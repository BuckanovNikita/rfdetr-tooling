# predict — Инференс RF-DETR

Команда `predict` выполняет инференс RF-DETR на изображениях с сохранением результатов в формате YOLO (txt + dataset.yaml) или CSV (cveta2).

## Быстрый старт

```bash
# Инференс на папке с изображениями
rfdetr-tool predict source=./images weights=model.pth

# Инференс на YOLO датасете с сохранением сплитов
rfdetr-tool predict source=./dataset weights=model.pth

# Вывод в CSV формат
rfdetr-tool predict source=./images weights=model.pth format=csv

# С визуализацией
rfdetr-tool predict source=./images weights=model.pth visualize=true
```

## Параметры

| Параметр | Тип | По умолчанию | Описание |
|----------|-----|-------------|----------|
| `source` | `str` | **обязательный** | Пути через запятую: папки, data.yaml, dataset.csv |
| `weights` | `str` | **обязательный** | Путь к файлу весов модели |
| `variant` | `str` | `base` | Вариант модели: `nano`, `small`, `base`, `medium`, `large` |
| `conf_threshold` | `float` | `0.01` | Минимальная уверенность класса (передаётся в `model.predict`) |
| `nms_threshold` | `float` | `0.25` | IoU порог для NMS post-processing |
| `agnostic_nms` | `bool` | `false` | Class-agnostic NMS (единый пул вместо per-class) |
| `resolution` | `int\|None` | `None` | Размер изображения для инференса (`None` → дефолт варианта) |
| `batch_size` | `int` | `4` | Количество изображений в одном батче |
| `device` | `str` | `auto` | Устройство: `auto`, `cpu`, `cuda`, `cuda:0`, `cuda:1`, `mps` |
| `output_dir` | `str` | `predict_output` | Директория для результатов |
| `format` | `str` | `yolo` | Формат выхода: `yolo` или `csv` |
| `visualize` | `bool` | `false` | Рисовать аннотированные изображения |
| `check_image_sizes` | `bool` | `false` | Читать размер каждого изображения индивидуально |

## Форматы входных данных

### Папка с изображениями

Рекурсивный обход всех файлов. Пропускаются: директории, скрытые файлы, файлы без расширения. Валидация через `PIL.Image.verify()`.

```bash
rfdetr-tool predict source=./images weights=model.pth
```

### YOLO dataset (папка с data.yaml)

Если директория содержит `data.yaml` или `dataset.yaml` — распознаётся как YOLO датасет. Сплиты (`train`, `val`, `test`) сохраняются в выходных данных.

```bash
rfdetr-tool predict source=./dataset weights=model.pth
```

### YOLO yaml (прямой путь)

```bash
rfdetr-tool predict source=./dataset/data.yaml weights=model.pth
```

### CSV (формат cveta2)

CSV с колонкой `image_path` (обязательная). Колонка `split` — опциональна.

```bash
rfdetr-tool predict source=./data.csv weights=model.pth
```

### Несколько источников

Источники через запятую:

```bash
rfdetr-tool predict source=dir1,dir2,data.yaml weights=model.pth
```

## Форматы выхода

### YOLO (`format=yolo`)

```
output_dir/
  dataset.yaml          # names, пути к сплитам
  labels/
    train/              # если есть сплиты
      img1.txt
    val/
      img2.txt
    unsplit/            # если split=None
      img3.txt
```

Формат строки в `.txt`: `class_id xc yc w h confidence` (6 полей, нормализованные координаты). Класс ID — 0-indexed contiguous.

### CSV (`format=csv`)

```
output_dir/
  predictions.csv
```

Колонки: `task_name`, `source`, `image_path`, `image_name`, `image_width`, `image_height`, `instance_shape`, `instance_label`, `bbox_x_tl`, `bbox_y_tl`, `bbox_x_br`, `bbox_y_br`, `confidence`, `split`.

Координаты в пикселях. Для изображений без детекций: `instance_shape=none`.

## Визуализация

```bash
rfdetr-tool predict source=./images weights=model.pth visualize=true
```

Аннотированные изображения сохраняются в `output_dir/visualize/`.

## Примеры

```bash
# Настройка порогов
rfdetr-tool predict source=./images weights=model.pth conf_threshold=0.25 nms_threshold=0.5

# Батч + разрешение + устройство
rfdetr-tool predict source=./images weights=model.pth batch_size=16 resolution=640 device=cuda:1

# Class-agnostic NMS
rfdetr-tool predict source=./images weights=model.pth agnostic_nms=true

# Точное определение размеров (для датасетов с разными разрешениями)
rfdetr-tool predict source=./images weights=model.pth check_image_sizes=true
```

## Поддерживаемые форматы изображений

`.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`, `.tiff`, `.tif`, `.heif`, `.heic`, `.avif`

Для HEIF/HEIC/AVIF используется `pillow-heif`.

## Ограничения

- Запятая в пути source не поддерживается
- Если `output_dir` уже существует — создаётся с постфиксом `_2`, `_3`...
- При `check_image_sizes=false` размер берётся по первому изображению
