# План: Рефакторинг predict

## Цель

Переработать `predict` — сделать визуализацию опциональной (по умолчанию выключена), добавить структурированный вывод в двух форматах (YOLO txt + dataset.yaml, или cveta2 CSV), поддержать разнообразные источники входных данных с сохранением сплитов.

---

## 1. Изменения в `PredictConfig` (`config.py`)

**Было:**
```python
class PredictConfig(BaseModel):
    source: str
    weights: str
    variant: Variant = "base"
    threshold: float = 0.5
    device: Literal["auto", "cpu", "cuda", "mps"] = "auto"
    output_dir: str = "predict_output"
    save: bool = True
```

**Станет:**
```python
OutputFormat = Literal["yolo", "csv"]

class PredictConfig(BaseModel):
    source: str              # пути через запятую: папки, data.yaml, dataset.csv
    weights: str
    variant: Variant = "base"
    conf_threshold: float = 0.01   # мин. уверенность класса (передаётся в model.predict)
    nms_threshold: float = 0.25    # IoU порог для NMS (post-processing через supervision)
    agnostic_nms: bool = False     # class-agnostic NMS (единый пул вместо per-class)
    resolution: int | None = None  # размер изображения для инференса (None → дефолт варианта)
    batch_size: int = 4            # количество изображений в одном батче
    device: str = "auto"           # "auto", "cpu", "cuda", "cuda:0", "cuda:1", "mps"
    output_dir: str = "predict_output"
    format: OutputFormat = "yolo"
    visualize: bool = False  # было save=True, теперь по умолчанию выключено
    check_image_sizes: bool = False  # читать размер каждого изображения (медленно)
```

- `source` — строка через запятую: `source=dir1,dir2,data.yaml,dataset.csv`. Парсится внутри predict.
- `conf_threshold` — минимальная уверенность класса. Передаётся как `threshold` в `model.predict()`. По умолчанию `0.01` (пропускаем почти всё, фильтрация — на NMS).
- `nms_threshold` — IoU порог для NMS post-processing. Применяется через `sv.Detections.with_nms(threshold=..., class_agnostic=...)` после `model.predict()`. По умолчанию `0.25`. RF-DETR (DETR-family) не использует NMS внутри, поэтому применяем снаружи.
- `agnostic_nms` — class-agnostic NMS: если `True`, NMS работает в едином пуле (боксы разных классов подавляют друг друга). По умолчанию `False` (per-class NMS).
- `resolution` — размер входного изображения для модели. `None` (по умолчанию) — используется дефолт варианта из `VARIANT_RESOLUTION`. Передаётся через `model.model.resolution = resolution` после создания модели.
- `batch_size` — количество изображений в одном батче инференса. По умолчанию `4`. rfdetr `predict()` принимает список изображений и стекает их в один тензор — мы нарезаем входные изображения на чанки по `batch_size` и вызываем `model.predict()` для каждого чанка.
- `device` — устройство для инференса. Тип изменён на `str` (вместо `Literal`) для поддержки `"cuda:0"`, `"cuda:1"` и т.д. `"auto"` — rfdetr сам выберет. Передаётся через конструктор модели (`model_cls(pretrain_weights=..., device=...)`), если не `"auto"`.
- `format` — формат выхода: `"yolo"` (txt + dataset.yaml) или `"csv"` (cveta2 CSV).
- `visualize` — рисовать аннотированные изображения (как раньше `save`). По умолчанию `False`.
- `check_image_sizes` — если `False` (по умолчанию), размер читается только у первого изображения и предполагается одинаковым для всех (как `_SizeCache` в cveta2). Если `True` — читается размер каждого файла индивидуально (медленнее, но корректно для датасетов с разными разрешениями).
- Удалено поле `save`, удалено поле `threshold` (заменено на `conf_threshold` + `nms_threshold`).

---

## 2. Определение типа источника

Новая функция `_parse_sources(source: str) -> list[SourceEntry]` разбирает запятую-разделённую строку и определяет тип каждого элемента:

```python
class SourceEntry(NamedTuple):
    path: Path
    kind: Literal["dir", "yolo_yaml", "csv"]
```

Логика определения:
1. Файл с расширением `.yaml` / `.yml` → проверяем наличие ключа `names` внутри → `"yolo_yaml"`
2. Файл с расширением `.csv` → `"csv"`
3. Директория, содержащая `data.yaml` или `dataset.yaml` → `"yolo_yaml"` (path указывает на yaml)
4. Директория без yaml → `"dir"` (рекурсивный поиск изображений)
5. Иначе — ошибка

---

## 3. Сбор изображений с сохранением сплитов

Новая структура для передачи изображений с метаинформацией:

```python
class ImageEntry(NamedTuple):
    path: Path
    split: str | None  # None для обычных папок
    width: int         # заполняется на этапе verify()
    height: int        # заполняется на этапе verify()
```

### Константа расширений

Старая `_IMAGE_EXTENSIONS` удаляется. Новая:
```python
_IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif",
    ".heif", ".heic", ".avif",
}
```
Используется в §3.2 (YOLO yaml) для фильтрации файлов в images-директориях. В §3.1 (обычные папки) — **не** используется, берутся все файлы.

> **Примечание**: для HEIF/HEIC/AVIF требуется `pillow-heif`. Добавляется как явная зависимость в `pyproject.toml` (см. §8). Регистрация плагина: `import pillow_heif; pillow_heif.register_heif_opener()` в начале `predict.py`.

### 3.1 `_collect_from_dir(path: Path) -> list[ImageEntry]`
- **Рекурсивный** обход (`rglob("*")`) для сбора **всех файлов**.
- `split = None` для всех.
- Пропускаются: директории, скрытые файлы (имя начинается с `.`), файлы без расширения.
- Фильтрация по расширению **не** производится — любой файл с расширением считается потенциальным изображением (см. §4 — валидация при открытии через `verify()`).

### 3.2 `_collect_from_yolo_yaml(yaml_path: Path) -> tuple[list[ImageEntry], dict[int, str]]`
- Читает `dataset.yaml` / `data.yaml`.
- Для каждого сплита (`train`, `val`, `test`) находит `images/` директорию.
- Собирает файлы с расширениями из `_IMAGE_EXTENSIONS` (включая `.heif`, `.heic`, `.avif`).
- `split = "train"` / `"val"` / `"test"`.
- Возвращает также `class_names` из yaml (пригодится для контекста, но не используется при предсказании — классы берутся из модели).

### 3.3 `_collect_from_csv(csv_path: Path) -> list[ImageEntry]`
- Читает CSV с помощью pandas.
- Требует колонку `image_path`. Колонка `split` — опциональна: если отсутствует, все изображения получают `split = None`.
- Берёт уникальные `image_path` + их `split`.
- Пути берутся из колонки `image_path` (абсолютные). Если путь относительный — разрешается от директории CSV.
- Если `image_path` пустой или файл не найден — warning, пропуск.

### 3.4 Объединение
Все `ImageEntry` из всех source объединяются в один список. Дубликаты по `path` — warning для каждого с указанием пути, первый встреченный сплит побеждает. После сбора, если итоговый список пуст — `raise ValueError("Не найдено ни одного изображения в указанных источниках")`.

---

## 4. Инференс

### Инициализация модели
```python
model_cls = _get_model_class(config.variant)
kwargs = {"pretrain_weights": config.weights}
if config.device != "auto":
    kwargs["device"] = config.device
model = model_cls(**kwargs)
if config.resolution is not None:
    model.model.resolution = config.resolution
```

### Легковесная валидация и определение размеров

Два режима определения размера изображений:

**`check_image_sizes=False` (по умолчанию)** — быстрый режим:
- `verify()` вызывается для **каждого** файла (валидация обязательна).
- Размер `(width, height)` читается только у **первого** изображения и предполагается одинаковым для всех остальных.
- Аналог `_SizeCache(read_all=False)` из cveta2.
- Подходит для типичных датасетов, где все изображения одного разрешения.

**`check_image_sizes=True`** — точный режим:
- `verify()` + `img.size` для **каждого** файла.
- Каждый `ImageEntry` получает индивидуальные `(width, height)`.
- Необходим если изображения имеют разные разрешения.

```python
failed: list[Path] = []
validated: list[ImageEntry] = []
cached_size: tuple[int, int] | None = None

for entry in images:
    try:
        with Image.open(entry.path) as img:
            if config.check_image_sizes or cached_size is None:
                w, h = img.size
                if cached_size is None:
                    cached_size = (w, h)
                    logger.debug(f"Размер изображений (по первому файлу): {w}x{h}")
            else:
                w, h = cached_size
            img.verify()
        validated.append(ImageEntry(entry.path, entry.split, w, h))
    except (OSError, SyntaxError):
        failed.append(entry.path)

if failed:
    raise ValueError(f"Не удалось распознать {len(failed)} файлов как изображения: {failed}")
images = validated
```

### Стадия 1: GPU-инференс (батчированный, чанками)

GPU-инференс по чанкам. Результаты каждого чанка сохраняются в `raw_chunks` — список чанков, каждый чанк содержит `list[tuple[ImageEntry, sv.Detections]]`.

```python
raw_chunks: list[list[tuple[ImageEntry, sv.Detections]]] = []
for batch_entries in chunked(images, config.batch_size):
    pil_images = [Image.open(e.path).convert("RGB") for e in batch_entries]
    det_list = model.predict(pil_images, threshold=config.conf_threshold)
    chunk = list(zip(batch_entries, det_list))
    raw_chunks.append(chunk)
    del pil_images  # освобождаем PIL-объекты сразу после батча
```

GPU не простаивает в ожидании CPU-NMS — весь инференс прогоняется непрерывно.

### Стадия 2: NMS post-processing (CPU) + стриминговая запись

NMS и запись результатов выполняются потоково по чанкам — не нужно держать все результаты в памяти одновременно. Writer инициализируется до начала обработки и принимает результаты инкрементально.

```python
writer = _create_writer(config, model.class_names)  # YOLO или CSV writer
for chunk in raw_chunks:
    for entry, detections in chunk:
        detections = detections.with_nms(
            threshold=config.nms_threshold, class_agnostic=config.agnostic_nms
        )
        writer.write(entry, detections)
        if config.visualize:
            _visualize(entry, detections, model.class_names, output_dir)
writer.finalize()  # dataset.yaml для YOLO, закрытие файла для CSV
```

**Для визуализации** — изображение переоткрывается из `entry.path` (единственное место где нужны пиксели, вызывается только при `visualize=True`).

**Writer-ы:**
- **YOLO**: `write()` — пишет один `.txt` файл. `finalize()` — пишет `dataset.yaml`.
- **CSV**: `write()` — дописывает строки в открытый файл (append-mode, заголовок пишется один раз при инициализации). `finalize()` — закрывает файл.

Разделение стадий позволяет GPU работать непрерывно без чередования с CPU-NMS на каждом батче. Стриминговая запись по чанкам ограничивает пиковое потребление памяти размером одного чанка результатов.

**Порядок:**
1. **Валидация** — `Image.open().verify()` проверяет заголовок без декодирования пикселей. Если хоть один файл не прошёл — raise с полным списком.
2. **GPU-инференс** — изображения нарезаются на чанки по `batch_size`, открываются лениво per-batch. PIL-объекты освобождаются после каждого батча. rfdetr `predict()` принимает список → стекает в один тензор. `threshold=conf_threshold` (default 0.01) — минимальная отсечка на GPU.
3. **NMS + запись** — по чанкам: `detections.with_nms(threshold=nms_threshold)` (default 0.25) → запись через writer. Память ограничена одним чанком. RF-DETR не применяет NMS внутри, поэтому делаем снаружи через supervision.

---

## 5. Запись результатов

### 5.1 Формат YOLO (`format="yolo"`) — `YoloWriter`

Структура выхода:
```
output_dir/
  dataset.yaml          # names, train/val/test пути
  labels/
    train/              # если есть сплиты
      img1.txt
    val/
      img2.txt
    unsplit/            # если split=None (обычные папки)
      img3.txt
```

- `__init__` — создаёт `output_dir` (с постфиксом `_2`, `_3`... если уже существует). Строит маппинг `pred_id_to_yolo_id` и `yolo_names` из `model.class_names`. Отслеживает встреченные сплиты и стемы для коллизий.

  **Маппинг классов** (работает с любым количеством классов):
  ```python
  # model.class_names: dict[int, str]
  # Finetuned: {1: "cat", 2: "dog", 3: "bird"} — 1-indexed contiguous
  # Pretrained COCO: {1: "person", 2: "bicycle", ..., 27: "backpack", ...} — 1-indexed с пробелами
  sorted_pred_ids = sorted(model.class_names.keys())
  pred_id_to_yolo_id = {pred_id: yolo_id for yolo_id, pred_id in enumerate(sorted_pred_ids)}
  yolo_names = {yolo_id: model.class_names[pred_id] for pred_id, yolo_id in pred_id_to_yolo_id.items()}
  # Результат: {0: "cat", 1: "dog", 2: "bird"} — всегда 0-indexed contiguous
  ```

- `write(entry, detections)` — пишет один `.txt` файл в `labels/{split}/`. Для каждой детекции `class_id` ремаппится через `pred_id_to_yolo_id`. Формат строки: `yolo_class_id xc yc w h confidence` (6 полей, YOLO normalized + confidence). Использует `entry.width` / `entry.height` для нормализации координат.
- `finalize()` — пишет `dataset.yaml` с `yolo_names` и путями к встреченным сплитам.

### 5.2 Формат CSV (`format="csv"`) — `CsvWriter`

- `__init__` — создаёт `output_dir` (с постфиксом если существует), открывает `predictions.csv` на запись, пишет заголовок (все колонки из `CSV_COLUMNS`).
- `write(entry, detections)` — дописывает строки в открытый файл (без буферизации в памяти). Для каждой детекции: `instance_shape="box"`, `instance_label` — строковое имя класса из `model.class_names[class_id]` (не числовой ID), координаты в пикселях из `detections.xyxy`, `image_width`/`image_height` из `entry`, `confidence`, `split`, `image_path` (абсолютный). Для изображений без детекций: `instance_shape="none"`. `task_name="rfdetr-predict"`, `source="auto"`.
- `finalize()` — закрывает файл, логирует итог.

### 5.3 Визуализация (`visualize=True`)

Если `visualize=True`:
- Изображение переоткрывается из `entry.path` только в момент визуализации (не хранится в памяти между стадиями).
- Аннотированные изображения сохраняются в `output_dir/visualize/`.
- Вызывается потоково в стадии 2 вместе с записью.

---

## 6. Изменения в CLI (`cli.py`)

- Обновить `_print_help()` — новые параметры и примеры.
- `_coerce_value` уже корректно обрабатывает `Literal` и `bool`, новых типов не нужно.
- Удалённое поле `save` → ошибка валидации если старый пользователь передаст `save=true` (ожидаемо, breaking change).

---

## 7. Обработка краевых случаев

| Случай | Поведение |
|--------|-----------|
| Все источники пустые (0 изображений) | `raise ValueError` |
| Пустая папка (нет изображений) | warning, пропуск |
| CSV без колонки `image_path` | error, exit 1 |
| CSV без колонки `split` | все изображения получают `split = None` |
| CSV с пустым `split` | `split = None`, кладётся в `unsplit/` (yolo) или split=None (csv) |
| `output_dir` уже существует | создаётся с постфиксом: `predict_output_2`, `_3`... |
| Запятая в пути source | не поддерживается, документируем ограничение |
| YOLO yaml без images dir | warning, пропуск этого сплита |
| Несколько source разных типов | объединяются; если один CSV и одна папка — ОК |
| Файл image_path не существует | warning, пропуск с подсчётом skipped |
| Модель с COCO-91 классами | маппинг 1-indexed с пробелами → 0-indexed contiguous для YOLO txt |
| Изображение без детекций | пустой .txt (yolo), строка none (csv) |
| Дубликат изображения (по абсолютному path) | warning с указанием пути, первый встреченный сплит побеждает |
| Дубликат имени файла из разных папок | в YOLO txt — stem коллизия, добавить суффикс `_1`, `_2`... |

---

## 8. Файлы для изменения

| Файл | Что меняется |
|------|-------------|
| `rfdetr_tooling/config.py` | `PredictConfig`: новые поля `format`, `visualize`; удаление `save` |
| `rfdetr_tooling/predict.py` | Полная переработка: парсинг источников, сбор с сплитами, запись YOLO/CSV |
| `rfdetr_tooling/cli.py` | Обновление help-текста и примеров для predict |
| `pyproject.toml` | Добавить `pandas`, `pillow-heif` в зависимости |
| `README.md` | Добавить секцию про predict с примерами CLI |
| `docs/predict.md` | **Новый файл** — полная документация команды predict (на русском) |
| `CLAUDE.md` | Обновить Validation Strategy: smoke-тесты для predict |

Новые зависимости в `pyproject.toml`:
- `pandas` — явная зависимость для CSV чтения/записи (ранее неявно через supervision)
- `pillow-heif` — декодирование HEIF/HEIC/AVIF через Pillow

`pyyaml` — уже есть.

---

## 9. Документация

### 9.1 `docs/predict.md` (новый, на русском)

Структура:
- **Быстрый старт** — минимальные примеры CLI
- **Параметры** — таблица всех полей `PredictConfig` с типами, дефолтами, описаниями
- **Форматы входных данных** — папки, YOLO dataset, YOLO yaml, CSV; как определяется тип; сохранение сплитов
- **Форматы выхода** — YOLO (структура labels/ + dataset.yaml), CSV (формат cveta2, колонки)
- **Визуализация** — опция `visualize=true`
- **Примеры** — типовые сценарии:
  - Инференс на папке: `rfdetr-tool predict source=./images weights=model.pth`
  - Инференс на YOLO датасете с сохранением сплитов: `rfdetr-tool predict source=./dataset weights=model.pth`
  - Вывод в CSV: `rfdetr-tool predict source=./images weights=model.pth format=csv`
  - Несколько источников: `rfdetr-tool predict source=dir1,dir2,data.yaml weights=model.pth`
  - Настройка порогов: `rfdetr-tool predict source=./images weights=model.pth conf_threshold=0.25 nms_threshold=0.5`
  - Батч + разрешение + устройство: `rfdetr-tool predict source=./images weights=model.pth batch_size=16 resolution=640 device=cuda:1`

### 9.2 `README.md`

Добавить после строки про `docs/train.md`:
- Краткий пример predict через CLI
- Ссылку на `docs/predict.md`

### 9.3 `CLAUDE.md`

Обновить секцию **Validation Strategy** — добавить smoke-тесты для predict:
```bash
rfdetr-tool predict                                    # pydantic error "Field required", exit 1
rfdetr-tool predict source=x weights=y format=invalid  # pydantic literal error, exit 1
rfdetr-tool predict source=x weights=y unknown=z       # "Неизвестный параметр", exit 1
```

---

## 10. Линтеры и проверки

После всех изменений — обязательный прогон:

```bash
# Форматирование
uv run ruff format rfdetr_tooling/

# Линтер (с автофиксом где возможно)
uv run ruff check --fix rfdetr_tooling/
uv run ruff check rfdetr_tooling/          # проверка что 0 ошибок

# Типизация
uv run mypy rfdetr_tooling/                # 0 ошибок в strict mode
```

Типичные проблемы, на которые обратить внимание:
- **`ANN401`** — не использовать `Any` без `noqa`, предпочитать конкретные типы
- **`PLR0913`** — слишком много аргументов функции → вынести в NamedTuple/dataclass или `noqa`
- **`PLC0415`** — lazy imports должны быть помечены `# noqa: PLC0415`
- **mypy `[no-any-return]`** — rfdetr не типизирован, `model.predict()` возвращает `Any` → явный cast или `# type: ignore[...]`
- **mypy `[import-untyped]`** — `import pillow_heif` без стабов → `# type: ignore[import-untyped]`
- **pandas стабы** — `pandas-stubs` уже в dev-зависимостях или `# type: ignore[import-untyped]`

---

## 11. Порядок реализации

1. Добавить `pandas`, `pillow-heif` в `pyproject.toml`, `uv sync`
2. Обновить `PredictConfig` в `config.py`
3. Реализовать парсинг источников (`_parse_sources`, `SourceEntry`)
4. Реализовать сборщики изображений (`_collect_from_dir`, `_collect_from_yolo_yaml`, `_collect_from_csv`)
5. Реализовать writer-ы (`YoloWriter`, `CsvWriter`)
6. Реализовать визуализацию (переиспользовать текущий код)
7. Собрать `predict_from_config` из компонентов
8. Обновить CLI help в `cli.py`
9. `uv run ruff format rfdetr_tooling/ && uv run ruff check --fix rfdetr_tooling/`
10. `uv run mypy rfdetr_tooling/` — исправить все ошибки
11. Написать `docs/predict.md`
12. Обновить `README.md` — добавить секцию predict
13. Обновить `CLAUDE.md` — smoke-тесты predict
14. Финальный прогон: `uv run ruff check rfdetr_tooling/ && uv run mypy rfdetr_tooling/`
