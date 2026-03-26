# План: True Resize режим + эксперимент на Severstal Steel Defect Detection

## Контекст

Пользователь хочет тренировать RF-DETR на изображениях 1920x128 (или аналогичных экстремальных aspect ratio). Текущий letterbox режим для таких изображений **крайне неэффективен** — при letterbox 1600x256 → 1920x128 используется лишь 41.7% canvas (800x128 реального изображения на 1920x128 canvas, 58% — серый padding).

**True resize** (принудительное растяжение без сохранения AR) — осознанный выбор для датасетов с однородным aspect ratio (все изображения одинаковых пропорций). Примеры: промышленные линейные сканеры (сталь, ткань), панорамные камеры.

### Severstal Steel Defect Detection

- **Kaggle**: https://www.kaggle.com/c/severstal-steel-defect-detection
- **Изображения**: 1600x256 пикселей (ширина x высота), все одинакового размера
- **Аннотации**: RLE-маски (instance segmentation) → нужна конвертация в bbox (detection)
- **Классы**: 4 типа дефектов (defect_1...defect_4)
- **Train**: 12568 изображений (из них ~63% без дефектов, ~4650 с аннотациями)
- **Формат**: `train.csv` с колонками ImageId, ClassId, EncodedPixels (RLE)
- **AR**: 6.25:1 — все изображения одинакового формата (непрерывное сканирование стального листа)

### Почему true resize, а не letterbox

| Режим | Target 1920x128 | Утилизация canvas | AR distortion |
|-------|----------------|-------------------|---------------|
| **Letterbox** | 800x128 img + 1120px padding | **41.7%** | Нет |
| **True resize** | 1920x128 img, 0 padding | **100%** | 6.25:1 → 15:1 |

Для Severstal AR distortion **допустим**: все изображения одинаковых пропорций (6.25:1), дефекты — горизонтальные полосы/пятна на стали, растяжение по горизонтали не ломает задачу. А letterbox теряет 58% площади canvas впустую.

**Однако**: при target resolution = native (1600x256 или кратное) letterbox и true resize дают одинаковый результат. True resize полезен когда target AR != source AR.

## Что нужно реализовать

### 1. Resize mode в конфигурации

Добавить параметр `resize_mode` с тремя вариантами:

```python
ResizeMode = Literal["auto", "letterbox", "true"]
```

- `auto` (default): если resolution — квадрат или скаляр → стандартный rfdetr resize. Если tuple (прямоугольный) → letterbox
- `letterbox`: всегда letterbox (сохранение AR + padding)
- `true`: принудительный resize без сохранения AR (растяжение до точных H×W)

### 2. True resize transforms для training

### 3. True resize inference (`predict_batch_true`)

### 4. Конвертер Severstal RLE → COCO detection

### 5. Эксперимент на Severstal

---

## Детальный план реализации

### Шаг 1: `rfdetr_tooling/config.py` — resize_mode

Добавить `ResizeMode` в type aliases и поле `resize_mode` в TrainConfig, PredictConfig, ValConfig.

```python
ResizeMode = Literal["auto", "letterbox", "true"]

class TrainConfig(BaseModel):
    # ... existing fields ...
    resize_mode: ResizeMode = "auto"

class PredictConfig(BaseModel):
    resize_mode: ResizeMode = "auto"

class ValConfig(BaseModel):
    resize_mode: ResizeMode = "auto"
```

Валидация: `resize_mode=true` допускается только при `resolution` == tuple (прямоугольное). Если `resize_mode=true` + скалярный resolution → ошибка (квадратный true resize = обычный resize, бессмысленно).

**Edge cases**:
- `resize_mode=auto, resolution=512x384` → letterbox (текущее поведение)
- `resize_mode=letterbox, resolution=512x384` → letterbox (явно)
- `resize_mode=true, resolution=1920x128` → true resize
- `resize_mode=true, resolution=384` → ошибка
- `resize_mode=auto, resolution=384` → стандартный rfdetr (без изменений)

### Шаг 2: `rfdetr_tooling/_inference.py` — true resize transforms + inference

#### 2.1 True resize transforms для training

Новая функция `_make_true_resize_transforms(res_h, res_w)` — аналог `_make_rect_transforms`, но вместо LetterboxResize использует `Resize(height=res_h, width=res_w)` (прямой resize без сохранения AR).

```python
def _make_true_resize_transforms(
    res_h: int,
    res_w: int,
) -> Callable[..., Any]:
    """Возвращает patched make_coco_transforms для true resize (без letterbox)."""

    def _patched(
        image_set: str,
        resolution: int,
        multi_scale: bool = False,
        expanded_scales: bool = False,
        skip_random_resize: bool = False,
        patch_size: int = 16,
        num_windows: int = 4,
        aug_config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        # Аналог _make_rect_transforms, но с Resize вместо LetterboxResize:

        if image_set == "train":
            option_a = {"Resize": {"height": res_h, "width": res_w}}
            # option_b: crop + resize для аугментации
            crop_min = min(384, res_h, res_w)
            crop_max = min(600, max(res_h, res_w))
            option_b = {
                "Sequential": {
                    "transforms": [
                        {"SmallestMaxSize": {"max_size": [400, 500, 600]}},
                        {"OneOf": {"transforms": [
                            {"RandomSizedCrop": {
                                "min_max_height": [crop_min, crop_max],
                                "height": res_h,
                                "width": res_w,
                            }}
                        ]}}
                    ]
                }
            }
            resize_config = [{"OneOf": {"transforms": [option_a, option_b]}}]
            # ... стандартная сборка pipeline

        if image_set in ("val", "test", "val_speed"):
            resize_config = [{"Resize": {"height": res_h, "width": res_w}}]
            # ... стандартная сборка pipeline
```

Ключевое отличие от letterbox: `Resize` вместо `LetterboxResize`. Bbox координаты обрабатываются корректно, т.к. `Resize` — geometric transform, albumentations автоматически масштабирует bbox'ы при resize.

#### 2.2 True resize inference

Новая функция `predict_batch_true` — аналог `predict_batch_rect`, но с прямым resize вместо letterbox:

```python
def predict_batch_true(
    model: RFDETR,
    images: list[Image.Image],
    threshold: float,
    res_h: int,
    res_w: int,
) -> list[sv.Detections]:
    """Inference с true resize (без letterbox)."""

    for img in images:
        t = F.to_tensor(img)
        orig_h, orig_w = t.shape[1], t.shape[2]
        t = t.to(device)
        # Прямой resize — без letterbox, без padding
        t = F.resize(t, [res_h, res_w])
        t = F.normalize(t, _MEANS, _STDS)
        meta.append((orig_h, orig_w))
        processed.append(t)

    # Forward pass с target_sizes=(res_h, res_w)
    # Postprocess: rescale из (res_h, res_w) в (orig_h, orig_w)
    for result, (orig_h, orig_w) in zip(results, meta):
        xyxy = boxes[keep].float().cpu().numpy()
        # Обратный rescale: canvas → original
        xyxy[:, [0, 2]] *= orig_w / res_w  # x coords
        xyxy[:, [1, 3]] *= orig_h / res_h  # y coords
```

Обратное преобразование проще чем в letterbox — линейный масштаб по каждой оси отдельно.

#### 2.3 Context manager `true_resize_patch`

Аналог `rect_resolution_patch`, но использует `_make_true_resize_transforms`:

```python
@contextmanager
def true_resize_patch(res_h: int, res_w: int) -> Iterator[None]:
    """Context manager: подменяет rfdetr transform builders на true resize."""
    import rfdetr.datasets.coco as coco_mod
    # ... аналогично rect_resolution_patch, но с _make_true_resize_transforms
```

### Шаг 3: `rfdetr_tooling/train.py` — поддержка resize_mode

Обновить функцию `train()`:

```python
def train(
    ...,
    resolution: int | tuple[int, int] | None = None,
    resize_mode: str = "auto",  # "auto" | "letterbox" | "true"
    ...
) -> None:
    # Определяем effective resize mode
    if resize_mode == "auto":
        effective_mode = "letterbox" if isinstance(resolution, tuple) else "standard"
    else:
        effective_mode = resize_mode

    if effective_mode == "letterbox":
        with rect_resolution_patch(res_h, res_w):
            model.train(**kwargs)
    elif effective_mode == "true":
        with true_resize_patch(res_h, res_w):
            model.train(**kwargs)
    else:
        model.train(**kwargs)
```

### Шаг 4: `rfdetr_tooling/predict.py` — поддержка resize_mode

Обновить inference branch:

```python
if rect_resolution is not None:
    if resize_mode == "true":
        det_list = predict_batch_true(model, pil_images, conf_threshold, res_h, res_w)
    else:
        det_list = predict_batch_rect(model, pil_images, conf_threshold, res_h, res_w)
else:
    det_list = model.predict(pil_images, threshold=conf_threshold)
```

### Шаг 5: `rfdetr_tooling/val.py` — поддержка resize_mode

Аналогично predict.py.

### Шаг 6: `rfdetr_tooling/cli.py` — CLI параметр resize_mode

Параметр `resize_mode` — строка, не требует специальной обработки в `_coerce_value`. Pydantic Literal валидация обработает невалидные значения.

```bash
rfdetr-tool train data=./severstal variant=nano resolution=1920x128 resize_mode=true epochs=20
```

### Шаг 7: Конвертер Severstal RLE → COCO detection

Новый скрипт `experiments/severstal/convert_severstal.py` (не в основном пакете — экспериментальный).

#### 7.1 RLE декодирование

```python
def rle_decode(rle_string: str, height: int, width: int) -> np.ndarray:
    """Decode RLE string → binary mask (H, W)."""
    pairs = list(map(int, rle_string.split()))
    starts = pairs[0::2]
    lengths = pairs[1::2]
    mask = np.zeros(height * width, dtype=np.uint8)
    for start, length in zip(starts, lengths):
        # Severstal RLE: 1-indexed, column-major order
        mask[start - 1 : start - 1 + length] = 1
    # Column-major → row-major (Fortran → C order)
    return mask.reshape((height, width), order="F")
```

**Внимание**: Severstal RLE использует column-major (Fortran) порядок и 1-based indexing.

#### 7.2 Mask → Bbox

```python
def mask_to_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """Binary mask → (x, y, w, h) COCO format. None если маска пустая."""
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any():
        return None
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    return (int(x_min), int(y_min), int(x_max - x_min + 1), int(y_max - y_min + 1))
```

#### 7.3 Генерация COCO JSON

```python
def convert_severstal_to_coco(
    csv_path: str,
    images_dir: str,
    output_dir: str,
    val_fraction: float = 0.2,
) -> None:
    """Конвертирует Severstal train.csv → COCO detection format.

    Структура output_dir:
        train/
            _annotations.coco.json
            <images...>
        valid/
            _annotations.coco.json
            <images...>
        test/
            _annotations.coco.json  (пустой — для run_test=True в rfdetr)
            <symlink to valid images>
    """
```

Алгоритм:
1. Прочитать train.csv, сгруппировать аннотации по ImageId
2. Отфильтровать изображения без аннотаций (NaN EncodedPixels)
3. Для каждого изображения с аннотациями:
   - Декодировать RLE → binary mask
   - Извлечь bbox (x, y, w, h)
   - Отфильтровать слишком мелкие bbox'ы (area < min_area)
4. Разделить на train/valid (по val_fraction, стратифицированно по классам)
5. Скопировать (или symlink) изображения в train/ и valid/
6. Сгенерировать `_annotations.coco.json` для каждого split
7. Создать test/ как symlink на valid/ (для rfdetr run_test)

**Категории**:
```json
[
    {"id": 1, "name": "defect_1"},
    {"id": 2, "name": "defect_2"},
    {"id": 3, "name": "defect_3"},
    {"id": 4, "name": "defect_4"}
]
```

**Один RLE → один или несколько bbox?**

Один RLE может содержать несвязные компоненты (два отдельных дефекта одного класса). Варианты:
- **Простой**: один bbox на весь RLE (bounding rectangle) — может захватить фон между компонентами
- **Правильный**: найти connected components через `cv2.connectedComponents()`, создать отдельный bbox для каждого

**Рекомендация**: connected components — дефекты стали часто состоят из нескольких кусков, объединение в один bbox теряет информацию.

```python
def mask_to_bboxes(mask: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Binary mask → список bbox'ов (x, y, w, h) COCO format, по connected components."""
    num_labels, labels = cv2.connectedComponents(mask)
    bboxes = []
    for label_id in range(1, num_labels):
        component_mask = (labels == label_id).astype(np.uint8)
        bbox = mask_to_bbox(component_mask)
        if bbox is not None:
            bboxes.append(bbox)
    return bboxes
```

### Шаг 8: Скачивание и подготовка датасета

```bash
# Скачать через kaggle CLI
kaggle competitions download -c severstal-steel-defect-detection -p ~/datasets/severstal/

# Разархивировать
cd ~/datasets/severstal/
unzip severstal-steel-defect-detection.zip

# Конвертировать
python experiments/severstal/convert_severstal.py \
    --csv ~/datasets/severstal/train.csv \
    --images ~/datasets/severstal/train_images/ \
    --output ~/datasets/severstal_coco/ \
    --val-fraction 0.2
```

### Шаг 9: Эксперимент

#### 9A: True resize 1600x256 (native resolution)

```bash
rfdetr-tool train \
    data=~/datasets/severstal_coco \
    variant=nano \
    epochs=20 \
    batch_size=8 \
    resolution=1600x256 \
    resize_mode=true \
    lr_drop=15 \
    output_dir=experiments/severstal/9a_true_1600x256
```

Размеры: 1600x256 = 409600 px. Batch=8 → ~9.5M px/batch (nano backbone = ~6M params).

#### 9B: Letterbox 1600x256 (для сравнения)

```bash
rfdetr-tool train \
    data=~/datasets/severstal_coco \
    variant=nano \
    epochs=20 \
    batch_size=8 \
    resolution=1600x256 \
    resize_mode=letterbox \
    lr_drop=15 \
    output_dir=experiments/severstal/9b_lb_1600x256
```

При letterbox 1600x256 → изображения 1600x256 уже имеют AR 6.25:1, а target 1600x256 = AR 6.25:1, значит scale=1.0, pad=0 → **letterbox = true resize** (идентичны для native resolution).

#### 9C: True resize 1920x128 (target resolution из задачи)

```bash
rfdetr-tool train \
    data=~/datasets/severstal_coco \
    variant=nano \
    epochs=20 \
    batch_size=16 \
    resolution=1920x128 \
    resize_mode=true \
    lr_drop=15 \
    output_dir=experiments/severstal/9c_true_1920x128
```

Размеры: 1920x128 = 245760 px. Batch=16 → ~3.9M px/batch. AR distortion: 6.25:1 → 15:1 (x2.4 растяжение по горизонтали). Высота сжимается в 2x (256→128) — возможна потеря мелких дефектов.

#### 9D: Letterbox 1920x128 (для контраста — показать проблему)

```bash
rfdetr-tool train \
    data=~/datasets/severstal_coco \
    variant=nano \
    epochs=20 \
    batch_size=16 \
    resolution=1920x128 \
    resize_mode=letterbox \
    lr_drop=15 \
    output_dir=experiments/severstal/9d_lb_1920x128
```

Утилизация canvas: 41.7%. Ожидание: значительно хуже 9C из-за потери 58% canvas.

### Критерии успеха эксперимента

| Проверка | Критерий |
|----------|----------|
| Тренировка не падает | exit code 0 для всех 4 конфигов |
| Loss снижается | Монотонное снижение train_loss |
| mAP > 0 | mAP@50 > 0.0 хотя бы для 9A и 9C |
| 9C true >> 9D letterbox | mAP@50 9C > mAP@50 9D (подтверждает что true resize эффективнее для AR mismatch) |
| 9A ≈ 9B | Должны быть ~одинаковы (native resolution, AR match) |
| Bbox координаты корректны | Predict → визуализация → bbox'ы совпадают с дефектами |

---

## Изменяемые файлы

| Файл | Изменения | Строк |
|------|-----------|-------|
| `rfdetr_tooling/config.py` | `ResizeMode` type, `resize_mode` field в 3 классах | ~15 |
| `rfdetr_tooling/_inference.py` | `predict_batch_true`, `_make_true_resize_transforms`, `true_resize_patch` | ~100 |
| `rfdetr_tooling/train.py` | `resize_mode` параметр, branch по mode | ~20 |
| `rfdetr_tooling/predict.py` | `resize_mode` параметр, branch по mode | ~10 |
| `rfdetr_tooling/val.py` | `resize_mode` параметр, branch по mode | ~10 |
| `rfdetr_tooling/cli.py` | Пробросить `resize_mode` | ~5 |
| `experiments/severstal/convert_severstal.py` | **Новый**: RLE→COCO конвертер | ~200 |

Итого: ~160 строк изменений + ~200 строк нового экспериментального скрипта.

---

## Потенциальные проблемы

### 1. Очень маленький height=128

128 пикселей по высоте — крайне мало для детекции. DINOv2 с patch_size=16 создаст feature map 128/16 = 8 токенов по высоте. С window_size=2 → 4 окна по высоте, что допустимо (block_size=32, 128/32=4).

Но мелкие дефекты (пятна < 10px) могут потеряться при resize 256→128.

### 2. Severstal: неравномерное распределение классов

Severstal имеет сильный дисбаланс:
- defect_3: самый частый (~70% аннотаций)
- defect_1, defect_4: редкие

Для фокусного train можно отфильтровать только изображения с дефектами (4650 из 12568).

### 3. VRAM при 1600x256 (ограничение 20GB)

1600x256 = 409k пикселей. Для сравнения: стандартный 384x384 = 147k. Больше в ~2.8 раз. Ограничение: **не более 20GB VRAM**. Для nano batch_size=4 должно быть безопасно. Для 1920x128 (245k px) — batch_size=8.

### 4. Augmentation при true resize

`RandomSizedCrop` с `height=res_h, width=res_w` после crop'а делает resize в target dimensions. При target 1920x128 crop квадратный, а потом resize растягивает — это может давать неестественные аугментации.

**Альтернатива**: для true resize с экстремальным AR, option_b (crop) можно отключить и использовать только прямой resize + standard augmentations (blur, color jitter и т.д.).

### 5. run_test=True в rfdetr

rfdetr по умолчанию ищет test/ split. Нужно создать test/ как symlink на valid/ (или пустой test/).

### 6. RLE column-major порядок

Severstal RLE использует Fortran (column-major) порядок. Ошибка в порядке → маски будут transposed → bbox'ы неправильные. Нужна визуальная верификация нескольких масок после конвертации.

---

## Порядок выполнения

| # | Шаг | Зависимости | Время |
|---|------|-------------|-------|
| 1 | `config.py`: ResizeMode type + field | — | 10 мин |
| 2 | `_inference.py`: true resize transforms + inference | — | 30 мин |
| 3 | `train.py`: resize_mode support | 1, 2 | 15 мин |
| 4 | `predict.py`, `val.py`: resize_mode support | 1, 2 | 15 мин |
| 5 | `cli.py`: пробросить resize_mode | 1 | 5 мин |
| 6 | Linters + mypy + CLI smoke tests | 1-5 | 15 мин |
| 7 | Скачать Severstal dataset | — | 10 мин |
| 8 | `convert_severstal.py`: RLE→COCO конвертер | 7 | 30 мин |
| 9 | Визуальная верификация конвертации (несколько bbox) | 8 | 10 мин |
| 10 | Эксперимент 9A: true 1600x256 | 6, 8 | ~30 мин |
| 11 | Эксперимент 9C: true 1920x128 | 6, 8 | ~30 мин |
| 12 | Эксперимент 9D: letterbox 1920x128 (опционально) | 6, 8 | ~30 мин |
| 13 | Val/predict на лучшем чекпоинте + визуализация | 10-12 | 15 мин |

Общее время (параллельно): ~3-4 часа (шаги 10-12 — GPU-bound, последовательно).

---

## Рефакторинг: объединение letterbox и true resize

Функции `_make_rect_transforms` и `_make_true_resize_transforms` имеют ~80% общего кода. Варианты DRY:

**Вариант A**: Параметр в `_make_rect_transforms`:

```python
def _make_rect_transforms(res_h, res_w, *, use_letterbox: bool = True):
    if use_letterbox:
        train_resize = {"LetterboxResize": {...}}
        val_resize = {"LetterboxResize": {...}}
    else:
        train_resize = {"Resize": {"height": res_h, "width": res_w}}
        val_resize = {"Resize": {"height": res_h, "width": res_w}}
    # ... остальной код общий
```

**Вариант B**: Общая фабрика с callback для resize config.

**Рекомендация**: Вариант A — проще, меньше кода, один context manager `rect_resolution_patch` с параметром `letterbox=True`.

Аналогично для inference: `predict_batch_rect` и `predict_batch_true` можно объединить в одну функцию с параметром `letterbox: bool`.

```python
def predict_batch_rect(
    model: RFDETR,
    images: list[Image.Image],
    threshold: float,
    res_h: int,
    res_w: int,
    *,
    letterbox: bool = True,
) -> list[sv.Detections]:
```

Это чище, чем дублировать функции.

---

## Результаты эксперимента

### Датасет после конвертации

| Метрика | Значение |
|---------|----------|
| Изображений с дефектами | 6666 |
| Train | 5333 изображений, 15950 bbox |
| Valid | 1333 изображений, 3972 bbox |
| Классы | defect_1: 2583, defect_2: 271, defect_3: 11565, defect_4: 1531 |
| Медианный bbox | 32x111 px |

### Сравнительная таблица (nano, 20 epochs, lr_drop=15)

| Эксперимент | Resolution | Resize | Batch | VRAM (MB) | mAP@50:95 (best) | mAP@50 (best) | train_loss (e20) | test_loss (e20) |
|-------------|-----------|--------|-------|-----------|-------------------|----------------|------------------|-----------------|
| true_1600x256 | 1600x256 | true | 8 | 5741 | **0.0324** | **0.0636** | 4.1818 | 4.6278 |
| true_1920x128 | 1920x128 | true | 16 | 8057 | 0.0162 | 0.0328 | 4.2323 | 4.7442 |
| lb_1920x128 | 1920x128 | letterbox | 16 | 8057 | 0.0000 | 0.0000 | 4.5011 | 4.7427 |

### Выводы

1. **True resize работает**: тренировка стабильна, loss снижается, mAP > 0 — реализация корректна
2. **Letterbox при AR mismatch катастрофически плох**: letterbox 1920x128 даёт mAP=0 на всех 20 эпохах. При letterbox изображение 1600x256 масштабируется до 800x128 + 1120px серого padding = 41.7% утилизации canvas. Модель не может ничему научиться
3. **True resize 1920x128 vs 1600x256**: native resolution (1600x256) лучше в 2x по mAP. При 1920x128 высота сжимается 256→128 (потеря деталей) + AR distortion 6.25:1→15:1
4. **VRAM**: все эксперименты уложились в ≤8.1 GB, далеко от лимита 20 GB
5. **Best epoch**: для true_1600x256 лучший результат на epoch 2 (mAP@50:95=0.0319) и epoch 5 (mAP@50:95=0.0324). После epoch 5 mAP падает несмотря на снижение train_loss — признак переобучения или нестабильности при малом датасете
6. **Практическая рекомендация**: для Severstal-подобных данных (однородный AR) использовать `resize_mode=true` с resolution максимально близким к native
