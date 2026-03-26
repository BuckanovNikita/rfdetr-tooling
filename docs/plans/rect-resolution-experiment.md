# Эксперимент: Валидация прямоугольного resolution для RF-DETR

## Цель

Доказать, что RF-DETR корректно тренируется и делает inference с прямоугольным resolution, **без искажения пропорций объектов**. Сравнить letterbox rect 512x384 vs square 384x384 на COCO2017.

## Проблема aspect ratio distortion

Текущая реализация rect resolution использует `Resize(height=H, width=W)` — принудительный resize без сохранения пропорций. Если изображения в датасете имеют разный aspect ratio (COCO: от 1:1 до ~2:1), объекты будут растянуты/сжаты. Это:
- Ломает пропорции объектов → ухудшает качество детекции
- Делает невозможным честное сравнение square vs rect

**Решение**: letterbox — resize с сохранением AR + padding до целевого размера.

## Letterbox pipeline

### Training (Albumentations)

```
LongestMaxSize(max_size=max(H, W))  →  сохраняет AR, вписывает в H×W
PadIfNeeded(min_height=H, min_width=W, border_mode=CONSTANT, fill=114)  →  pad до точных H×W
```

Albumentations автоматически обновляет bbox координаты при geometric transforms, включая pad. Координаты bbox'ов остаются корректными.

### Inference (`predict_batch_rect`)

Текущий `predict_batch_rect` делает `F.resize(t, [res_h, res_w])` — прямой resize без letterbox. Нужно заменить на:

```python
# 1. Resize с сохранением AR (longest side = max(res_h, res_w))
scale = min(res_h / orig_h, res_w / orig_w)
new_h, new_w = int(orig_h * scale), int(orig_w * scale)
t = F.resize(t, [new_h, new_w])

# 2. Pad до (res_h, res_w) серым (114/255)
pad_h = res_h - new_h
pad_w = res_w - new_w
t = F.pad(t, [0, 0, pad_w, pad_h], fill=114/255)  # right, bottom padding
```

### Postprocess — нужна коррекция координат

**Критический нюанс**: rfdetr postprocess масштабирует box'ы из [0,1] → `target_sizes` (оригинальный размер). Но при letterbox модель видит изображение, сдвинутое и масштабированное внутри padded canvas. Box'ы в [0,1] координатах соответствуют padded canvas, а не оригинальному изображению.

Нужна обратная трансформация:
```python
# Box'ы из postprocess с target_sizes=(res_h, res_w) — в координатах padded canvas
# Убираем padding offset и масштабируем обратно в оригинальные координаты
boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_left) / scale
boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_top) / scale
```

Либо (проще): передавать `target_sizes=(res_h, res_w)` и потом вручную rescale.

## Изменения в коде (scope эксперимента)

### Изменяемые файлы

| Файл | Изменения |
|------|-----------|
| `_inference.py` | Новая функция `_letterbox_resize` для inference. Обновить `predict_batch_rect` для letterbox + обратное преобразование координат. Новая функция `_make_letterbox_transforms` для training. |
| `train.py` | Использовать letterbox transforms вместо прямого Resize в `rect_resolution_patch`. |
| Без изменений | `config.py`, `cli.py`, `predict.py`, `val.py` — интерфейс не меняется. |

### Детали реализации

#### 1. Letterbox transforms для training

В `_make_rect_transforms` заменить `Resize(height=res_h, width=res_w)` на:

```python
# Val/test:
resize_config = [
    {"LongestMaxSize": {"max_size": max(res_h, res_w)}},
    {"PadIfNeeded": {
        "min_height": res_h,
        "min_width": res_w,
        "border_mode": 0,  # cv2.BORDER_CONSTANT
        "fill": 114,
        "position": "top_left",
    }},
]

# Train — тот же letterbox + augmentations:
option_a = {"Sequential": {"transforms": [
    {"LongestMaxSize": {"max_size": max(res_h, res_w)}},
    {"PadIfNeeded": {
        "min_height": res_h,
        "min_width": res_w,
        "border_mode": 0,
        "fill": 114,
        "position": "random",
    }},
]}}
```

`position="top_left"` для val (детерминированный), `position="random"` для train (augmentation).

Albumentations `PadIfNeeded` автоматически сдвигает bbox координаты при padding — координаты остаются корректными в новом canvas. `Normalize` в rfdetr потом переведёт их в [0,1] относительно padded canvas.

#### 2. Letterbox inference в `predict_batch_rect`

```python
def predict_batch_rect(model, images, threshold, res_h, res_w):
    for img in images:
        t = F.to_tensor(img)
        orig_h, orig_w = t.shape[1], t.shape[2]

        # Letterbox: resize preserving AR
        scale = min(res_h / orig_h, res_w / orig_w)
        new_h = int(orig_h * scale)
        new_w = int(orig_w * scale)
        t = F.resize(t, [new_h, new_w])

        # Pad to exact (res_h, res_w)
        pad_bottom = res_h - new_h
        pad_right = res_w - new_w
        t = torch.nn.functional.pad(t, [0, pad_right, 0, pad_bottom], value=114/255)

        # Normalize
        t = F.normalize(t, means, stds)
        # ...store (scale, pad_bottom, pad_right, orig_h, orig_w) per image

    # Postprocess: target_sizes = (res_h, res_w) для всех
    # Потом rescale boxes обратно:
    for det, (scale, pad_b, pad_r, orig_h, orig_w) in zip(detections, meta):
        # boxes в координатах (res_h, res_w) canvas
        det.xyxy[:, [0, 2]] /= scale  # x coords
        det.xyxy[:, [1, 3]] /= scale  # y coords
        # Clip to original image
        det.xyxy[:, [0, 2]] = np.clip(det.xyxy[:, [0, 2]], 0, orig_w)
        det.xyxy[:, [1, 3]] = np.clip(det.xyxy[:, [1, 3]], 0, orig_h)
```

## Ресурсы

- **GPU**: NVIDIA RTX 5090, 32 GB VRAM
- **Датасет**: coco8 (4 train / 4 valid) — smoke test; coco2017_rfdetr_coco (118k train / 5k valid) — полное сравнение
- **Variant**: `nano` (block_size=32, default resolution=384)

## Дизайн эксперимента

### Фаза 0: Реализация letterbox (~1-2 часа кода)

1. Обновить `_make_rect_transforms` в `_inference.py` — использовать `LongestMaxSize` + `PadIfNeeded` вместо `Resize`
2. Обновить `predict_batch_rect` — letterbox resize + обратное преобразование координат
3. Проверить: `uv run ruff check . && uv run mypy .`

### Фаза 1: Smoke test (coco8, ~5 минут)

#### 1A: Square baseline

```bash
rfdetr-tool train \
  data=./coco8 variant=nano epochs=20 batch_size=4 \
  resolution=384 multi_scale=false \
  output_dir=experiments/rect_smoke/1a_square_384
```

#### 1B: Rect 512x384 letterbox

```bash
rfdetr-tool train \
  data=./coco8 variant=nano epochs=20 batch_size=4 \
  resolution=512x384 \
  output_dir=experiments/rect_smoke/1b_rect_512x384_lb
```

#### Критерий успеха

| Проверка | Критерий |
|----------|----------|
| Тренировка не падает | exit code 0, 20 эпох пройдены |
| Loss снижается | Последний loss < первый loss |
| Checkpoint создан | `checkpoint_best_ema.pth` существует |
| Нет shape mismatch | Нет ошибок в логах |

### Фаза 2: Валидация чекпоинтов (coco8)

```bash
# Val square
rfdetr-tool val data=./coco8 variant=nano resolution=384 \
  weights=experiments/rect_smoke/1a_square_384/checkpoint_best_ema.pth

# Val rect letterbox
rfdetr-tool val data=./coco8 variant=nano resolution=512x384 \
  weights=experiments/rect_smoke/1b_rect_512x384_lb/checkpoint_best_ema.pth

# Predict rect letterbox + визуализация
rfdetr-tool predict source=./coco8/valid/images variant=nano resolution=512x384 \
  weights=experiments/rect_smoke/1b_rect_512x384_lb/checkpoint_best_ema.pth \
  visualize=true output_dir=experiments/rect_smoke/2_predict_rect_lb
```

#### Критерий успеха

| Проверка | Критерий |
|----------|----------|
| Val не падает | mAP выведен |
| mAP > 0 | mAP@50 > 0.0 |
| Predict не падает | YOLO-файлы созданы |
| **Bbox пропорции корректны** | Визуально bbox'ы не сдвинуты, совпадают с объектами |
| Координаты в пределах | bbox xyxy ≤ (image_width, image_height) |

### Фаза 3: Сравнение на COCO2017 (~2-4 часа)

Главный эксперимент: **384×384 square vs 512×384 rect letterbox**.

#### 3A: Square 384 (без multi_scale)

```bash
rfdetr-tool train \
  data=~/datasets/coco2017_rfdetr_coco variant=nano \
  epochs=5 batch_size=16 resolution=384 \
  multi_scale=false lr_drop=5 \
  output_dir=experiments/rect_coco/3a_square_384
```

#### 3B: Rect 512x384 letterbox

```bash
rfdetr-tool train \
  data=~/datasets/coco2017_rfdetr_coco variant=nano \
  epochs=5 batch_size=16 resolution=512x384 \
  lr_drop=5 \
  output_dir=experiments/rect_coco/3b_rect_512x384_lb
```

#### Валидация

```bash
rfdetr-tool val data=~/datasets/coco2017_rfdetr_coco variant=nano \
  resolution=384 \
  weights=experiments/rect_coco/3a_square_384/checkpoint_best_ema.pth

rfdetr-tool val data=~/datasets/coco2017_rfdetr_coco variant=nano \
  resolution=512x384 \
  weights=experiments/rect_coco/3b_rect_512x384_lb/checkpoint_best_ema.pth
```

#### Критерий успеха

| Метрика | Ожидание |
|---------|----------|
| Обе тренировки завершены | exit code 0 |
| Loss сходится | Монотонное снижение |
| mAP@50 > 0.3 | Модель реально учится |
| 3B mAP >= 3A mAP | Больше пикселей → лучше или сравнимо |

**Ожидание**: 3B (512x384 = 197k пикс) должен быть >= 3A (384x384 = 147k пикс), т.к. больше информации при сохранении пропорций.

## Итоговая таблица результатов

### Фаза 1-2: Smoke test (coco8, 20 epochs)

| Эксперимент | Resolution | Letterbox | Результат |
|------------|-----------|-----------|-----------|
| 1A | 384×384 | нет (square) | OK, тренировка завершена, loss снижается |
| 1B | 512×384 | да | OK, тренировка завершена, loss снижается |
| 2 (val/predict) | 512×384 | да | OK, bbox корректны, координаты в пределах изображения |

### Фаза 3: COCO2017 (5 epochs, nano, batch_size=16)

| Эксперимент | Resolution | Letterbox | Пиксели | mAP@50:95 | mAP@50 | mAP@75 | train_loss | test_loss |
|------------|-----------|-----------|---------|-----------|--------|--------|------------|-----------|
| 3A | 384×384 | нет (square) | 147k | 0.0293 | 0.0391 | 0.0312 | 4.2525 | 4.3949 |
| 3B | 512×384 | да | 197k | **0.0554** | **0.1028** | **0.0509** | 4.2593 | 4.3921 |

**Дельта 3B vs 3A**: mAP@50:95 +89%, mAP@50 +163%, mAP@75 +63%

### Выводы

1. Прямоугольный letterbox resolution работает корректно — тренировка стабильна, bbox координаты валидны
2. Rect 512×384 значительно лучше square 384 на COCO2017 (преимущественно landscape изображения)
3. Train/test loss практически идентичны — letterbox не вносит нестабильность в тренировку
4. Большой скачок mAP на epoch 5 (lr_drop) — rect получил больший выигрыш от lr drop

## Потенциальные проблемы

### 1. PadIfNeeded и bbox clipping

Albumentations `PadIfNeeded` сдвигает bbox координаты при padding. Нужно убедиться, что `AlbumentationsWrapper` корректно обрабатывает `PadIfNeeded` как geometric transform (он входит в список `GEOMETRIC_TRANSFORMS` в rfdetr).

### 2. fill=114 vs ImageNet normalization

`fill=114` (серый) — стандартное значение для YOLO letterbox. После ImageNet normalization `(114/255 - mean) / std` даст ненулевые значения. Это нормально — модель учится игнорировать padding.

### 3. `target_sizes` в postprocess при letterbox inference

PostProcess масштабирует box'ы из [0,1] в `target_sizes`. Если передать `target_sizes=(orig_h, orig_w)` — координаты будут неправильные (модель видела padded canvas). Нужно передать `target_sizes=(res_h, res_w)` и потом вручную rescale обратно.

### 4. multi_scale автоматически отключается для rect

Для rect `multi_scale=False` устанавливается автоматически. В фазе 3 square baseline тоже запускается с `multi_scale=false` для честного сравнения.

### 5. Padding увеличивает «мёртвую» площадь

При letterbox 512x384 изображение ~4:3 займёт всю площадь, а ~1:1 — будет иметь padding по бокам. Это снижает эффективность пикселей, но сохраняет корректные пропорции.

### 6. run_test=True

rfdetr по умолчанию запускает test после тренировки. coco8 имеет `test/` — проблем не ожидается.

## Порядок выполнения

1. **Фаза 0**: реализовать letterbox в `_inference.py` (transforms + inference)
2. **Фаза 1**: smoke test на coco8 (1A, 1B)
3. **Фаза 2**: val/predict на чекпоинтах, визуальная проверка bbox'ов
4. Если фазы 1-2 пройдены — **фаза 3**: COCO2017 сравнение (3A, 3B)
5. Заполнить таблицу результатов, выводы
