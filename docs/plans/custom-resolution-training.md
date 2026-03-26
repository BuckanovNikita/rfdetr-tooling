# План: Прямоугольное разрешение для тренировки/инференса RF-DETR (Вариант D)

## Цель

Поддержка прямоугольных разрешений (e.g. `1920x1080`, `960x608`) в `rfdetr-tool train/val/predict` без fork rfdetr. Monkey-patch rfdetr в runtime.

## Предпосылки

### Что уже работает

- **Backbone (DINOv2)** нативно поддерживает прямоугольные входы:
  - `interpolate_pos_encoding(embeddings, H, W)` интерполирует PE в `(H//ps, W//ps)` через bicubic — работает при `H != W` (`dinov2_with_windowed_attn.py:245-297`)
  - Windowed attention делит оба axis на `num_windows` независимо (`dinov2_with_windowed_attn.py:318-331`)
  - Forward assertion проверяет divisibility обеих осей: `x.shape[2] % block_size == 0 and x.shape[3] % block_size == 0` (`dinov2.py:200-203`)
- **NestedTensor** padding'ит до max size в batch с mask — автоматически handle разные размеры (`misc.py:333-355`)
- **LWDETR forward** принимает `NestedTensor` с произвольным `(H, W)` — нет квадратных ограничений (`lwdetr.py:147-240`)

### Что блокирует

| Что | Где | Проблема |
|-----|-----|----------|
| Transform pipeline | `coco.py:473-481` | `Resize(height=S, width=S)` — всегда квадрат |
| Multi-scale scales | `coco.py:43-57` | `compute_multi_scale_scales(resolution: int)` — один скаляр |
| `_build_train_resize_config` | `coco.py:286-306` | `Resize(height=s, width=s)` / `RandomSizedCrop(..., height=s, width=s)` |
| Inference resize | `detr.py:369` | `F.resize(img, (resolution, resolution))` |
| `backbone[0].encoder.shape` | `dinov2.py:73` | Используется в `export()` для PE pre-bake (не влияет на training, но влияет на inference) |
| `model.resolution` | `main.py:85` | Скаляр `int`, используется в `detr.py:369` |

### Divisibility constraints

`block_size = patch_size × num_windows` — обе оси должны быть кратны.

| Variant | patch_size | num_windows | block_size |
|---------|-----------|-------------|-----------|
| nano | 16 | 2 | 32 |
| small | 16 | 2 | 32 |
| base | 14 | 4 | 56 |
| medium | 16 | 2 | 32 |
| large | 16 | 2 | 32 |

Примеры ближайших кратных к 1920×1080:
- block_size=32: **1920×1088** (1080→1088) или 1920×1056
- block_size=56: **1904×1064** (1920→1904, 1080→1064) или 1960×1120

### DINOv2 pretrained weights

`positional_encoding_size` в `rfdetr/config.py` определяет initial PE size. При создании модели (`dinov2.py:97-104`):
```python
implied_resolution = positional_encoding_size * patch_size
if implied_resolution != dino_config["image_size"]:
    load_dinov2_weights = False  # НЕ загружаются DINOv2 backbone weights!
```
Дефолтные `positional_encoding_size`: nano=24, small=32, base=37, medium=36, large=44.

**Но**: при finetuning rfdetr checkpoint (не DINOv2) — PE загружаются из rfdetr checkpoint через `model.load_state_dict()`, а потом интерполируются в forward pass. Так что для finetuning **это не проблема**. Проблема только для training from scratch без rfdetr pretrain.

---

## Детальный план

### Шаг 1: `rfdetr_tooling/config.py` — тип resolution + валидация

**Изменения**:

1. Добавить type alias `Resolution = int | tuple[int, int]`
2. Заменить тип `resolution` на `Resolution | None` во всех трёх config-классах
3. Заменить validator `_resolution_divisible_by_32` на два validator'а:
   - `mode="before"`: парсинг строки `"960x608"` → `(608, 960)` (WxH input → (H,W) internal)
   - `mode="after"`: валидация кратности 32 для каждого измерения

```python
Resolution = int | tuple[int, int]


class TrainConfig(BaseModel):
    # ...
    resolution: Resolution | None = Field(default=None)

    @field_validator("resolution", mode="before")
    @classmethod
    def _parse_resolution(cls, v: Any) -> Any:
        if v is None:
            return v
        if isinstance(v, str):
            if "x" in v.lower():
                w_str, h_str = v.split("x", 1)
                return (int(h_str), int(w_str))  # WxH → (H, W)
            return int(v)
        if isinstance(v, (list, tuple)) and len(v) == 2:
            return (int(v[0]), int(v[1]))
        return v

    @field_validator("resolution")
    @classmethod
    def _resolution_divisible_by_32(cls, v: Resolution | None) -> Resolution | None:
        if v is None:
            return v
        dims = (v,) if isinstance(v, int) else v
        for d in dims:
            if d <= 0:
                msg = f"resolution должен быть > 0, получено {d}"
                raise ValueError(msg)
            if d % 32 != 0:
                msg = f"resolution должен быть кратен 32, получено {d}"
                raise ValueError(msg)
        return v
```

**Повторить** одинаковые validators для `PredictConfig` и `ValConfig`. Вынести в mixin или standalone-функции, чтобы не дублировать.

**Edge cases**:
- `resolution=960` → `int(960)` — без изменений, обратная совместимость
- `resolution=960x608` → `(608, 960)` — (H, W) внутри
- `resolution=[608, 960]` из YAML → `(608, 960)`
- `resolution=961` → ошибка "кратен 32"
- `resolution=960x607` → ошибка "кратен 32"

### Шаг 2: `rfdetr_tooling/cli.py` — обновить `_coerce_value`

Проблема: `_coerce_value` в `cli.py:85-106` приводит raw string к типу по annotation. Для `resolution: int | tuple[int, int] | None` он увидит `Union[int, tuple, None]`, возьмёт первый non-None тип (`int`), и вызовет `int("960x608")` → crash.

**Решение**: не трогать `_coerce_value`. Передать raw string `"960x608"` в pydantic as-is — `mode="before"` validator из шага 1 обработает строку. Нужно только убрать преждевременное приведение к `int`.

Вариант: в `_coerce_value`, если `annotation` — Union содержащий `tuple`, **не** приводить к int, а вернуть raw string:

```python
def _coerce_value(value: str, annotation: Any) -> Any:
    origin = typing.get_origin(annotation)

    if origin is typing.Union or isinstance(annotation, types.UnionType):
        args = typing.get_args(annotation)
        non_none = [a for a in args if a is not type(None)]
        # Если Union содержит tuple — не приводить, пусть pydantic разбирается
        if any(typing.get_origin(a) is tuple or a is tuple for a in non_none):
            return value
        if non_none:
            return _coerce_value(value, non_none[0])

    # ... rest unchanged
```

Альтернативный (более простой) вариант: специальная обработка для `resolution` в `_build_config`:
```python
# В _build_config, перед coerce:
if key == "resolution" and "x" in raw_value.lower():
    data[key] = raw_value  # пусть pydantic validator разберётся
else:
    data[key] = _coerce_value(raw_value, hints[key])
```

**Рекомендация**: первый вариант (generic tuple detection) — чище, не хардкодит имя поля.

### Шаг 3: `rfdetr_tooling/train.py` — monkey-patch transforms + backbone shape

Это основной шаг. Порядок действий внутри `train()`:

#### 3.1 Определить `res_h, res_w` и `scalar_resolution`

```python
rect_resolution: tuple[int, int] | None = None
if isinstance(resolution, tuple):
    rect_resolution = resolution  # (H, W)
    res_h, res_w = resolution
    scalar_resolution = max(res_h, res_w)
    resolution_for_rfdetr: int | None = scalar_resolution
elif isinstance(resolution, int):
    resolution_for_rfdetr = resolution
else:
    resolution_for_rfdetr = None
```

#### 3.2 Monkey-patch `make_coco_transforms_square_div_64`

rfdetr training по умолчанию использует `square_resize_div_64=True` (из `rfdetr/config.py:304`). Функция `make_coco_transforms_square_div_64` вызывается из `build_roboflow_from_coco` (`coco.py:569-586`).

Нужно подменить **обе** функции (`make_coco_transforms_square_div_64` и `make_coco_transforms`) — rfdetr может вызвать любую в зависимости от `square_resize_div_64`.

Подменная функция должна:
1. Заменить `Resize(height=S, width=S)` на `Resize(height=res_h, width=res_w)` в val/test
2. Заменить scales в `_build_train_resize_config` на пары `(h_i, w_i)` для train
3. Или — **проще** — принудительно отключить multi_scale и использовать single rect resize

**Самый простой подход** — переписать transform pipeline целиком в monkey-patch:

```python
def _make_rect_transforms(
    res_h: int,
    res_w: int,
) -> Callable:
    """Возвращает patched make_coco_transforms_square_div_64 для прямоугольного resize."""
    from rfdetr.datasets.aug_config import AUG_CONFIG
    from rfdetr.datasets.transforms import AlbumentationsWrapper, Normalize

    def _patched(
        image_set: str,
        resolution: int,  # ignored — используем res_h, res_w
        multi_scale: bool = False,  # ignored — rect не поддерживает multi_scale
        expanded_scales: bool = False,
        skip_random_resize: bool = False,
        patch_size: int = 16,
        num_windows: int = 4,
        aug_config: dict | None = None,
    ) -> Compose:
        to_image = ToImage()
        to_float = ToDtype(torch.float32, scale=True)
        normalize = Normalize()

        if image_set == "train":
            resolved_aug_config = aug_config if aug_config is not None else AUG_CONFIG

            # Прямоугольный resize: option A = direct resize, option B = resize + crop + resize
            option_a = {
                "OneOf": {
                    "transforms": [{"Resize": {"height": res_h, "width": res_w}}],
                }
            }
            # Crop: используем min(res_h, res_w) как размер crop, потом resize в (res_h, res_w)
            crop_size = min(res_h, res_w)
            option_b = {
                "Sequential": {
                    "transforms": [
                        {"SmallestMaxSize": {"max_size": [400, 500, 600]}},
                        {
                            "OneOf": {
                                "transforms": [
                                    {"RandomSizedCrop": {
                                        "min_max_height": [384, 600],
                                        "height": res_h,
                                        "width": res_w,
                                    }},
                                ],
                            }
                        },
                    ]
                }
            }
            resize_config = [{"OneOf": {"transforms": [option_a, option_b]}}]
            resize_wrappers = AlbumentationsWrapper.from_config(resize_config)
            aug_wrappers = AlbumentationsWrapper.from_config(resolved_aug_config)
            return Compose([*resize_wrappers, *aug_wrappers, to_image, to_float, normalize])

        if image_set in ("val", "test", "val_speed"):
            resize_wrappers = AlbumentationsWrapper.from_config(
                [{"Resize": {"height": res_h, "width": res_w}}]
            )
            return Compose([*resize_wrappers, to_image, to_float, normalize])

        raise ValueError(f"unknown {image_set}")

    return _patched
```

#### 3.3 Применить monkey-patch через context manager

```python
@contextmanager
def _rect_resolution_patch(res_h: int, res_w: int):
    """Context manager: подменяет rfdetr transform builders на прямоугольные."""
    import rfdetr.datasets.coco as coco_mod

    orig_square = coco_mod.make_coco_transforms_square_div_64
    orig_normal = coco_mod.make_coco_transforms
    patched = _make_rect_transforms(res_h, res_w)

    coco_mod.make_coco_transforms_square_div_64 = patched
    coco_mod.make_coco_transforms = patched
    try:
        yield
    finally:
        coco_mod.make_coco_transforms_square_div_64 = orig_square
        coco_mod.make_coco_transforms = orig_normal
```

#### 3.4 Вызов train с патчем

```python
def train(..., resolution: int | tuple[int, int] | None = None, ...):
    model_cls = _get_model_class(variant)
    model = model_cls()

    # ...build all_params...

    if isinstance(resolution, tuple):
        res_h, res_w = resolution
        all_params["resolution"] = max(res_h, res_w)
        all_params["multi_scale"] = False  # rect не поддерживает multi_scale

        # Backbone shape для export() (inference после тренировки)
        model.model.model.backbone[0].encoder.shape = (res_h, res_w)

        with _rect_resolution_patch(res_h, res_w):
            model.train(**kwargs)
    else:
        if resolution is not None:
            all_params["resolution"] = resolution
        model.train(**kwargs)
```

#### 3.5 Обновить сигнатуру `train()`

Изменить `resolution: int | None = None` → `resolution: int | tuple[int, int] | None = None`.

### Шаг 4: `rfdetr_tooling/predict.py` — прямоугольный inference

`predict.py` вызывает `model.predict(pil_images, threshold=...)` (`predict.py:603`).

`model.predict()` внутри делает `F.resize(img, (self.model.resolution, self.model.resolution))` — **квадрат**.

**Два варианта**:

#### 4A: Monkey-patch `RFDETR.predict()` (хрупкий)

Переопределить `model.predict` на нашу версию. Проблема: метод большой (100+ строк), дублировать нежелательно.

#### 4B: Monkey-patch `model.model.resolution` + resize logic (точечный)

`model.model.resolution` — скаляр `int`. Нельзя сделать его кортежем без сломания `detr.py:369`.

#### 4C: Monkey-patch только resize строку (рекомендуется)

Подменить `torchvision.transforms.functional.resize` в scope `rfdetr.detr`:

```python
@contextmanager
def _rect_inference_patch(res_h: int, res_w: int):
    """Подменяет F.resize в rfdetr.detr для прямоугольного inference."""
    import rfdetr.detr as detr_mod
    import torchvision.transforms.functional as orig_F

    orig_resize = detr_mod.F.resize

    def patched_resize(img, size, *args, **kwargs):
        # Если вызвано с (resolution, resolution) — подменяем на (res_h, res_w)
        if isinstance(size, (list, tuple)) and len(size) == 2 and size[0] == size[1]:
            size = (res_h, res_w)
        return orig_resize(img, size, *args, **kwargs)

    detr_mod.F.resize = patched_resize
    try:
        yield
    finally:
        detr_mod.F.resize = orig_resize
```

Проблема: `detr_mod.F` — это сам модуль `torchvision.transforms.functional`, подмена `F.resize` подменит его глобально. Лучше подменить на уровне `detr_mod`:

```python
# rfdetr/detr.py импортирует: import torchvision.transforms.functional as F
# и использует: F.resize(img_tensor, (self.model.resolution, self.model.resolution))
```

Можно подменить весь `F` в namespace `detr_mod`:

```python
import types

@contextmanager
def _rect_inference_patch(res_h: int, res_w: int):
    import rfdetr.detr as detr_mod
    orig_F = detr_mod.F

    # Создаём обёртку модуля с подменённым resize
    class PatchedF:
        def __getattr__(self, name):
            if name == "resize":
                def patched_resize(img, size, *a, **kw):
                    if isinstance(size, (list, tuple)) and len(size) == 2 and size[0] == size[1]:
                        size = [res_h, res_w]
                    return orig_F.resize(img, size, *a, **kw)
                return patched_resize
            return getattr(orig_F, name)

    detr_mod.F = PatchedF()
    try:
        yield
    finally:
        detr_mod.F = orig_F
```

Но это хрупко — `size[0] == size[1]` ловит только квадратные вызовы. Если rfdetr вызовет resize с квадратным кропом для другой цели — тоже подменится.

#### 4D: Переписать inference loop в predict.py (рекомендуется — самый чистый)

Вместо вызова `model.predict()` — делать свой inference loop:

```python
def _predict_batch_rect(
    model: RFDETR,
    images: list[Image.Image],
    threshold: float,
    res_h: int,
    res_w: int,
) -> list[sv.Detections]:
    """Inference с прямоугольным resize."""
    import torch
    import torchvision.transforms.functional as F

    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]

    orig_sizes = []
    processed = []
    for img in images:
        t = F.to_tensor(img)
        orig_sizes.append((t.shape[1], t.shape[2]))
        t = t.to(model.model.device)
        t = F.normalize(t, means, stds)
        t = F.resize(t, (res_h, res_w))  # прямоугольный resize
        processed.append(t)

    batch = torch.stack(processed)
    with torch.no_grad():
        model.model.model.eval()
        predictions = model.model.model(batch)
        target_sizes = torch.tensor(orig_sizes, device=model.model.device)
        results = model.model.postprocess(predictions, target_sizes=target_sizes)

    detections = []
    for result in results:
        scores = result["scores"]
        labels = result["labels"]
        boxes = result["boxes"]
        keep = scores > threshold
        det = sv.Detections(
            xyxy=boxes[keep].float().cpu().numpy(),
            confidence=scores[keep].float().cpu().numpy(),
            class_id=labels[keep].cpu().numpy(),
        )
        detections.append(det)
    return detections
```

Потом в `predict()`:

```python
if isinstance(resolution, tuple):
    res_h, res_w = resolution
    for batch_entries in _chunked(images, batch_size):
        pil_images = [Image.open(e.path).convert("RGB") for e in batch_entries]
        det_list = _predict_batch_rect(model, pil_images, conf_threshold, res_h, res_w)
        # ...
else:
    # Существующий код через model.predict()
    ...
```

**Рекомендация**: вариант 4D — самый чистый, полностью контролируемый, не зависит от internal API rfdetr.predict().

### Шаг 5: `rfdetr_tooling/val.py` — прямоугольный inference

`val.py` тоже вызывает `model.predict()` (`val.py:186`). Аналогично predict.py — либо monkey-patch, либо свой inference loop.

**Решение**: переиспользовать `_predict_batch_rect` из шага 4D. Вынести в общий модуль `rfdetr_tooling/_inference.py`.

```python
# val.py
if isinstance(resolution, tuple):
    res_h, res_w = resolution
    for img_path in image_paths:
        image = Image.open(img_path).convert("RGB")
        [detections] = _predict_batch_rect(model, [image], threshold, res_h, res_w)
        # ...
else:
    # Существующий код
    detections = model.predict(image, threshold=threshold)
```

### Шаг 6: `rfdetr_tooling/_inference.py` — общий модуль

Новый файл. Содержит:

1. `_predict_batch_rect(model, images, threshold, res_h, res_w) -> list[sv.Detections]`
2. `_rect_resolution_patch(res_h, res_w)` — context manager для monkey-patch transforms
3. `_make_rect_transforms(res_h, res_w)` — factory для patched transform builder

Используется из `train.py`, `predict.py`, `val.py`.

### Шаг 7: Обновить `_cmd_cfg` в cli.py

`_cmd_cfg` генерирует YAML конфиг. Если `resolution` — кортеж, нужно сериализовать как `"960x608"` (или `[608, 960]`).

YAML-dump для tuple:
```python
if isinstance(data["resolution"], tuple):
    h, w = data["resolution"]
    data["resolution"] = f"{w}x{h}"  # обратно в WxH формат для человека
```

### Шаг 8: Mypy — типизация

Обновить type hints:
- `train()`: `resolution: int | tuple[int, int] | None`
- `predict()`: аналогично
- `val()`: аналогично
- Новые функции в `_inference.py`

### Шаг 9: Тесты

#### CLI smoke tests

```bash
# Парсинг прямоугольного resolution
rfdetr-tool cfg variant=nano resolution=960x608
# → YAML должен содержать resolution: 960x608

# Ошибка — не кратно 32
rfdetr-tool train data=x resolution=960x607
# → "resolution должен быть кратен 32, получено 607", exit 1

# Ошибка — невалидный формат
rfdetr-tool train data=x resolution=abcxdef
# → pydantic error, exit 1

# Обратная совместимость
rfdetr-tool train data=x resolution=960
# → работает как раньше (квадрат)
```

#### Функциональные тесты (требуют GPU + датасет)

```bash
# Тренировка 1 эпоха с прямоугольным resolution
rfdetr-tool train \
  data=./test_dataset \
  variant=nano \
  epochs=1 \
  batch_size=2 \
  resolution=960x608 \
  output_dir=output_rect

# Инференс с прямоугольным resolution
rfdetr-tool predict \
  source=./test_dataset/valid \
  weights=output_rect/checkpoint_best_ema.pth \
  variant=nano \
  resolution=960x608

# Валидация
rfdetr-tool val \
  data=./test_dataset \
  weights=output_rect/checkpoint_best_ema.pth \
  variant=nano \
  resolution=960x608
```

#### Юнит-тесты (без GPU)

```python
# test_config.py
def test_resolution_parse_rect():
    cfg = TrainConfig(data="x", resolution="960x608")
    assert cfg.resolution == (608, 960)  # (H, W)

def test_resolution_parse_int():
    cfg = TrainConfig(data="x", resolution=960)
    assert cfg.resolution == 960

def test_resolution_rect_not_div_32():
    with pytest.raises(ValidationError):
        TrainConfig(data="x", resolution="960x607")

def test_resolution_yaml_list():
    cfg = TrainConfig(data="x", resolution=[608, 960])
    assert cfg.resolution == (608, 960)
```

---

## Ограничения и компромиссы

### Multi-scale отключается для rect

При `resolution=(H, W)` автоматически устанавливаем `multi_scale=False`. Причина: `compute_multi_scale_scales()` работает только со скаляром. Для поддержки rect multi-scale нужно генерировать пары `(h_i, w_i)` с сохранением aspect ratio и кратности block_size — отдельная задача.

### Формат WxH vs HxW

- **CLI input**: `WxH` — стандарт для разрешений (1920x1080 = ширина×высота)
- **Internal**: `(H, W)` — PyTorch convention
- **YAML**: `WxH` строка для читаемости, или `[H, W]` список

### Export / ONNX

`model.export()` в `dinov2.py:140-197` pre-bake PE используя `self.shape`. Monkey-patch `backbone[0].encoder.shape = (res_h, res_w)` обеспечит правильный export. Но ONNX export не тестировался с rect — потенциальные проблемы в `forward_export`.

### Checkpoint совместимость

`args.resolution` сохраняется в checkpoint как скаляр. При загрузке rect-чекпоинта пользователь должен повторно указать `resolution=WxH` — оно не восстанавливается автоматически.

---

## Порядок реализации

| # | Шаг | Файлы | Зависимости |
|---|------|-------|-------------|
| 1 | `Resolution` type + validators | `config.py` | — |
| 2 | CLI `_coerce_value` для tuple | `cli.py` | 1 |
| 3 | `_inference.py` — rect predict + transform patch | `_inference.py` (новый) | — |
| 4 | `train.py` — rect resolution support | `train.py` | 1, 3 |
| 5 | `predict.py` — rect inference | `predict.py` | 1, 3 |
| 6 | `val.py` — rect inference | `val.py` | 1, 3 |
| 7 | `_cmd_cfg` — YAML сериализация | `cli.py` | 1 |
| 8 | Mypy strict | все | 1-7 |
| 9 | CLI smoke tests | — | 1-7 |
| 10 | Ruff + pre-commit | — | 8 |

---

## Граф изменений по файлам

```
rfdetr_tooling/
  config.py         — Resolution type alias, validators в 3 классах
  cli.py            — _coerce_value tuple detection, _cmd_cfg rect serialization
  _inference.py     — NEW: _predict_batch_rect, _rect_resolution_patch, _make_rect_transforms
  train.py          — resolution: tuple support, monkey-patch через context manager
  predict.py        — branch: rect → _predict_batch_rect, else → model.predict()
  val.py            — branch: rect → _predict_batch_rect, else → model.predict()
```

Всего: ~200-300 строк нового кода, ~50 строк изменений в существующих файлах.
