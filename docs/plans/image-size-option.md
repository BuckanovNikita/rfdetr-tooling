# План: настройка image size для тренировки

## Контекст

RF-DETR принимает параметр `resolution: int` — размер входного изображения. У каждого варианта модели есть дефолт (nano=384, small=512, base=560, medium=576, large=704). Параметр `resolution` уже существует в `TrainConfig` и `PredictConfig` как `int | None` и передаётся в `model.train()`.

## Исследование: поддержка не-квадратных изображений

**RF-DETR поддерживает не-квадратные изображения** — но только на уровне препроцессинга, а не как параметр resolution.

Как это работает внутри rfdetr:
- **Дефолтный режим** (`square_resize_div_64=False`): `SmallestMaxSize(resolution)` + `LongestMaxSize(1333)` — сохраняет aspect ratio, короткая сторона = resolution, длинная ≤ 1333. Батч дополняется нулями до максимального размера в батче.
- **Квадратный режим** (`square_resize_div_64=True`): resize до `(resolution, resolution)` — нужен для segmentation head.

**Параметр `resolution` в rfdetr — всегда один int** (не tuple). Он задаёт масштаб ресайза, но не финальную форму тензора. Не-квадратные изображения поддерживаются автоматически через aspect-ratio-preserving transforms.

Связь с `positional_encoding_size`:
- Каждый вариант имеет захардкоженный `positional_encoding_size` в конфиге rfdetr (например, base: 37, large: 44)
- Он рассчитывается как `resolution // patch_size` (patch_size обычно 16)
- При изменении resolution rfdetr **не пересчитывает** positional_encoding_size автоматически — он берётся из конфига модели
- Однако `model.train()` принимает `resolution` как override и rfdetr сам адаптирует positional embeddings через интерполяцию

## Текущее состояние

- `TrainConfig.resolution: int | None = None` — уже есть
- `PredictConfig.resolution: int | None = None` — уже есть
- `ValConfig` — **нет** поля resolution
- В `train.py` resolution передаётся в `model.train(**kwargs)` — работает
- В `cli.py` при генерации конфига (`cfg` команда) resolution заполняется из `VARIANT_RESOLUTION` если None

## Что нужно сделать

### Задача 1: Добавить resolution в ValConfig

`ValConfig` не имеет поля `resolution`, хотя predict и train имеют.

**Файл**: `rfdetr_tooling/config.py`
- Добавить `resolution: int | None = None` в `ValConfig`

**Файл**: `rfdetr_tooling/val.py`
- Проверить, передаётся ли resolution в модель при валидации. Если нет — добавить.

### Задача 2: Валидация resolution

Сейчас resolution принимает любой int. Нужна базовая валидация:

**Файл**: `rfdetr_tooling/config.py`
- Добавить pydantic validator на `resolution` в `TrainConfig`, `PredictConfig`, `ValConfig`:
  - `resolution` должен быть > 0
  - `resolution` должен быть кратен 32 (rfdetr multi-scale использует деление на 64, но 32 — безопасный минимум; rfdetr сам рассчитывает scales)
  - Использовать `Field(default=None, gt=0)` + кастомный validator для кратности

### Задача 3: Логирование resolution при старте

**Файл**: `rfdetr_tooling/train.py`
- Добавить resolution в лог-сообщение при старте тренировки (строка 198), чтобы пользователь видел какой resolution используется (или "default" если None)

## Что НЕ нужно делать

- Добавлять поддержку tuple `(width, height)` — rfdetr принимает только один int
- Менять дефолтные resolution — они уже корректны в `VARIANT_RESOLUTION`
- Добавлять параметр `square_resize_div_64` — это внутренний параметр rfdetr для segmentation, не нужен в tooling

## Проверка

```bash
uv run ruff check rfdetr_tooling/
uv run ruff format --check rfdetr_tooling/
uv run mypy rfdetr_tooling/
rfdetr-tool cfg variant=base  # resolution=560 в выводе
rfdetr-tool train data=x resolution=100  # ошибка: не кратно 32
rfdetr-tool train data=x resolution=-1   # ошибка: > 0
rfdetr-tool train data=x resolution=640  # ok (если бы датасет был)
```
