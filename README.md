# rfdetr-tooling

Инструменты для дообучения моделей [RF-DETR](https://github.com/roboflow/rf-detr) на пользовательских датасетах. Поддерживает форматы COCO и YOLO, а также CSV-определения датасетов (формат cveta2) с конвертацией.

## Установка

```bash
uv sync
```

## Использование

```python
from rfdetr_tooling.train import train

train("path/to/dataset", variant="base", epochs=50, batch_size=8)
```

Подробная документация по обучению: [docs/train.md](docs/train.md) — все параметры, варианты моделей и требования к формату датасетов.

### Инференс

```bash
rfdetr-tool predict source=./images weights=model.pth
rfdetr-tool predict source=./dataset weights=model.pth format=csv
```

Подробная документация по инференсу: [docs/predict.md](docs/predict.md) — все параметры, форматы входных данных, форматы выхода.
