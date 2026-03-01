# rfdetr-tooling

Python tooling for finetuning [RF-DETR](https://github.com/roboflow/rf-detr) object detection models on custom datasets. Supports COCO and YOLO dataset formats, with CSV-based dataset definitions (cveta2 format) and conversion utilities.

## Installation

```bash
uv sync
```

## Usage

```python
from rfdetr_tooling.train import train

train("path/to/dataset", variant="base", epochs=50, batch_size=8)
```

See [docs/train.md](docs/train.md) for full training documentation including all parameters, model variants, and dataset format requirements.
