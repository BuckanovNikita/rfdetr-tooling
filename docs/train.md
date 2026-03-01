# Training RF-DETR Models

## Quick start

```python
from rfdetr_tooling.train import train

train("path/to/dataset", epochs=50, batch_size=8)
```

This trains an RF-DETR Base model for 50 epochs. The dataset directory must be in COCO or YOLO format — rfdetr auto-detects which one.

## `train()` function

```python
def train(
    dataset_dir: str | Path,
    *,
    variant: str = "base",
    epochs: int = 100,
    batch_size: int = 4,
    output_dir: str | Path = "output",
    **kwargs,  # passed through to rfdetr TrainConfig
) -> None
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dataset_dir` | `str \| Path` | *required* | Path to dataset in COCO or YOLO format |
| `variant` | `str` | `"base"` | Model variant (see table below) |
| `epochs` | `int` | `100` | Number of training epochs |
| `batch_size` | `int` | `4` | Batch size per step |
| `output_dir` | `str \| Path` | `"output"` | Directory for checkpoints and logs |

All additional keyword arguments are forwarded to rfdetr's `TrainConfig`.

## Model variants

| Variant | Class | Resolution | Encoder |
|---------|-------|-----------|---------|
| `"nano"` | `RFDETRNano` | 384 | DINOv2-S (windowed) |
| `"small"` | `RFDETRSmall` | 512 | DINOv2-S (windowed) |
| `"base"` | `RFDETRBase` | 560 | DINOv2-S (windowed) |
| `"medium"` | `RFDETRMedium` | 576 | DINOv2-S (windowed) |
| `"large"` | `RFDETRLarge` | 704 | DINOv2-S (windowed) |

Smaller variants train faster and use less VRAM. Larger variants are more accurate.

## Dataset format

rfdetr auto-detects COCO vs YOLO format based on directory structure.

### COCO format

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

The JSON file follows the COCO annotation format with `bbox: [x, y, width, height]` (top-left corner + size, in pixels).

### YOLO format

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

Label files use YOLO normalized format: `class_id xc yc w h` (center + size, values in [0, 1]).

`data.yaml` specifies class names and paths:

```yaml
train: train/images
val: valid/images
nc: 3
names: ["cat", "dog", "bird"]
```

## rfdetr pass-through parameters

These are forwarded via `**kwargs` to rfdetr's `TrainConfig`.

### Learning rate

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `lr` | `float` | `1e-4` | Decoder learning rate |
| `lr_encoder` | `float` | `1.5e-4` | Encoder (backbone) learning rate |
| `lr_drop` | `int` | `100` | Epoch to drop learning rate |
| `lr_vit_layer_decay` | `float` | `0.8` | Layer-wise LR decay for ViT encoder |
| `lr_component_decay` | `float` | `0.7` | Component-wise LR decay |

### Optimization

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `weight_decay` | `float` | `1e-4` | Weight decay |
| `grad_accum_steps` | `int` | `4` | Gradient accumulation steps |
| `warmup_epochs` | `float` | `0.0` | Warmup epochs (0 = disabled) |

### EMA

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_ema` | `bool` | `True` | Use exponential moving average |
| `ema_decay` | `float` | `0.993` | EMA decay rate |
| `ema_tau` | `int` | `100` | EMA time constant |

### Data

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_workers` | `int` | `2` | Dataloader workers |
| `multi_scale` | `bool` | `True` | Multi-scale training augmentation |
| `resolution` | `int` | *(per variant)* | Input resolution (overrides variant default) |

### Checkpointing

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `resume` | `str \| None` | `None` | Path to checkpoint to resume from |
| `checkpoint_interval` | `int` | `10` | Save checkpoint every N epochs |

### Early stopping

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `early_stopping` | `bool` | `False` | Enable early stopping |
| `early_stopping_patience` | `int` | `10` | Epochs without improvement before stopping |
| `early_stopping_min_delta` | `float` | `0.001` | Minimum improvement threshold |

### Logging

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tensorboard` | `bool` | `True` | Log to TensorBoard |
| `wandb` | `bool` | `False` | Log to Weights & Biases |
| `mlflow` | `bool` | `False` | Log to MLflow |
| `clearml` | `bool` | `False` | Log to ClearML |
| `project` | `str \| None` | `None` | Project name for experiment trackers |
| `run` | `str \| None` | `None` | Run name for experiment trackers |

### Device & precision

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `device` | `str` | `"auto"` | Device: `"auto"`, `"cpu"`, `"cuda"`, or `"mps"` |
| `amp` | `bool` | `True` | Automatic mixed precision |

### Advanced

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `gradient_checkpointing` | `bool` | `False` | Trade compute for memory savings |
| `drop_path` | `float` | `0.0` | Drop path rate for regularization |
| `group_detr` | `int` | `13` | Number of DETR groups |

## Full example

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
    early_stopping=True,
    early_stopping_patience=15,
    wandb=True,
    project="my-detection",
    run="base-80ep",
)
```
