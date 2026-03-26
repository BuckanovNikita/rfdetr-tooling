# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**rfdetr-tooling** — Python tooling for finetuning RF-DETR object detection models on custom datasets (COCO and YOLO formats). Supports CSV-based dataset definitions (cveta2 format) with conversion to COCO/YOLO for training.

**Language**: Python 3.12+

## Development Commands

```bash
uv sync                    # install dependencies
uv run ruff format .       # format code
uv run ruff check .        # lint
uv run ruff check --fix .  # auto-fix lint
uv run mypy .              # type check
```

## Key Constraints

- **rfdetr API**: training uses `model.train(dataset_dir=..., epochs=..., batch_size=..., ...)` — rfdetr auto-detects COCO vs YOLO format
- **COCO format**: `train/_annotations.coco.json` + images in same dir; `valid/` same structure
- **YOLO format**: `data.yaml` + `train/images/`, `train/labels/`, `valid/images/`, `valid/labels/`
- **Variants**: `nano` (384), `small` (512), `base` (560), `medium` (576), `large` (704) — default resolutions in parentheses

## Architecture

- `config.py` — pydantic models: `TrainConfig`, `ValConfig`, `PredictConfig`
- `cli.py` — CLI entry point, argument parsing, YAML config generation
- `train.py` — training wrapper around rfdetr API
- `val.py` — validation with mAP calculation via supervision
- `predict.py` — batch inference with YOLO/CSV output, visualization
- `_inference.py` — rectangular resolution: letterbox/true resize, monkey-patch transforms for rfdetr
- `ddp.py` — DDP launch helper (torchrun relaunch for multi-GPU)
- `test_runner.py` — built-in smoke tests (`rfdetr-tool test`)

## CLI

Entry point: `rfdetr-tool` (defined in `rfdetr_tooling/cli.py`). Syntax: `rfdetr-tool <command> [key=value ...]`

Commands: `train`, `val`, `predict`, `cfg`, `test`

Config priority: CLI args > YAML (`cfg=path.yaml`) > pydantic defaults

### Resolution & resize_mode

- `resolution` accepts int (square) or `WxH` string (rect, e.g. `960x608` → internal tuple `(608, 960)` as `(H, W)`)
- `resize_mode`: `auto` (default, uses letterbox for rect), `letterbox` (explicit), `true` (stretch without AR preservation)
- Resolution must be > 0 and divisible by 32

### DDP (multi-GPU training)

- `gpus=N` (N > 1) triggers automatic relaunch via `torchrun --standalone --nproc_per_node=N`
- Incompatible with `device=cpu` or `device=mps`

## Validation Strategy

After making changes, run these checks in order:

```bash
# 1. Install entry point
uv sync

# 2. CLI smoke tests (automated)
rfdetr-tool test                        # all tests
rfdetr-tool test category=cli           # only CLI tests
rfdetr-tool test category=linter        # only ruff checks
rfdetr-tool test category=typecheck     # only mypy

# 3. Or manually:
rfdetr-tool                                          # help output, exit 0
rfdetr-tool cfg variant=base                         # generates YAML to stdout
rfdetr-tool cfg variant=large output=config.yaml     # writes YAML to file
rfdetr-tool train                                    # pydantic error "Field required", exit 1
rfdetr-tool train data=x unknown_param=y             # "Неизвестный параметр", exit 1
rfdetr-tool train data=x variant=invalid             # pydantic literal error, exit 1
rfdetr-tool train cfg=nonexistent.yaml data=x        # "Файл не найден", exit 1
rfdetr-tool badcommand                               # "Неизвестная команда", exit 1
rfdetr-tool train cfg=config.yaml data=./ds epochs=5 # YAML + CLI overrides work
rfdetr-tool predict                                    # pydantic error "Field required", exit 1
rfdetr-tool predict source=x weights=y format=invalid  # pydantic literal error, exit 1
rfdetr-tool predict source=x weights=y unknown=z       # "Неизвестный параметр", exit 1

# 4. Linters and type checking
uv run ruff check rfdetr_tooling/
uv run ruff format --check rfdetr_tooling/
uv run mypy rfdetr_tooling/
```

All CLI error cases must exit with code 1 and a loguru error message. All linter/mypy checks must pass with zero errors.

## Style

- Use `loguru` for logging (never `print`)
- Use `pydantic` for configs and data models
- Never use bare `except Exception` — catch specific types
- Follow cveta2 conventions: f-strings, explicit error handling with tracking of skipped items

## Dataset Formats

**CSV (cveta2)**: columns include `image_name`, `image_width`, `image_height`, `instance_label`, `bbox_x_tl`, `bbox_y_tl`, `bbox_x_br`, `bbox_y_br`, `split` (train/val/test), `image_path`

**Coordinate systems**:
- CSV/COCO pixel coords: `(x_tl, y_tl, x_br, y_br)` or `(x, y, w, h)`
- YOLO normalized: `(xc, yc, w, h)` in [0, 1]
- COCO annotations: `bbox: [x, y, width, height]` (top-left + size, pixels)
