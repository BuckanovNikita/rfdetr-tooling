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

- **PyTorch**: rfdetr requires `torch<=2.8.0` — do not upgrade beyond this
- **rfdetr API**: training uses `model.train(dataset_dir=..., epochs=..., batch_size=..., ...)` — rfdetr auto-detects COCO vs YOLO format
- **COCO format**: `train/_annotations.coco.json` + images in same dir; `valid/` same structure
- **YOLO format**: `data.yaml` + `train/images/`, `train/labels/`, `valid/images/`, `valid/labels/`

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
