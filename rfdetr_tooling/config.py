"""Pydantic configuration models for rfdetr-tooling CLI."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

Variant = Literal["nano", "small", "base", "medium", "large"]

VARIANT_RESOLUTION: dict[str, int] = {
    "nano": 384,
    "small": 512,
    "base": 560,
    "medium": 576,
    "large": 704,
}


class TrainConfig(BaseModel):
    """Конфигурация тренировки RF-DETR."""

    data: str
    variant: Variant = "base"

    # Обучение
    epochs: int = 100
    batch_size: int = 4
    lr: float = 1e-4
    lr_encoder: float = 1.5e-4
    lr_drop: int = 100
    weight_decay: float = 1e-4
    grad_accum_steps: int = 4
    warmup_epochs: float = 0.0

    # EMA
    use_ema: bool = True
    ema_decay: float = 0.993
    ema_tau: int = 100

    # Early stopping
    early_stopping: bool = False
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 0.001

    # Выходные данные
    output_dir: str = "output"
    device: Literal["auto", "cpu", "cuda", "mps"] = "auto"
    resume: str | None = None
    checkpoint_interval: int = 10

    # Логирование
    tensorboard: bool = True
    wandb: bool = False
    mlflow: bool = False
    clearml: bool = False
    project: str | None = None
    run: str | None = None

    # Прочее
    gradient_checkpointing: bool = False
    drop_path: float = 0.0
    seed: int | None = None
    num_workers: int = 2
    multi_scale: bool = True
    resolution: int | None = None
    progress_bar: bool = False


OutputFormat = Literal["yolo", "csv"]


class PredictConfig(BaseModel):
    """Конфигурация инференса RF-DETR."""

    source: str  # пути через запятую: папки, data.yaml, dataset.csv
    weights: str
    variant: Variant = "base"
    conf_threshold: float = 0.01
    nms_threshold: float = 0.25
    agnostic_nms: bool = False
    resolution: int | None = None
    batch_size: int = 4
    device: str = "auto"
    output_dir: str = "predict_output"
    format: OutputFormat = "yolo"
    visualize: bool = False
    check_image_sizes: bool = False


class ValConfig(BaseModel):
    """Конфигурация валидации RF-DETR."""

    data: str
    weights: str
    variant: Variant = "base"
    threshold: float = 0.5
    device: Literal["auto", "cpu", "cuda", "mps"] = "auto"
    batch_size: int = 4
