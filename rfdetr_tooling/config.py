"""Pydantic configuration models for rfdetr-tooling CLI."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

Variant = Literal["nano", "small", "base", "medium", "large"]

Resolution = int | tuple[int, int]

ResizeMode = Literal["auto", "letterbox", "true"]

VARIANT_RESOLUTION: dict[str, int] = {
    "nano": 384,
    "small": 512,
    "base": 560,
    "medium": 576,
    "large": 704,
}


def _parse_resolution(v: Any) -> Any:  # noqa: ANN401
    """Парсинг resolution: строка '960x608' → (608, 960) (H, W), список → tuple."""
    if v is None:
        return v
    if isinstance(v, str):
        low = v.lower()
        if "x" in low:
            parts = low.split("x", 1)
            return (int(parts[1]), int(parts[0]))  # WxH → (H, W)
        return int(v)
    if isinstance(v, (list, tuple)) and len(v) == 2:
        return (int(v[0]), int(v[1]))
    return v


def _validate_resolution_div32(v: Resolution | None) -> Resolution | None:
    """Валидация: resolution > 0 и кратен 32 (для каждого измерения)."""
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

    # Распределённое обучение (DDP)
    gpus: int = Field(default=1, ge=1)
    sync_bn: bool = True

    # Прочее
    gradient_checkpointing: bool = False
    drop_path: float = 0.0
    seed: int | None = None
    num_workers: int = 2
    multi_scale: bool = True
    resolution: Resolution | None = Field(default=None)
    resize_mode: ResizeMode = "auto"
    progress_bar: bool = False

    @field_validator("resolution", mode="before")
    @classmethod
    def _parse_resolution(cls, v: Any) -> Any:  # noqa: ANN401
        return _parse_resolution(v)

    @field_validator("resolution")
    @classmethod
    def _resolution_divisible_by_32(cls, v: Resolution | None) -> Resolution | None:
        return _validate_resolution_div32(v)


OutputFormat = Literal["yolo", "csv"]


class PredictConfig(BaseModel):
    """Конфигурация инференса RF-DETR."""

    source: str  # пути через запятую: папки, data.yaml, dataset.csv
    weights: str
    variant: Variant = "base"
    conf_threshold: float = 0.01
    nms_threshold: float = 0.25
    agnostic_nms: bool = False
    resolution: Resolution | None = Field(default=None)
    resize_mode: ResizeMode = "auto"
    batch_size: int = 4
    device: str = "auto"
    output_dir: str = "predict_output"
    format: OutputFormat = "yolo"
    visualize: bool = False
    check_image_sizes: bool = False

    @field_validator("resolution", mode="before")
    @classmethod
    def _parse_resolution(cls, v: Any) -> Any:  # noqa: ANN401
        return _parse_resolution(v)

    @field_validator("resolution")
    @classmethod
    def _resolution_divisible_by_32(cls, v: Resolution | None) -> Resolution | None:
        return _validate_resolution_div32(v)


class ValConfig(BaseModel):
    """Конфигурация валидации RF-DETR."""

    data: str
    weights: str
    variant: Variant = "base"
    threshold: float = 0.5
    device: Literal["auto", "cpu", "cuda", "mps"] = "auto"
    batch_size: int = 4
    resolution: Resolution | None = Field(default=None)
    resize_mode: ResizeMode = "auto"

    @field_validator("resolution", mode="before")
    @classmethod
    def _parse_resolution(cls, v: Any) -> Any:  # noqa: ANN401
        return _parse_resolution(v)

    @field_validator("resolution")
    @classmethod
    def _resolution_divisible_by_32(cls, v: Resolution | None) -> Resolution | None:
        return _validate_resolution_div32(v)
