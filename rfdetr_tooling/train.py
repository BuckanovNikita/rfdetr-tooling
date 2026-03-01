"""Thin training wrapper around the rfdetr API."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    from rfdetr_tooling.config import TrainConfig

_VARIANT_MAP: dict[str, str] = {
    "nano": "RFDETRNano",
    "small": "RFDETRSmall",
    "base": "RFDETRBase",
    "medium": "RFDETRMedium",
    "large": "RFDETRLarge",
}

# Поля TrainConfig, которые НЕ передаются в rfdetr model.train()
_EXCLUDED_FIELDS = {"variant", "gradient_checkpointing", "seed"}


def _get_model_class(variant: str) -> type:
    """Import and return the rfdetr model class for *variant*."""
    import rfdetr  # noqa: PLC0415

    class_name = _VARIANT_MAP.get(variant)
    if class_name is None:
        msg = f"Unknown variant {variant!r}, choose from: {sorted(_VARIANT_MAP)}"
        raise ValueError(msg)
    return getattr(rfdetr, class_name)  # type: ignore[no-any-return]


def _upload_clearml_artifacts(config: TrainConfig) -> None:
    """Загрузка артефактов в ClearML после тренировки."""
    try:
        from clearml import OutputModel, Task  # noqa: PLC0415
    except ImportError:
        logger.warning("clearml не установлен, артефакты не загружены")
        return

    task = Task.current_task()
    if not task:
        logger.warning("ClearML Task не найден, артефакты не загружены")
        return

    output_path = Path(config.output_dir)

    # Загрузка лучшего чекпоинта
    best_ckpt = output_path / "checkpoint_best_ema.pth"
    if best_ckpt.exists():
        output_model = OutputModel(task=task, framework="PyTorch")
        output_model.update_weights(str(best_ckpt))
        logger.info(f"ClearML: загружен чекпоинт {best_ckpt}")
    else:
        logger.warning(f"Чекпоинт {best_ckpt} не найден")

    # Загрузка графика метрик
    plot = output_path / "metrics_plot.png"
    if plot.exists():
        task.upload_artifact("metrics_plot", str(plot))
        logger.info("ClearML: загружен metrics_plot.png")


def train_from_config(config: TrainConfig) -> None:
    """Тренировка RF-DETR из pydantic-конфига."""
    model_cls = _get_model_class(config.variant)
    model = model_cls()

    # Собираем kwargs для rfdetr, фильтруя None и исключённые поля
    kwargs: dict[str, Any] = {}
    raw = config.model_dump()
    for key, value in raw.items():
        if key in _EXCLUDED_FIELDS:
            continue
        if key == "data":
            kwargs["dataset_dir"] = str(Path(value).resolve())
            continue
        if value is None:
            continue
        # "auto" не поддерживается torch.device(), rfdetr сам определит устройство
        if key == "device" and value == "auto":
            continue
        if key == "output_dir":
            kwargs[key] = str(Path(value).resolve())
            continue
        kwargs[key] = value

    logger.info(
        f"Тренировка RF-DETR: variant={config.variant}, "
        f"dataset={kwargs['dataset_dir']}, epochs={kwargs.get('epochs')}, "
        f"batch_size={kwargs.get('batch_size')}"
    )

    model.train(**kwargs)

    logger.info(f"Тренировка завершена. Результаты в {kwargs['dataset_dir']}")

    if config.clearml:
        _upload_clearml_artifacts(config)


def train(
    dataset_dir: str | Path,
    *,
    variant: str = "base",
    epochs: int = 100,
    batch_size: int = 4,
    output_dir: str | Path = "output",
    **kwargs: Any,  # noqa: ANN401
) -> None:
    """Train an RF-DETR model on a COCO or YOLO dataset."""
    model_cls = _get_model_class(variant)
    model = model_cls()

    dataset_dir = str(Path(dataset_dir).resolve())
    output_dir = str(Path(output_dir).resolve())

    logger.info(
        f"Starting RF-DETR training: variant={variant}, "
        f"dataset={dataset_dir}, epochs={epochs}, batch_size={batch_size}"
    )

    model.train(
        dataset_dir=dataset_dir,
        epochs=epochs,
        batch_size=batch_size,
        output_dir=output_dir,
        **kwargs,
    )

    logger.info(f"Training complete. Output saved to {output_dir}")
