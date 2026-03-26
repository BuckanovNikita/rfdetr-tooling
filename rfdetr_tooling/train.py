"""Thin training wrapper around the rfdetr API."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from loguru import logger

_VARIANT_MAP: dict[str, str] = {
    "nano": "RFDETRNano",
    "small": "RFDETRSmall",
    "base": "RFDETRBase",
    "medium": "RFDETRMedium",
    "large": "RFDETRLarge",
}

# Поля, которые НЕ передаются в rfdetr model.train()
_EXCLUDED_FIELDS = {
    "variant",
    "gradient_checkpointing",
    "seed",
    "gpus",
    "clearml",
    "sync_bn",
    "resize_mode",
}


def _get_model_class(variant: str) -> type:
    """Import and return the rfdetr model class for *variant*."""
    import rfdetr  # noqa: PLC0415

    class_name = _VARIANT_MAP.get(variant)
    if class_name is None:
        msg = f"Unknown variant {variant!r}, choose from: {sorted(_VARIANT_MAP)}"
        raise ValueError(msg)
    return getattr(rfdetr, class_name)  # type: ignore[no-any-return]


def _upload_clearml_artifacts(output_dir: str) -> None:
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

    output_path = Path(output_dir)

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


def train(  # noqa: PLR0913, C901
    data: str,
    *,
    variant: Literal["nano", "small", "base", "medium", "large"] = "base",
    epochs: int = 100,
    batch_size: int = 4,
    lr: float = 1e-4,
    lr_encoder: float = 1.5e-4,
    lr_drop: int = 100,
    weight_decay: float = 1e-4,
    grad_accum_steps: int = 4,
    warmup_epochs: float = 0.0,
    use_ema: bool = True,
    ema_decay: float = 0.993,
    ema_tau: int = 100,
    early_stopping: bool = False,
    early_stopping_patience: int = 10,
    early_stopping_min_delta: float = 0.001,
    output_dir: str = "output",
    device: Literal["auto", "cpu", "cuda", "mps"] = "auto",
    resume: str | None = None,
    checkpoint_interval: int = 10,
    tensorboard: bool = True,
    wandb: bool = False,
    mlflow: bool = False,
    clearml: bool = False,
    project: str | None = None,
    run: str | None = None,
    gpus: int = 1,  # noqa: ARG001
    sync_bn: bool = True,  # noqa: ARG001
    gradient_checkpointing: bool = False,  # noqa: ARG001
    drop_path: float = 0.0,
    seed: int | None = None,  # noqa: ARG001
    num_workers: int = 2,
    multi_scale: bool = True,
    resolution: int | tuple[int, int] | None = None,
    resize_mode: str = "auto",
    progress_bar: bool = False,
    **model_extra: Any,  # noqa: ANN401
) -> None:
    """Тренировка RF-DETR.

    Args:
        data: Путь к директории датасета (COCO или YOLO формат).
        variant: Вариант архитектуры RF-DETR.
        epochs: Количество эпох.
        batch_size: Размер батча.
        lr: Learning rate для декодера.
        lr_encoder: Learning rate для энкодера.
        lr_drop: Эпоха для снижения LR.
        weight_decay: Weight decay.
        grad_accum_steps: Шаги градиентной аккумуляции.
        warmup_epochs: Количество warmup-эпох.
        use_ema: Использовать EMA.
        ema_decay: EMA decay.
        ema_tau: EMA tau.
        early_stopping: Включить early stopping.
        early_stopping_patience: Терпение early stopping.
        early_stopping_min_delta: Минимальный delta для early stopping.
        output_dir: Директория для результатов.
        device: Устройство ("auto", "cpu", "cuda", "mps").
        resume: Путь к чекпоинту для продолжения тренировки.
        checkpoint_interval: Интервал сохранения чекпоинтов (эпохи).
        tensorboard: Логирование в TensorBoard.
        wandb: Логирование в W&B.
        mlflow: Логирование в MLflow.
        clearml: Логирование в ClearML + загрузка артефактов.
        project: Имя проекта для логгеров.
        run: Имя запуска для логгеров.
        gpus: Количество GPU для DDP.
        sync_bn: Синхронизировать BatchNorm при DDP.
        gradient_checkpointing: Gradient checkpointing (не передаётся в rfdetr).
        drop_path: Drop path rate.
        seed: Random seed.
        num_workers: Количество DataLoader workers.
        multi_scale: Multi-scale аугментация.
        resolution: Разрешение входа модели.
        resize_mode: Режим resize ("auto", "letterbox", "true").
        progress_bar: Показывать progress bar.
        **model_extra: Дополнительные kwargs для rfdetr model.train().

    """
    # Определяем rect resolution
    rect_resolution: tuple[int, int] | None = None
    scalar_resolution: int | None = None
    if isinstance(resolution, tuple):
        rect_resolution = resolution
        scalar_resolution = max(resolution)
    elif isinstance(resolution, int):
        scalar_resolution = resolution

    model_cls = _get_model_class(variant)
    model = model_cls()

    # Собираем kwargs для rfdetr
    all_params: dict[str, Any] = {
        "dataset_dir": str(Path(data).resolve()),
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "lr_encoder": lr_encoder,
        "lr_drop": lr_drop,
        "weight_decay": weight_decay,
        "grad_accum_steps": grad_accum_steps,
        "warmup_epochs": warmup_epochs,
        "use_ema": use_ema,
        "ema_decay": ema_decay,
        "ema_tau": ema_tau,
        "early_stopping": early_stopping,
        "early_stopping_patience": early_stopping_patience,
        "early_stopping_min_delta": early_stopping_min_delta,
        "output_dir": str(Path(output_dir).resolve()),
        "checkpoint_interval": checkpoint_interval,
        "tensorboard": tensorboard,
        "wandb": wandb,
        "mlflow": mlflow,
        "project": project,
        "run": run,
        "drop_path": drop_path,
        "num_workers": num_workers,
        "multi_scale": multi_scale,
        "resolution": scalar_resolution,
        "progress_bar": progress_bar,
        "resume": resume,
        **model_extra,
    }

    # Rect: отключаем multi_scale, передаём max(H,W) как скалярный resolution
    if rect_resolution is not None:
        all_params["multi_scale"] = False

    # Фильтруем None и device=auto
    kwargs: dict[str, Any] = {}
    for key, value in all_params.items():
        if value is None:
            continue
        kwargs[key] = value

    if device != "auto":
        kwargs["device"] = device

    res_str: str
    if rect_resolution is not None:
        res_str = f"{rect_resolution[1]}x{rect_resolution[0]}"
    elif scalar_resolution is not None:
        res_str = str(scalar_resolution)
    else:
        res_str = "default"
    logger.info(
        f"Тренировка RF-DETR: variant={variant}, "
        f"dataset={kwargs['dataset_dir']}, epochs={epochs}, "
        f"batch_size={batch_size}, resolution={res_str}"
    )

    if rect_resolution is not None:
        from rfdetr_tooling._inference import rect_resolution_patch  # noqa: PLC0415

        use_letterbox = resize_mode != "true"
        res_h, res_w = rect_resolution
        with rect_resolution_patch(res_h, res_w, letterbox=use_letterbox):
            model.train(**kwargs)
    else:
        model.train(**kwargs)

    logger.info(f"Тренировка завершена. Результаты в {kwargs['dataset_dir']}")

    if clearml:
        _upload_clearml_artifacts(output_dir)
