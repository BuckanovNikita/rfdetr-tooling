"""Thin training wrapper around the rfdetr API."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from loguru import logger

_VARIANT_MAP: dict[str, str] = {
    "nano": "RFDETRNano",
    "small": "RFDETRSmall",
    "base": "RFDETRBase",
    "medium": "RFDETRMedium",
    "large": "RFDETRLarge",
}


def _get_model_class(variant: str) -> type:
    """Import and return the rfdetr model class for *variant*."""
    import rfdetr  # noqa: PLC0415

    class_name = _VARIANT_MAP.get(variant)
    if class_name is None:
        msg = f"Unknown variant {variant!r}, choose from: {sorted(_VARIANT_MAP)}"
        raise ValueError(msg)
    return getattr(rfdetr, class_name)  # type: ignore[no-any-return]


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
