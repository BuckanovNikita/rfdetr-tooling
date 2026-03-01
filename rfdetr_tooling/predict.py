"""Инференс RF-DETR на изображениях."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import supervision as sv
from loguru import logger
from PIL import Image

from rfdetr_tooling.train import _get_model_class

if TYPE_CHECKING:
    from rfdetr_tooling.config import PredictConfig

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _collect_images(source: str) -> list[Path]:
    """Собирает пути к изображениям из файла или директории."""
    path = Path(source)
    if path.is_file():
        return [path]
    if path.is_dir():
        images = [
            p for p in sorted(path.iterdir()) if p.suffix.lower() in _IMAGE_EXTENSIONS
        ]
        if not images:
            logger.warning(f"Изображения не найдены в {source}")
        return images
    logger.error(f"Источник не найден: {source}")
    return []


def predict_from_config(config: PredictConfig) -> None:
    """Инференс RF-DETR из pydantic-конфига."""
    model_cls = _get_model_class(config.variant)
    model = model_cls(pretrain_weights=config.weights)

    images = _collect_images(config.source)
    if not images:
        return

    output_dir = Path(config.output_dir)
    if config.save:
        output_dir.mkdir(parents=True, exist_ok=True)

    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    logger.info(
        f"Инференс: {len(images)} изображений, "
        f"variant={config.variant}, threshold={config.threshold}"
    )

    for img_path in images:
        image = Image.open(img_path).convert("RGB")
        detections: sv.Detections = model.predict(image, threshold=config.threshold)

        n_dets = len(detections)
        logger.info(f"{img_path.name}: {n_dets} детекций")

        if config.save:
            import numpy as np  # noqa: PLC0415

            frame = np.array(image)
            labels = []
            if detections.class_id is not None and detections.confidence is not None:
                labels = [
                    f"cls_{class_id} {conf:.2f}"
                    for class_id, conf in zip(
                        detections.class_id, detections.confidence, strict=True
                    )
                ]
            annotated = box_annotator.annotate(frame.copy(), detections)
            annotated = label_annotator.annotate(annotated, detections, labels=labels)

            out_path = output_dir / img_path.name
            Image.fromarray(annotated).save(out_path)
            logger.info(f"Сохранено: {out_path}")

    logger.info("Инференс завершён")
