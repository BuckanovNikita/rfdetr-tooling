"""Валидация RF-DETR на val-сете с расчётом mAP."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import supervision as sv
from loguru import logger
from PIL import Image
from supervision.metrics import MeanAveragePrecision

from rfdetr_tooling.train import _get_model_class

if TYPE_CHECKING:
    from rfdetr_tooling.config import ValConfig

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _find_val_dir(data: str) -> Path | None:
    """Ищет директорию валидации (valid/ или val/) в датасете."""
    base = Path(data)
    for name in ("valid", "val"):
        candidate = base / name
        if candidate.is_dir():
            return candidate
    return None


def _load_coco_annotations(
    val_dir: Path,
) -> tuple[dict[str, sv.Detections], list[Path]]:
    """Загружает GT-аннотации из COCO JSON.

    Возвращает dict {filename: Detections} и список путей к изображениям.
    """
    ann_file = val_dir / "_annotations.coco.json"
    if not ann_file.exists():
        logger.error(f"Файл аннотаций не найден: {ann_file}")
        return {}, []

    with ann_file.open() as f:
        coco = json.load(f)

    # Маппинг id -> filename
    id_to_info: dict[int, dict[str, str | int]] = {}
    for img in coco["images"]:
        id_to_info[img["id"]] = {
            "file_name": img["file_name"],
            "width": img["width"],
            "height": img["height"],
        }

    # Маппинг category_id -> contiguous index
    cat_ids = sorted({c["id"] for c in coco["categories"]})
    cat_id_to_idx = {cid: idx for idx, cid in enumerate(cat_ids)}

    # Группировка аннотаций по image_id
    img_anns: dict[int, list[dict[str, Any]]] = {}
    for ann in coco["annotations"]:
        img_anns.setdefault(ann["image_id"], []).append(ann)

    gt_map: dict[str, sv.Detections] = {}
    image_paths: list[Path] = []

    for img_id, info in id_to_info.items():
        fname = str(info["file_name"])
        img_path = val_dir / fname
        if not img_path.exists():
            continue
        image_paths.append(img_path)

        anns = img_anns.get(img_id, [])
        if not anns:
            gt_map[fname] = sv.Detections.empty()
            continue

        bboxes = []
        class_ids = []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            bboxes.append([x, y, x + w, y + h])
            class_ids.append(cat_id_to_idx[ann["category_id"]])

        gt_map[fname] = sv.Detections(
            xyxy=np.array(bboxes, dtype=np.float32),
            class_id=np.array(class_ids, dtype=int),
        )

    return gt_map, image_paths


def val_from_config(config: ValConfig) -> None:
    """Валидация RF-DETR из pydantic-конфига."""
    val_dir = _find_val_dir(config.data)
    if val_dir is None:
        logger.error(
            f"Директория валидации (valid/ или val/) не найдена в {config.data}"
        )
        return

    gt_map, image_paths = _load_coco_annotations(val_dir)
    if not image_paths:
        logger.error("Нет изображений для валидации")
        return

    model_cls = _get_model_class(config.variant)
    model = model_cls(pretrain_weights=config.weights)

    logger.info(
        f"Валидация: {len(image_paths)} изображений, "
        f"variant={config.variant}, threshold={config.threshold}"
    )

    all_predictions: list[sv.Detections] = []
    all_targets: list[sv.Detections] = []

    for img_path in image_paths:
        image = Image.open(img_path).convert("RGB")
        detections: sv.Detections = model.predict(image, threshold=config.threshold)

        fname = img_path.name
        gt = gt_map.get(fname, sv.Detections.empty())

        all_predictions.append(detections)
        all_targets.append(gt)

    # Расчёт mAP
    metric = MeanAveragePrecision()
    result = metric.update(all_predictions, all_targets).compute()

    logger.info(f"mAP@50:95 = {result.map50_95:.4f}")
    logger.info(f"mAP@50    = {result.map50:.4f}")
    logger.info(f"mAP@75    = {result.map75:.4f}")

    logger.info("Валидация завершена")
