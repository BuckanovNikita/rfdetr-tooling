"""Валидация RF-DETR на val-сете с расчётом mAP."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import numpy as np
import supervision as sv
from loguru import logger
from PIL import Image
from supervision.metrics import MeanAveragePrecision

from rfdetr_tooling.train import _get_model_class

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
) -> tuple[dict[str, sv.Detections], list[Path], dict[int, str]]:
    """Загружает GT-аннотации из COCO JSON.

    Возвращает (dict {filename: Detections}, список путей, dict {cat_id: name}).
    """
    ann_file = val_dir / "_annotations.coco.json"
    if not ann_file.exists():
        logger.error(f"Файл аннотаций не найден: {ann_file}")
        return {}, [], {}

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

    gt_cat_names: dict[int, str] = {c["id"]: c["name"] for c in coco["categories"]}

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
            class_ids.append(ann["category_id"])

        gt_map[fname] = sv.Detections(
            xyxy=np.array(bboxes, dtype=np.float32),
            class_id=np.array(class_ids, dtype=int),
        )

    return gt_map, image_paths, gt_cat_names


def _build_pred_to_gt_map(
    model_class_names: dict[int, str],
    gt_cat_names: dict[int, str],
) -> dict[int, int]:
    """Маппинг pred_class_id → gt_category_id через совпадение имён классов."""
    name_to_gt_id: dict[str, int] = {name: cid for cid, name in gt_cat_names.items()}
    return {
        pred_id: name_to_gt_id[name]
        for pred_id, name in model_class_names.items()
        if name in name_to_gt_id
    }


def _remap_class_ids(
    detections: sv.Detections,
    mapping: dict[int, int],
) -> sv.Detections:
    """Переводит class_id детекций по маппингу, убирая классы без соответствия."""
    if detections.class_id is None or len(detections) == 0:
        return detections

    new_ids = np.array([mapping.get(int(cid), -1) for cid in detections.class_id])
    keep = new_ids >= 0
    if not keep.all():
        detections = sv.Detections(
            xyxy=detections.xyxy[keep],
            confidence=(
                detections.confidence[keep]
                if detections.confidence is not None
                else None
            ),
            class_id=new_ids[keep],
        )
    else:
        detections.class_id = new_ids
    return detections


def val(  # noqa: PLR0913
    data: str,
    weights: str,
    *,
    variant: Literal["nano", "small", "base", "medium", "large"] = "base",
    threshold: float = 0.5,
    device: Literal["auto", "cpu", "cuda", "mps"] = "auto",
    batch_size: int = 4,  # noqa: ARG001
    resolution: int | tuple[int, int] | None = None,
    resize_mode: str = "auto",
    **model_extra: Any,  # noqa: ANN401
) -> None:
    """Валидация RF-DETR на val-сете с расчётом mAP.

    Args:
        data: Путь к директории датасета (с valid/ или val/ внутри).
        weights: Путь к файлу весов модели.
        variant: Вариант архитектуры RF-DETR.
        threshold: Порог уверенности для детекций.
        device: Устройство ("auto", "cpu", "cuda", "mps").
        batch_size: Размер батча (зарезервировано).
        resolution: Разрешение входа модели (None = по умолчанию).
        resize_mode: Режим resize ("auto", "letterbox", "true").
        **model_extra: Дополнительные kwargs для конструктора модели.

    """
    val_dir = _find_val_dir(data)
    if val_dir is None:
        logger.error(f"Директория валидации (valid/ или val/) не найдена в {data}")
        return

    gt_map, image_paths, gt_cat_names = _load_coco_annotations(val_dir)
    if not image_paths:
        logger.error("Нет изображений для валидации")
        return

    model_cls = _get_model_class(variant)
    model_kwargs: dict[str, Any] = {
        "pretrain_weights": weights,
        **model_extra,
    }
    if device != "auto":
        model_kwargs["device"] = device
    model = model_cls(**model_kwargs)
    rect_resolution: tuple[int, int] | None = None
    if isinstance(resolution, tuple):
        rect_resolution = resolution
    elif resolution is not None:
        model.model.resolution = resolution

    pred_to_gt = _build_pred_to_gt_map(model.class_names, gt_cat_names)

    logger.info(
        f"Валидация: {len(image_paths)} изображений, "
        f"variant={variant}, threshold={threshold}"
    )

    all_predictions: list[sv.Detections] = []
    all_targets: list[sv.Detections] = []

    for img_path in image_paths:
        image = Image.open(img_path).convert("RGB")
        if rect_resolution is not None:
            from rfdetr_tooling._inference import predict_batch_rect  # noqa: PLC0415

            use_letterbox = resize_mode != "true"
            [detections] = predict_batch_rect(
                model,
                [image],
                threshold,
                rect_resolution[0],
                rect_resolution[1],
                letterbox=use_letterbox,
            )
        else:
            detections = model.predict(image, threshold=threshold)
        detections = _remap_class_ids(detections, pred_to_gt)

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
