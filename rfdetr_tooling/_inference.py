"""Общие утилиты для прямоугольного инференса и monkey-patch transforms."""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

import torch
import torchvision.transforms.functional as F  # noqa: N812

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    import supervision as sv
    from PIL import Image
    from rfdetr import RFDETR

_MEANS = [0.485, 0.456, 0.406]
_STDS = [0.229, 0.224, 0.225]


def _letterbox_resize(
    t: torch.Tensor,
    res_h: int,
    res_w: int,
) -> tuple[torch.Tensor, float, int, int]:
    """Letterbox resize: сохраняет AR, добавляет padding до (res_h, res_w).

    Returns:
        (resized_padded_tensor, scale, pad_bottom, pad_right)

    """
    orig_h, orig_w = t.shape[1], t.shape[2]
    scale = min(res_h / orig_h, res_w / orig_w)
    new_h = int(orig_h * scale)
    new_w = int(orig_w * scale)
    t = F.resize(t, [new_h, new_w])

    pad_bottom = res_h - new_h
    pad_right = res_w - new_w
    if pad_bottom > 0 or pad_right > 0:
        t = torch.nn.functional.pad(t, [0, pad_right, 0, pad_bottom], value=114 / 255)
    return t, scale, pad_bottom, pad_right


def predict_batch_rect(
    model: RFDETR,
    images: list[Image.Image],
    threshold: float,
    res_h: int,
    res_w: int,
) -> list[sv.Detections]:
    """Inference с прямоугольным letterbox resize."""
    import numpy as np  # noqa: PLC0415
    import supervision as sv  # noqa: PLC0415

    meta: list[tuple[float, int, int]] = []  # (scale, orig_h, orig_w)
    processed: list[torch.Tensor] = []
    device = model.model.device

    for img in images:
        t = F.to_tensor(img)
        orig_h, orig_w = t.shape[1], t.shape[2]
        t = t.to(device)
        t, scale, _, _ = _letterbox_resize(t, res_h, res_w)
        t = F.normalize(t, _MEANS, _STDS)
        meta.append((scale, orig_h, orig_w))
        processed.append(t)

    canvas_sizes = torch.tensor([[res_h, res_w]] * len(processed), device=device)
    batch = torch.stack(processed)
    with torch.no_grad():
        model.model.model.eval()
        predictions = model.model.model(batch)
        results: list[dict[str, Any]] = model.model.postprocess(
            predictions, target_sizes=canvas_sizes
        )

    detections: list[sv.Detections] = []
    for result, (scale, orig_h, orig_w) in zip(results, meta, strict=True):
        scores = result["scores"]
        labels = result["labels"]
        boxes = result["boxes"]
        keep = scores > threshold

        xyxy = boxes[keep].float().cpu().numpy()
        # Обратное преобразование: canvas coords → original image coords
        xyxy[:, [0, 2]] /= scale
        xyxy[:, [1, 3]] /= scale
        xyxy[:, [0, 2]] = np.clip(xyxy[:, [0, 2]], 0, orig_w)
        xyxy[:, [1, 3]] = np.clip(xyxy[:, [1, 3]], 0, orig_h)

        det = sv.Detections(
            xyxy=xyxy,
            confidence=scores[keep].float().cpu().numpy(),
            class_id=labels[keep].cpu().numpy(),
        )
        detections.append(det)
    return detections


def _make_rect_transforms(
    res_h: int,
    res_w: int,
) -> Callable[..., Any]:
    """Возвращает patched make_coco_transforms для прямоугольного resize."""

    def _patched(  # noqa: PLR0913
        image_set: str,
        resolution: int,  # noqa: ARG001
        multi_scale: bool = False,  # noqa: ARG001, FBT001, FBT002
        expanded_scales: bool = False,  # noqa: ARG001, FBT001, FBT002
        skip_random_resize: bool = False,  # noqa: ARG001, FBT001, FBT002
        patch_size: int = 16,  # noqa: ARG001
        num_windows: int = 4,  # noqa: ARG001
        aug_config: dict[str, Any] | None = None,
        **kwargs: Any,  # noqa: ANN401, ARG001
    ) -> Any:  # noqa: ANN401
        from rfdetr.datasets.aug_config import AUG_CONFIG  # noqa: PLC0415
        from rfdetr.datasets.transforms import (  # noqa: PLC0415
            AlbumentationsWrapper,
            Normalize,
        )
        from torchvision.transforms.v2 import Compose, ToDtype, ToImage  # noqa: PLC0415

        to_image = ToImage()
        to_float = ToDtype(torch.float32, scale=True)
        normalize = Normalize()

        max_side = max(res_h, res_w)
        pad_cfg = {
            "min_height": res_h,
            "min_width": res_w,
            "border_mode": 0,
            "fill": 114,
        }

        if image_set == "train":
            resolved_aug = aug_config if aug_config is not None else AUG_CONFIG

            option_a = {
                "Sequential": {
                    "transforms": [
                        {"LongestMaxSize": {"max_size": max_side}},
                        {
                            "PadIfNeeded": {
                                **pad_cfg,
                                "position": "random",
                            }
                        },
                    ]
                }
            }
            crop_min = min(384, res_h, res_w)
            crop_max = min(600, max_side)
            option_b = {
                "Sequential": {
                    "transforms": [
                        {"SmallestMaxSize": {"max_size": [400, 500, 600]}},
                        {
                            "OneOf": {
                                "transforms": [
                                    {
                                        "RandomSizedCrop": {
                                            "min_max_height": [
                                                crop_min,
                                                crop_max,
                                            ],
                                            "height": res_h,
                                            "width": res_w,
                                        }
                                    },
                                ],
                            }
                        },
                    ]
                }
            }
            resize_config: list[dict[str, Any]] = [
                {"OneOf": {"transforms": [option_a, option_b]}}
            ]
            resize_wrappers = AlbumentationsWrapper.from_config(resize_config)
            aug_wrappers = AlbumentationsWrapper.from_config(resolved_aug)
            return Compose(
                [*resize_wrappers, *aug_wrappers, to_image, to_float, normalize]
            )

        if image_set in ("val", "test", "val_speed"):
            resize_config_val: list[dict[str, Any]] = [
                {"LongestMaxSize": {"max_size": max_side}},
                {
                    "PadIfNeeded": {
                        **pad_cfg,
                        "position": "top_left",
                    }
                },
            ]
            resize_wrappers = AlbumentationsWrapper.from_config(resize_config_val)
            return Compose([*resize_wrappers, to_image, to_float, normalize])

        msg = f"unknown image_set: {image_set}"
        raise ValueError(msg)

    return _patched


@contextmanager
def rect_resolution_patch(res_h: int, res_w: int) -> Iterator[None]:
    """Context manager: подменяет rfdetr transform builders на прямоугольные."""
    import rfdetr.datasets.coco as coco_mod  # noqa: PLC0415

    orig_square = coco_mod.make_coco_transforms_square_div_64
    orig_normal = coco_mod.make_coco_transforms
    patched = _make_rect_transforms(res_h, res_w)

    coco_mod.make_coco_transforms_square_div_64 = patched
    coco_mod.make_coco_transforms = patched
    try:
        yield
    finally:
        coco_mod.make_coco_transforms_square_div_64 = orig_square
        coco_mod.make_coco_transforms = orig_normal
