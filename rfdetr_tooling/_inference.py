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


def predict_batch_rect(
    model: RFDETR,
    images: list[Image.Image],
    threshold: float,
    res_h: int,
    res_w: int,
) -> list[sv.Detections]:
    """Inference с прямоугольным resize."""
    import supervision as sv  # noqa: PLC0415

    orig_sizes: list[tuple[int, int]] = []
    processed: list[torch.Tensor] = []
    device = model.model.device

    for img in images:
        t = F.to_tensor(img)
        orig_sizes.append((t.shape[1], t.shape[2]))  # (H, W)
        t = t.to(device)
        t = F.normalize(t, _MEANS, _STDS)
        t = F.resize(t, [res_h, res_w])
        processed.append(t)

    batch = torch.stack(processed)
    with torch.no_grad():
        model.model.model.eval()
        predictions = model.model.model(batch)
        target_sizes = torch.tensor(orig_sizes, device=device)
        results: list[dict[str, Any]] = model.model.postprocess(
            predictions, target_sizes=target_sizes
        )

    detections: list[sv.Detections] = []
    for result in results:
        scores = result["scores"]
        labels = result["labels"]
        boxes = result["boxes"]
        keep = scores > threshold
        det = sv.Detections(
            xyxy=boxes[keep].float().cpu().numpy(),
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

        if image_set == "train":
            resolved_aug = aug_config if aug_config is not None else AUG_CONFIG

            option_a = {
                "OneOf": {
                    "transforms": [
                        {"Resize": {"height": res_h, "width": res_w}},
                    ],
                }
            }
            crop_min = min(384, res_h, res_w)
            crop_max = min(600, max(res_h, res_w))
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
            resize_wrappers = AlbumentationsWrapper.from_config(
                [{"Resize": {"height": res_h, "width": res_w}}]
            )
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
