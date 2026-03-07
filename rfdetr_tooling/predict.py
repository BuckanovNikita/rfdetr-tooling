"""Инференс RF-DETR на изображениях."""

from __future__ import annotations

import abc
import csv
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal, NamedTuple

import pillow_heif
import supervision as sv
import yaml
from loguru import logger
from PIL import Image

from rfdetr_tooling.train import _get_model_class

if TYPE_CHECKING:
    import numpy as np

pillow_heif.register_heif_opener()

_IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tiff",
    ".tif",
    ".heif",
    ".heic",
    ".avif",
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


class SourceEntry(NamedTuple):
    """Элемент источника данных с типом."""

    path: Path
    kind: str  # "dir" | "yolo_yaml" | "csv"


class ImageEntry(NamedTuple):
    """Запись об изображении с метаинформацией."""

    path: Path
    split: str | None
    width: int
    height: int


# ---------------------------------------------------------------------------
# Source parsing (§2)
# ---------------------------------------------------------------------------


def _parse_sources(source: str) -> list[SourceEntry]:
    """Разбирает запятую-разделённую строку и определяет тип каждого элемента."""
    entries: list[SourceEntry] = []
    for raw_part in source.split(","):
        raw = raw_part.strip()
        if not raw:
            continue
        p = Path(raw)

        if p.is_file():
            ext = p.suffix.lower()
            if ext in {".yaml", ".yml"}:
                with p.open() as f:
                    data = yaml.safe_load(f)
                if isinstance(data, dict) and "names" in data:
                    entries.append(SourceEntry(p, "yolo_yaml"))
                else:
                    msg = f"YAML-файл {p} не содержит ключа 'names'"
                    raise ValueError(msg)
            elif ext == ".csv":
                entries.append(SourceEntry(p, "csv"))
            else:
                msg = f"Неподдерживаемый тип файла: {p}"
                raise ValueError(msg)
        elif p.is_dir():
            yaml_candidates = [p / "data.yaml", p / "dataset.yaml"]
            yaml_found = next((y for y in yaml_candidates if y.is_file()), None)
            if yaml_found is not None:
                entries.append(SourceEntry(yaml_found, "yolo_yaml"))
            else:
                entries.append(SourceEntry(p, "dir"))
        else:
            msg = f"Источник не найден: {raw}"
            raise ValueError(msg)
    return entries


# ---------------------------------------------------------------------------
# Image collectors (§3)
# ---------------------------------------------------------------------------


def _collect_from_dir(path: Path) -> list[ImageEntry]:
    """Рекурсивный сбор всех файлов из директории."""
    results: list[ImageEntry] = []
    for p in sorted(path.rglob("*")):
        if p.is_dir():
            continue
        if p.name.startswith("."):
            continue
        if not p.suffix:
            continue
        results.append(ImageEntry(p.resolve(), None, 0, 0))
    if not results:
        logger.warning(f"Изображения не найдены в {path}")
    return results


def _collect_from_yolo_yaml(
    yaml_path: Path,
) -> tuple[list[ImageEntry], dict[int, str]]:
    """Собирает изображения из YOLO dataset yaml с сохранением сплитов."""
    with yaml_path.open() as f:
        data = yaml.safe_load(f)

    base_dir = yaml_path.parent
    class_names: dict[int, str] = {}
    raw_names = data.get("names", {})
    if isinstance(raw_names, dict):
        class_names = {int(k): str(v) for k, v in raw_names.items()}
    elif isinstance(raw_names, list):
        class_names = dict(enumerate(raw_names))

    results: list[ImageEntry] = []
    for split in ("train", "val", "test"):
        split_path_raw = data.get(split)
        if split_path_raw is None:
            continue
        split_path = base_dir / split_path_raw
        images_dir = (
            split_path / "images" if (split_path / "images").is_dir() else split_path
        )
        if not images_dir.is_dir():
            logger.warning(
                f"Директория изображений не найдена для сплита {split}: {images_dir}"
            )
            continue
        results.extend(
            ImageEntry(p.resolve(), split, 0, 0)
            for p in sorted(images_dir.rglob("*"))
            if p.is_file() and p.suffix.lower() in _IMAGE_EXTENSIONS
        )

    if not results:
        logger.warning(f"Изображения не найдены в YOLO yaml: {yaml_path}")
    return results, class_names


def _collect_from_csv(csv_path: Path) -> list[ImageEntry]:
    """Собирает изображения из CSV с колонкой image_path."""
    import pandas as pd  # noqa: PLC0415

    df = pd.read_csv(csv_path)
    if "image_path" not in df.columns:
        msg = f"CSV {csv_path} не содержит колонку 'image_path'"
        raise ValueError(msg)

    has_split = "split" in df.columns
    base_dir = csv_path.parent
    results: list[ImageEntry] = []
    skipped = 0

    if has_split:
        unique = df[["image_path", "split"]].drop_duplicates()
    else:
        unique = df[["image_path"]].drop_duplicates()
        unique = unique.assign(split=None)

    for row in unique.itertuples(index=False):
        raw_path = str(row.image_path)
        if not raw_path or raw_path == "nan":
            skipped += 1
            continue
        p = Path(raw_path)
        if not p.is_absolute():
            p = base_dir / p
        if not p.is_file():
            logger.warning(f"Файл не найден, пропускаю: {p}")
            skipped += 1
            continue
        split_val = row.split if has_split else None
        if isinstance(split_val, float):
            split_val = None
        results.append(
            ImageEntry(
                p.resolve(),
                str(split_val) if split_val is not None else None,
                0,
                0,
            )
        )

    if skipped:
        logger.warning(f"Пропущено {skipped} записей из CSV {csv_path}")
    if not results:
        logger.warning(f"Изображения не найдены в CSV: {csv_path}")
    return results


def _collect_all(sources: list[SourceEntry]) -> list[ImageEntry]:
    """Объединяет изображения из всех источников, дедупликация по path."""
    all_entries: list[ImageEntry] = []
    for src in sources:
        if src.kind == "dir":
            all_entries.extend(_collect_from_dir(src.path))
        elif src.kind == "yolo_yaml":
            entries, _ = _collect_from_yolo_yaml(src.path)
            all_entries.extend(entries)
        elif src.kind == "csv":
            all_entries.extend(_collect_from_csv(src.path))

    seen: dict[Path, ImageEntry] = {}
    for entry in all_entries:
        if entry.path in seen:
            logger.warning(f"Дубликат изображения, пропускаю: {entry.path}")
        else:
            seen[entry.path] = entry

    result = list(seen.values())
    if not result:
        msg = "Не найдено ни одного изображения в указанных источниках"  # noqa: RUF001
        raise ValueError(msg)
    return result


# ---------------------------------------------------------------------------
# Validation & size detection (§4)
# ---------------------------------------------------------------------------


def _validate_images(
    images: list[ImageEntry], *, check_image_sizes: bool
) -> list[ImageEntry]:
    """Валидация изображений и определение размеров."""
    failed: list[Path] = []
    validated: list[ImageEntry] = []
    cached_size: tuple[int, int] | None = None

    for entry in images:
        try:
            with Image.open(entry.path) as img:
                if check_image_sizes or cached_size is None:
                    w, h = img.size
                    if cached_size is None:
                        cached_size = (w, h)
                        logger.debug(f"Размер изображений (по первому файлу): {w}x{h}")
                else:
                    w, h = cached_size
                img.verify()
            validated.append(ImageEntry(entry.path, entry.split, w, h))
        except (OSError, SyntaxError):
            failed.append(entry.path)

    if failed:
        msg = f"Не удалось распознать {len(failed)} файлов как изображения: {failed}"  # noqa: RUF001
        raise ValueError(msg)
    return validated


# ---------------------------------------------------------------------------
# Output dir helper
# ---------------------------------------------------------------------------


def _make_output_dir(base: str) -> Path:
    """Создаёт output_dir с постфиксом _2, _3... если уже существует."""
    p = Path(base)
    if not p.exists():
        p.mkdir(parents=True)
        return p
    idx = 2
    while True:
        candidate = Path(f"{base}_{idx}")
        if not candidate.exists():
            candidate.mkdir(parents=True)
            return candidate
        idx += 1


# ---------------------------------------------------------------------------
# Writers (§5)
# ---------------------------------------------------------------------------


class _Writer(abc.ABC):
    @abc.abstractmethod
    def write(self, entry: ImageEntry, detections: sv.Detections) -> None: ...

    @abc.abstractmethod
    def finalize(self) -> None: ...


class YoloWriter(_Writer):
    """Записывает результаты в формате YOLO (txt + dataset.yaml)."""

    def __init__(
        self,
        output_dir: Path,
        class_names: dict[int, str],
    ) -> None:
        """Инициализация YOLO writer."""
        self._output_dir = output_dir
        self._labels_dir = output_dir / "labels"
        self._labels_dir.mkdir(parents=True, exist_ok=True)

        sorted_pred_ids = sorted(class_names.keys())
        self._pred_id_to_yolo_id = {
            pred_id: yolo_id for yolo_id, pred_id in enumerate(sorted_pred_ids)
        }
        self._yolo_names = {
            yolo_id: class_names[pred_id]
            for pred_id, yolo_id in self._pred_id_to_yolo_id.items()
        }
        self._seen_splits: set[str] = set()
        self._seen_stems: dict[str, dict[str, int]] = {}
        self._count = 0

    def write(self, entry: ImageEntry, detections: sv.Detections) -> None:
        """Записывает один txt-файл с детекциями."""
        split_name = entry.split or "unsplit"
        self._seen_splits.add(split_name)

        split_dir = self._labels_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)

        stem_counts = self._seen_stems.setdefault(split_name, {})
        stem = entry.path.stem
        if stem in stem_counts:
            stem_counts[stem] += 1
            stem = f"{stem}_{stem_counts[stem]}"
        else:
            stem_counts[stem] = 0

        txt_path = split_dir / f"{stem}.txt"
        lines: list[str] = []

        if (
            detections.xyxy is not None
            and len(detections.xyxy) > 0
            and detections.class_id is not None
            and detections.confidence is not None
        ):
            w_img = entry.width
            h_img = entry.height
            for xyxy, class_id, conf in zip(
                detections.xyxy,
                detections.class_id,
                detections.confidence,
                strict=True,
            ):
                x1, y1, x2, y2 = xyxy
                xc = ((x1 + x2) / 2) / w_img
                yc = ((y1 + y2) / 2) / h_img
                bw = (x2 - x1) / w_img
                bh = (y2 - y1) / h_img
                yolo_id = self._pred_id_to_yolo_id.get(int(class_id), 0)
                lines.append(
                    f"{yolo_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f} {conf:.6f}"
                )

        txt_path.write_text("\n".join(lines) + ("\n" if lines else ""))
        self._count += 1

    def finalize(self) -> None:
        """Записывает dataset.yaml."""
        yaml_data: dict[str, object] = {"names": self._yolo_names}
        for split_name in sorted(self._seen_splits):
            yaml_data[split_name] = f"labels/{split_name}"

        yaml_path = self._output_dir / "dataset.yaml"
        with yaml_path.open("w") as f:
            yaml.dump(yaml_data, f, default_flow_style=False, allow_unicode=True)

        logger.info(f"YOLO: {self._count} файлов записано в {self._output_dir}")


class CsvWriter(_Writer):
    """Записывает результаты в формате CSV (cveta2)."""

    _CSV_COLUMNS: ClassVar[list[str]] = [
        "task_name",
        "source",
        "image_path",
        "image_name",
        "image_width",
        "image_height",
        "instance_shape",
        "instance_label",
        "bbox_x_tl",
        "bbox_y_tl",
        "bbox_x_br",
        "bbox_y_br",
        "confidence",
        "split",
    ]

    def __init__(
        self,
        output_dir: Path,
        class_names: dict[int, str],
    ) -> None:
        """Инициализация CSV writer."""
        self._output_dir = output_dir
        self._class_names = class_names
        self._csv_path = output_dir / "predictions.csv"
        self._file = self._csv_path.open("w", newline="")
        self._csv_writer = csv.DictWriter(self._file, fieldnames=self._CSV_COLUMNS)
        self._csv_writer.writeheader()
        self._count = 0

    def write(self, entry: ImageEntry, detections: sv.Detections) -> None:
        """Дописывает строки детекций в CSV."""
        base_row = {
            "task_name": "rfdetr-predict",
            "source": "auto",
            "image_path": str(entry.path),
            "image_name": entry.path.name,
            "image_width": entry.width,
            "image_height": entry.height,
            "split": entry.split or "",
        }

        if (
            detections.xyxy is not None
            and len(detections.xyxy) > 0
            and detections.class_id is not None
            and detections.confidence is not None
        ):
            for xyxy, class_id, conf in zip(
                detections.xyxy,
                detections.class_id,
                detections.confidence,
                strict=True,
            ):
                x1, y1, x2, y2 = xyxy
                row = {
                    **base_row,
                    "instance_shape": "box",
                    "instance_label": self._class_names.get(
                        int(class_id), f"cls_{class_id}"
                    ),
                    "bbox_x_tl": f"{x1:.2f}",
                    "bbox_y_tl": f"{y1:.2f}",
                    "bbox_x_br": f"{x2:.2f}",
                    "bbox_y_br": f"{y2:.2f}",
                    "confidence": f"{conf:.6f}",
                }
                self._csv_writer.writerow(row)
        else:
            row = {
                **base_row,
                "instance_shape": "none",
                "instance_label": "",
                "bbox_x_tl": "",
                "bbox_y_tl": "",
                "bbox_x_br": "",
                "bbox_y_br": "",
                "confidence": "",
            }
            self._csv_writer.writerow(row)

        self._count += 1

    def finalize(self) -> None:
        """Закрывает CSV файл."""
        self._file.close()
        logger.info(f"CSV: {self._count} изображений записано в {self._csv_path}")


# ---------------------------------------------------------------------------
# Visualization (§5.3)
# ---------------------------------------------------------------------------


def _visualize(
    entry: ImageEntry,
    detections: sv.Detections,
    class_names: dict[int, str],
    vis_dir: Path,
) -> None:
    """Рисует аннотированное изображение и сохраняет в vis_dir."""
    import numpy as np  # noqa: PLC0415

    image = Image.open(entry.path).convert("RGB")
    frame: np.ndarray[tuple[int, ...], np.dtype[np.uint8]] = np.array(image)

    labels: list[str] = []
    if detections.class_id is not None and detections.confidence is not None:
        labels = [
            f"{class_names.get(int(cid), f'cls_{cid}')} {conf:.2f}"
            for cid, conf in zip(
                detections.class_id, detections.confidence, strict=True
            )
        ]

    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()
    annotated = box_annotator.annotate(frame.copy(), detections)
    annotated = label_annotator.annotate(annotated, detections, labels=labels)

    out_path = vis_dir / entry.path.name
    Image.fromarray(annotated).save(out_path)


# ---------------------------------------------------------------------------
# Chunking helper
# ---------------------------------------------------------------------------


def _chunked(lst: list[ImageEntry], n: int) -> list[list[ImageEntry]]:
    """Разбивает список на чанки размером n."""
    return [lst[i : i + n] for i in range(0, len(lst), n)]


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def predict(  # noqa: PLR0913, C901
    source: str,
    weights: str,
    *,
    variant: Literal["nano", "small", "base", "medium", "large"] = "base",
    conf_threshold: float = 0.01,
    nms_threshold: float = 0.25,
    agnostic_nms: bool = False,
    resolution: int | tuple[int, int] | None = None,
    batch_size: int = 4,
    device: str = "auto",
    output_dir: str = "predict_output",
    format: Literal["yolo", "csv"] = "yolo",  # noqa: A002
    visualize: bool = False,
    check_image_sizes: bool = False,
    **model_extra: Any,  # noqa: ANN401
) -> None:
    """Инференс RF-DETR.

    Args:
        source: Пути через запятую — папки, data.yaml, dataset.csv.
        weights: Путь к файлу весов модели.
        variant: Вариант архитектуры RF-DETR.
        conf_threshold: Порог уверенности для детекций.
        nms_threshold: IoU-порог для NMS.
        agnostic_nms: Class-agnostic NMS.
        resolution: Разрешение входа модели (None = по умолчанию).
        batch_size: Размер батча для инференса.
        device: Устройство ("auto", "cpu", "cuda", "mps").
        output_dir: Директория для результатов.
        format: Формат выходных данных ("yolo" или "csv").
        visualize: Сохранять визуализации детекций.
        check_image_sizes: Проверять размер каждого изображения.
        **model_extra: Дополнительные kwargs для конструктора модели.

    """
    # 1. Парсинг источников
    sources = _parse_sources(source)
    logger.info(f"Источники: {sources}")

    # 2. Сбор изображений
    images = _collect_all(sources)
    logger.info(f"Найдено {len(images)} изображений")

    # 3. Валидация и размеры
    images = _validate_images(images, check_image_sizes=check_image_sizes)

    # 4. Инициализация модели
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

    class_names: dict[int, str] = model.class_names

    logger.info(
        f"Инференс: {len(images)} изображений, variant={variant}, "
        f"conf={conf_threshold}, nms={nms_threshold}, "
        f"batch_size={batch_size}"
    )

    # 5. GPU-инференс (батчированный, чанками)
    raw_chunks: list[list[tuple[ImageEntry, sv.Detections]]] = []
    for batch_entries in _chunked(images, batch_size):
        pil_images = [Image.open(e.path).convert("RGB") for e in batch_entries]
        if rect_resolution is not None:
            from rfdetr_tooling._inference import predict_batch_rect  # noqa: PLC0415

            det_list: list[sv.Detections] = predict_batch_rect(
                model,
                pil_images,
                conf_threshold,
                rect_resolution[0],
                rect_resolution[1],
            )
        else:
            det_list = model.predict(pil_images, threshold=conf_threshold)
        chunk = list(zip(batch_entries, det_list, strict=True))
        raw_chunks.append(chunk)
        del pil_images

    # 6. Создание output_dir и writer
    out_path = _make_output_dir(output_dir)

    vis_dir: Path | None = None
    if visualize:
        vis_dir = out_path / "visualize"
        vis_dir.mkdir(parents=True, exist_ok=True)

    writer: _Writer
    if format == "yolo":
        writer = YoloWriter(out_path, class_names)
    else:
        writer = CsvWriter(out_path, class_names)

    # 7. NMS + запись (по чанкам)
    for chunk in raw_chunks:
        for entry, raw_dets in chunk:
            filtered = raw_dets.with_nms(
                threshold=nms_threshold,
                class_agnostic=agnostic_nms,
            )
            writer.write(entry, filtered)
            if vis_dir is not None:
                _visualize(entry, filtered, class_names, vis_dir)

    writer.finalize()
    logger.info("Инференс завершён")
