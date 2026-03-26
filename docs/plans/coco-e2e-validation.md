# План: Скачивание COCO, конвертация через cveta2, тренировка 3 эпохи + val + predict

## Цель

Скачать полный COCO 2017 dataset, сконвертировать через cveta2 в оба формата (COCO rfdetr и YOLO), верифицировать `rfdetr-tool` end-to-end: тренировка 3 эпохи, валидация, предсказание. После predict — round-trip конвертация предсказаний обратно через `cveta2 convert --from-yolo`.

---

## 1. Скачивание COCO 2017

**Источник**: https://cocodataset.org/#download

| Файл | Размер | URL |
|------|--------|-----|
| train2017.zip | ~18 GB | http://images.cocodataset.org/zips/train2017.zip |
| val2017.zip | ~1 GB | http://images.cocodataset.org/zips/val2017.zip |
| annotations_trainval2017.zip | ~252 MB | http://images.cocodataset.org/annotations/annotations_trainval2017.zip |

**Итого**: ~20 GB (830 GB свободно — ОК).

```bash
mkdir -p /home/nkt/datasets/coco2017
cd /home/nkt/datasets/coco2017

wget -c http://images.cocodataset.org/zips/train2017.zip
wget -c http://images.cocodataset.org/zips/val2017.zip
wget -c http://images.cocodataset.org/annotations/annotations_trainval2017.zip

unzip train2017.zip      # → train2017/ (118 287 изображений)
unzip val2017.zip        # → val2017/ (5 000 изображений)
unzip annotations_trainval2017.zip  # → annotations/
```

Итоговая структура:
```
/home/nkt/datasets/coco2017/
  train2017/          # 118k изображений
  val2017/            # 5k изображений
  annotations/
    instances_train2017.json
    instances_val2017.json
```

---

## 2. COCO JSON → cveta2 CSV (одноразовый скрипт)

Промежуточный шаг: из нативного COCO JSON в cveta2 CSV формат. Далее из CSV — конвертация в COCO (rfdetr) и YOLO через `cveta2 convert`.

```python
"""coco_to_cveta2_csv.py — COCO 2017 instances JSON → cveta2 dataset.csv."""
import json
import csv
from pathlib import Path

SRC = Path("/home/nkt/datasets/coco2017")
DST_CSV = Path("/home/nkt/datasets/coco2017_cveta2.csv")

rows: list[dict] = []

for split, ann_name, img_dir_name in [
    ("train", "instances_train2017.json", "train2017"),
    ("val",   "instances_val2017.json",   "val2017"),
]:
    ann_path = SRC / "annotations" / ann_name
    img_dir = SRC / img_dir_name

    with ann_path.open() as f:
        coco = json.load(f)

    cat_names = {c["id"]: c["name"] for c in coco["categories"]}
    img_info = {img["id"]: img for img in coco["images"]}

    img_anns: dict[int, list] = {}
    for ann in coco["annotations"]:
        img_anns.setdefault(ann["image_id"], []).append(ann)

    for img_id, info in img_info.items():
        fname = info["file_name"]
        w, h = info["width"], info["height"]
        img_path = str(img_dir / fname)

        anns = img_anns.get(img_id, [])
        if not anns:
            rows.append({
                "image_name": fname,
                "image_width": w, "image_height": h,
                "instance_shape": "none", "instance_label": "",
                "bbox_x_tl": "", "bbox_y_tl": "",
                "bbox_x_br": "", "bbox_y_br": "",
                "split": split, "image_path": img_path,
            })
        else:
            for ann in anns:
                x, y, bw, bh = ann["bbox"]
                rows.append({
                    "image_name": fname,
                    "image_width": w, "image_height": h,
                    "instance_shape": "box",
                    "instance_label": cat_names[ann["category_id"]],
                    "bbox_x_tl": x, "bbox_y_tl": y,
                    "bbox_x_br": x + bw, "bbox_y_br": y + bh,
                    "split": split, "image_path": img_path,
                })

    print(f"  {split}: {len(img_info)} images, {len(img_anns)} with anns")

fieldnames = [
    "image_name", "image_width", "image_height",
    "instance_shape", "instance_label",
    "bbox_x_tl", "bbox_y_tl", "bbox_x_br", "bbox_y_br",
    "split", "image_path",
]
with DST_CSV.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"\nИтого: {len(rows)} строк → {DST_CSV}")
```

### Верификация

```bash
wc -l /home/nkt/datasets/coco2017_cveta2.csv
head -3 /home/nkt/datasets/coco2017_cveta2.csv
# Ожидаемый вывод: ~897k строк (860k train anns + 36k val anns + images без аннотаций)
```

---

## 3. CSV → COCO формат (rfdetr) через cveta2

```bash
cveta2 convert --to-coco \
  -d /home/nkt/datasets/coco2017_cveta2.csv \
  -o /home/nkt/datasets/coco2017_rfdetr_coco/ \
  --link-mode symlink
```

cveta2 `--to-coco` создаёт rfdetr-совместимую структуру:
```
coco2017_rfdetr_coco/
  train/
    _annotations.coco.json
    <image symlinks>
  valid/               # cveta2 автоматически переименовывает "val" → "valid"
    _annotations.coco.json
    <image symlinks>
```

### Добавление test-сплита

rfdetr по умолчанию запускает `run_test=True` и упадёт с ZeroDivisionError без test/. Workaround — симлинки:

```bash
COCO_DST=/home/nkt/datasets/coco2017_rfdetr_coco
ln -s $COCO_DST/valid $COCO_DST/test
```

### Верификация

```bash
COCO_DST=/home/nkt/datasets/coco2017_rfdetr_coco

ls $COCO_DST/train/*.jpg 2>/dev/null | wc -l   # → 118287
ls $COCO_DST/valid/*.jpg 2>/dev/null | wc -l   # → 5000

python3 -c "
import json
for split in ['train', 'valid']:
    with open(f'$COCO_DST/{split}/_annotations.coco.json') as f:
        d = json.load(f)
    print(f'{split}: {len(d[\"images\"])} images, {len(d[\"annotations\"])} anns, {len(d[\"categories\"])} cats')
"
# train: 118287 images, ~860001 anns, 80 cats
# valid: 5000 images, ~36781 anns, 80 cats
```

**Нюанс**: cveta2 `--to-coco` использует 1-indexed category_id (COCO стандарт), но нумерация будет **контигуальная** (1..80), а не с пропусками как в оригинальном COCO (1..90). Это нормально для rfdetr — он работает с любыми category_id.

---

## 4. CSV → YOLO формат через cveta2

```bash
cveta2 convert --to-yolo \
  -d /home/nkt/datasets/coco2017_cveta2.csv \
  -o /home/nkt/datasets/coco2017_rfdetr_yolo/ \
  --link-mode symlink
```

Ожидаемая структура:
```
coco2017_rfdetr_yolo/
  dataset.yaml
  images/
    train/   → симлинки
    val/     → симлинки
  labels/
    train/   → .txt (class_id xc yc w h)
    val/     → .txt
```

### Добавление test-сплита

```bash
YOLO_DST=/home/nkt/datasets/coco2017_rfdetr_yolo
ln -s $YOLO_DST/images/val $YOLO_DST/images/test
ln -s $YOLO_DST/labels/val $YOLO_DST/labels/test
# Добавить test: в dataset.yaml
echo "test: images/test" >> $YOLO_DST/dataset.yaml
```

### Верификация

```bash
YOLO_DST=/home/nkt/datasets/coco2017_rfdetr_yolo

cat $YOLO_DST/dataset.yaml | head -10
ls $YOLO_DST/images/train/ | wc -l   # → 118287
ls $YOLO_DST/labels/train/ | wc -l   # → 118287
ls $YOLO_DST/images/val/ | wc -l     # → 5000
head -3 $YOLO_DST/labels/train/000000000009.txt
```

**Нюанс**: cveta2 сортирует классы **алфавитно** при генерации YOLO class_id (0-indexed). Порядок может отличаться от стандартного COCO80 (person=0, bicycle=1...). Это нормально для тренировки/инференса, но нужно учитывать при сравнении с эталонными COCO метриками.

---

## 5. Тренировка (3 эпохи)

### На COCO-формате

```bash
rfdetr-tool train \
  data=/home/nkt/datasets/coco2017_rfdetr_coco \
  variant=base \
  epochs=3 \
  batch_size=4 \
  output_dir=/home/nkt/datasets/coco2017_output_coco \
  checkpoint_interval=1 \
  progress_bar=true
```

### На YOLO-формате (альтернатива)

```bash
rfdetr-tool train \
  data=/home/nkt/datasets/coco2017_rfdetr_yolo \
  variant=base \
  epochs=3 \
  batch_size=4 \
  output_dir=/home/nkt/datasets/coco2017_output_yolo \
  checkpoint_interval=1 \
  progress_bar=true
```

### Ожидаемый результат
- `checkpoint_best_ema.pth` — лучший чекпоинт (EMA)
- `checkpoint.pth` — последний чекпоинт
- TensorBoard логи

### Ускорение (опционально)
- `variant=nano` — самая быстрая модель (384px)
- `batch_size=8`/`16` — если хватает VRAM
- `gpus=N` — DDP

---

## 6. Валидация

```bash
rfdetr-tool val \
  data=/home/nkt/datasets/coco2017_rfdetr_coco \
  weights=/home/nkt/datasets/coco2017_output_coco/checkpoint_best_ema.pth \
  variant=base \
  threshold=0.5
```

- val.py ожидает COCO-формат (`valid/_annotations.coco.json`)
- После 3 эпох mAP будет низкий — это нормально, цель — верификация пайплайна

---

## 7. Предсказание

### YOLO вывод + визуализация

```bash
rfdetr-tool predict \
  source=/home/nkt/datasets/coco2017/val2017 \
  weights=/home/nkt/datasets/coco2017_output_coco/checkpoint_best_ema.pth \
  variant=base \
  format=yolo \
  output_dir=/home/nkt/datasets/coco2017_predict_yolo \
  batch_size=8 \
  visualize=true
```

### CSV вывод (на YOLO dataset с сохранением сплитов)

```bash
rfdetr-tool predict \
  source=/home/nkt/datasets/coco2017_rfdetr_yolo/dataset.yaml \
  weights=/home/nkt/datasets/coco2017_output_coco/checkpoint_best_ema.pth \
  variant=base \
  format=csv \
  output_dir=/home/nkt/datasets/coco2017_predict_csv \
  batch_size=8
```

---

## 8. Round-trip: предсказания YOLO → CSV через cveta2

Конвертируем предсказания обратно в CSV — верификация полного round-trip.

```bash
cveta2 convert --from-yolo \
  -i /home/nkt/datasets/coco2017_predict_yolo/ \
  -o /home/nkt/datasets/coco2017_predict_roundtrip.csv \
  --names-file /home/nkt/datasets/coco2017_rfdetr_yolo/dataset.yaml \
  --image-dir /home/nkt/datasets/coco2017/val2017
```

### Верификация

```bash
head -5 /home/nkt/datasets/coco2017_predict_roundtrip.csv
wc -l /home/nkt/datasets/coco2017_predict_roundtrip.csv
# Должны быть строки с confidence, bbox_x_tl/y_tl/x_br/y_br в пикселях
```

**Полный data flow**:
```
COCO JSON → [скрипт] → cveta2 CSV → [cveta2 --to-coco] → rfdetr COCO → train → weights
                                    → [cveta2 --to-yolo] → rfdetr YOLO → train (alt)
                                                           weights → predict (YOLO out) → [cveta2 --from-yolo] → CSV
                                                           weights → predict (CSV out)
                                                           weights → val (mAP)
```

---

## 9. Порядок выполнения

| # | Шаг | Зависимости |
|---|------|-------------|
| 1 | Скачать + распаковать COCO 2017 | — |
| 2 | COCO JSON → cveta2 CSV (скрипт) | 1 |
| 3 | `cveta2 convert --to-coco` | 2 |
| 4 | `cveta2 convert --to-yolo` | 2 |
| 5 | Добавить test-сплиты (симлинки) | 3, 4 |
| 6 | Верифицировать оба формата | 5 |
| 7 | `rfdetr-tool train` (3 эпохи) | 6 |
| 8 | `rfdetr-tool val` | 7 |
| 9 | `rfdetr-tool predict` (YOLO + CSV) | 7 |
| 10 | `cveta2 convert --from-yolo` (round-trip) | 9 |

---

## 10. Edge cases и риски

1. **cveta2 алфавитная сортировка классов**: `--to-yolo` сортирует class names алфавитно (airplane=0, apple=1, ..., toothbrush=79). Это отличается от стандартного COCO порядка (person=0). Не влияет на работоспособность, но мешает сравнению с эталонными COCO метриками.

2. **cveta2 `--to-coco` category_id**: использует 1-indexed контигуальные ID (1..80) вместо оригинальных COCO ID (1..90 с пропусками). rfdetr и val.py это поддерживают — val.py маппит по именам классов.

3. **YOLO структура**: cveta2 создаёт `images/{split}/` + `labels/{split}/`, а некоторые YOLO-реализации ожидают `{split}/images/` + `{split}/labels/`. rfdetr использует `data.yaml` paths, поэтому должен работать с обеими структурами — но нужно проверить.

4. **run_test=True**: rfdetr по умолчанию тестирует на test-сплите. Без test/ — ZeroDivisionError. Решение: симлинки val → test.

5. **VRAM**: `variant=base` (560px) + `batch_size=4` → ~8-12 GB. Fallback: `batch_size=2` или `variant=nano`.

6. **Дисковое пространство**: ~45 GB из 830 GB свободных.

7. **Симлинки**: rfdetr/PIL должен следовать симлинкам — проверено на coco8.

8. **val.py**: работает только с COCO-форматом. Для val после YOLO-тренировки нужен COCO-датасет.

---

## 11. Артефакты после выполнения

```
/home/nkt/datasets/
  coco2017/                          # Оригинал
    train2017/, val2017/, annotations/
  coco2017_cveta2.csv                # Промежуточный CSV (cveta2 формат)
  coco2017_rfdetr_coco/              # COCO-формат (cveta2 --to-coco)
    train/ valid/ test→valid/
  coco2017_rfdetr_yolo/              # YOLO-формат (cveta2 --to-yolo)
    dataset.yaml, images/, labels/
  coco2017_output_coco/              # Результат тренировки
    checkpoint_best_ema.pth, checkpoint.pth
  coco2017_predict_yolo/             # Предсказания YOLO
    labels/, dataset.yaml, visualize/
  coco2017_predict_csv/              # Предсказания CSV
    predictions.csv
  coco2017_predict_roundtrip.csv     # Round-trip: YOLO → CSV (cveta2)
```
