import json

log_path = "/home/nkt/rfdetr-tooling/experiments/rect_coco/3b_rect_512x384_lb/log.txt"
lines = open(log_path).readlines()
for i, line in enumerate(lines):
    data = json.loads(line)
    bbox = data.get("test_coco_eval_bbox", [])
    if bbox:
        print(f"epoch {i+1}: mAP@50:95={bbox[0]:.4f}, mAP@50={bbox[1]:.4f}, mAP@75={bbox[2]:.4f}")
    else:
        print(f"epoch {i+1}: no eval data")
    print(f"  train_loss={data.get('train_loss', '?'):.4f}, test_loss={data.get('test_loss', '?'):.4f}")
print(f"Total epochs logged: {len(lines)}")
