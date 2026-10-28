"""Shared helpers: YOLO label I/O, coordinate conversion, box drawing, IoU."""
from pathlib import Path
import cv2

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def list_images(folder):
    return sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in IMG_EXTS)


def read_yolo(path):
    """Return list of (class_id, xc, yc, w, h), all normalized to [0, 1]."""
    path = Path(path)
    boxes = []
    if not path.exists():
        return boxes
    for line in path.read_text().splitlines():
        p = line.split()
        if len(p) >= 5:
            boxes.append((int(float(p[0])), *map(float, p[1:5])))
    return boxes


def write_yolo(path, boxes):
    with open(path, "w") as f:
        for c, x, y, w, h in boxes:
            f.write(f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")


def yolo_to_xyxy(box, W, H):
    """Normalized (cls, xc, yc, w, h) -> pixel (x_min, y_min, x_max, y_max)."""
    _, xc, yc, w, h = box
    return (int(round((xc - w / 2) * W)), int(round((yc - h / 2) * H)),
            int(round((xc + w / 2) * W)), int(round((yc + h / 2) * H)))


def xyxy_to_yolo(x1, y1, x2, y2, W, H, cls=0):
    return (cls, (x1 + x2) / 2 / W, (y1 + y2) / 2 / H, (x2 - x1) / W, (y2 - y1) / H)


def draw_yolo_boxes(img, boxes, names=None, color=(0, 255, 0), thickness=None):
    """Draw YOLO-format boxes with cv2.rectangle on a copy of img."""
    out = img.copy()
    H, W = out.shape[:2]
    t = thickness or max(2, W // 400)
    for b in boxes:
        x1, y1, x2, y2 = yolo_to_xyxy(b, W, H)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, t)
        label = names[b[0]] if names and b[0] < len(names) else str(b[0])
        cv2.putText(out, label, (x1, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX,
                    max(0.5, W / 1600), color, max(1, t - 1), cv2.LINE_AA)
    return out


def iou_xyxy(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0
