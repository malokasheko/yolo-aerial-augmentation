"""Suggest an HSV range for the object by sampling pixels inside the GT boxes
(and checking how much of the background that range would also catch).

Usage: python find_hsv_range.py --images raw/images --labels raw/labels
"""
import argparse
import cv2, numpy as np
from utils import list_images, read_yolo, yolo_to_xyxy

ap = argparse.ArgumentParser()
ap.add_argument("--images", required=True)
ap.add_argument("--labels", required=True)
ap.add_argument("--n", type=int, default=20)
ap.add_argument("--pct", type=float, default=5, help="trim this percentile on each side")
args = ap.parse_args()

inside, outside = [], []
for p in list_images(args.images)[:args.n]:
    img = cv2.imread(str(p))
    boxes = read_yolo(f"{args.labels}/{p.stem}.txt")
    if not boxes:
        continue
    H, W = img.shape[:2]
    hsv = cv2.cvtColor(cv2.GaussianBlur(img, (5, 5), 0), cv2.COLOR_BGR2HSV)
    x1, y1, x2, y2 = yolo_to_xyxy(boxes[0], W, H)
    m = np.zeros((H, W), bool); m[max(0, y1):y2, max(0, x1):x2] = True
    inside.append(hsv[m]); outside.append(hsv[~m][::50])
inside, outside = np.vstack(inside), np.vstack(outside)

lo = np.percentile(inside, args.pct, axis=0).astype(int)
hi = np.percentile(inside, 100 - args.pct, axis=0).astype(int)
inr = np.all((inside >= lo) & (inside <= hi), axis=1).mean()
bg = np.all((outside >= lo) & (outside <= hi), axis=1).mean()
print(f"Suggested range:  --hsv-lower {lo[0]} {lo[1]} {lo[2]} --hsv-upper {hi[0]} {hi[1]} {hi[2]}")
print(f"Captures {inr:.0%} of pixels inside GT boxes and {bg:.1%} of background pixels.")
print("Good: high first number, very low second number. If background is high, the object colour overlaps the ground.")
