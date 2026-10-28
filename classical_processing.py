"""Part 1: GT box visualisation + classical pipeline
(filter -> threshold -> morphology -> Canny + contours) with comparison grids.

Usage:
  python classical_processing.py --images raw/images --labels raw/labels --out results/part1 --n 5
  python classical_processing.py ... --method hsv --hsv-lower 0 80 40 --hsv-upper 20 255 255
"""
import argparse, csv, random
from pathlib import Path
import cv2, numpy as np
from utils import list_images, read_yolo, yolo_to_xyxy, draw_yolo_boxes, iou_xyxy

PANEL = 400  # side of each tile in the grid


def titled(img, title):
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    tile = cv2.resize(img, (PANEL, PANEL))
    cv2.rectangle(tile, (0, 0), (PANEL, 26), (0, 0, 0), -1)
    cv2.putText(tile, title, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return tile


def segment(blur_gray, hsv, gt_xyxy, args):
    """Return a binary mask (255 = object)."""
    if args.method == "hsv":
        return cv2.inRange(hsv, np.array(args.hsv_lower), np.array(args.hsv_upper))
    if args.method == "bgdiff":
        # Background estimate = very large median blur (removes the small object),
        # then threshold |image - background| with Otsu.
        k = args.bg_kernel or (max(51, blur_gray.shape[1] // 10) | 1)
        bg = cv2.medianBlur(blur_gray, k)
        diff = cv2.absdiff(blur_gray, bg)
        _, mask = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return mask
    if args.method == "adaptive":
        return cv2.adaptiveThreshold(blur_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY_INV, 51, 5)
    # Otsu: decide polarity by comparing mean intensity inside vs outside the GT box
    x1, y1, x2, y2 = gt_xyxy
    inside = blur_gray[max(0, y1):y2, max(0, x1):x2].mean()
    m = np.ones(blur_gray.shape, bool); m[max(0, y1):y2, max(0, x1):x2] = False
    outside = blur_gray[m].mean()
    flag = cv2.THRESH_BINARY if inside > outside else cv2.THRESH_BINARY_INV
    _, mask = cv2.threshold(blur_gray, 0, 255, flag + cv2.THRESH_OTSU)
    return mask


def process(img, boxes, args):
    H, W = img.shape[:2]
    t = max(2, W // 400)
    gt = yolo_to_xyxy(boxes[0], W, H) if boxes else (0, 0, W, H)

    gt_img = draw_yolo_boxes(img, boxes, color=(0, 255, 0))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Edge-preserving smoothing, then a light Gaussian to flatten leftover texture
    filt = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
    filt_gray = cv2.GaussianBlur(cv2.cvtColor(filt, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    filt_hsv = cv2.cvtColor(filt, cv2.COLOR_BGR2HSV)

    mask = segment(filt_gray, filt_hsv, gt, args)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=2)    # remove specks
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, k, iterations=3)  # fill holes

    edges = cv2.Canny(filt_gray, 50, 150)

    cnts, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    final = gt_img.copy()
    edge_vis = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    iou = 0.0
    if cnts:
        c = max(cnts, key=cv2.contourArea)
        cv2.drawContours(final, [c], -1, (0, 0, 255), t)               # red: largest contour
        cv2.drawContours(edge_vis, [c], -1, (0, 0, 255), t)
        rect = cv2.minAreaRect(c)                                      # blue: min-area rotated rect
        pts = cv2.boxPoints(rect).astype(np.int32)
        cv2.polylines(final, [pts], True, (255, 0, 0), t)
        x, y, w, h = cv2.boundingRect(c)                               # yellow: axis-aligned box
        cv2.rectangle(final, (x, y), (x + w, y + h), (0, 255, 255), t)
        iou = iou_xyxy(gt, (x, y, x + w, y + h))

    tiles = [titled(gt_img, "1. Original + GT box"),
             titled(filt, "2. Bilateral filtered"),
             titled(mask, f"3. Mask ({args.method})"),
             titled(clean, "4. Morph open+close"),
             titled(edge_vis, "5. Canny + contour"),
             titled(final, f"6. GT(g) cnt(r) rect(b) IoU={iou:.2f}")]
    return cv2.hconcat(tiles), iou


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", default="results/part1")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--method", choices=["otsu", "adaptive", "hsv", "bgdiff"], default="otsu")
    ap.add_argument("--hsv-lower", type=int, nargs=3, default=[0, 60, 40])
    ap.add_argument("--hsv-upper", type=int, nargs=3, default=[179, 255, 255])
    ap.add_argument("--bg-kernel", type=int, default=0, help="odd median kernel for bgdiff (0 = auto)")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    imgs = list_images(args.images)
    random.Random(args.seed).shuffle(imgs)
    rows, results = [], []
    for p in imgs[:args.n]:
        img = cv2.imread(str(p))
        boxes = read_yolo(Path(args.labels) / f"{p.stem}.txt")
        grid, iou = process(img, boxes, args)
        cv2.imwrite(str(out / f"grid_{p.stem}.jpg"), grid)
        rows.append(grid); results.append((p.name, round(iou, 3)))
        print(f"{p.name}: IoU(classical bbox, GT) = {iou:.3f}")
    if rows:
        cv2.imwrite(str(out / "comparison_all.jpg"), cv2.vconcat(rows))
    with open(out / "iou_summary.csv", "w", newline="") as f:
        csv.writer(f).writerows([("image", "iou")] + results)


if __name__ == "__main__":
    main()
