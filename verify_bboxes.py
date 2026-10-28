"""Draw YOLO boxes on augmented images to prove labels stay aligned.

Usage: python verify_bboxes.py --dataset dataset --split train --n 10 --out results/part2/verification_grid.jpg
"""
import argparse, random
from pathlib import Path
import cv2, numpy as np
from utils import list_images, read_yolo, draw_yolo_boxes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="dataset")
    ap.add_argument("--split", default="train")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--cols", type=int, default=5)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="results/part2/verification_grid.jpg")
    args = ap.parse_args()

    root = Path(args.dataset) / args.split
    files = [p for p in list_images(root / "images") if "_aug" in p.stem]
    random.Random(args.seed).shuffle(files)
    tiles = []
    for p in files[:args.n]:
        img = cv2.imread(str(p))
        boxes = read_yolo(root / "labels" / f"{p.stem}.txt")
        vis = draw_yolo_boxes(img, boxes, thickness=2)
        cv2.rectangle(vis, (0, 0), (img.shape[1], 24), (0, 0, 0), -1)
        cv2.putText(vis, p.stem, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(cv2.resize(vis, (400, 400)))
    while len(tiles) % args.cols:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [cv2.hconcat(tiles[i:i + args.cols]) for i in range(0, len(tiles), args.cols)]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(args.out, cv2.vconcat(rows))
    print(f"Saved {args.out} ({min(args.n, len(files))} augmented images)")


if __name__ == "__main__":
    main()
