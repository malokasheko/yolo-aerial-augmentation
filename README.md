# Task 2 – Image Preprocessing, Classical Processing & YOLO Augmentation

## Setup
```bash
pip install -r requirements.txt
```

## Part 1 – Classical processing
```bash
python classical_processing.py --images raw/images --labels raw/labels --out results/part1 --n 5
# optional HSV masking: --method hsv --hsv-lower H S V --hsv-upper H S V
```
Pipeline: GT box (`cv2.rectangle`) → bilateral + Gaussian filter → Otsu / adaptive / HSV mask →
opening + closing → Canny + `findContours` (largest contour, `minAreaRect`, `boundingRect`) → IoU vs GT.

![Part 1 grid](results/part1/comparison_all.jpg)

Mean IoU of classical box vs GT: **<fill in from results/part1/iou_summary.csv>**
Discussion: <where thresholding works / fails on the textured PBR background>

## Part 2 – Augmentation & split
```bash
python augment_dataset.py --images raw/images --labels raw/labels --out dataset --names <class_name> --variants 5
python verify_bboxes.py --dataset dataset --split train --n 10 --out results/part2/verification_grid.jpg
```
- Letterbox to 640×640 (labels remapped), split **by base image** 70/20/10, then augment inside each split (no leakage).
- Photometric: brightness/contrast, HSV shift, Gaussian / salt-and-pepper noise, Gaussian / motion blur.
- Geometric: h-flip, v-flip, 90/180/270° rotation, random scale + translation; boxes recomputed (corner transform, clipping, drop if <40 % visible).
- Every file's applied ops are listed in `dataset/augmentation_log.csv`.

![Part 2 verification](results/part2/verification_grid.jpg)

## Dataset
Google Drive: <link>
