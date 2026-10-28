"""Part 2: letterbox to 640x640, split by BASE image (no leakage), then create
photometric + geometric augmentations with exactly-updated YOLO labels.

Usage:
  python augment_dataset.py --images raw/images --labels raw/labels --out dataset \
         --names drone --variants 5
Output per base image: 1 clean letterboxed copy + N augmented variants (default 6 total).
"""
import argparse, csv, random, shutil
from pathlib import Path
import cv2, numpy as np
from utils import list_images, read_yolo, write_yolo

FILL = (114, 114, 114)


# ---------- preprocessing ----------
def letterbox(img, boxes, size=640):
    """Aspect-preserving resize + padding. Labels are remapped to the padded canvas."""
    h, w = img.shape[:2]
    r = size / max(h, w)
    nw, nh = round(w * r), round(h * r)
    res = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR)
    top, left = (size - nh) // 2, (size - nw) // 2
    canvas = np.full((size, size, 3), FILL, np.uint8)
    canvas[top:top + nh, left:left + nw] = res
    nb = [(c, (x * nw + left) / size, (y * nh + top) / size, bw * nw / size, bh * nh / size)
          for c, x, y, bw, bh in boxes]
    return canvas, nb


# ---------- photometric (labels unchanged) ----------
def p_bright_contrast(img, rng):
    return cv2.convertScaleAbs(img, alpha=rng.uniform(0.7, 1.3), beta=rng.uniform(-40, 40))


def p_hsv(img, rng):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] + rng.integers(-8, 9)) % 180
    hsv[..., 1] = np.clip(hsv[..., 1] * rng.uniform(0.7, 1.3), 0, 255)
    hsv[..., 2] = np.clip(hsv[..., 2] * rng.uniform(0.8, 1.2), 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def p_noise(img, rng):
    if rng.random() < 0.5:  # Gaussian
        n = rng.normal(0, rng.uniform(3, 12), img.shape)
        return np.clip(img.astype(np.float32) + n, 0, 255).astype(np.uint8)
    out = img.copy()        # salt & pepper
    p = rng.uniform(0.002, 0.01)
    m = rng.random(img.shape[:2])
    out[m < p / 2] = 0
    out[m > 1 - p / 2] = 255
    return out


def p_blur(img, rng):
    k = int(rng.choice([3, 5, 7]))
    if rng.random() < 0.5:
        return cv2.GaussianBlur(img, (k, k), 0)
    ker = np.zeros((k, k), np.float32); ker[k // 2, :] = 1.0          # motion blur
    R = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), rng.uniform(0, 180), 1)
    ker = cv2.warpAffine(ker, R, (k, k)); ker /= max(ker.sum(), 1e-6)
    return cv2.filter2D(img, -1, ker)


PHOTO = [p_bright_contrast, p_hsv, p_noise, p_blur]


# ---------- geometric (labels recomputed) ----------
def g_hflip(img, b):
    return cv2.flip(img, 1), [(c, 1 - x, y, w, h) for c, x, y, w, h in b]


def g_vflip(img, b):
    return cv2.flip(img, 0), [(c, x, 1 - y, w, h) for c, x, y, w, h in b]


def g_rot90(img, b):   # clockwise: (x, y) -> (1-y, x); w/h swap
    return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE), [(c, 1 - y, x, h, w) for c, x, y, w, h in b]


def g_rot180(img, b):
    return cv2.rotate(img, cv2.ROTATE_180), [(c, 1 - x, 1 - y, w, h) for c, x, y, w, h in b]


def g_rot270(img, b):  # counter-clockwise: (x, y) -> (y, 1-x); w/h swap
    return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE), [(c, y, 1 - x, h, w) for c, x, y, w, h in b]


def g_affine(img, b, rng, min_vis=0.4, min_px=4):
    """Random scale + translation via warpAffine; boxes via transformed corners, clipped."""
    S = img.shape[0]
    s = rng.uniform(0.7, 1.3)
    tx, ty = rng.uniform(-0.15, 0.15, 2) * S
    M = np.array([[s, 0, (1 - s) * S / 2 + tx], [0, s, (1 - s) * S / 2 + ty]], np.float32)
    out = cv2.warpAffine(img, M, (S, S), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=FILL)
    nb = []
    for c, x, y, w, h in b:
        x1, y1, x2, y2 = (x - w / 2) * S, (y - h / 2) * S, (x + w / 2) * S, (y + h / 2) * S
        pts = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]]) @ M[:, :2].T + M[:, 2]
        nx1, ny1 = pts.min(0); nx2, ny2 = pts.max(0)
        full = (nx2 - nx1) * (ny2 - ny1)
        cx1, cy1, cx2, cy2 = np.clip([nx1, ny1, nx2, ny2], 0, S)       # clip to image
        if (cx2 - cx1) < min_px or (cy2 - cy1) < min_px or full <= 0:
            continue
        if (cx2 - cx1) * (cy2 - cy1) / full < min_vis:                 # mostly cropped out
            continue
        nb.append((c, (cx1 + cx2) / 2 / S, (cy1 + cy2) / 2 / S, (cx2 - cx1) / S, (cy2 - cy1) / S))
    return out, nb


GEO_FIXED = [g_hflip, g_vflip, g_rot90, g_rot180, g_rot270]


def augment(img, boxes, rng):
    for _ in range(10):  # retry if the object gets cropped away
        ops = []
        out, b = img, boxes
        for f in rng.choice(PHOTO, size=int(rng.integers(2, 4)), replace=False):
            out = f(out, rng); ops.append(f.__name__)
        geo = []
        if rng.random() < 0.8:
            geo.append(GEO_FIXED[int(rng.integers(len(GEO_FIXED)))])
        if rng.random() < 0.5 or not geo:
            geo.append(g_affine)
        for g in geo:
            out, b = g(out, b, rng) if g is g_affine else g(out, b)
            ops.append(g.__name__)
        if b or not boxes:
            return out, [tuple(map(lambda v: min(max(v, 0.0), 1.0) if isinstance(v, float) else v, bb)) for bb in b], ops
    return img, boxes, ["none"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", default="dataset")
    ap.add_argument("--size", type=int, default=640)
    ap.add_argument("--variants", type=int, default=5, help="augmented copies per base image")
    ap.add_argument("--names", nargs="+", default=["object"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split", type=float, nargs=3, default=[0.7, 0.2, 0.1])
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    imgs = list_images(args.images)
    random.Random(args.seed).shuffle(imgs)
    n = len(imgs)
    n_tr, n_va = round(n * args.split[0]), round(n * args.split[1])
    splits = {"train": imgs[:n_tr], "val": imgs[n_tr:n_tr + n_va], "test": imgs[n_tr + n_va:]}

    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    log = [("file", "split", "base", "ops")]
    for sp, files in splits.items():
        (out / sp / "images").mkdir(parents=True); (out / sp / "labels").mkdir(parents=True)
        for p in files:
            img = cv2.imread(str(p))
            boxes = read_yolo(Path(args.labels) / f"{p.stem}.txt")
            base, base_b = letterbox(img, boxes, args.size)
            items = [(f"{p.stem}_orig", base, base_b, ["letterbox"])]
            for k in range(args.variants):
                a, ab, ops = augment(base, base_b, rng)
                items.append((f"{p.stem}_aug{k}", a, ab, ops))
            for name, im, bb, ops in items:
                cv2.imwrite(str(out / sp / "images" / f"{name}.jpg"), im, [cv2.IMWRITE_JPEG_QUALITY, 95])
                write_yolo(out / sp / "labels" / f"{name}.txt", bb)
                log.append((f"{name}.jpg", sp, p.name, "+".join(ops)))
        print(f"{sp}: {len(files)} base images -> {len(files) * (args.variants + 1)} images")

    with open(out / "augmentation_log.csv", "w", newline="") as f:
        csv.writer(f).writerows(log)
    names = "\n".join(f"  {i}: {nm}" for i, nm in enumerate(args.names))
    (out / "data.yaml").write_text(
        f"# Paths are relative to this file's folder. In Colab/Drive, add:  path: /content/drive/MyDrive/<your>/dataset\n"
        f"train: train/images\nval: val/images\ntest: test/images\n\nnc: {len(args.names)}\nnames:\n{names}\n")
    print(f"Total images: {len(log) - 1}  ->  {out}")


if __name__ == "__main__":
    main()
