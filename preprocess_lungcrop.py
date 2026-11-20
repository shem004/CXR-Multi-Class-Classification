#!/usr/bin/env python
"""
Lung-ROI cropping for the Lab3 CXR 4-class competition (Normal / Bacteria / Virus / COVID-19).

Pipeline per image
  1. Load (.jpeg/.jpg/.png), fix EXIF orientation, convert ANY mode (L, RGB, RGBA, P, LA, I;16, CMYK)
     to 8-bit grayscale. Alpha is composited onto black, colour tints from screenshots are removed.
  2. Segment anatomy with TorchXRayVision's ChestX-Det PSPNet (14 structures; we use Left/Right Lung,
     Heart, Spine, Mediastinum). Input is letterboxed (pad-to-square) so anatomy is never stretched.
  3. Build a lung bounding box with fallbacks for diseased "white-out" lungs:
       a) both lungs found               -> union(left lung, right lung, heart)
       b) one lung missing / much smaller -> mirror the healthy lung across the spine/mediastinum midline
       c) nothing usable, or box too small / weird aspect -> centre crop
  4. Add a margin (max(margin_px, margin_frac * box side)), optionally expand to a square box
     (pixels outside the image are filled black), crop, resize (INTER_AREA), save 3-channel RGB .jpg.
  5. Write manifest.csv (one row per image: label, crop box, method used, areas) + debug overlays.

Hidden/macOS files (.DS_Store, ._AppleDouble, __MACOSX/) are skipped. CLAHE and augmentation are
NOT baked in here on purpose - they live in cxr_data.py so they can be ablated at training time.

Colab:   !pip -q install torchxrayvision
         !python preprocess_lungcrop.py --data_root /content/data --out_root /content/processed_data
Kaggle:  (Settings -> Internet ON, needed once to download the segmentation weights)
         !pip -q install torchxrayvision
         !python preprocess_lungcrop.py --data_root /kaggle/input/<dataset> --out_root /kaggle/working/processed_data
"""

import argparse
import csv
import os
import random
import sys
import time
import warnings
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps

try:
    from tqdm.auto import tqdm
except ImportError:  # minimal fallback so the script never breaks
    def tqdm(it, total=None, **kw):
        return it

IMG_EXTS = {".jpeg", ".jpg", ".png"}
SEG_SIZE = 512  # PSPNet native resolution


# --------------------------------------------------------------------------------------
# File discovery
# --------------------------------------------------------------------------------------
def is_hidden(p: Path) -> bool:
    """True for .DS_Store, macOS '._xxx' resource forks, and anything inside hidden/__MACOSX dirs."""
    return any(part.startswith(".") or part == "__MACOSX" for part in p.parts)


def canonical_split(folder_name: str) -> str:
    n = folder_name.lower()
    for s in ("train", "val", "test"):
        if s in n:  # 'test_images' -> test, 'validation' -> val
            return s
    return folder_name


def discover(data_root: Path, splits, exclude=None):
    """Return list of dicts: split, label, src (abs Path), rel (Path relative to data_root).
    label = name of the image's parent folder (train/<class>/x.jpeg, also train/train/<class>/x.jpeg);
    empty for flat folders such as test/x.jpeg."""
    if not splits:  # auto-detect: every visible sub-folder of data_root
        splits = []
        for d in sorted(data_root.iterdir()):
            if not d.is_dir() or is_hidden(Path(d.name)) or d.name.startswith("_"):  # _debug etc.
                continue
            if exclude is not None and d.resolve() == exclude.resolve():
                continue
            if (d / "manifest.csv").exists() or (d / "_debug").is_dir():  # output of an earlier run
                print(f"[info] skipping '{d.name}' (looks like a previous preprocessing output)")
                continue
            splits.append(d.name)
        print(f"Splits detected: {splits}")
    items = []
    for split in splits:
        split_dir = data_root / split
        if not split_dir.is_dir():
            print(f"[warn] split folder not found, skipping: {split_dir}")
            continue
        for p in sorted(split_dir.rglob("*")):
            rel = p.relative_to(data_root)
            if is_hidden(rel) or not p.is_file():
                continue
            if p.suffix.lower() not in IMG_EXTS:
                continue
            label = p.parent.name if p.parent != split_dir else ""
            if canonical_split(label) == canonical_split(split):  # double-nested test/test/x.jpeg
                label = ""
            items.append({"split": canonical_split(split), "label": label, "src": p, "rel": rel})
    return items


def assign_output_paths(items, out_root: Path):
    """<out_root>/<same relative dir>/<stem>.jpg ; resolves clashes like a.png + a.jpeg in one folder."""
    used = set()
    for it in items:
        rel_dir = it["rel"].parent
        cand = rel_dir / (it["src"].stem + ".jpg")
        if cand in used:  # clash -> keep original extension in the name
            cand = rel_dir / (it["src"].stem + "_" + it["src"].suffix.lower().lstrip(".") + ".jpg")
        used.add(cand)
        it["out_rel"] = cand
        it["dst"] = out_root / cand


# --------------------------------------------------------------------------------------
# Image loading
# --------------------------------------------------------------------------------------
def load_gray_uint8(path: Path) -> np.ndarray:
    """Any PIL mode -> HxW uint8 grayscale."""
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        if im.mode in ("I;16", "I;16B", "I;16L", "I", "F"):
            a = np.asarray(im, dtype=np.float32)
            lo, hi = np.percentile(a, 0.5), np.percentile(a, 99.5)
            a = np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1) * 255.0
            return a.astype(np.uint8)
        if im.mode == "P":
            im = im.convert("RGBA")
        if im.mode in ("RGBA", "LA"):
            bg = Image.new("RGBA", im.size, (0, 0, 0, 255))      # alpha -> composite on black
            im = Image.alpha_composite(bg, im.convert("RGBA"))
        return np.asarray(im.convert("L"), dtype=np.uint8).copy()


def letterbox(gray: np.ndarray, size: int = SEG_SIZE, max_aspect: float = 1.5):
    """Prepare the 512x512 segmentation input.
    Near-normal films (aspect <= max_aspect): pad to square (no distortion).
    Very wide/tall films (often stored stretched): resize anisotropically so the long side is at most
    max_aspect x the short side, then pad -> anatomy is big enough for the segmenter.
    Returns img, sx, sy, pad_x, pad_y, new_w, new_h  (seg_coord = orig_coord * s + pad)."""
    h, w = gray.shape
    ar = w / h
    if ar > max_aspect:          # too wide -> squeeze width
        nw, nh = size, max(1, round(size / max_aspect))
    elif ar < 1 / max_aspect:    # too tall -> squeeze height
        nh, nw = size, max(1, round(size / max_aspect))
    else:
        scale = size / max(h, w)
        nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    sx, sy = nw / w, nh / h
    resized = cv2.resize(gray, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((size, size), dtype=np.uint8)
    py, px = (size - nh) // 2, (size - nw) // 2
    canvas[py:py + nh, px:px + nw] = resized
    return canvas, sx, sy, px, py, nw, nh


class SegDataset(torch.utils.data.Dataset):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        try:
            gray = load_gray_uint8(it["src"])
            if gray.shape[0] < 32 or gray.shape[1] < 32:
                raise ValueError(f"image too small {gray.shape}")
        except Exception as e:  # corrupted / unreadable file
            return {"idx": i, "ok": False, "err": repr(e)}
        lb, sx, sy, px, py, nw, nh = letterbox(gray)
        # TorchXRayVision convention: [0,255] -> [-1024, 1024]
        x = torch.from_numpy(lb).float().div(255.0).mul(2048.0).sub(1024.0)[None]
        return {"idx": i, "ok": True, "x": x, "gray": gray, "geo": (sx, sy, px, py, nw, nh)}


def collate(batch):
    return batch  # variable-size originals -> keep as list


# --------------------------------------------------------------------------------------
# Mask -> box logic
# --------------------------------------------------------------------------------------
def largest_cc(mask: np.ndarray, keep_frac: float = 0.15) -> np.ndarray:
    """Keep the largest connected component (+ any component >= keep_frac of it)."""
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if n <= 1:
        return np.zeros_like(mask, dtype=bool)
    areas = stats[1:, cv2.CC_STAT_AREA]
    big = areas.max()
    keep = [k + 1 for k, a in enumerate(areas) if a >= keep_frac * big]
    return np.isin(lab, keep)


def mask_box(m: np.ndarray):
    ys, xs = np.nonzero(m)
    if len(xs) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def union(a, b):
    if a is None:
        return b
    if b is None:
        return a
    return [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]


def compute_box(probs: np.ndarray, targets, geo, orig_hw, args):
    """probs: [14, 512, 512] sigmoid outputs. Returns (box in original px, method, info dict)."""
    sx, sy, px, py, nw, nh = geo
    H, W = orig_hw
    valid = np.zeros((SEG_SIZE, SEG_SIZE), dtype=bool)
    valid[py:py + nh, px:px + nw] = True  # ignore predictions on the padding

    img_area_seg = nw * nh

    def get(name, thr=args.thr, min_frac=0.0):
        m = largest_cc((probs[targets.index(name)] > thr) & valid)
        return m if m.sum() >= min_frac * img_area_seg else np.zeros_like(m)

    def lungs_ok(a, b):
        return (a >= args.min_lung_frac * img_area_seg and b >= args.min_lung_frac * img_area_seg
                and min(a, b) >= args.lung_ratio * max(a, b))

    lungA, lungB = get("Left Lung"), get("Right Lung")
    aA, aB = int(lungA.sum()), int(lungB.sum())
    info = {"area_L": aA / img_area_seg, "area_R": aB / img_area_seg}
    if not lungs_ok(aA, aB):
        # opacified lung tissue often gets low (not zero) probability -> retry with a looser threshold
        lA2, lB2 = get("Left Lung", args.thr_low), get("Right Lung", args.thr_low)
        if lungs_ok(int(lA2.sum()), int(lB2.sum())):
            lungA, lungB, aA, aB = lA2, lB2, int(lA2.sum()), int(lB2.sum())
            info["low_thr"] = 1
    heart = get("Heart", min_frac=0.005)

    box, method = None, "center"
    if lungs_ok(aA, aB):
        box, method = union(mask_box(lungA), mask_box(lungB)), "lungs"
    elif max(aA, aB) >= args.min_mirror_lung_frac * img_area_seg:  # mirror only from a real lung, not a fragment
        # one lung missing / much smaller (white-out): mirror the healthy lung across the midline
        good, weak = (lungA, lungB) if aA >= aB else (lungB, lungA)
        gb = mask_box(good)
        spine, medi = get("Spine", min_frac=0.002), get("Mediastinum", min_frac=0.002)
        mid_src = spine if spine.any() else medi
        mid = float(np.nonzero(mid_src)[1].mean()) if mid_src.any() else px + nw / 2.0
        mirrored = [int(2 * mid - gb[2]), gb[1], int(2 * mid - gb[0]), gb[3]]
        box = union(union(gb, mirrored), mask_box(weak) if weak.any() else None)
        # the surviving lung may itself be partly opacified -> take vertical extent from stable anatomy:
        # top = clavicles / mediastinum top, bottom = mediastinum / heart bottom
        anchors = [mask_box(get(n, min_frac=0.001)) for n in ("Left Clavicle", "Right Clavicle", "Mediastinum")]
        anchors = [a for a in anchors if a is not None] + ([mask_box(heart)] if heart.any() else [])
        if anchors:
            box[1] = min([box[1]] + [a[1] for a in anchors])
            box[3] = max([box[3]] + [a[3] for a in anchors])
        method = "mirror"
    if box is not None and heart.any():
        box = union(box, mask_box(heart))  # heart stays visible in white-out, extends to lung bases

    if box is not None:
        # seg (512 letterbox) coords -> original image coords
        x0 = (box[0] - px) / sx
        y0 = (box[1] - py) / sy
        x1 = (box[2] - px) / sx
        y1 = (box[3] - py) / sy
        x0, y0 = max(0.0, x0), max(0.0, y0)
        x1, y1 = min(float(W), x1), min(float(H), y1)
        bw, bh = x1 - x0, y1 - y0
        frac = (bw * bh) / (W * H)
        aspect = bw / max(bh, 1e-6)
        rel_aspect = aspect / (W / H)  # stretched films stretch the box too -> compare to the image itself
        info["raw_box_frac"] = frac
        info["box_aspect"] = aspect
        too_short = method == "mirror" and bh / H < args.min_mirror_h_frac
        if frac < args.min_box_frac or too_short or not (args.min_rel_aspect <= rel_aspect <= args.max_rel_aspect):
            box, method = None, "center"
        else:
            box = [x0, y0, x1, y1]
    else:
        info["raw_box_frac"] = 0.0

    if box is None:
        cw, ch = W * args.center_frac, H * args.center_frac
        box = [(W - cw) / 2, (H - ch) / 2, (W + cw) / 2, (H + ch) / 2]
        return box, "center", info  # no extra margin on centre crop

    # margin: max(fixed px, fraction of box side), per axis
    bw, bh = box[2] - box[0], box[3] - box[1]
    mx = max(args.margin_px, args.margin_frac * bw)
    my = max(args.margin_px, args.margin_frac * bh)
    box = [max(0.0, box[0] - mx), max(0.0, box[1] - my), min(float(W), box[2] + mx), min(float(H), box[3] + my)]
    return box, method, info


def squarify(box, W, H):
    """Grow the short side to make a square, then slide the window back inside the image when possible,
    so we only pad with black when the image itself is too small in that direction."""
    x0, y0, x1, y1 = box
    side = max(x1 - x0, y1 - y0)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    x0, x1 = cx - side / 2, cx + side / 2
    y0, y1 = cy - side / 2, cy + side / 2
    for lo, hi, lim in ((0, 2, W), (1, 3, H)):
        b = [x0, y0, x1, y1]
        if side <= lim:
            if b[lo] < 0:
                b[hi] -= b[lo]; b[lo] = 0.0
            if b[hi] > lim:
                b[lo] -= b[hi] - lim; b[hi] = float(lim)
        else:  # image too small along this axis -> centre on the image, pad both sides
            c = lim / 2
            b[lo], b[hi] = c - side / 2, c + side / 2
        x0, y0, x1, y1 = b
    return [x0, y0, x1, y1]


def thorax_mask(probs, targets, geo, orig_hw, method, args):
    """Soft mask (H x W, values 0..1) of the chest contents: convex hull of both lungs + heart
    (for 'mirror' the healthy lung mirrored across the spine stands in for the missing one),
    generously dilated and blurred so that opacified lung at the edges is never darkened.
    None for 'center' (nothing reliable to mask)."""
    if method not in ("lungs", "mirror"):
        return None
    sx, sy, px, py, nw, nh = geo
    H, W = orig_hw
    valid = np.zeros((SEG_SIZE, SEG_SIZE), dtype=bool)
    valid[py:py + nh, px:px + nw] = True
    get = lambda n, t=args.thr: largest_cc((probs[targets.index(n)] > t) & valid)  # noqa: E731
    lA, lB, heart = get("Left Lung"), get("Right Lung"), get("Heart")
    if min(lA.sum(), lB.sum()) < args.lung_ratio * max(lA.sum(), lB.sum()):
        lA, lB = get("Left Lung", args.thr_low), get("Right Lung", args.thr_low)
    union_m = lA | lB | heart
    if method == "mirror":
        good = lA if lA.sum() >= lB.sum() else lB
        spine = get("Spine")
        mid = float(np.nonzero(spine)[1].mean()) if spine.sum() > 0.002 * nw * nh else px + nw / 2.0
        ys, xs = np.nonzero(good)
        xm = np.clip(np.round(2 * mid - xs).astype(int), 0, SEG_SIZE - 1)
        union_m[ys, xm] = True
    pts = np.argwhere(union_m)[:, ::-1].astype(np.int32)  # (x, y)
    if len(pts) < 10:
        return None
    # flat bottom at the lowest lung/heart point across the full chest width: lung bases and
    # costophrenic angles (where consolidation often hides from the segmenter) are never dimmed
    x0m, x1m, y1m = pts[:, 0].min(), pts[:, 0].max(), pts[:, 1].max()
    pts = np.vstack([pts, [[x0m, y1m], [x1m, y1m]]]).astype(np.int32)
    hull = np.zeros((SEG_SIZE, SEG_SIZE), dtype=np.uint8)
    cv2.fillConvexPoly(hull, cv2.convexHull(pts), 1)
    m = cv2.resize(hull[py:py + nh, px:px + nw].astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
    k = max(3, int(args.mask_dilate * max(W, H)) | 1)
    m = cv2.dilate((m > 0.5).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))).astype(np.float32)
    return cv2.GaussianBlur(m, (0, 0), sigmaX=max(1.0, k / 3))


def crop_resize(gray: np.ndarray, box, out_size: int, square: bool, max_pad_aspect: float = 1.5) -> np.ndarray:
    H, W = gray.shape
    bw, bh = box[2] - box[0], box[3] - box[1]
    if square and max(bw, bh) / max(min(bw, bh), 1e-6) <= max_pad_aspect:
        box = squarify(box, W, H)  # near-square box: pad (no distortion)
    # else: very elongated box (stretched film) -> resized straight to a square, which un-stretches it
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    out = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)  # out-of-image area = black
    sx0, sy0, sx1, sy1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
    out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = gray[sy0:sy1, sx0:sx1]
    # --no_square stretches the box to a square output (aspect distortion) - only for the ablation
    interp = cv2.INTER_AREA if max(out.shape) > out_size else cv2.INTER_LINEAR
    out = cv2.resize(out, (out_size, out_size), interpolation=interp)
    return np.repeat(out[..., None], 3, axis=2)  # 3-channel RGB (identical channels)


# --------------------------------------------------------------------------------------
# Debug overlay
# --------------------------------------------------------------------------------------
def save_debug(path: Path, gray, probs, targets, geo, box, method, thr, square=True):
    sx, sy, px, py, nw, nh = geo
    lb, *_ = letterbox(gray)
    vis = cv2.cvtColor(lb, cv2.COLOR_GRAY2BGR)
    colors = {"Left Lung": (0, 200, 0), "Right Lung": (0, 160, 255), "Heart": (255, 0, 200), "Spine": (200, 200, 0)}
    for name, c in colors.items():
        m = probs[targets.index(name)] > thr
        layer = vis.copy()
        layer[m] = c
        vis = cv2.addWeighted(layer, 0.35, vis, 0.65, 0)
    b = [int(round(box[0] * sx + px)), int(round(box[1] * sy + py)),
         int(round(box[2] * sx + px)), int(round(box[3] * sy + py))]
    cv2.rectangle(vis, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)  # red = lung box (+margin)
    bw, bh = box[2] - box[0], box[3] - box[1]
    if square and max(bw, bh) / max(min(bw, bh), 1e-6) <= 1.5:
        H, W = gray.shape
        s = squarify(box, W, H)
        s = [int(round(s[0] * sx + px)), int(round(s[1] * sy + py)),
             int(round(s[2] * sx + px)), int(round(s[3] * sy + py))]
        cv2.rectangle(vis, (s[0], s[1]), (s[2], s[3]), (255, 255, 0), 1)  # cyan = final saved crop
    cv2.putText(vis, method, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), vis)


def stratified_sample(items, n, seed=0, min_per_group=3):
    """Random subset that keeps every (split, label) group, rare classes (COVID) get >= min_per_group."""
    rng = random.Random(seed)
    groups = {}
    for it in items:
        groups.setdefault((it["split"], it["label"]), []).append(it)
    out = []
    for key, g in sorted(groups.items()):
        k = max(min_per_group, round(n * len(g) / len(items)))
        out += rng.sample(g, min(k, len(g)))
    print("Sample per split/class: " + ", ".join(f"{s}/{l or '-'}={sum(1 for o in out if (o['split'], o['label']) == (s, l))}"
                                              for s, l in sorted(groups)))
    return out


def make_contact_sheet(debug_dir: Path, out_path: Path, max_imgs=60, tile=256, cols=6, only=None):
    """One image with overlays (fallback cases first) so you can eyeball them in Colab.
    only = overlays written in this run (if any), otherwise everything in _debug."""
    if not debug_dir.is_dir():
        return
    order = {"center": 0, "mirror": 1, "lungs": 2}
    pool = [Path(f) for f in only] if only else list(debug_dir.glob("*/*.jpg"))
    files = sorted(pool, key=lambda f: (order.get(f.parent.name, 3), f.name))
    files = files[:max_imgs]
    if not files:
        return
    tiles = []
    for f in files:
        im = cv2.imread(str(f))
        if im is None:
            continue
        im = cv2.resize(im, (tile, tile))
        cv2.putText(im, f.name[:34], (4, tile - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)
        tiles.append(im)
    while len(tiles) % cols:
        tiles.append(np.zeros((tile, tile, 3), np.uint8))
    sheet = np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])
    cv2.imwrite(str(out_path), sheet)
    print(f"Contact sheet -> {out_path}  ({len(files)} overlays; red = lung box, cyan = saved crop)")


# --------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_root", required=True, help="folder containing train/ val/ test/")
    ap.add_argument("--out_root", default="processed_data")
    ap.add_argument("--splits", nargs="+", default=None,
                    help="sub-folders to process (default: all sub-folders, e.g. train val test / test_images)")
    ap.add_argument("--out_size", type=int, default=512,
                    help="saved size. 512 lets you train at 224/384/448 later; 224 locks you in.")
    ap.add_argument("--no_square", action="store_true", help="stretch the box instead of padding it to square")
    ap.add_argument("--margin_px", type=float, default=15)
    ap.add_argument("--margin_frac", type=float, default=0.05, help="margin as fraction of box side (wins if larger)")
    ap.add_argument("--thr", type=float, default=0.5, help="sigmoid threshold for masks")
    ap.add_argument("--thr_low", type=float, default=0.2, help="retry threshold when a lung looks missing")
    ap.add_argument("--lung_ratio", type=float, default=0.35,
                    help="if smaller lung < ratio * larger lung -> treat as white-out and mirror")
    ap.add_argument("--min_lung_frac", type=float, default=0.01, help="lung mask below this frac of image = missing")
    ap.add_argument("--min_mirror_lung_frac", type=float, default=0.05,
                    help="healthy lung must cover >= this frac of the image to be mirrored (healthy ~0.11-0.19)")
    ap.add_argument("--min_mirror_h_frac", type=float, default=0.45,
                    help="mirrored box must span >= this frac of image height, else centre crop")
    ap.add_argument("--min_box_frac", type=float, default=0.15,
                    help="final box smaller than this frac of the image -> centre crop (your proposal: 0.30)")
    ap.add_argument("--min_rel_aspect", type=float, default=0.4,
                    help="(box w/h) / (image w/h) below this -> centre crop")
    ap.add_argument("--max_rel_aspect", type=float, default=2.5)
    ap.add_argument("--max_pad_aspect", type=float, default=1.5,
                    help="boxes wider/taller than this ratio are resized to square instead of padded "
                         "(undoes horizontally stretched films instead of saving 60%% black padding)")
    ap.add_argument("--redo", nargs="+", default=[],
                    help="re-process rows of an existing manifest by method, e.g. --redo center skipped_existing wide "
                         "('wide' = saved boxes with aspect > --max_pad_aspect)")
    ap.add_argument("--center_frac", type=float, default=0.85, help="centre-crop keeps this frac of W and H")
    ap.add_argument("--jpeg_quality", type=int, default=95)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--debug_n", type=int, default=30, help="random overlays to save (+ all fallbacks, capped)")
    ap.add_argument("--debug_fallback_max", type=int, default=200)
    ap.add_argument("--limit", type=int, default=0, help="process only the first N images")
    ap.add_argument("--sample", type=int, default=0,
                    help="TRIAL RUN: random N images stratified over every split/class (each class >= 3); "
                         "saves an overlay for every image + _debug/contact_sheet.jpg. Use a separate --out_root.")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--mask_outside", type=float, default=1.0,
                    help="<1 = multiply everything outside the chest (lungs+heart hull) by this, e.g. 0.3; 1 = off")
    ap.add_argument("--mask_dilate", type=float, default=0.08, help="hull dilation as a fraction of the image size")
    ap.add_argument("--mask_only", action="store_true",
                    help="--data_root is an already-processed folder: segment each crop and dim outside the chest "
                         "(use with --mask_outside), no re-cropping, source manifest copied")
    ap.add_argument("--mask_root", default="", help="with --mask_outside: write the masked copy HERE and the normal "
                                                     "crop to --out_root (identical boxes)")
    ap.add_argument("--full_image", action="store_true",
                    help="NO-CROP baseline for the ablation: save the whole film (padded square) in the same format")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    try:
        import torchxrayvision as xrv
    except ImportError:
        sys.exit("pip install torchxrayvision")

    data_root, out_root = Path(args.data_root), Path(args.out_root)
    items = discover(data_root, args.splits, exclude=out_root)
    if args.limit:
        items = items[: args.limit]
    if args.sample:
        items = stratified_sample(items, args.sample, args.seed)
        args.debug_n = len(items)  # overlay for every sampled image
    if not items:
        sys.exit(f"No images found under {data_root} for splits {args.splits}")
    assign_output_paths(items, out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"Found {len(items)} images: " + ", ".join(f"{k}={v}" for k, v in Counter(i['split'] for i in items).items()))

    man_path = out_root / "manifest.csv"
    prev = {}
    if man_path.exists():
        with open(man_path, newline="") as f:
            prev = {r["orig_relpath"]: r for r in csv.DictReader(f)}

    def needs_redo(it):
        r = prev.get(it["rel"].as_posix())
        if r is None or not args.redo:
            return False
        if r["method"] in args.redo:
            return True
        if "wide" in args.redo and r.get("x1"):
            bw, bh = float(r["x1"]) - float(r["x0"]), float(r["y1"]) - float(r["y0"])
            return max(bw, bh) / max(min(bw, bh), 1e-6) > args.max_pad_aspect
        return False

    todo = [i for i, it in enumerate(items) if args.overwrite or not it["dst"].exists() or needs_redo(items[i])]
    if args.redo:
        args.debug_n = len(todo)  # overlay for every re-processed image
        for i in todo:  # remove stale overlays of these images
            for f in (out_root / "_debug").glob("*/" + str(items[i]["rel"]).replace(os.sep, "__") + ".jpg"):
                f.unlink()
        print(f"--redo {args.redo}: {len(todo)} images selected")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} | to process: {len(todo)} (existing outputs skipped unless --overwrite)")

    seg, targets = None, None
    if not args.full_image:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            seg = xrv.baseline_models.chestx_det.PSPNet().to(device).eval()
        targets = list(seg.targets)

    rng = random.Random(args.seed)
    debug_ids = set(rng.sample(todo, min(args.debug_n, len(todo))))
    n_fallback_dbg = 0
    written_dbg = []

    ds = SegDataset([items[i] for i in todo])
    dl = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, num_workers=args.num_workers,
                                     collate_fn=collate, pin_memory=(device == "cuda"))

    results = {}
    t0 = time.time()
    counts = Counter()
    pbar = tqdm(dl, total=len(dl), desc="Lung-crop", unit="batch", dynamic_ncols=True)
    for batch in pbar:
        good = [b for b in batch if b["ok"]]
        for b in batch:
            if not b["ok"]:
                gi = todo[b["idx"]]
                results[gi] = {"method": "failed", "error": b["err"]}
                counts["failed"] += 1
                getattr(tqdm, "write", print)(f"[error] {items[gi]['rel']}: {b['err']}")
        if good and args.full_image:  # no-crop baseline: whole film, padded to square, same format
            for b in good:
                gi = todo[b["idx"]]
                it = items[gi]
                gray = b["gray"]
                box = [0.0, 0.0, float(gray.shape[1]), float(gray.shape[0])]
                img = crop_resize(gray, box, args.out_size, square=True, max_pad_aspect=args.max_pad_aspect)
                it["dst"].parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(it["dst"]), img, [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality])
                results[gi] = {"method": "full", "box": box, "W": gray.shape[1], "H": gray.shape[0],
                               "area_L": 0.0, "area_R": 0.0}
                counts["full"] += 1
        elif good:
            x = torch.stack([b["x"] for b in good]).to(device)
            with torch.no_grad(), torch.autocast(device_type="cuda", enabled=(device == "cuda")):
                logits = seg(x)
            probs = torch.sigmoid(logits.float()).cpu().numpy()
            for b, p in zip(good, probs):
                gi = todo[b["idx"]]
                it = items[gi]
                gray = b["gray"]
                box, method, info = compute_box(p, targets, b["geo"], gray.shape, args)
                if args.mask_only:  # input is already a crop: keep its framing, only dim outside the chest
                    box = [0.0, 0.0, float(gray.shape[1]), float(gray.shape[0])]
                gray_src = gray
                if args.mask_outside < 1.0:  # lung-attenuated variant for the "right signal" experiment
                    tm = thorax_mask(p, targets, b["geo"], gray.shape, method, args)
                    gray_m = gray if tm is None else \
                        (gray.astype(np.float32) * (args.mask_outside + (1 - args.mask_outside) * tm)).astype(np.uint8)
                    if args.mask_root:  # same box, written to a second folder -> clean A/B comparison
                        mdst = Path(args.mask_root) / it["out_rel"]
                        mdst.parent.mkdir(parents=True, exist_ok=True)
                        cv2.imwrite(str(mdst), crop_resize(gray_m, box, args.out_size, square=not args.no_square,
                                                           max_pad_aspect=args.max_pad_aspect),
                                    [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality])
                    else:
                        gray = gray_m
                img = crop_resize(gray, box, args.out_size, square=not args.no_square, max_pad_aspect=args.max_pad_aspect)
                gray = gray_src
                it["dst"].parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(it["dst"]), img, [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality])
                results[gi] = {"method": method, "box": box, "W": gray.shape[1], "H": gray.shape[0], **info}
                counts[method] += 1
                is_fb = method != "lungs"
                if gi in debug_ids or (is_fb and n_fallback_dbg < args.debug_fallback_max):
                    n_fallback_dbg += int(is_fb and gi not in debug_ids)
                    dbg = out_root / "_debug" / method / (str(it["rel"]).replace(os.sep, "__") + ".jpg")
                    save_debug(dbg, gray, p, targets, b["geo"], box, method, args.thr, not args.no_square)
                    written_dbg.append(dbg)
        if hasattr(pbar, "set_postfix"):
            pbar.set_postfix(lungs=counts["lungs"], mirror=counts["mirror"], center=counts["center"],
                             failed=counts["failed"])
    if todo:
        el = time.time() - t0
        print(f"Processed {len(todo)} images in {el / 60:.1f} min ({len(todo) / max(el, 1e-6):.1f} img/s)")
    make_contact_sheet(out_root / "_debug", out_root / "_debug" / "contact_sheet.jpg", only=written_dbg)

    # ---------------- manifest (merge with any previous run) ----------------
    old = prev
    fields = ["split", "label", "orig_relpath", "out_relpath", "orig_w", "orig_h", "method",
              "x0", "y0", "x1", "y1", "box_frac", "raw_box_frac", "box_aspect", "area_L", "area_R", "low_thr", "error"]
    rows = []
    for gi, it in enumerate(items):
        key = it["rel"].as_posix()
        r = results.get(gi)
        if r is None and key in old:
            rows.append(old[key])
            continue
        if r is None:
            r = {"method": "skipped_existing"}
        row = {"split": it["split"], "label": it["label"], "orig_relpath": key,
               "out_relpath": it["out_rel"].as_posix(), "method": r["method"], "error": r.get("error", "")}
        if "box" in r:
            x0, y0, x1, y1 = r["box"]
            row.update(orig_w=r["W"], orig_h=r["H"], x0=round(x0), y0=round(y0), x1=round(x1), y1=round(y1),
                       box_frac=round((x1 - x0) * (y1 - y0) / (r["W"] * r["H"]), 4),
                       raw_box_frac=round(r.get("raw_box_frac", 0), 4),
                       area_L=round(r["area_L"], 4), area_R=round(r["area_R"], 4), low_thr=r.get("low_thr", 0),
                       box_aspect=round(r.get("box_aspect", 0), 3))
        rows.append(row)
    with open(man_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    if args.mask_only and (data_root / "manifest.csv").exists():
        import shutil  # keep the source manifest: same out_relpath / orig_relpath -> same labels and folds
        shutil.copy(data_root / "manifest.csv", man_path)
    if args.mask_root:
        import shutil
        shutil.copy(man_path, Path(args.mask_root) / "manifest.csv")

    # ---------------- summary ----------------
    print("\nMethod counts per split / label:")
    cnt = Counter((r["split"], r["label"] or "-", r["method"]) for r in rows)
    for k in sorted(cnt):
        print(f"  {k[0]:>5} | {k[1]:<10} | {k[2]:<16} {cnt[k]}")
    failed = [r for r in rows if r["method"] == "failed"]
    if failed:
        print(f"\n[!] {len(failed)} files failed to load (see 'error' column). "
              f"Test-set failures must be fixed manually before submission!")
    print(f"\nSaved crops -> {out_root}\nManifest   -> {man_path}\nDebug      -> {out_root / '_debug'}")


if __name__ == "__main__":
    main()
