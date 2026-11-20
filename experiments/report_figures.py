#!/usr/bin/env python
"""
Figures for the report (written to <out>/):
  fig_class_distribution.png   images per class and split
  fig_class_examples.png       lung crops, a row per class
  fig_pipeline.png             original | lung segmentation + box | crop | CLAHE | two training augmentations
  fig_fallbacks.png            the three crop methods (lungs / mirror / center) with their overlays
  fig_models.png               OOF macro-F1 and per-class F1 per model and for the ensemble (if tables exist)

python report_figures.py --img_root DATA --proc_root PROC --label_dir DATA --out OUT [--tables_dir WORK/ensembles/final]
"""

import argparse
import sys
import tempfile
import warnings
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))         # experiments/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root (shared modules)
import preprocess_lungcrop as pp  # noqa: E402
from cxr_data import CLASSES, apply_clahe_mode, build_transforms, load_manifest  # noqa: E402

MEAN, STD = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])


def overlay(seg, targets, src, row, device):
    """segmentation overlay (512 letterbox) with the saved crop box, as an RGB array"""
    gray = pp.load_gray_uint8(src)
    lb, sx, sy, px, py, nw, nh = pp.letterbox(gray)
    x = torch.from_numpy(lb).float().div(255).mul(2048).sub(1024)[None, None].to(device)
    with torch.no_grad():
        probs = torch.sigmoid(seg(x).float())[0].cpu().numpy()
    box = [float(row.x0), float(row.y0), float(row.x1), float(row.y1)]
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "o.jpg"
        pp.save_debug(f, gray, probs, targets, (sx, sy, px, py, nw, nh), box, row.method, 0.5, True)
        ov = cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2RGB)
    return cv2.cvtColor(lb, cv2.COLOR_GRAY2RGB), ov


def show(ax, img, title=None):
    ax.imshow(img, cmap="gray", vmin=0, vmax=255)
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img_root", required=True, help="folder that holds the original image folders")
    ap.add_argument("--proc_root", required=True)
    ap.add_argument("--label_dir", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tables_dir", default="")
    ap.add_argument("--seed", type=int, default=3)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    proc, img_root = Path(a.proc_root), Path(a.img_root)
    df = load_manifest(proc, ("train", "val"), label_dir=a.label_dir)
    df = df[df.label != ""]
    classes = [c for c in CLASSES if (df.label.str.lower() == c.lower()).any()]
    df["label"] = df.label.map(lambda l: next(c for c in CLASSES if c.lower() == l.lower()))
    rng = np.random.RandomState(a.seed)

    # 1. class distribution
    ct = pd.crosstab(df.label, df.split).reindex(classes)
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    bottom = np.zeros(len(ct))
    for sp, col in zip(ct.columns, ("#4c78a8", "#9ecae9")):
        ax.bar(ct.index, ct[sp], bottom=bottom, label=sp, color=col)
        bottom += ct[sp].to_numpy()
    for i, v in enumerate(bottom):
        ax.text(i, v + 20, f"{int(v)} ({v / bottom.sum():.1%})", ha="center", fontsize=9)
    ax.set_ylabel("images")
    ax.set_title("Labelled images per class (train + val)")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "fig_class_distribution.png", dpi=150)
    plt.close(fig)

    # 2. class examples (crops)
    n = 6
    fig, axes = plt.subplots(len(classes), n, figsize=(2.1 * n, 2.2 * len(classes)))
    axes = np.atleast_2d(axes)
    for r, c in enumerate(classes):
        sub = df[df.label == c]
        for k, (_, row) in enumerate(sub.sample(min(n, len(sub)), random_state=a.seed).iterrows()):
            show(axes[r, k], cv2.imread(str(proc / row.out_relpath), 0), f"{c}  ({Path(row.orig_relpath).name})" if k == 0 else Path(row.orig_relpath).name)
        for k in range(len(sub.sample(min(n, len(sub)), random_state=a.seed)), n):
            axes[r, k].axis("off")
    fig.suptitle("Lung crops by class", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "fig_class_examples.png", dpi=130)
    plt.close(fig)

    # segmentation model (same one as the preprocessing)
    import torchxrayvision as xrv
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        seg = xrv.baseline_models.chestx_det.PSPNet().to(device).eval()
    targets = list(seg.targets)
    tf = build_transforms(384, train=True)

    # 3. pipeline, one row per class
    cols = ["original (padded to square)", "lung segmentation + crop box", "lung crop (512 px)", "CLAHE", "augmented (train)", "augmented (train)"]
    fig, axes = plt.subplots(len(classes), 6, figsize=(13.5, 2.35 * len(classes)))
    axes = np.atleast_2d(axes)
    for r, c in enumerate(classes):
        sub = df[(df.label == c) & (df.method == "lungs")]
        sub = sub if len(sub) else df[df.label == c]
        row = sub.sample(1, random_state=a.seed + r).iloc[0]
        orig, ov = overlay(seg, targets, img_root / row.orig_relpath, row, device)
        crop = cv2.imread(str(proc / row.out_relpath), 0)
        cl = apply_clahe_mode(crop, "clahe")
        imgs = [orig, ov, crop, cl]
        for _ in range(2):
            t = tf(cl).permute(1, 2, 0).numpy() * STD + MEAN
            imgs.append((np.clip(t, 0, 1) * 255).astype(np.uint8))
        for k, im in enumerate(imgs):
            show(axes[r, k], im, cols[k] if r == 0 else None)
        axes[r, 0].text(-0.08, 0.5, c, transform=axes[r, 0].transAxes, rotation=90, va="center", ha="right", fontsize=10)
    fig.suptitle("Preprocessing and augmentation pipeline (red box = lung box with margin, cyan = saved crop)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "fig_pipeline.png", dpi=130)
    plt.close(fig)

    # 4. crop methods
    methods = [m for m in ("lungs", "mirror", "center") if (df.method == m).any()]
    fig, axes = plt.subplots(2, 2 * len(methods), figsize=(5.2 * len(methods), 5.4))
    axes = np.atleast_2d(axes)
    for j, m in enumerate(methods):
        sub = df[df.method == m]
        for r, (_, row) in enumerate(sub.sample(min(2, len(sub)), random_state=a.seed).iterrows()):
            _, ov = overlay(seg, targets, img_root / row.orig_relpath, row, device)
            show(axes[r, 2 * j], ov, f"{m}: {len(sub)} images" if r == 0 else None)
            show(axes[r, 2 * j + 1], cv2.imread(str(proc / row.out_relpath), 0), "saved crop" if r == 0 else None)
    for ax in axes.ravel():
        ax.axis("off")
    fig.suptitle("Crop methods: both lungs found / one lung mirrored (white-out) / centre-crop fallback", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "fig_fallbacks.png", dpi=130)
    plt.close(fig)

    # 5. model comparison
    td = Path(a.tables_dir) if a.tables_dir else None
    if td and (td / "models_table.csv").exists():
        m = pd.read_csv(td / "models_table.csv").rename(columns={"model": "name"})
        rt = pd.read_csv(td / "rules_table.csv").rename(columns={"rule": "name"})
        rt = rt[rt.name.str.startswith("1_")].assign(name="ensemble (argmax)")
        t = pd.concat([m, rt[m.columns]], ignore_index=True)
        metrics = ["oof_macro_f1"] + CLASSES
        fig, ax = plt.subplots(figsize=(9, 3.8))
        w = 0.8 / len(t)
        for i, row in t.iterrows():
            xs = np.arange(len(metrics)) + i * w
            ax.bar(xs, [row[k] for k in metrics], w, label=row["name"])
            for x, k in zip(xs, metrics):
                ax.text(x, row[k] + 0.004, f"{row[k]:.3f}", ha="center", fontsize=6.5, rotation=90)
        ax.set_xticks(np.arange(len(metrics)) + 0.4 - w / 2, ["macro-F1"] + CLASSES)
        ax.set_ylim(0.6, 1.03)
        ax.set_ylabel("out-of-fold F1")
        ax.set_title("Models and ensemble, out-of-fold")
        ax.legend(fontsize=8, ncol=len(t), loc="upper center")
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(out / "fig_models.png", dpi=150)
        plt.close(fig)
    print("figures ->", out, sorted(p.name for p in out.glob("fig_*.png")))


if __name__ == "__main__":
    main()
