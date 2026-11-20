"""
Training-time data module for the Lab3 CXR competition. Reads the manifest.csv written by
preprocess_lungcrop.py.

Contents
  * CLAHE as a deterministic *preprocessing* step (applied identically at train AND test time),
    with three modes so it can be ablated:  none | clahe | stack
        stack = 3 input channels [raw, CLAHE(clip 2), CLAHE(clip 4)] -> gives ImageNet-pretrained
                backbones different contrast views instead of 3 identical channels.
  * Train augmentation (mild, anatomy-preserving): affine (rot +-10, shift 6%, scale 0.9-1.1),
    brightness/contrast jitter, optional horizontal flip, optional light RandomErasing.
  * Batch-level Mixup (alpha 0.2) + soft-target cross-entropy. CutMix deliberately left out.
  * CXRDataset reading the manifest.

Preview the augmentations (good figure for the report):
    python cxr_data.py --root processed_data --preview aug_preview.jpg --clahe stack
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torchvision.transforms import v2 as T

CLASSES = ["normal", "bacteria", "virus", "COVID-19"]  # column order of the submission CSV
_ALIASES = {"normal": 0, "bacteria": 1, "bacterial": 1, "virus": 2, "viral": 2,
            "covid-19": 3, "covid19": 3, "covid": 3, "covid_19": 3}
IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def label_to_idx(name: str) -> int:
    key = str(name).strip().lower()
    if key not in _ALIASES:
        raise ValueError(f"Unknown class folder '{name}'. Add it to _ALIASES in cxr_data.py")
    return _ALIASES[key]


# ----------------------------------------------------------------------------------------
# Intensity preprocessing (deterministic)
# ----------------------------------------------------------------------------------------
def apply_clahe_mode(gray: np.ndarray, mode: str = "clahe", clip: float = 2.0, tile: int = 8) -> np.ndarray:
    """uint8 HxW -> uint8 HxWx3."""
    if mode == "none":
        return np.repeat(gray[..., None], 3, axis=2)
    c1 = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile)).apply(gray)
    if mode == "clahe":
        return np.repeat(c1[..., None], 3, axis=2)
    if mode == "stack":
        c2 = cv2.createCLAHE(clipLimit=2 * clip, tileGridSize=(tile, tile)).apply(gray)
        return np.stack([gray, c1, c2], axis=2)
    raise ValueError(f"clahe mode must be none|clahe|stack, got {mode}")


# ----------------------------------------------------------------------------------------
# Geometric / photometric augmentation (tensor based, torchvision v2)
# ----------------------------------------------------------------------------------------
def build_transforms(img_size: int = 384, train: bool = True, hflip: bool = False,
                     erasing: float = 0.25, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    ops = [T.ToImage(), T.Resize((img_size, img_size), antialias=True)]
    if train:
        ops += [
            T.RandomApply([T.RandomAffine(degrees=10, translate=(0.06, 0.06), scale=(0.9, 1.1), fill=0)], p=0.8),
            T.RandomApply([T.ColorJitter(brightness=0.15, contrast=0.15)], p=0.5),
        ]
        if hflip:
            ops.append(T.RandomHorizontalFlip(p=0.5))
    ops += [T.ToDtype(torch.float32, scale=True), T.Normalize(mean, std)]
    if train and erasing > 0:
        black = [-m / sd for m, sd in zip(mean, std)]  # erase to black, not to the ImageNet-mean grey
        ops.append(T.RandomErasing(p=erasing, scale=(0.02, 0.08), ratio=(0.5, 2.0), value=black))
    return T.Compose(ops)


# ----------------------------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------------------------
class CXRDataset(torch.utils.data.Dataset):
    """df rows come from manifest.csv. Returns (image_tensor, label_idx or -1, out_relpath)."""

    def __init__(self, df: pd.DataFrame, root, transform, clahe: str = "clahe", cache: bool = False):
        self.df = df.reset_index(drop=True)
        self.root = Path(root)
        self.transform = transform
        self.clahe = clahe
        self.labels = [label_to_idx(l) if (l and sp != "test") else -1
                       for l, sp in zip(self.df["label"].fillna(""), self.df["split"])]
        self._cache = {} if cache else None

    def __len__(self):
        return len(self.df)

    def _load(self, i):
        if self._cache is not None and i in self._cache:
            return self._cache[i]
        p = self.root / self.df.loc[i, "out_relpath"]
        gray = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise FileNotFoundError(p)
        img = apply_clahe_mode(gray, self.clahe)
        if self._cache is not None:
            self._cache[i] = img
        return img

    def __getitem__(self, i):
        x = self.transform(self._load(i))  # HxWx3 uint8 numpy -> CxHxW float tensor
        return x, self.labels[i], self.df.loc[i, "out_relpath"]


def _csv_split(name: str):
    n = name.lower()
    for sp in ("train", "val", "test"):
        if sp in n:
            return sp
    return None


def read_label_csvs(label_dir):
    """Find the competition CSVs (e.g. train_data.csv / val_data.csv) and return
    {(split or None, filename): class_name}. Supports one-hot columns (normal, bacteria, virus, COVID-19)
    or a single label/class column. Test/sample-submission CSVs are ignored."""
    label_dir = Path(label_dir)
    out, used = {}, []
    for f in sorted(label_dir.rglob("*.csv")):
        if any(part.startswith(".") or part == "__MACOSX" for part in f.parts) or f.name == "manifest.csv":
            continue
        sp = _csv_split(f.name)
        if sp == "test" or "sample" in f.name.lower() or "submission" in f.name.lower():
            continue
        df = pd.read_csv(f)
        cols = {c.strip().lower(): c for c in df.columns}
        fcol = cols.get("new_filename") or cols.get("filename") or cols.get("file") or cols.get("image") \
            or df.columns[0]
        onehot = [cols.get(c.lower()) for c in CLASSES]
        if all(onehot):
            idx = df[onehot].to_numpy().argmax(1)
            bad = (df[onehot].to_numpy().sum(1) != 1).sum()
            if bad:
                print(f"[warn] {f.name}: {bad} rows are not exactly one-hot")
            labels = [CLASSES[i] for i in idx]
        else:
            lcol = cols.get("label") or cols.get("class") or cols.get("target")
            if lcol is None:
                print(f"[skip] {f.name}: no one-hot or label column found ({list(df.columns)})")
                continue
            labels = [CLASSES[label_to_idx(v)] for v in df[lcol]]
        for fn, lab in zip(df[fcol].astype(str), labels):
            out[(sp, Path(fn).name)] = lab
        used.append(f"{f.relative_to(label_dir)} ({len(df)} rows, split={sp})")
    print("Label CSVs used:", *used, sep="\n  ")
    return out


def load_manifest(root, splits=("train", "val"), label_dir=None) -> pd.DataFrame:
    """Merged train+val (for K-fold) or test. Drops failed rows loudly.
    label_dir: folder with the competition CSVs; needed when images are not in class sub-folders."""
    df = pd.read_csv(Path(root) / "manifest.csv", keep_default_na=False)
    df = df[df["split"].isin(splits)].copy()
    bad = df[df["method"] == "failed"]
    if len(bad):
        print(f"[warn] dropping {len(bad)} failed images:", bad["orig_relpath"].tolist()[:10])
    df = df[df["method"] != "failed"].reset_index(drop=True)
    if label_dir is not None and any(sp != "test" for sp in splits):
        lab = read_label_csvs(label_dir)
        names = df["orig_relpath"].map(lambda p: Path(p).name)
        df["label"] = [lab.get((sp, n), lab.get((None, n), l)) if sp != "test" else l
                       for sp, n, l in zip(df["split"], names, df["label"])]
        missing = df[(df["split"] != "test") & (df["label"] == "")]
        if len(missing):
            raise ValueError(f"{len(missing)} train/val images have no label, e.g. {missing.orig_relpath.tolist()[:5]}")
    if any(sp != "test" for sp in splits):
        tr = df[df["split"] != "test"]
        print("Class counts:\n" + pd.crosstab(tr["split"], tr["label"], margins=True).to_string())
    return df


# ----------------------------------------------------------------------------------------
# Mixup (batch level) + soft-target loss
# ----------------------------------------------------------------------------------------
def mixup_batch(x, y, num_classes=4, alpha=0.2, p=0.5, label_smoothing=0.0):
    """Returns (x_mixed, soft_targets). With prob 1-p returns the batch unchanged (one-hot targets).
    Switch off for the last ~20% of epochs."""
    t = F.one_hot(y, num_classes).float()
    if label_smoothing > 0:
        t = t * (1 - label_smoothing) + label_smoothing / num_classes
    if alpha <= 0 or np.random.rand() > p:
        return x, t
    lam = float(np.random.beta(alpha, alpha))
    lam = max(lam, 1 - lam)  # keep the original image dominant
    perm = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[perm], lam * t + (1 - lam) * t[perm]


def soft_ce(logits, soft_targets, class_weight=None):
    logp = F.log_softmax(logits, dim=1)
    if class_weight is not None:
        logp = logp * class_weight[None, :]
    return -(soft_targets * logp).sum(dim=1).mean()


# ----------------------------------------------------------------------------------------
def _preview(root, out, clahe, img_size, n_imgs=4, n_aug=5, hflip=False):
    df = load_manifest(root, ("test",))
    df = df.sample(min(n_imgs, len(df)), random_state=0)
    tf = build_transforms(img_size, train=True, hflip=hflip)
    mean, std = torch.tensor(IMAGENET_MEAN)[:, None, None], torch.tensor(IMAGENET_STD)[:, None, None]
    rows = []
    for row in df.itertuples():
        g = cv2.imread(str(Path(root) / row.out_relpath), cv2.IMREAD_GRAYSCALE)
        base = cv2.resize(apply_clahe_mode(g, "none"), (img_size, img_size))
        cl = cv2.resize(apply_clahe_mode(g, clahe), (img_size, img_size))
        tiles = [base, cl]
        for _ in range(n_aug):
            x = tf(apply_clahe_mode(g, clahe)) * std + mean
            tiles.append((x.clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8))
        rows.append(np.hstack(tiles))
    grid = np.vstack(rows)
    cv2.imwrite(out, cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
    print(f"saved {out}: columns = original | CLAHE({clahe}) | {n_aug} random train augmentations")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="processed_data")
    ap.add_argument("--preview", default="aug_preview.jpg")
    ap.add_argument("--clahe", default="clahe", choices=["none", "clahe", "stack"])
    ap.add_argument("--img_size", type=int, default=256)
    ap.add_argument("--hflip", action="store_true")
    a = ap.parse_args()
    _preview(a.root, a.preview, a.clahe, a.img_size, hflip=a.hflip)


# ----------------------------------------------------------------------------------------
# Submission
# ----------------------------------------------------------------------------------------
def write_submission(pred, template_csv, out_csv):
    """pred: dict {original test filename (e.g. '1736.jpeg') -> class index 0..3 or class name}.
    Keeps the template's row order exactly, writes integer one-hot columns, refuses to write if
    any template file is missing a prediction (Kaggle rejects null values)."""
    tpl = pd.read_csv(template_csv)
    names = tpl["new_filename"].astype(str)
    missing = [n for n in names if n not in pred]
    if missing:
        raise ValueError(f"{len(missing)} test files have no prediction, e.g. {missing[:5]}")
    idx = [p if isinstance(p, (int, np.integer)) else label_to_idx(p) for p in (pred[n] for n in names)]
    onehot = np.eye(len(CLASSES), dtype=int)[idx]
    sub = pd.DataFrame(onehot, columns=CLASSES)
    sub.insert(0, "new_filename", names.values)
    sub.to_csv(out_csv, index=False)
    assert sub[CLASSES].sum(1).eq(1).all() and not sub.isna().any().any()
    print(f"wrote {out_csv}: {len(sub)} rows | predicted counts: "
          + ", ".join(f"{c}={int(v)}" for c, v in sub[CLASSES].sum().items()))
    return sub
