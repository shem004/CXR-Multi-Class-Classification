#!/usr/bin/env python
"""
Analysis of one finished run (needs its folds' oof.csv / history.csv / best_fp16.pt).

Writes PNG figures + summary.json to <work_dir>/analysis/<run>/ :
  confusion.png            counts and row-% (argmax and, if present, tuned offsets)
  curves.png               mean validation macro-F1 / loss / per-class F1 per epoch
  bv_probability.png       P(virus) for true bacteria vs true virus  -> how separable the hard pair is
  errors_B_as_V.png        most confident bacteria -> virus mistakes
  errors_V_as_B.png        most confident virus -> bacteria mistakes
  covid_cases.png          every COVID error (missed or false alarm)
  gradcam_covid.png        Grad-CAM of the out-of-fold model on COVID images (where does it look?)
  gradcam_errors.png       Grad-CAM on the most confident bacteria/virus mistakes
  metadata_probe.txt       can image size / file type alone predict the class? (shortcut check)
  by_crop_method.csv       accuracy split by preprocessing method (lungs / mirror / center)

python analyze.py --work_dir WORK --proc_root PROC --run convnext_s_384
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from sklearn.metrics import confusion_matrix  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cxr_data import CLASSES, apply_clahe_mode, build_transforms  # noqa: E402
from train import SHORT, build_model, log_softmax_np, macro_f1, per_class_f1  # noqa: E402


def load(work, run):
    rdir = Path(work) / run
    folds = sorted(int(p.parent.name[4:]) for p in rdir.glob("fold*/DONE.json"))
    if not folds:
        raise SystemExit(f"no finished folds in {rdir}")
    oof = pd.concat([pd.read_csv(rdir / f"fold{k}" / "oof.csv") for k in folds], ignore_index=True)
    hist = pd.concat([pd.read_csv(rdir / f"fold{k}" / "history.csv").assign(fold=k) for k in folds])
    args = json.load(open(rdir / f"fold{folds[0]}" / "DONE.json"))["args"]
    offsets = np.array(json.load(open(rdir / "offsets.json"))["offsets"]) if (rdir / "offsets.json").exists() else None
    return rdir, folds, oof, hist, args, offsets


def plot_confusion(y, preds, titles, path):
    fig, axes = plt.subplots(1, len(preds), figsize=(5.2 * len(preds), 4.6))
    axes = np.atleast_1d(axes)
    for ax, p, t in zip(axes, preds, titles):
        cm = confusion_matrix(y, p, labels=range(4))
        pct = cm / cm.sum(1, keepdims=True)
        ax.imshow(pct, cmap="Blues", vmin=0, vmax=1)
        for i in range(4):
            for j in range(4):
                ax.text(j, i, f"{cm[i, j]}\n{pct[i, j]:.0%}", ha="center", va="center", fontsize=9,
                        color="white" if pct[i, j] > 0.6 else "black")
        ax.set_xticks(range(4), CLASSES, rotation=20)
        ax.set_yticks(range(4), CLASSES)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.set_title(f"{t}\nmacro-F1 {macro_f1(y, p):.4f}", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_curves(hist, path):
    m = hist.groupby("epoch").mean(numeric_only=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for k, g in hist.groupby("fold"):
        axes[0].plot(g.epoch, g.val_macro_f1, color="0.8", lw=1)
        axes[1].plot(g.epoch, g.val_loss, color="0.8", lw=1)
    axes[0].plot(m.index, m.val_macro_f1, lw=2.5, label="mean of folds")
    axes[1].plot(m.index, m.val_loss, lw=2.5, label="val (mean)")
    axes[1].plot(m.index, m.train_loss, lw=1.5, ls="--", label="train (mean)")
    for c, nm in zip(SHORT, CLASSES):
        axes[2].plot(m.index, m[f"f1_{c}"], lw=2, label=nm)
    for ax, t in zip(axes, ["validation macro-F1", "loss", "validation F1 per class (mean of folds)"]):
        ax.set_title(t)
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_bv(oof, prob, path):
    pv = prob[:, 2] / (prob[:, 1] + prob[:, 2] + 1e-9)  # P(virus | bacteria or virus)
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(0, 1, 41)
    for lab, col in (("bacteria", "tab:orange"), ("virus", "tab:blue")):
        ax.hist(pv[oof.label == lab], bins=bins, alpha=0.55, label=f"true {lab}", color=col)
    ax.axvline(0.5, color="k", ls="--", lw=1)
    ax.set_xlabel("P(virus) / (P(virus) + P(bacteria))  (out-of-fold)")
    ax.set_ylabel("images")
    ax.set_title("How separable bacteria and virus are")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def gallery(rows, proc_root, path, title, ncol=6, extra=None):
    if len(rows) == 0:
        return
    n = len(rows)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.6 * ncol, 2.9 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for ax, (_, r) in zip(axes, rows.iterrows()):
        img = cv2.imread(str(Path(proc_root) / r.out_relpath), cv2.IMREAD_GRAYSCALE)
        ax.imshow(img, cmap="gray")
        if extra is not None:
            ax.imshow(extra[r.name], cmap="jet", alpha=0.35)
        ax.set_title(f"{Path(r.orig_relpath).name}\ntrue {r.label} | pred {CLASSES[r.pred]} {r.conf:.2f}", fontsize=7.5)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.35 / nrow))
    fig.savefig(path, dpi=110)
    plt.close(fig)


def gradcam(rdir, rows, proc_root, args, device):
    """Grad-CAM of each image's OUT-OF-FOLD model, for its predicted class."""
    cams = {}
    tf = None
    for k, g in rows.groupby("fold"):
        wpath = rdir / f"fold{k}" / "best_fp16.pt"
        if not wpath.exists():
            wpath = rdir / f"fold{k}" / "best.pt"  # older runs: full-precision checkpoint
        if not wpath.exists():
            print(f"[skip gradcam] fold {k}: no saved weights ({len(g)} images)")
            continue
        model = build_model(args["model"], pretrained=False, drop=args.get("drop", 0.2),
                            drop_path=args.get("drop_path", 0.2), img_size=args.get("img_size", 384))
        sd = torch.load(wpath, map_location="cpu", weights_only=False)
        model.load_state_dict(sd["state_dict"] if "state_dict" in sd else sd)
        model = model.float().to(device).eval()
        cfg = model.pretrained_cfg
        tf = build_transforms(args["img_size"], False, mean=cfg.get("mean"), std=cfg.get("std"))
        for idx, r in g.iterrows():
            gray = cv2.imread(str(Path(proc_root) / r.out_relpath), cv2.IMREAD_GRAYSCALE)
            x = tf(apply_clahe_mode(gray, args.get("clahe", "clahe")))[None].to(device)
            feats = model.forward_features(x)
            feats.retain_grad()
            out = model.forward_head(feats)
            model.zero_grad()
            out[0, int(r.pred)].backward()
            w = feats.grad.mean(dim=(2, 3), keepdim=True)
            cam = F.relu((w * feats).sum(1))[0].detach().cpu().numpy()
            cam = cv2.resize(cam, gray.shape[::-1])
            cams[idx] = cam / (cam.max() + 1e-9)
        del model
    return cams


def metadata_probe(oof, manifest, out_txt):
    """Shortcut check: predict the class from image size / file type only (no pixels)."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_predict
    m = oof.merge(manifest[["out_relpath", "orig_w", "orig_h"]], on="out_relpath", how="left")
    X = np.c_[m.orig_w, m.orig_h, m.orig_w / m.orig_h,
              m.orig_relpath.str.lower().str.endswith(".png"), m.orig_relpath.str.lower().str.endswith(".jpg")]
    X = np.nan_to_num(X.astype(float))
    p = cross_val_predict(RandomForestClassifier(300, min_samples_leaf=3, random_state=0), X, m.y, cv=5)
    pcf = per_class_f1(m.y, p)
    txt = ("Metadata-only classifier (width, height, aspect, file type; no pixels), 5-fold CV:\n"
           f"  macro-F1 {macro_f1(m.y, p):.4f} | per-class F1 "
           + ", ".join(f"{c} {v:.3f}" for c, v in zip(CLASSES, pcf))
           + "\nA high COVID F1 here means COVID images can be told apart by their source alone (shortcut risk).\n")
    open(out_txt, "w").write(txt)
    print(txt)
    return {f"metadata_f1_{c}": float(v) for c, v in zip(CLASSES, pcf)}


def signal_checks(oof, manifest, out_csv):
    """Per true class: things that are NOT disease but could still differ between classes (shortcuts)."""
    m = oof.merge(manifest, on="out_relpath", how="left", suffixes=("", "_m"))
    for c in ("orig_w", "orig_h", "x0", "x1", "y0", "y1"):
        m[c] = pd.to_numeric(m[c], errors="coerce") if c in m else np.nan
    if "method" not in m:
        m["method"] = "lungs"
    bw, bh = m.x1 - m.x0, m.y1 - m.y0
    side = np.maximum(bw, bh)
    elongated = np.maximum(bw, bh) / np.minimum(bw, bh).clip(lower=1) > 1.5  # resized, not padded
    m["padded"] = (~elongated) & ((side > m.orig_w + 1) | (side > m.orig_h + 1))
    m["non_jpeg"] = ~m.orig_relpath.str.lower().str.endswith(".jpeg")
    m["aspect"] = m.orig_w / m.orig_h
    t = m.groupby("label").agg(n=("y", "size"), median_w=("orig_w", "median"), median_h=("orig_h", "median"),
                               median_aspect=("aspect", "median"), padded_share=("padded", "mean"),
                               non_jpeg_share=("non_jpeg", "mean"),
                               mirror_or_center=("method", lambda v: float((v != "lungs").mean())))
    t = t.round(3)
    t.to_csv(out_csv)
    print("Signal checks per true class (none of these should matter for the diagnosis):\n" + t.to_string())
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work_dir", required=True)
    ap.add_argument("--proc_root", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--gallery_n", type=int, default=18)
    ap.add_argument("--gradcam_errors", type=int, default=12)
    ap.add_argument("--no_gradcam", action="store_true")
    ap.add_argument("--covid_like_thr", type=float, default=0.05)
    a = ap.parse_args()

    rdir, folds, oof, hist, args, offsets = load(a.work_dir, a.run)
    out = Path(a.work_dir) / "analysis" / a.run
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    lp = log_softmax_np(oof[[f"logit_{c}" for c in SHORT]].to_numpy())
    prob = np.exp(lp)
    y = oof["y"].to_numpy()
    oof["pred"] = lp.argmax(1)
    oof["conf"] = prob.max(1)
    print(f"{a.run}: folds {folds}, {len(oof)} OOF images")

    preds, titles = [oof.pred.to_numpy()], ["argmax"]
    if offsets is not None:
        preds.append((lp + offsets).argmax(1))
        titles.append(f"tuned offsets {np.round(offsets, 2).tolist()}")
    plot_confusion(y, preds, titles, out / "confusion.png")
    plot_curves(hist, out / "curves.png")
    plot_bv(oof, prob, out / "bv_probability.png")

    err = oof[oof.pred != oof.y]
    b_as_v = err[(err.label == "bacteria") & (err.pred == 2)].sort_values("conf", ascending=False).head(a.gallery_n)
    v_as_b = err[(err.label == "virus") & (err.pred == 1)].sort_values("conf", ascending=False).head(a.gallery_n)
    covid = oof[((oof.y == 3) | (oof.pred == 3)) & (oof.pred != oof.y)]
    gallery(b_as_v, a.proc_root, out / "errors_B_as_V.png", "Most confident: bacteria predicted as virus")
    gallery(v_as_b, a.proc_root, out / "errors_V_as_B.png", "Most confident: virus predicted as bacteria")
    gallery(covid, a.proc_root, out / "covid_cases.png", "COVID errors (missed or false alarm)")

    manifest = pd.read_csv(Path(a.proc_root) / "manifest.csv", keep_default_na=False)
    manifest = manifest.drop_duplicates("out_relpath")
    by = oof.merge(manifest[["out_relpath", "method"]], on="out_relpath", how="left")
    by_m = by.assign(correct=by.pred == by.y).groupby("method").agg(n=("y", "size"), accuracy=("correct", "mean"))
    by_m.to_csv(out / "by_crop_method.csv")
    print("Accuracy by crop method:\n" + by_m.to_string())
    meta = metadata_probe(oof, manifest, out / "metadata_probe.txt")
    signal_checks(oof, manifest, out / "signal_checks.csv")

    # COVID-like negatives: hard-negative candidates (adult / outside-source non-COVID films)
    oof["p_covid"] = prob[:, 3]
    cand = oof[(oof.y != 3) & ((oof.p_covid >= a.covid_like_thr) |
                               ~oof.orig_relpath.str.lower().str.endswith(".jpeg"))].sort_values("p_covid", ascending=False)
    cand[["out_relpath", "orig_relpath", "label", "p_covid", "pred"]].to_csv(out / "covid_like_negatives.csv", index=False)
    gallery(cand.head(36).assign(conf=cand.head(36).p_covid, pred=3), a.proc_root, out / "covid_like_negatives.png",
            f"Non-COVID films the model finds COVID-like (P(COVID) >= {a.covid_like_thr} or not .jpeg): "
            f"{len(cand)} candidates; check visually, keep the adult ones")
    print(f"COVID-like negatives: {len(cand)} -> {out / 'covid_like_negatives.csv'}")

    # label-noise estimate: errors the model is very sure about
    conf_err = oof[(oof.pred != oof.y) & (oof.conf >= 0.9)]
    print("Confident errors (wrong with p >= 0.9) per true class:\n" + conf_err.label.value_counts().to_string())

    if not a.no_gradcam and not args["model"].startswith(("swin_", "fusion")):  # Grad-CAM needs one conv feature map
        cov_rows = oof[oof.y == 3]
        cams = gradcam(rdir, cov_rows, a.proc_root, args, device)
        gallery(cov_rows.loc[list(cams)], a.proc_root, out / "gradcam_covid.png",
                "Grad-CAM, true COVID images (out-of-fold model, predicted class)", extra=cams)
        e_rows = pd.concat([b_as_v.head(a.gradcam_errors // 2), v_as_b.head(a.gradcam_errors // 2)])
        cams_e = gradcam(rdir, e_rows, a.proc_root, args, device)
        gallery(e_rows.loc[list(cams_e)], a.proc_root, out / "gradcam_errors.png",
                "Grad-CAM, most confident bacteria/virus mistakes", extra=cams_e)

    pcf = per_class_f1(y, oof.pred)
    summary = {"run": a.run, "folds": folds, "n": int(len(oof)), "oof_macro_f1": float(macro_f1(y, oof.pred)),
               **{f"f1_{c}": float(v) for c, v in zip(CLASSES, pcf)},
               "errors": int(len(err)), "bacteria_to_virus": int(((oof.y == 1) & (oof.pred == 2)).sum()),
               "virus_to_bacteria": int(((oof.y == 2) & (oof.pred == 1)).sum()),
               "covid_missed": int(((oof.y == 3) & (oof.pred != 3)).sum()),
               "covid_false_alarm": int(((oof.y != 3) & (oof.pred == 3)).sum()),
               "confident_errors_p90": int(len(conf_err)), "covid_like_negatives": int(len(cand)), **meta}
    json.dump(summary, open(out / "summary.json", "w"), indent=2)
    print(json.dumps(summary, indent=2))
    print(f"\nfigures -> {out}")


if __name__ == "__main__":
    main()
