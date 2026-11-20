#!/usr/bin/env python
"""
5-fold training for the Lab3 CXR competition (Normal / Bacteria / Virus / COVID-19, macro-F1).

  * merged train+val -> StratifiedKFold (folds.csv is created once and reused by every model)
  * timm backbone (default ConvNeXt-Small IN-22k, 384 px), AdamW + warmup/cosine, AMP, EMA
  * label-smoothed CE, COVID oversampled x5, mild augmentation + CLAHE (cxr_data.py)
  * one fold per run (free Colab friendly): checkpoint every epoch -> resumes after a disconnect
  * --finalize: out-of-fold (OOF) report, per-class logit-offset tuning for macro-F1,
                fold-averaged test predictions -> submission CSVs

Colab (T4):
  !pip -q install timm
  !python train.py --proc_root /content/processed_data --data_root $DATA --work_dir $WORK --fold 0
  ... --fold 1 ... --fold 4      (or --fold all)
  !python train.py --proc_root /content/processed_data --data_root $DATA --work_dir $WORK --finalize \
                   --template_csv $DATA/test_data_sample.csv
"""

import argparse
import copy
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cxr_data import (CLASSES, CXRDataset, build_transforms, load_manifest,  # noqa: E402
                      mixup_batch, soft_ce, write_submission)

try:
    from tqdm.auto import tqdm as _tqdm
except ImportError:
    _tqdm = None
NO_BAR = os.environ.get("NO_BAR", "0") == "1"


def tqdm(it, **kw):
    """progress bar, or the plain iterable when NO_BAR=1 (log files) / tqdm missing"""
    return it if (NO_BAR or _tqdm is None) else _tqdm(it, **kw)

SHORT = ["N", "B", "V", "C"]


# --------------------------------------------------------------------------------------
# utils
# --------------------------------------------------------------------------------------
def seed_all(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def macro_f1(y, pred):
    return f1_score(y, pred, average="macro", labels=list(range(4)), zero_division=0)


def per_class_f1(y, pred):
    return f1_score(y, pred, average=None, labels=list(range(4)), zero_division=0)


def atomic_save(obj, path):
    tmp = str(path) + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


class EMA:
    """Exponential moving average of weights (with warm-up so early epochs are not frozen)."""

    def __init__(self, model, decay):
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)
        self.decay, self.updates = decay, 0

    @torch.no_grad()
    def update(self, model):
        self.updates += 1
        d = min(self.decay, (1 + self.updates) / (10 + self.updates))
        msd = model.state_dict()
        for k, v in self.module.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(d).add_(msd[k].detach(), alpha=1 - d)
            else:
                v.copy_(msd[k])


def get_folds(work_dir, df, n_folds, seed):
    """folds.csv is the single source of truth so every model uses identical splits."""
    path = Path(work_dir) / ("folds.csv" if n_folds == 5 else f"folds{n_folds}.csv")
    if path.exists():
        f = pd.read_csv(path)
        if set(f["out_relpath"]) != set(df["out_relpath"]):
            raise ValueError(f"{path} does not match the current manifest - delete it to re-split")
        return df.merge(f[["out_relpath", "fold"]], on="out_relpath")
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    df = df.copy()
    df["fold"] = -1
    for k, (_, va) in enumerate(skf.split(df, df["label"])):
        df.loc[df.index[va], "fold"] = k
    df[["out_relpath", "orig_relpath", "split", "label", "fold"]].to_csv(path, index=False)
    print(f"Created {path}\n" + pd.crosstab(df["fold"], df["label"]).to_string())
    return df


def make_loader(ds, bs, shuffle, workers, drop_last=False):
    return torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=shuffle, num_workers=workers,
                                       pin_memory=torch.cuda.is_available(), drop_last=drop_last,
                                       persistent_workers=workers > 0)


@torch.no_grad()
def predict(model, loader, device, tta=False, desc="predict"):
    model.eval()
    out = []
    for x, _, _ in tqdm(loader, desc=desc, leave=False, dynamic_ncols=True):
        x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device == "cuda"):
            lg = model(x).float()
            if tta:
                lg = (lg + model(torch.flip(x, dims=[3])).float()) / 2
        out.append(lg.cpu())
    return torch.cat(out).numpy()


# --------------------------------------------------------------------------------------
# models
# --------------------------------------------------------------------------------------
class XRVDenseNet(torch.nn.Module):
    """TorchXRayVision DenseNet121 pretrained on ~8 public chest X-ray datasets, new 4-class head.
    Expects 1-channel input in [-1024, 1024]; the transform's mean/std below produce exactly that
    (pixel/255 -> (x - 0.5) * 2048) and the 3 identical channels are averaged."""

    def __init__(self, weights="densenet121-res224-all", pretrained=True, drop=0.2):
        super().__init__()
        import torchxrayvision as xrv
        base = xrv.models.DenseNet(weights=weights if pretrained else None)
        self.features = base.features
        self.head = torch.nn.Sequential(torch.nn.Dropout(drop), torch.nn.Linear(1024, 4))
        self.pretrained_cfg = {"mean": (0.5, 0.5, 0.5), "std": (1 / 2048,) * 3}

    def forward_features(self, x):
        return F.relu(self.features(x.mean(1, keepdim=True)))

    def forward_head(self, f):
        return self.head(F.adaptive_avg_pool2d(f, 1).flatten(1))

    def forward(self, x):
        return self.forward_head(self.forward_features(x))


CNX = "convnext_small.fb_in22k_ft_in1k_384"
SWIN = "swin_small_patch4_window7_224.ms_in22k_ft_in1k"


def _timm(name, pretrained, drop_path, img_size, **kw):
    import timm
    if name.startswith("swin_"):  # Swin needs the input size up front; the 224 weights are resampled to it
        kw.update(img_size=img_size, window_size=max(7, img_size // 32))
    return timm.create_model(name, pretrained=pretrained, drop_path_rate=drop_path, **kw)


class FusionNet(torch.nn.Module):
    """Late fusion: a CNN and a transformer run side by side on the same image; their pooled
    embeddings (768 + 768) are concatenated and classified by a small MLP. Trained end to end."""

    def __init__(self, a=CNX, b=SWIN, pretrained=True, drop=0.2, drop_path=0.2, img_size=384):
        super().__init__()
        self.a = _timm(a, pretrained, drop_path, img_size, num_classes=0)
        self.b = _timm(b, pretrained, drop_path, img_size, num_classes=0)
        dim = self.a.num_features + self.b.num_features
        self.head = torch.nn.Sequential(torch.nn.LayerNorm(dim), torch.nn.Dropout(drop), torch.nn.Linear(dim, 512),
                                        torch.nn.GELU(), torch.nn.Dropout(drop), torch.nn.Linear(512, 4))
        self.pretrained_cfg = self.a.pretrained_cfg  # both use ImageNet mean/std

    def forward(self, x):
        return self.head(torch.cat([self.a(x), self.b(x)], dim=1))


class HierNet(torch.nn.Module):
    """Hierarchical head on one backbone. Head 1: normal / pneumonia / COVID-19. Head 2: bacteria vs
    virus. Output = 4 log-probabilities: log P(normal), log P(pneu) + log P(bact | pneu),
    log P(pneu) + log P(virus | pneu), log P(COVID). Cross-entropy on this output equals
    CE(head 1) + CE(head 2 on pneumonia images), so training, ensembling and offsets are unchanged."""

    def __init__(self, name=CNX, pretrained=True, drop=0.2, drop_path=0.2, img_size=384):
        super().__init__()
        self.backbone = _timm(name, pretrained, drop_path, img_size, num_classes=0)
        f = self.backbone.num_features
        self.head1 = torch.nn.Sequential(torch.nn.Dropout(drop), torch.nn.Linear(f, 3))
        self.head2 = torch.nn.Sequential(torch.nn.Dropout(drop), torch.nn.Linear(f, 256), torch.nn.GELU(),
                                         torch.nn.Dropout(drop), torch.nn.Linear(256, 2))
        self.pretrained_cfg = self.backbone.pretrained_cfg

    def forward_features(self, x):
        return self.backbone.forward_features(x)

    def forward_head(self, feats):
        z = self.backbone.forward_head(feats, pre_logits=True)
        g, b = F.log_softmax(self.head1(z).float(), 1), F.log_softmax(self.head2(z).float(), 1)
        return torch.stack([g[:, 0], g[:, 1] + b[:, 0], g[:, 1] + b[:, 1], g[:, 2]], dim=1)

    def forward(self, x):
        return self.forward_head(self.forward_features(x))


def build_model(name, pretrained=True, drop=0.2, drop_path=0.2, img_size=384):
    """timm model name, 'xrv_densenet121[:weights]', 'fusion' (ConvNeXt-S + Swin-S) or 'hier[:<timm name>]'."""
    if name == "fusion":
        return FusionNet(pretrained=pretrained, drop=drop, drop_path=drop_path, img_size=img_size)
    if name.startswith("hier"):
        return HierNet(name.split(":", 1)[1] if ":" in name else CNX, pretrained, drop, drop_path, img_size)
    if name.startswith("xrv_densenet121"):
        tag = name.split(":", 1)[1] if ":" in name else "densenet121-res224-all"
        return XRVDenseNet(tag, pretrained, drop)
    import timm
    kw = {}
    if name.startswith("swin_"):  # Swin needs the input size up front; the 224 weights are resampled to it
        kw = {"img_size": img_size, "window_size": max(7, img_size // 32)}
    return timm.create_model(name, pretrained=pretrained, num_classes=4, drop_rate=drop, drop_path_rate=drop_path, **kw)


def safe_load(path, device):
    """None for missing / empty / truncated checkpoints (e.g. a write that failed on a full Drive)."""
    try:
        if not Path(path).exists() or Path(path).stat().st_size < 1024:
            return None
        return torch.load(path, map_location=device, weights_only=False)
    except Exception as e:  # noqa: BLE001
        print(f"[warn] ignoring unreadable checkpoint {path}: {e!r}")
        return None


# --------------------------------------------------------------------------------------
# training one fold
# --------------------------------------------------------------------------------------
def train_fold(args, df, test_df, fold, device):
    fdir = Path(args.work_dir) / args.run_name / f"fold{fold}"
    fdir.mkdir(parents=True, exist_ok=True)
    if (fdir / "DONE.json").exists() and not args.overwrite:
        print(f"[fold {fold}] already done -> skip ({fdir / 'DONE.json'})")
        return
    seed_all(args.seed + fold)

    tr_df, va_df = df[df.fold != fold].reset_index(drop=True), df[df.fold == fold].reset_index(drop=True)
    # COVID oversampling: repeat the rare class inside each epoch
    if args.covid_oversample > 1:
        cov = tr_df[tr_df.label == "COVID-19"]
        tr_df = pd.concat([tr_df] + [cov] * (args.covid_oversample - 1), ignore_index=True)
    if args.hard_neg_csv and args.hard_neg_oversample > 1:
        # e.g. adult non-COVID films that the model confuses with COVID: repeat them so that
        # "adult-looking" stops being a shortcut for COVID (validation images are never repeated)
        hn = set(pd.read_csv(args.hard_neg_csv)["out_relpath"])
        hard = tr_df[tr_df.out_relpath.isin(hn) & (tr_df.label != "COVID-19")].drop_duplicates("out_relpath")
        tr_df = pd.concat([tr_df] + [hard] * (args.hard_neg_oversample - 1), ignore_index=True)
        print(f"[fold {fold}] hard negatives in train: {len(hard)} (x{args.hard_neg_oversample})")

    model = build_model(args.model, args.pretrained, args.drop, args.drop_path, args.img_size)
    cfg = model.pretrained_cfg
    mean, std = cfg.get("mean", (0.485, 0.456, 0.406)), cfg.get("std", (0.229, 0.224, 0.225))
    model = model.to(device).to(memory_format=torch.channels_last)

    tf_tr = build_transforms(args.img_size, True, hflip=args.hflip, erasing=args.erasing, mean=mean, std=std)
    tf_ev = build_transforms(args.img_size, False, mean=mean, std=std)
    ds_tr = CXRDataset(tr_df, args.proc_root, tf_tr, clahe=args.clahe)
    ds_va = CXRDataset(va_df, args.proc_root, tf_ev, clahe=args.clahe)
    ds_te = CXRDataset(test_df, args.proc_root, tf_ev, clahe=args.clahe)
    dl_tr = make_loader(ds_tr, args.batch_size, True, args.num_workers, drop_last=True)
    dl_va = make_loader(ds_va, args.batch_size * 2, False, args.num_workers)

    head = [p for n, p in model.named_parameters() if n.startswith(("head", "fc", "classifier"))]
    head_ids = {id(p) for p in head}
    body = [p for p in model.parameters() if id(p) not in head_ids]
    opt = torch.optim.AdamW([{"params": body, "lr": args.lr},
                             {"params": head, "lr": args.lr * args.head_lr_mult}], weight_decay=args.wd)
    steps_per_epoch = max(1, len(dl_tr) // args.accum)
    total, warm = args.epochs * steps_per_epoch, args.warmup_epochs * steps_per_epoch
    min_ratio = args.min_lr / args.lr

    def lr_lambda(step):
        if step < warm:
            return (step + 1) / max(1, warm)
        t = (step - warm) / max(1, total - warm)
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * min(t, 1.0)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")
    ema = EMA(model, args.ema_decay) if args.ema_decay > 0 else None

    start_epoch, best, history = 0, {"f1": -1.0, "epoch": -1}, []
    # big, frequently overwritten checkpoints live on LOCAL disk (Drive keeps every overwritten
    # version in its Trash, which counts against the quota). Only small results + one fp16 copy of
    # the best weights go to work_dir.
    cdir = Path(args.ckpt_dir) / args.run_name / f"fold{fold}"
    cdir.mkdir(parents=True, exist_ok=True)
    last_path, best_path = cdir / "last.pt", cdir / "best.pt"
    if not last_path.exists() and (fdir / "last.pt").exists():  # checkpoint left by the old version
        last_path = fdir / "last.pt"
    if not best_path.exists() and (fdir / "best.pt").exists():
        best_path = fdir / "best.pt"
    ck = None if args.overwrite else safe_load(last_path, device)
    if ck is not None and safe_load(best_path, "cpu") is None:
        print(f"[fold {fold}] last checkpoint found but no readable best.pt -> starting fresh")
        ck = None
    if ck is not None:
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"])
        if ema is not None and ck.get("ema") is not None:
            ema.module.load_state_dict(ck["ema"])
            ema.updates = ck["ema_updates"]
        start_epoch, best, history = ck["epoch"] + 1, ck["best"], ck["history"]
        print(f"[fold {fold}] resumed from epoch {start_epoch} (best mF1 {best['f1']:.4f} @ ep {best['epoch']})")

    y_va = np.array(ds_va.labels)
    print(f"[fold {fold}] train {len(ds_tr)} (incl. COVID x{args.covid_oversample}) | val {len(ds_va)} | "
          f"{args.model} @ {args.img_size}px | bs {args.batch_size}x{args.accum} | {args.epochs} ep | {device}")
    mix_off_epoch = int(args.epochs * 0.8)

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        model.train()
        run_loss, n_seen = 0.0, 0
        opt.zero_grad(set_to_none=True)
        pbar = tqdm(dl_tr, desc=f"fold {fold} ep {epoch + 1}/{args.epochs}", leave=False, dynamic_ncols=True)
        for it, (x, y, _) in enumerate(pbar):
            x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            y = torch.as_tensor(y, device=device)
            alpha = args.mixup_alpha if epoch < mix_off_epoch else 0.0
            x, t = mixup_batch(x, y, 4, alpha=alpha, p=args.mixup_p, label_smoothing=args.label_smoothing)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device == "cuda"):
                out = model(x).float()
                if args.focal_gamma > 0:  # focal loss: down-weight images the model already gets right
                    logp = F.log_softmax(out, 1)
                    loss = -(t * (1 - logp.exp()).pow(args.focal_gamma) * logp).sum(1).mean() / args.accum
                else:
                    loss = soft_ce(out, t) / args.accum
            scaler.scale(loss).backward()
            if (it + 1) % args.accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()
                if ema is not None:
                    ema.update(model)
            run_loss += loss.item() * args.accum * x.size(0)
            n_seen += x.size(0)
            if hasattr(pbar, "set_postfix"):
                pbar.set_postfix(loss=f"{run_loss / n_seen:.4f}", lr=f"{opt.param_groups[0]['lr']:.1e}")

        eval_model = ema.module if ema is not None else model
        logits = predict(eval_model, dl_va, device, desc="val")
        pred = logits.argmax(1)
        vloss = F.cross_entropy(torch.from_numpy(logits), torch.from_numpy(y_va)).item()
        f1 = macro_f1(y_va, pred)
        pcf = per_class_f1(y_va, pred)
        rec = {"epoch": epoch + 1, "train_loss": run_loss / max(n_seen, 1), "val_loss": vloss,
               "val_acc": float((pred == y_va).mean()), "val_macro_f1": f1,
               **{f"f1_{c}": float(v) for c, v in zip(SHORT, pcf)}, "lr": opt.param_groups[0]["lr"],
               "sec": time.time() - t0}
        history.append(rec)
        star = ""
        if f1 > best["f1"]:
            best = {"f1": float(f1), "epoch": epoch + 1}
            best_path = cdir / "best.pt"
            atomic_save({"state_dict": eval_model.state_dict(), "args": vars(args), "epoch": epoch + 1}, best_path)
            oof = va_df[["out_relpath", "orig_relpath", "label"]].copy()
            oof["y"] = y_va
            oof[[f"logit_{c}" for c in SHORT]] = logits
            oof["fold"] = fold
            oof.to_csv(fdir / "oof.csv", index=False)
            star = " *best"
        print(f"[fold {fold}] ep {epoch + 1:>2}/{args.epochs} | train {rec['train_loss']:.4f} | val {vloss:.4f} "
              f"acc {rec['val_acc']:.3f} mF1 {f1:.4f} [" + " ".join(f"{c} {v:.2f}" for c, v in zip(SHORT, pcf))
              + f"] | {rec['sec']:.0f}s{star}", flush=True)
        pd.DataFrame(history).to_csv(fdir / "history.csv", index=False)
        last_path = cdir / "last.pt"
        atomic_save({"model": model.state_dict(), "ema": ema.module.state_dict() if ema else None,
                     "ema_updates": ema.updates if ema else 0, "opt": opt.state_dict(),
                     "sched": sched.state_dict(), "scaler": scaler.state_dict(), "epoch": epoch,
                     "best": best, "history": history}, last_path)

    # ---- test predictions with the best checkpoint ----
    eval_model = ema.module if ema is not None else model
    best_sd = torch.load(best_path, map_location=device, weights_only=False)["state_dict"]
    eval_model.load_state_dict(best_sd)
    dl_te = make_loader(ds_te, args.batch_size * 2, False, args.num_workers)
    te_logits = predict(eval_model, dl_te, device, tta=args.tta, desc="test")
    np.save(fdir / "test_logits.npy", te_logits)
    test_df[["out_relpath", "orig_relpath"]].to_csv(fdir / "test_names.csv", index=False)
    if args.tta:  # keep OOF consistent with test-time TTA
        oof = pd.read_csv(fdir / "oof.csv")
        oof[[f"logit_{c}" for c in SHORT]] = predict(eval_model, dl_va, device, tta=True, desc="val-tta")
        oof.to_csv(fdir / "oof.csv", index=False)
    json.dump({"fold": fold, "best_epoch": best["epoch"], "best_val_macro_f1": best["f1"],
               "args": vars(args)}, open(fdir / "DONE.json", "w"), indent=2)
    if args.save_weights:  # one compact copy (~100 MB for ConvNeXt-S) for later Grad-CAM / re-inference
        torch.save({k: (v.half() if v.is_floating_point() else v) for k, v in best_sd.items()},
                   fdir / "best_fp16.pt")
    for f in {cdir / "last.pt", cdir / "best.pt", fdir / "last.pt", fdir / "best.pt"}:
        if f.exists() and not args.keep_last:
            f.unlink()
    print(f"[fold {fold}] done: best mF1 {best['f1']:.4f} @ epoch {best['epoch']} -> {fdir}")


# --------------------------------------------------------------------------------------
# finalize: OOF report, offset tuning, submissions
# --------------------------------------------------------------------------------------
def log_softmax_np(z):
    z = z - z.max(1, keepdims=True)
    return z - np.log(np.exp(z).sum(1, keepdims=True))


def tune_offsets(logp, y, classes=(1, 2), grid=np.arange(-2.0, 2.0001, 0.05), passes=4):
    """Coordinate ascent on per-class additive offsets (normal fixed at 0) to maximise macro-F1."""
    b = np.zeros(4)
    best = macro_f1(y, (logp + b).argmax(1))
    for _ in range(passes):
        changed = False
        for c in classes:
            for v in grid:
                bb = b.copy()
                bb[c] = v
                s = macro_f1(y, (logp + bb).argmax(1))
                if s > best + 1e-9:
                    best, b, changed = s, bb, True
        if not changed:
            break
    return b, best


def report(name, y, pred):
    pcf = per_class_f1(y, pred)
    print(f"\n{name}: macro-F1 {macro_f1(y, pred):.4f} | acc {(pred == y).mean():.4f} | per-class F1 "
          + ", ".join(f"{c} {v:.3f}" for c, v in zip(CLASSES, pcf)))
    cm = pd.DataFrame(confusion_matrix(y, pred, labels=range(4)), index=[f"true {c}" for c in CLASSES],
                      columns=[f"pred {c}" for c in CLASSES])
    print(cm.to_string())


def finalize(args):
    rdir = Path(args.work_dir) / args.run_name
    folds = sorted(int(p.parent.name[4:]) for p in rdir.glob("fold*/DONE.json"))
    if not folds:
        sys.exit(f"No finished folds in {rdir}")
    print(f"Finished folds: {folds}" + ("" if len(folds) == args.n_folds else "  (partial! OOF covers only these)"))
    oof = pd.concat([pd.read_csv(rdir / f"fold{k}" / "oof.csv") for k in folds], ignore_index=True)
    lcols = [f"logit_{c}" for c in SHORT]
    logp, y = log_softmax_np(oof[lcols].to_numpy()), oof["y"].to_numpy()

    report("OOF argmax", y, logp.argmax(1))
    classes = (1, 2, 3) if args.tune_covid else (1, 2)
    b, tuned = tune_offsets(logp, y, classes)
    report(f"OOF tuned offsets {np.round(b, 2).tolist()}", y, (logp + b).argmax(1))

    # honest estimate: tune on the other folds, evaluate on the held-out fold
    if len(folds) > 1:
        raw, nested = [], []
        for k in folds:
            m = oof["fold"].to_numpy() == k
            bk, _ = tune_offsets(logp[~m], y[~m], classes)
            raw.append(macro_f1(y[m], logp[m].argmax(1)))
            nested.append(macro_f1(y[m], (logp[m] + bk).argmax(1)))
        print(f"\nNested check (offsets tuned on other folds): argmax {np.mean(raw):.4f} -> tuned "
              f"{np.mean(nested):.4f} (mean over folds). If tuned <= argmax, submit the argmax file.")

    # test: average fold log-probs
    names = pd.read_csv(rdir / f"fold{folds[0]}" / "test_names.csv")
    te = np.mean([log_softmax_np(np.load(rdir / f"fold{k}" / "test_logits.npy")) for k in folds], axis=0)
    fn = names["orig_relpath"].map(lambda p: Path(p).name).tolist()
    np.save(rdir / "test_logp_mean.npy", te)
    json.dump({"offsets": b.tolist(), "oof_macro_f1_tuned": tuned, "folds": folds},
              open(rdir / "offsets.json", "w"), indent=2)
    if not args.template_csv:
        print("\n(no --template_csv given -> submissions not written)")
        return
    out_a, out_t = rdir / "submission_argmax.csv", rdir / "submission_tuned.csv"
    sub_a = write_submission(dict(zip(fn, te.argmax(1).tolist())), args.template_csv, out_a)
    sub_t = write_submission(dict(zip(fn, (te + b).argmax(1).tolist())), args.template_csv, out_t)
    print("Expected test counts from the spec priors: ~ normal 316, bacteria 556, virus 299, COVID-19 12")
    # sanity check only (NOT used for prediction): COVID files are numbered 0-56 in train/val
    for nm, sub in (("argmax", sub_a), ("tuned", sub_t)):
        num = sub["new_filename"].str.extract(r"^(\d+)")[0].astype(int)
        cov = sub["COVID-19"] == 1
        print(f"[sanity, {nm}] predicted COVID: {int(cov.sum())} | of those numbered <=57: "
              f"{int((cov & (num <= 57)).sum())} | files numbered <=57 not predicted COVID: "
              f"{int((~cov & (num <= 57)).sum())}")


# --------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--proc_root", required=True, help="processed_data folder (manifest.csv)")
    ap.add_argument("--data_root", required=True, help="original dataset folder with train_data.csv / val_data.csv")
    ap.add_argument("--work_dir", required=True, help="where folds.csv, checkpoints, OOF go (use Drive)")
    ap.add_argument("--run_name", default="convnext_s_384")
    ap.add_argument("--fold", default="0", help="fold index, comma list, or 'all'")
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--finalize", action="store_true")
    ap.add_argument("--template_csv", default="")
    ap.add_argument("--tune_covid", action="store_true", help="also tune the COVID offset (risky with 46 cases)")
    # model / data
    ap.add_argument("--model", default="convnext_small.fb_in22k_ft_in1k_384")
    ap.add_argument("--pretrained", type=int, default=1)
    ap.add_argument("--img_size", type=int, default=384)
    ap.add_argument("--clahe", default="clahe", choices=["none", "clahe", "stack"])
    ap.add_argument("--hflip", action="store_true")
    ap.add_argument("--erasing", type=float, default=0.25)
    ap.add_argument("--covid_oversample", type=int, default=5)
    ap.add_argument("--hard_neg_csv", default="", help="CSV with out_relpath of non-COVID images to oversample")
    ap.add_argument("--hard_neg_oversample", type=int, default=5)
    ap.add_argument("--mixup_alpha", type=float, default=0.0, help="0 = off (baseline); try 0.2")
    ap.add_argument("--mixup_p", type=float, default=0.5)
    ap.add_argument("--label_smoothing", type=float, default=0.1)
    ap.add_argument("--drop", type=float, default=0.2)
    ap.add_argument("--drop_path", type=float, default=0.2)
    # optimisation
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--warmup_epochs", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=24)
    ap.add_argument("--accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--head_lr_mult", type=float, default=10.0)
    ap.add_argument("--min_lr", type=float, default=1e-6)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--clip", type=float, default=2.0)
    ap.add_argument("--focal_gamma", type=float, default=0.0, help="0 = cross-entropy; 2 = focal loss")
    ap.add_argument("--ema_decay", type=float, default=0.999, help="0 = off")
    ap.add_argument("--tta", action="store_true", help="hflip TTA (only sensible together with --hflip)")
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--keep_last", action="store_true", help="keep the full checkpoints after the fold finishes")
    ap.add_argument("--prepare", action="store_true", help="only create folds.csv and exit")
    ap.add_argument("--ckpt_dir", default="/content/ckpt" if Path("/content").is_dir() else
                    ("/tmp/ckpt" if Path("/kaggle").is_dir() else ""),
                    help="LOCAL disk for per-epoch checkpoints (default /content/ckpt on Colab)")
    ap.add_argument("--save_weights", type=int, default=1, help="save best weights as fp16 to work_dir")
    args = ap.parse_args()

    Path(args.work_dir, args.run_name).mkdir(parents=True, exist_ok=True)
    if not args.ckpt_dir:
        args.ckpt_dir = str(Path(args.work_dir) / "_ckpt")
    if args.finalize:
        finalize(args)
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("[warn] no GPU found - Colab: Runtime > Change runtime type > T4 GPU | Kaggle: Settings > Accelerator")
    torch.backends.cudnn.benchmark = True
    df = load_manifest(args.proc_root, ("train", "val"), label_dir=args.data_root)
    df = get_folds(args.work_dir, df, args.n_folds, args.seed)
    if args.prepare:
        # also download the pretrained weights ONCE here, so the parallel per-GPU jobs don't race on the same file
        if args.pretrained:
            build_model(args.model, True, args.drop, args.drop_path, args.img_size)
        print("folds + pretrained weights ready")
        return
    test_df = load_manifest(args.proc_root, ("test",))
    folds = range(args.n_folds) if args.fold == "all" else [int(f) for f in args.fold.split(",")]
    for k in folds:
        train_fold(args, df, test_df, k, device)
        if device == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
