#!/usr/bin/env python
"""
Combine several finished runs (each = 5 folds trained by train.py on the same folds.csv).

  * per-run OOF macro-F1 + how often runs disagree (diversity)
  * blend = weighted sum of per-run log-probabilities; weights searched on OOF (or equal)
  * per-class offsets tuned on the blended OOF
  * nested check: weights + offsets tuned on 4 folds, scored on the 5th -> honest estimate
  * writes submissions + a one-line-per-run summary CSV (paste into the report)

python ensemble.py --work_dir WORK --runs convnext_s_384 effv2s_384 xrv_dn121_384 \
                   --template_csv TEMPLATE --name ens_v1
"""

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cxr_data import CLASSES, write_submission  # noqa: E402
from train import SHORT, log_softmax_np, macro_f1, per_class_f1, report, tune_offsets  # noqa: E402


def load_run(rdir, n_folds):
    folds = sorted(int(p.parent.name[4:]) for p in rdir.glob("fold*/DONE.json"))
    if len(folds) != n_folds:
        raise SystemExit(f"{rdir.name}: finished folds {folds}, need all {n_folds} (wrong --n_folds?) -> finish it first")
    oof = pd.concat([pd.read_csv(rdir / f"fold{k}" / "oof.csv") for k in folds], ignore_index=True)
    oof = oof.sort_values("out_relpath").reset_index(drop=True)
    names = pd.read_csv(rdir / f"fold{folds[0]}" / "test_names.csv")
    te = np.mean([log_softmax_np(np.load(rdir / f"fold{k}" / "test_logits.npy")) for k in folds], axis=0)
    args = json.load(open(rdir / f"fold{folds[0]}" / "DONE.json"))["args"]
    return oof, names, te, args


def search_weights(lps, y, grid=(0, 0.5, 1, 1.5, 2), passes=3):
    """coordinate ascent on run weights (first run fixed at 1) for OOF macro-F1 (argmax)"""
    w = np.ones(len(lps))
    blend = lambda ww: sum(wi * lp for wi, lp in zip(ww, lps)) / max(ww.sum(), 1e-9)  # noqa: E731
    best = macro_f1(y, blend(w).argmax(1))
    for _ in range(passes):
        changed = False
        for i in range(1, len(lps)):
            for g in grid:
                ww = w.copy()
                ww[i] = g
                s = macro_f1(y, blend(ww).argmax(1))
                if s > best + 1e-9:
                    best, w, changed = s, ww, True
        if not changed:
            break
    return w / w.sum(), best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work_dir", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--name", default="ensemble")
    ap.add_argument("--template_csv", default="")
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--weights", default="search", choices=["search", "equal"])
    ap.add_argument("--tune_covid", action="store_true")
    args = ap.parse_args()

    work = Path(args.work_dir)
    out = work / "ensembles" / args.name
    out.mkdir(parents=True, exist_ok=True)
    runs = {r: load_run(work / r, args.n_folds) for r in args.runs}

    base = runs[args.runs[0]][0]
    for r, (oof, names, _, _) in runs.items():
        if not oof["out_relpath"].equals(base["out_relpath"]):
            raise SystemExit(f"{r}: OOF images differ from {args.runs[0]} - were they trained on the same folds.csv?")
        if not names["out_relpath"].equals(runs[args.runs[0]][1]["out_relpath"]):
            raise SystemExit(f"{r}: test image order differs")
    y, fold = base["y"].to_numpy(), base["fold"].to_numpy()
    lcols = [f"logit_{c}" for c in SHORT]
    lps = [log_softmax_np(runs[r][0][lcols].to_numpy()) for r in args.runs]
    tes = [runs[r][2] for r in args.runs]

    # ---- per-run summary + diversity ----
    rows = []
    for r, lp in zip(args.runs, lps):
        p = lp.argmax(1)
        pcf = per_class_f1(y, p)
        a = runs[r][3]
        rows.append({"run": r, "model": a.get("model"), "img_size": a.get("img_size"), "epochs": a.get("epochs"),
                     "oof_macro_f1": round(macro_f1(y, p), 4), "oof_acc": round(float((p == y).mean()), 4),
                     **{f"f1_{c}": round(float(v), 4) for c, v in zip(CLASSES, pcf)}})
    summ = pd.DataFrame(rows)
    print("\nPer-run OOF (argmax):\n" + summ.to_string(index=False))
    if len(lps) > 1:
        print("\nDisagreement between runs (share of OOF images where argmax differs):")
        for (i, a), (j, b) in itertools.combinations(enumerate(args.runs), 2):
            d = float((lps[i].argmax(1) != lps[j].argmax(1)).mean())
            print(f"  {a} vs {b}: {d:.3f}")

    # ---- blend ----
    eq = np.ones(len(lps)) / len(lps)
    w = search_weights(lps, y)[0] if (args.weights == "search" and len(lps) > 1) else eq
    blend = lambda ww, L: sum(wi * lp for wi, lp in zip(ww, L))  # noqa: E731
    oof_eq, oof_w = blend(eq, lps), blend(w, lps)
    classes = (1, 2, 3) if args.tune_covid else (1, 2)
    b, _ = tune_offsets(oof_w, y, classes)
    print(f"\nWeights: " + ", ".join(f"{r}={wi:.2f}" for r, wi in zip(args.runs, w)))
    report("Ensemble, equal weights, argmax", y, oof_eq.argmax(1))
    report("Ensemble, searched weights, argmax", y, oof_w.argmax(1))
    report(f"Ensemble, searched weights + offsets {np.round(b, 2).tolist()}", y, (oof_w + b).argmax(1))

    # ---- nested check: everything tuned without the scored fold ----
    nest_eq, nest_full = [], []
    for k in np.unique(fold):
        m = fold == k
        ww = search_weights([lp[~m] for lp in lps], y[~m])[0] if (args.weights == "search" and len(lps) > 1) else eq
        bb, _ = tune_offsets(blend(ww, [lp[~m] for lp in lps]), y[~m], classes)
        nest_eq.append(macro_f1(y[m], blend(eq, [lp[m] for lp in lps]).argmax(1)))
        nest_full.append(macro_f1(y[m], (blend(ww, [lp[m] for lp in lps]) + bb).argmax(1)))
    ne, nf = float(np.mean(nest_eq)), float(np.mean(nest_full))
    best_rule = "weighted_tuned" if nf > ne else "equal_argmax"
    print(f"\nNested check (mean over folds): equal+argmax {ne:.4f} | weights+offsets {nf:.4f} -> submit '{best_rule}'")

    summ = pd.concat([summ, pd.DataFrame([
        {"run": f"ENSEMBLE {args.name} (equal, argmax)", "oof_macro_f1": round(macro_f1(y, oof_eq.argmax(1)), 4),
         "nested_macro_f1": round(ne, 4)},
        {"run": f"ENSEMBLE {args.name} (weights+offsets)", "oof_macro_f1": round(macro_f1(y, (oof_w + b).argmax(1)), 4),
         "nested_macro_f1": round(nf, 4)}])], ignore_index=True)
    summ.to_csv(out / "summary.csv", index=False)
    json.dump({"runs": args.runs, "weights": w.tolist(), "offsets": b.tolist(), "nested_equal": ne,
               "nested_weighted_tuned": nf, "recommended": best_rule}, open(out / "ensemble.json", "w"), indent=2)
    print(f"\nsummary -> {out / 'summary.csv'}")

    if args.template_csv:
        fn = runs[args.runs[0]][1]["orig_relpath"].map(lambda p: Path(p).name).tolist()
        te_eq, te_w = blend(eq, tes), blend(w, tes)
        write_submission(dict(zip(fn, te_eq.argmax(1).tolist())), args.template_csv, out / "submission_equal_argmax.csv")
        write_submission(dict(zip(fn, (te_w + b).argmax(1).tolist())), args.template_csv,
                         out / "submission_weighted_tuned.csv")
        print(f"Recommended file: {out / ('submission_' + best_rule + '.csv')}")


if __name__ == "__main__":
    main()
