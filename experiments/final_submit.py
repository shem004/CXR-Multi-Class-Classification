#!/usr/bin/env python
"""
Final decision rules on top of the cross-family ensemble, each one scored on out-of-fold (OOF)
predictions before it is used on the test set.

Uses the class counts as knowledge: the spec says train / val / test share the same class
distribution, so the test set should hold about 315 normal, 556 bacteria, 299 virus, 12 COVID-19.

Rules compared (same blended probabilities, different decisions):
  1 argmax                       plain ensemble
  2 covid_topk_ens               COVID = the k films with the highest ensemble P(COVID)
  3 covid_topk_ver               COVID = the k films with the highest second-stage (verifier) score
  4 covid_topk_mix               COVID = top-k by sqrt(ensemble x verifier)
  5 <best of 2-4> + prior_match  remaining films: offsets fitted so predicted counts match the class counts

OOF simulation: inside each validation fold the true class counts of that fold play the role of
the known test counts. The verifier scores are leave-one-out.

python final_submit.py --work_dir WORK --proc_root PROC --runs convnext_s_384 swin_s_384 xrv_dn121_384 \
       --n_folds 5 --template_csv TEMPLATE --name final --test_counts 315 556 299 12
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from sklearn.model_selection import LeaveOneOut  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))         # experiments/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root (shared modules)
from analyze import plot_confusion  # noqa: E402
from covid_verify import gallery, make_clf, xrv_features  # noqa: E402
from cxr_data import CLASSES, write_submission  # noqa: E402
from ensemble import load_run  # noqa: E402
from train import SHORT, log_softmax_np, macro_f1, per_class_f1  # noqa: E402


def prior_match(logp3, counts, iters=300):
    """Offsets for the 3 non-COVID classes so that argmax counts match `counts` as closely as possible."""
    counts = np.asarray(counts, dtype=float)
    b, best_b, best_err = np.zeros(3), np.zeros(3), np.inf
    for i in range(iters):
        c = np.bincount((logp3 + b).argmax(1), minlength=3)
        err = np.abs(c - counts).sum()
        if err < best_err:
            best_err, best_b = err, b.copy()
        if err == 0:
            break
        b += (0.5 / (1 + i / 50)) * (np.log(counts + 0.5) - np.log(c + 0.5))
    return best_b


def decide(logp, score, k, counts3=None):
    """COVID = top-k by `score` (None -> argmax rule); the rest = best non-COVID class (+ prior matching)."""
    n = len(logp)
    if score is None:
        covid = logp.argmax(1) == 3
    else:
        covid = np.zeros(n, dtype=bool)
        covid[np.argsort(-score)[:k]] = True
    pred = np.full(n, 3)
    rest = ~covid
    l3 = logp[rest, :3]
    b = prior_match(l3, counts3) if counts3 is not None else np.zeros(3)
    pred[rest] = (l3 + b).argmax(1)
    return pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work_dir", required=True)
    ap.add_argument("--proc_root", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--name", default="final")
    ap.add_argument("--template_csv", required=True)
    ap.add_argument("--test_counts", type=int, nargs=4, default=[315, 556, 299, 12],
                    help="expected test counts: normal bacteria virus COVID-19")
    ap.add_argument("--gate", type=float, default=0.05)
    ap.add_argument("--C", type=float, default=0.01)
    a = ap.parse_args()

    work, proc = Path(a.work_dir), Path(a.proc_root)
    out = work / "ensembles" / a.name
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    lcols = [f"logit_{c}" for c in SHORT]

    runs = [load_run(work / r, a.n_folds) for r in a.runs]
    oof, names = runs[0][0], runs[0][1]
    for r, rr in zip(a.runs, runs):
        assert rr[0]["out_relpath"].equals(oof["out_relpath"]), f"{r}: different folds file than {a.runs[0]}"
    lp = np.mean([log_softmax_np(r[0][lcols].to_numpy()) for r in runs], axis=0)
    te = np.mean([r[2] for r in runs], axis=0)
    y, fold = oof["y"].to_numpy(), oof["fold"].to_numpy()
    p_cov, p_cov_te = np.exp(lp)[:, 3], np.exp(te)[:, 3]

    # per-run table
    rows = []
    for r, rr in zip(a.runs, runs):
        p = log_softmax_np(rr[0][lcols].to_numpy()).argmax(1)
        rows.append({"model": r, "oof_macro_f1": round(macro_f1(y, p), 4),
                     **{c: round(float(v), 3) for c, v in zip(CLASSES, per_class_f1(y, p))}})

    # ---- second-stage verifier scores (leave-one-out on OOF candidates; fitted on all of them for test) ----
    cand, cand_te = np.where(p_cov >= a.gate)[0], np.where(p_cov_te >= a.gate)[0]
    yc = (y[cand] == 3).astype(int)
    print(f"COVID candidates (P >= {a.gate}): train+val {len(cand)} ({yc.sum()} COVID), test {len(cand_te)}")
    X = xrv_features([proc / p for p in oof.out_relpath.iloc[cand]], device)
    Xte = xrv_features([proc / p for p in names.out_relpath.iloc[cand_te]], device)
    s_ver, s_ver_te = np.zeros(len(y)), np.zeros(len(te))
    for tr, ho in LeaveOneOut().split(X):
        s_ver[cand[ho]] = make_clf(a.C).fit(X[tr], yc[tr]).predict_proba(X[ho])[:, 1]
    s_ver_te[cand_te] = make_clf(a.C).fit(X, yc).predict_proba(Xte)[:, 1]

    def scores(kind, test=False):
        e, v = (p_cov_te, s_ver_te) if test else (p_cov, s_ver)
        return {"ens": e, "ver": v + 1e-6 * e, "mix": np.sqrt(e * v) + 1e-6 * e}[kind]

    # ---- OOF simulation, fold by fold ----
    def oof_eval(kind, prior):
        pred = np.zeros(len(y), dtype=int)
        for f in np.unique(fold):
            m = fold == f
            cnt = np.bincount(y[m], minlength=4)
            sc = None if kind is None else scores(kind)[m]
            pred[m] = decide(lp[m], sc, int(cnt[3]), cnt[:3] if prior else None)
        return pred

    variants = [("1_argmax", None, False), ("2_covid_topk_ens", "ens", False), ("3_covid_topk_ver", "ver", False),
                ("4_covid_topk_mix", "mix", False)]
    res = {}
    for nm, kind, prior in variants:
        res[nm] = (kind, prior, oof_eval(kind, prior))
    best_covid = max(list(res)[1:], key=lambda n: (per_class_f1(y, res[n][2])[3], macro_f1(y, res[n][2])))
    k_best = res[best_covid][0]
    res[f"5_{best_covid[2:]}_prior"] = (k_best, True, oof_eval(k_best, True))
    res["6_argmax_covid_prior"] = (None, True, oof_eval(None, True))

    print("\nOOF comparison of decision rules (same probabilities):")
    print(f"{'rule':<32}{'macro-F1':>9}{'normal':>8}{'bact':>7}{'virus':>7}{'COVID':>7}{'COVID found/false/missed':>27}")
    table = []
    for nm, (_, _, pred) in res.items():
        f = per_class_f1(y, pred)
        tp, fp, fn = int(((pred == 3) & (y == 3)).sum()), int(((pred == 3) & (y != 3)).sum()), int(((pred != 3) & (y == 3)).sum())
        print(f"{nm:<32}{macro_f1(y, pred):>9.4f}{f[0]:>8.3f}{f[1]:>7.3f}{f[2]:>7.3f}{f[3]:>7.3f}{f'{tp} / {fp} / {fn}':>27}")
        table.append({"rule": nm, "oof_macro_f1": round(macro_f1(y, pred), 4), **{c: round(float(v), 3) for c, v in zip(CLASSES, f)},
                      "covid_found": tp, "covid_false_alarm": fp, "covid_missed": fn})
    pd.DataFrame(rows).to_csv(out / "models_table.csv", index=False)
    pd.DataFrame(table).to_csv(out / "rules_table.csv", index=False)
    print("\nPer-model OOF (argmax):\n" + pd.DataFrame(rows).to_string(index=False))

    best = max(res, key=lambda n: macro_f1(y, res[n][2]))
    base = macro_f1(y, res["1_argmax"][2])
    print(f"\nBest rule on OOF: {best} ({macro_f1(y, res[best][2]):.4f} vs argmax {base:.4f}). "
          f"Differences under ~0.005 are within noise.")
    plot_confusion(y, [res["1_argmax"][2], res[best][2]], ["ensemble, argmax", f"ensemble, {best}"],
                   out / "final_confusion.png")

    # ---- test submissions ----
    fn_ = names["orig_relpath"].map(lambda p: Path(p).name).tolist()
    k, c3 = a.test_counts[3], a.test_counts[:3]
    picks = {}
    for nm, (kind, prior, _) in res.items():
        sc = None if kind is None else scores(kind, test=True)
        pred = decide(te, sc, k, c3 if prior else None)
        write_submission(dict(zip(fn_, pred.tolist())), a.template_csv, out / f"submission_{nm}.csv")
        picks[nm] = sorted(np.array(fn_)[pred == 3].tolist(), key=lambda s: int("".join(ch for ch in s.split(".")[0] if ch.isdigit()) or 0))
    print("\nTest films called COVID by each rule:")
    for nm, pk in picks.items():
        print(f"  {nm:<32} {len(pk):>2}: {' '.join(pk)}")

    t = names.iloc[cand_te].copy()
    t["ens"], t["ver"] = p_cov_te[cand_te], s_ver_te[cand_te]
    t = t.sort_values("ens", ascending=False)
    t["title"] = [f"{Path(o).name}\nensemble {e:.2f} | verifier {v:.2f}" for o, e, v in zip(t.orig_relpath, t.ens, t.ver)]
    gallery(t.head(36), proc, out / "test_covid_candidates.png", "Test COVID candidates (sorted by ensemble P(COVID))")
    json.dump({"runs": a.runs, "best_rule_oof": best, "test_counts": a.test_counts, "picks": picks},
              open(out / "final.json", "w"), indent=2)
    print(f"\nfiles -> {out}\nSubmit first: submission_{best}.csv ; then submission_1_argmax.csv as the reference.")


if __name__ == "__main__":
    main()
