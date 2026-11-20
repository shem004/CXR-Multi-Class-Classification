#!/usr/bin/env python
"""
Second-stage COVID check.

Problem: the 4-class models call every adult-looking film COVID (8 false alarms on train+val,
and 15 COVID predictions on a test set that should hold ~12). They rarely miss a real COVID film.

Idea: only for "COVID candidates" (ensemble P(COVID) >= --gate), ask a second, small classifier
"COVID or not?", built on FROZEN features of a network pretrained on ~800k adult chest X-rays
(TorchXRayVision DenseNet121). Those features describe adult lung findings, so they are not
dominated by "adult vs child" the way the fine-tuned 4-class model is.

  * candidates: OOF images with P(COVID) >= gate (true COVID + the COVID-like negatives); same rule on test
  * features: 1024-d pooled XRV features (image + horizontal flip averaged) + the 18 XRV pathology scores
  * classifier: standardise -> logistic regression (L2, class-balanced)
  * honest estimate: leave-one-out over the candidates; macro-F1 on all OOF images before / after
  * writes submission_verified.csv; says which file to submit based on the leave-one-out result

python covid_verify.py --work_dir WORK --proc_root PROC --runs cnx_k4 cnx_k4_hardneg ... --n_folds 4 \
                       --template_csv TEMPLATE --name ens_k4
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
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import LeaveOneOut  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))         # experiments/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root (shared modules)
from cxr_data import CLASSES, write_submission  # noqa: E402
from ensemble import load_run  # noqa: E402
from train import SHORT, log_softmax_np, macro_f1, per_class_f1  # noqa: E402


@torch.no_grad()
def xrv_features(paths, device, bs=16):
    import torchxrayvision as xrv
    model = xrv.models.DenseNet(weights="densenet121-res224-all").to(device).eval()
    feats = []
    for i in range(0, len(paths), bs):
        batch = []
        for p in paths[i:i + bs]:
            g = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            g = cv2.resize(g, (224, 224), interpolation=cv2.INTER_AREA).astype(np.float32)
            batch.append((g / 255.0 - 0.5) * 2048.0)  # XRV convention: [-1024, 1024]
        x = torch.from_numpy(np.stack(batch))[:, None].to(device)
        out = []
        for xx in (x, torch.flip(x, dims=[3])):
            f = F.adaptive_avg_pool2d(F.relu(model.features(xx)), 1).flatten(1)
            out.append(torch.cat([f, model.classifier(f)], dim=1))
        feats.append(((out[0] + out[1]) / 2).cpu().numpy())
    return np.concatenate(feats)


def make_clf(C):
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, class_weight="balanced", max_iter=5000))


def gallery(df, proc_root, path, title, ncol=6):
    n = len(df)
    if n == 0:
        return
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.6 * ncol, 2.9 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for ax, r in zip(axes, df.itertuples()):
        ax.imshow(cv2.imread(str(Path(proc_root) / r.out_relpath), cv2.IMREAD_GRAYSCALE), cmap="gray")
        ax.set_title(r.title, fontsize=7.5)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.35 / nrow))
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work_dir", required=True)
    ap.add_argument("--proc_root", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--n_folds", type=int, default=4)
    ap.add_argument("--name", default="ens")
    ap.add_argument("--template_csv", default="")
    ap.add_argument("--gate", type=float, default=0.05, help="P(COVID) above this = candidate")
    ap.add_argument("--C", type=float, nargs="+", default=[0.001, 0.01, 0.1])
    ap.add_argument("--thr", type=float, default=0.5, help="verifier probability needed to keep COVID")
    a = ap.parse_args()

    work, proc = Path(a.work_dir), Path(a.proc_root)
    out = work / "ensembles" / a.name
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    lcols = [f"logit_{c}" for c in SHORT]

    runs = [load_run(work / r, a.n_folds) for r in a.runs]
    oof = runs[0][0][["out_relpath", "orig_relpath", "label", "y"]].copy()
    names = runs[0][1]
    lp = np.mean([log_softmax_np(r[0][lcols].to_numpy()) for r in runs], axis=0)   # equal-weight blend
    te = np.mean([r[2] for r in runs], axis=0)
    y = oof["y"].to_numpy()
    base_pred, base_te = lp.argmax(1), te.argmax(1)
    p_cov, p_cov_te = np.exp(lp)[:, 3], np.exp(te)[:, 3]

    cand, cand_te = np.where(p_cov >= a.gate)[0], np.where(p_cov_te >= a.gate)[0]
    yc = (y[cand] == 3).astype(int)
    print(f"Candidates (P(COVID) >= {a.gate}): train+val {len(cand)} = {yc.sum()} COVID + {len(yc) - yc.sum()} "
          f"non-COVID | test {len(cand_te)}")
    print(f"COVID films outside the candidate set (cannot be rescued): {int(((y == 3) & (p_cov < a.gate)).sum())}")

    X = xrv_features([proc / p for p in oof.out_relpath.iloc[cand]], device)
    Xte = xrv_features([proc / p for p in names.out_relpath.iloc[cand_te]], device)

    def apply(pred, idx, pv, logp):
        """candidates: COVID iff verifier >= thr; otherwise the best non-COVID class of the ensemble"""
        new = pred.copy()
        non_covid = logp[:, :3].argmax(1)
        new[idx] = np.where(pv >= a.thr, 3, non_covid[idx])
        return new

    # leave-one-out for each C; the reported result uses the C chosen INSIDE each left-out split? -> too few
    # samples for that, so we report every C and pick the best one openly (optimistic by a little; said below)
    results = {}
    for C in a.C:
        pv = np.zeros(len(cand))
        for tr, ho in LeaveOneOut().split(X):
            pv[ho] = make_clf(C).fit(X[tr], yc[tr]).predict_proba(X[ho])[:, 1]
        new = apply(base_pred, cand, pv, lp)
        results[C] = (macro_f1(y, new), pv, new)
    f_base = macro_f1(y, base_pred)

    def covid_stats(pred):
        tp = int(((pred == 3) & (y == 3)).sum())
        return tp, int(((pred == 3) & (y != 3)).sum()), int(((pred != 3) & (y == 3)).sum())

    print("\n                         macro-F1  COVID F1   found  false alarms  missed")
    tp, fp, fn = covid_stats(base_pred)
    print(f"ensemble only            {f_base:.4f}    {per_class_f1(y, base_pred)[3]:.3f}     {tp:>3}      {fp:>3}        {fn:>3}")
    for C, (f, pv, new) in results.items():
        tp, fp, fn = covid_stats(new)
        print(f"+ verifier (C={C:<6})    {f:.4f}    {per_class_f1(y, new)[3]:.3f}     {tp:>3}      {fp:>3}        {fn:>3}")
    best_C = max(results, key=lambda c: results[c][0])
    f_best, pv, new = results[best_C]
    gain = f_best - f_base
    print(f"\nBest C = {best_C}: macro-F1 {f_base:.4f} -> {f_best:.4f} ({gain:+.4f}), leave-one-out over "
          f"{len(cand)} candidates. C was picked on the same data, so treat gains under ~0.005 as noise.")

    # which candidates change
    c_df = oof.iloc[cand].copy()
    c_df["p_covid_ensemble"], c_df["p_covid_verifier"] = p_cov[cand], pv
    c_df["before"], c_df["after"] = [CLASSES[i] for i in base_pred[cand]], [CLASSES[i] for i in new[cand]]
    c_df.sort_values("p_covid_verifier", ascending=False).to_csv(out / "verifier_candidates_trainval.csv", index=False)
    changed = c_df[c_df.before != c_df.after]
    print(f"\nTrain+val predictions changed by the verifier: {len(changed)}")
    if len(changed):
        print(changed[["orig_relpath", "label", "before", "after", "p_covid_ensemble", "p_covid_verifier"]]
              .round(3).to_string(index=False))
    g = c_df.sort_values("p_covid_verifier", ascending=False)
    g["title"] = [f"{Path(o).name} | true {l}\nens {e:.2f} | verifier {v:.2f}" for o, l, e, v in
                  zip(g.orig_relpath, g.label, g.p_covid_ensemble, g.p_covid_verifier)]
    gallery(g.head(72), proc, out / "verifier_trainval.png",
            "Train+val COVID candidates, sorted by verifier probability (leave-one-out)")

    # test
    clf = make_clf(best_C).fit(X, yc)
    pv_te = clf.predict_proba(Xte)[:, 1] if len(cand_te) else np.zeros(0)
    new_te = apply(base_te, cand_te, pv_te, te)
    t_df = names.iloc[cand_te].copy()
    t_df["p_covid_ensemble"], t_df["p_covid_verifier"] = p_cov_te[cand_te], pv_te
    t_df["before"], t_df["after"] = [CLASSES[i] for i in base_te[cand_te]], [CLASSES[i] for i in new_te[cand_te]]
    t_df = t_df.sort_values("p_covid_verifier", ascending=False)
    t_df.to_csv(out / "verifier_candidates_test.csv", index=False)
    print(f"\nTest: COVID predictions {int((base_te == 3).sum())} -> {int((new_te == 3).sum())} "
          f"(the spec's class share suggests ~12)")
    print(t_df[["orig_relpath", "before", "after", "p_covid_ensemble", "p_covid_verifier"]].round(3).to_string(index=False))
    t_df["title"] = [f"{Path(o).name}\nens {e:.2f} | verifier {v:.2f} -> {aft}" for o, e, v, aft in
                     zip(t_df.orig_relpath, t_df.p_covid_ensemble, t_df.p_covid_verifier, t_df.after)]
    gallery(t_df.head(48), proc, out / "verifier_test.png", "Test COVID candidates, sorted by verifier probability")

    rec = "verified" if gain > 0.005 else "ensemble"
    json.dump({"gate": a.gate, "best_C": best_C, "oof_macro_f1_ensemble": f_base, "oof_macro_f1_verified": f_best,
               "test_covid_before": int((base_te == 3).sum()), "test_covid_after": int((new_te == 3).sum()),
               "recommended": rec}, open(out / "verifier.json", "w"), indent=2)
    if a.template_csv:
        fn = names["orig_relpath"].map(lambda p: Path(p).name).tolist()
        write_submission(dict(zip(fn, new_te.tolist())), a.template_csv, out / "submission_verified.csv")
    print(f"\nRecommendation from leave-one-out: submit the '{rec}' file "
          f"({'gain above the noise level' if rec == 'verified' else 'no clear gain from the verifier'}).")


if __name__ == "__main__":
    main()
