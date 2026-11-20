#!/usr/bin/env python
"""
Final decision rule: average of several k-fold runs + majority vote for COVID-19.

  normal / bacteria / virus = argmax of the averaged log-probabilities (COVID column ignored)
  COVID-19                  = more than half of the runs predict COVID-19 on their own

No parameter is tuned. Prints the out-of-fold (OOF) comparison with plain averaging and with each
single run, and writes the submission in the template's row order.

python vote_ensemble.py --work_dir WORK --runs convnext_s_384 cnx_s_512 ... --n_folds 5 \
                        --template_csv test_data_sample.csv --out submission_vote.csv
(--runs all = every run in WORK with all folds finished)
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cxr_data import CLASSES, write_submission  # noqa: E402
from ensemble import load_run  # noqa: E402
from train import SHORT, log_softmax_np, macro_f1, per_class_f1  # noqa: E402


def vote(logps, k=None):
    """logps: list of (N, 4) log-probability arrays, one per run."""
    k = k or len(logps) // 2 + 1
    avg = sum(logps) / len(logps)
    votes = sum((lp.argmax(1) == 3).astype(int) for lp in logps)
    pred = avg[:, :3].argmax(1)
    pred[votes >= k] = 3
    return pred, votes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work_dir", required=True)
    ap.add_argument("--runs", nargs="+", default=["all"])
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--template_csv", default="")
    ap.add_argument("--out", default="submission_vote.csv")
    a = ap.parse_args()

    work = Path(a.work_dir)
    runs = a.runs
    if runs == ["all"]:
        runs = sorted(p.name for p in work.iterdir()
                      if len(list(p.glob("fold*/DONE.json"))) == a.n_folds)
    R = {r: load_run(work / r, a.n_folds) for r in runs}
    base = R[runs[0]]
    for r in runs:
        assert R[r][0]["out_relpath"].equals(base[0]["out_relpath"]), f"{r}: different folds file"
        assert R[r][1]["out_relpath"].equals(base[1]["out_relpath"]), f"{r}: different test order"
    lcols = [f"logit_{c}" for c in SHORT]
    y = base[0]["y"].to_numpy()
    oof = [log_softmax_np(R[r][0][lcols].to_numpy()) for r in runs]

    def line(name, p):
        f = per_class_f1(y, p)
        fa, miss = int(((p == 3) & (y != 3)).sum()), int(((p != 3) & (y == 3)).sum())
        print(f"{name:<44}{macro_f1(y, p):.4f}  " + " ".join(f"{c[:4]} {v:.3f}" for c, v in zip(CLASSES, f))
              + f"  COVID false/missed {fa}/{miss}")

    print(f"{len(runs)} runs, {len(y)} OOF images\n")
    for r, lp in zip(runs, oof):
        line(r, lp.argmax(1))
    line("average", (sum(oof) / len(oof)).argmax(1))
    pred, _ = vote(oof)
    line(f"average + COVID vote (>= {len(runs) // 2 + 1} of {len(runs)})", pred)

    if a.template_csv:
        te = [R[r][2] for r in runs]  # fold-averaged test log-probabilities per run
        p_te, votes = vote(te)
        names = base[1]["orig_relpath"].map(lambda s: Path(s).name).tolist()
        write_submission(dict(zip(names, p_te.tolist())), a.template_csv, a.out)
        split = [(n, int(v)) for n, v in zip(names, votes) if 0 < v < len(runs)]
        print("test images with split COVID votes:", split)


if __name__ == "__main__":
    main()
