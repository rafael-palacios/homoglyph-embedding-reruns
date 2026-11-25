#!/usr/bin/env python3
"""Does the process-name test set carry the character-set shortcut? (CPU only, a few minutes)

Two checks:
1. Composition: for positives and negatives of each pair file, the share of candidates with at least one
   non-ASCII character and the mean number of non-ASCII characters. If positives have them and negatives
   do not, the class can be read off the candidate alone, as in the original Woodbridge benchmark.
2. Baselines (also scored on the ASCII-only candidates, a subset without the shortcut): the gradient-boosting baselines of shortcut_baseline.py, trained on the training split of
   each --train dataset (the original and the shortcut-controlled Woodbridge data) and scored on the
   process-name test set, as the visual models trained on those data were. The candidate-only feature
   sets (candidate_non_ascii, candidate_character_set) measure the shortcut; character_set_edit is the
   string-edit baseline of the paper.

Writes $WORK/results/process_baselines.csv and process_composition.csv.

Example:
  python process_baselines.py --train woodbridge_p0 woodbridge_ctrl_p0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
from shortcut_baseline import features  # noqa: E402

KINDS = ("candidate_non_ascii", "candidate_character_set", "character_set", "lengths", "character_set_edit")


def composition(name: str, rows) -> list[dict]:
    out = []
    for label, cls in ((1, "positive"), (0, "negative")):
        n = np.array([sum(ord(ch) > 127 for ch in cand) for cand, _, y in rows if y == label])
        out.append({"pairs": name, "class": cls, "count": len(n),
                    "share_with_non_ascii": float((n > 0).mean()) if len(n) else float("nan"),
                    "mean_non_ascii": float(n.mean()) if len(n) else float("nan")})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", nargs="+", default=["woodbridge_p0", "woodbridge_ctrl_p0"],
                    help="dataset folders under $WORK/data whose training split trains the baselines")
    ap.add_argument("--test", default="process", help="dataset folder of the test pairs")
    ap.add_argument("--max-train", type=int, default=300_000, help="training pairs used (random subsample)")
    ap.add_argument("--work", default=None)
    args = ap.parse_args()
    work = common.work_root(args.work)
    res = work / "results"
    res.mkdir(parents=True, exist_ok=True)

    test = common.read_rows(work / "data" / args.test / "test.pkl")
    y_test = np.array([r[2] for r in test])
    # Shortcut-free subset: candidates without non-ASCII characters (all negatives of the process set, and
    # the ASCII-only spoofs), so the candidate's character set carries no class information
    ascii_only = np.array([all(ord(ch) < 128 for ch in cand) for cand, _, _ in test])

    comp = composition(f"{args.test}/test", test)
    for name in args.train:
        comp += composition(f"{name}/test", common.read_rows(work / "data" / name / "test.pkl"))
    comp = pd.DataFrame(comp)
    comp.to_csv(res / "process_composition.csv", index=False)
    print(comp.to_string(index=False), flush=True)

    results = []
    for name in args.train:
        train = common.read_rows(work / "data" / name / "train.pkl")
        rng = np.random.default_rng(0)  # same subsample as shortcut_baseline.py and subsets.py
        if len(train) > args.max_train:
            train = [train[i] for i in rng.choice(len(train), args.max_train, replace=False)]
        y_train = np.array([r[2] for r in train])
        for kind in KINDS:
            model = HistGradientBoostingClassifier(random_state=0).fit(features(train, kind), y_train)
            scores = model.predict_proba(features(test, kind))[:, 1]
            auc = roc_auc_score(y_test, scores)
            sub = roc_auc_score(y_test[ascii_only], scores[ascii_only])
            results.append({"trained_on": name, "tested_on": args.test, "features": kind, "test_roc_auc": auc,
                            "test_tpr_at_fpr_0.01": common.tpr_at_fpr(y_test, scores, 0.01),
                            "ascii_only_roc_auc": sub})
            print(f"trained on {name:20s} {kind:24s} {args.test} ROC-AUC {auc:.4f}, ASCII-only candidates {sub:.4f}",
                  flush=True)
    pd.DataFrame(results).to_csv(res / "process_baselines.csv", index=False)
    print(f"wrote {res / 'process_baselines.csv'} and {res / 'process_composition.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
