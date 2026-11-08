#!/usr/bin/env python3
"""Measure how far non-visual cues go on each dataset (character-set shortcut).

Gradient-boosting classifiers are trained on the training split and scored on the test split, each with a
different feature set. None of them sees an image.

  candidate_non_ascii      number of non-ASCII characters in the candidate, and whether there is any
  candidate_character_set  composition of the candidate alone: counts of non-ASCII, non-Latin, combining,
                           NFKC-changing, digit and hyphen characters
  character_set            the same composition for both strings (the classifier can now compare the
                           counts of the two strings, e.g. equal digit counts in a spoof pair)
  lengths                  the two string lengths
  character_set_edit       all of the above plus Levenshtein distance, NFKC-casefold Levenshtein distance and
                           the number of candidate characters absent from the reference

The character-set shortcut is a cue in the candidate alone: if candidate_non_ascii or candidate_character_set
reaches a high ROC-AUC, the class can be read off the candidate's character set without comparing it with
the reference. On the shortcut-controlled data (make_controlled.py) both should fall to about 0.5.

Writes $WORK/results/shortcut_baseline.csv.

Example:
  python shortcut_baseline.py --data woodbridge_p0 synthetic_p0 synthetic_ctrl_p0
"""

from __future__ import annotations

import argparse
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz.distance import Levenshtein
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402


def composition(text: str) -> list[float]:
    non_ascii = sum(ord(ch) > 127 for ch in text)
    non_latin = sum(ch.isalpha() and not unicodedata.name(ch, "").startswith("LATIN") for ch in text)
    combining = sum(unicodedata.combining(ch) > 0 for ch in text)
    nfkc = sum(unicodedata.normalize("NFKC", ch) != ch for ch in text)
    return [non_ascii, non_latin, combining, nfkc, sum(ch.isdigit() for ch in text), text.count("-")]


def features(rows, kind: str) -> np.ndarray:
    out = []
    for cand, ref, _ in rows:
        n = sum(ord(ch) > 127 for ch in cand)
        if kind == "candidate_non_ascii":
            out.append([n, float(n > 0)])
            continue
        if kind == "candidate_character_set":
            out.append(composition(cand))
            continue
        if kind == "lengths":
            out.append([len(cand), len(ref)])
            continue
        row = composition(cand) + composition(ref)
        if kind == "character_set_edit":
            row += [len(cand), len(ref)]
            norm = lambda s: unicodedata.normalize("NFKC", s).casefold()  # noqa: E731
            row += [Levenshtein.distance(cand, ref), Levenshtein.distance(norm(cand), norm(ref)),
                    sum(ch not in ref for ch in cand)]
        out.append(row)
    return np.asarray(out, dtype=np.float32)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", nargs="+", required=True, help="dataset folders under $WORK/data")
    ap.add_argument("--max-train", type=int, default=300_000, help="training pairs used (random subsample)")
    ap.add_argument("--work", default=None)
    args = ap.parse_args()
    work = common.work_root(args.work)

    results = []
    for name in args.data:
        train = common.read_rows(work / "data" / name / "train.pkl")
        test = common.read_rows(work / "data" / name / "test.pkl")
        rng = np.random.default_rng(0)
        if len(train) > args.max_train:
            train = [train[i] for i in rng.choice(len(train), args.max_train, replace=False)]
        y_train = np.array([r[2] for r in train])
        y_test = np.array([r[2] for r in test])
        for kind in ("candidate_non_ascii", "candidate_character_set", "character_set", "lengths", "character_set_edit"):
            model = HistGradientBoostingClassifier(random_state=0).fit(features(train, kind), y_train)
            scores = model.predict_proba(features(test, kind))[:, 1]
            auc = roc_auc_score(y_test, scores)
            results.append({"data": name, "features": kind, "test_roc_auc": auc,
                            "test_tpr_at_fpr_0.01": common.tpr_at_fpr(y_test, scores, 0.01)})
            print(f"{name:22s} {kind:20s} test ROC-AUC {auc:.4f}", flush=True)

    out = work / "results" / "shortcut_baseline.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(results)
    if out.exists():
        old = pd.read_csv(out)
        frame = pd.concat([old[~old.data.isin(frame.data)], frame], ignore_index=True)
    frame.to_csv(out, index=False)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
