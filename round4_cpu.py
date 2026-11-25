#!/usr/bin/env python3
"""Round-4 CPU analyses (no GPU): string baselines on every partition, fusion with the visual models, and
the overlap of names between splits.

1. String baselines on every partition (string_baselines.csv, string_vs_visual.csv)
   The string-edit baseline (gradient boosting on composition, lengths and edit distances, the
   character_set_edit features of shortcut_baseline.py) is trained on the FULL training split of each
   partition, the same pairs the visual models are trained on, and scored on that partition's test split with
   ROC-AUC and TPR at FPR 0.1% and 1%. The training-free UTS #39 skeleton baseline is scored too (with
   --confusables). string_vs_visual.csv compares each visual model with the string-edit baseline partition by
   partition (model: mean over its seeds) and over the partitions (mean difference, 95% t interval).

2. Fusion of visual and string evidence (fusion.csv, fusion_runs.csv)
   For every visual run, its test scores are combined with the string-edit scores of the same test pairs by a
   logistic regression on the two scores. The combiner is cross-fitted on the test split in two folds grouped
   by reference name (fit on one half, score the other), so no pair is scored by a combiner that saw it and no
   reference name is shared between the halves. Visual, string and fused scores are compared on the same pairs.

3. Overlap of names between splits (overlap.csv)
   For every benchmark and partition: test reference names found among the training references; test
   negative candidates (legitimate names) found in the training split, as a reference or as a candidate;
   test positive candidates found in the training split; test pairs present in the training split; and test
   pairs that contain no string of the training split.

Writes $WORK/results/{string_baselines,string_vs_visual,fusion,fusion_runs,overlap}.csv and prints a report.

Example:
  python round4_cpu.py --confusables $WORK/data/confusables.txt
"""

from __future__ import annotations

import argparse
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz.distance import Levenshtein
from scipy.stats import t as student_t
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
from shortcut_baseline import features  # noqa: E402

FAMILIES = ("woodbridge", "woodbridge_ctrl", "synthetic", "synthetic_ctrl")


def load_confusables(path: Path) -> dict[str, str]:
    table = {}
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            source, target = (f.strip() for f in line.split(";")[:2])
            table[chr(int(source, 16))] = "".join(chr(int(h, 16)) for h in target.split())
    return table


def skeleton_scores(rows, table: dict[str, str]) -> np.ndarray:
    import unicodedata

    def skel(text: str) -> str:
        text = unicodedata.normalize("NFD", text)
        return unicodedata.normalize("NFD", "".join(table.get(ch, ch) for ch in text))

    out = []
    for cand, ref, _ in rows:
        a, b = skel(cand), skel(ref)
        out.append(-Levenshtein.distance(a, b) / max(len(a), len(b), 1))  # closer skeletons = more likely a spoof
    return np.array(out)


def metrics(y: np.ndarray, s: np.ndarray) -> dict:
    return {"roc_auc": roc_auc_score(y, s), "tpr_at_fpr_0.001": common.tpr_at_fpr(y, s, 0.001),
            "tpr_at_fpr_0.01": common.tpr_at_fpr(y, s, 0.01)}


def t_interval(values: pd.Series) -> tuple[float, float]:
    k = len(values)
    if k < 2:
        return float("nan"), float("nan")
    half = student_t.ppf(0.975, k - 1) * values.std() / np.sqrt(k)
    return values.mean() - half, values.mean() + half


def to_feature(s: np.ndarray) -> np.ndarray:
    """Probabilities are mapped to logits; other scores (logits) are used as they are."""
    s = s.astype(np.float64)
    if s.min() >= 0 and s.max() <= 1:
        s = np.clip(s, 1e-7, 1 - 1e-7)
        s = np.log(s / (1 - s))
    return s


def cross_fit(y: np.ndarray, x: np.ndarray, refs: list[str]) -> np.ndarray:
    fold = np.array([zlib.crc32(r.encode("utf-8")) % 2 for r in refs])
    out = np.empty(len(y))
    for k in (0, 1):
        train, test = fold != k, fold == k
        mu, sd = x[train].mean(0), x[train].std(0) + 1e-12
        model = LogisticRegression(max_iter=1000).fit((x[train] - mu) / sd, y[train])
        out[test] = model.decision_function((x[test] - mu) / sd)
    return out


def overlap(work: Path, family: str, partition: int) -> dict | None:
    folder = work / "data" / f"{family}_p{partition}"
    if not (folder / "test.pkl").exists():
        return None
    train = common.read_rows(folder / "train.pkl")
    test = common.read_rows(folder / "test.pkl")
    train_refs = {r for _, r, _ in train}
    train_strings = train_refs | {c for c, _, _ in train}
    train_pairs = {(c, r) for c, r, _ in train}
    neg = [(c, r) for c, r, y in test if y == 0]
    pos = [(c, r) for c, r, y in test if y == 1]
    test_refs = {r for _, r, _ in test}
    return {
        "family": family, "partition": partition, "test_pairs": len(test),
        "test_refs_in_train_refs": len(test_refs & train_refs),
        "test_negative_candidates": len(neg),
        "test_negative_candidates_are_train_refs": sum(c in train_refs for c, _ in neg),
        "test_negative_candidates_in_train": sum(c in train_strings for c, _ in neg),
        "test_positive_candidates_in_train": sum(c in train_strings for c, _ in pos),
        "test_pairs_in_train": sum((c, r) in train_pairs for c, r, _ in test),
        "share_test_pairs_no_train_string": float(np.mean([c not in train_strings and r not in train_strings
                                                           for c, r, _ in test])),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--confusables", type=Path, default=None, help="Unicode confusables.txt (skeleton baseline)")
    ap.add_argument("--families", nargs="+", default=list(FAMILIES))
    ap.add_argument("--partitions", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--work", default=None)
    args = ap.parse_args()
    work = common.work_root(args.work)
    res = work / "results"
    res.mkdir(parents=True, exist_ok=True)
    table = load_confusables(args.confusables) if args.confusables else None

    # 3. Overlap (cheap, first)
    ov = [o for f in args.families for p in args.partitions if (o := overlap(work, f, p))]
    ov = pd.DataFrame(ov)
    ov.to_csv(res / "overlap.csv", index=False)
    print("Overlap of names between the training and test splits")
    print(ov.to_string(index=False), flush=True)

    # 1. String baselines on every partition, and their scores for the fusion
    base, string_scores = [], {}
    for family in args.families:
        for p in args.partitions:
            folder = work / "data" / f"{family}_p{p}"
            if not (folder / "train.pkl").exists():
                continue
            train = common.read_rows(folder / "train.pkl")
            test = common.read_rows(folder / "test.pkl")
            y = np.array([r[2] for r in test])
            model = HistGradientBoostingClassifier(random_state=0).fit(
                features(train, "character_set_edit"), np.array([r[2] for r in train]))
            s = model.predict_proba(features(test, "character_set_edit"))[:, 1]
            string_scores[(family, p)] = (s, y, [r[1] for r in test])
            base.append({"family": family, "partition": p, "method": "string_edit", "train_pairs": len(train),
                         **metrics(y, s)})
            if table is not None:
                base.append({"family": family, "partition": p, "method": "skeleton", "train_pairs": 0,
                             **metrics(y, skeleton_scores(test, table))})
            print(f"{family}_p{p}: string-edit ROC-AUC {base[-2 if table else -1]['roc_auc']:.5f} "
                  f"(trained on {len(train):,} pairs)", flush=True)
    base = pd.DataFrame(base)
    base.to_csv(res / "string_baselines.csv", index=False)

    # Visual runs (test split, configurations of the paper)
    runs = []
    for path in sorted((work / "runs").glob("*_p*/*/seed*/evals/test/scores.npy")):
        run = path.parents[2]
        data, model = run.parent.parent.name, run.parent.name
        family, _, part = data.rpartition("_p")
        if "__" in model or family not in args.families or int(part) not in args.partitions:
            continue
        if (family, int(part)) not in string_scores:
            continue
        runs.append((family, int(part), model, int(run.name.removeprefix("seed")), path))

    # 2. Fusion, and per-run visual metrics for the comparison with the string baseline
    fusion = []
    for family, p, model, seed, path in runs:
        s_str, y, refs = string_scores[(family, p)]
        labels = np.load(path.parent / "labels.npy")
        if not np.array_equal(labels, y):
            raise SystemExit(f"labels of {path.parent} do not match {family}_p{p}/test.pkl")
        s_vis = np.load(path).astype(np.float64)
        fused = cross_fit(y, np.column_stack([to_feature(s_vis), to_feature(s_str)]), refs)
        row = {"family": family, "partition": p, "model": model, "seed": seed}
        for name, s in (("visual", s_vis), ("string", s_str), ("fused", fused)):
            row.update({f"{name}_{k}": v for k, v in metrics(y, s).items()})
        fusion.append(row)
    fusion = pd.DataFrame(fusion)
    fusion.to_csv(res / "fusion_runs.csv", index=False)
    cols = [c for c in fusion.columns if c.startswith(("visual_", "string_", "fused_"))]
    g = fusion.groupby(["family", "model"])[cols]
    summary = g.mean().add_suffix("_mean").join(g.std().add_suffix("_sd")).join(g.size().rename("runs")).reset_index()
    summary.to_csv(res / "fusion.csv", index=False)

    # Visual model vs string-edit baseline, partition by partition and over the partitions
    vs = []
    se = base[base.method == "string_edit"].set_index(["family", "partition"])
    per_part = fusion.groupby(["family", "model", "partition"])[["visual_roc_auc", "visual_tpr_at_fpr_0.001",
                                                                 "visual_tpr_at_fpr_0.01"]].mean().reset_index()
    for (family, model), g in per_part.groupby(["family", "model"]):
        row = {"family": family, "model": model, "partitions": len(g)}
        for metric in ("roc_auc", "tpr_at_fpr_0.001", "tpr_at_fpr_0.01"):
            diff = pd.Series([r[f"visual_{metric}"] - se.loc[(family, r.partition), metric] for _, r in g.iterrows()])
            low, high = t_interval(diff)
            row.update({f"{metric}_visual": g[f"visual_{metric}"].mean(),
                        f"{metric}_string": float(np.mean([se.loc[(family, q), metric] for q in g.partition])),
                        f"{metric}_diff": diff.mean(), f"{metric}_diff_ci95_low": low,
                        f"{metric}_diff_ci95_high": high, f"{metric}_partitions_visual_better": int((diff > 0).sum())})
        vs.append(row)
    vs = pd.DataFrame(vs)
    vs.to_csv(res / "string_vs_visual.csv", index=False)

    print("\nString baselines (full training split of each partition)")
    print(base.to_string(index=False))
    print("\nVisual model minus string-edit baseline (model: mean over seeds; interval over partitions)")
    for _, r in vs.iterrows():
        print(f"  {r.family:16s} {r.model:24s} ROC-AUC {r.roc_auc_visual:.5f} vs {r.roc_auc_string:.5f}: "
              f"{r.roc_auc_diff:+.5f} [{r.roc_auc_diff_ci95_low:+.5f}, {r.roc_auc_diff_ci95_high:+.5f}], "
              f"better in {r.roc_auc_partitions_visual_better}/{r.partitions} partitions; "
              f"TPR@0.1%FPR {r['tpr_at_fpr_0.001_visual']:.4f} vs {r['tpr_at_fpr_0.001_string']:.4f}")
    print("\nFusion (cross-fitted logistic regression on the visual and string-edit scores; mean over runs)")
    for _, r in summary.iterrows():
        print(f"  {r.family:16s} {r.model:24s} ROC-AUC visual {r.visual_roc_auc_mean:.5f}  string "
              f"{r.string_roc_auc_mean:.5f}  fused {r.fused_roc_auc_mean:.5f}   TPR@0.1%FPR "
              f"{r['visual_tpr_at_fpr_0.001_mean']:.4f} / {r['string_tpr_at_fpr_0.001_mean']:.4f} / "
              f"{r['fused_tpr_at_fpr_0.001_mean']:.4f} (n={int(r.runs)})")
    print(f"\nwrote {res}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
