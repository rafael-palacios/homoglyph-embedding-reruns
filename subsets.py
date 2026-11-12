#!/usr/bin/env python3
"""Score models and non-visual baselines on subsets of the test pairs that remove length and width cues.

Every finished evaluation under $WORK/runs (except the unseen-font ones) is re-scored from its saved
scores.npy on these subsets of its pair file:

  all                 every pair
  equal_length        candidate and reference have the same number of characters
  equal_width         the rendered text widths differ by at most --width-tolerance pixels (DejaVu Sans,
                      32-pixel images, as in training); the image width, the number of slices and the
                      amount of padding then carry no information about the class
  equal_length_width  both of the above
  edit_1, edit_2, edit_3+
                      Levenshtein distance between candidate and reference of 1, 2, and 3 or more

Non-visual baselines are scored on the same subsets:

  lengths, widths     gradient boosting on the two string lengths, or on the two rendered text widths
  string_edit         gradient boosting on the character composition of both strings, their lengths and
                      edit distances (the "all of the above" features of shortcut_baseline.py)
  skeleton            training-free: Levenshtein distance between the Unicode confusable skeletons of the
                      two strings (UTS #39), normalized by the longer skeleton; needs confusables.txt
                      (https://www.unicode.org/Public/security/latest/confusables.txt)

The gradient-boosting baselines are trained on the training split of the same data folder (300,000 pairs
sampled at random). For the process-name test split, they are trained on --process-train.

Writes $WORK/results/subsets_runs.csv (one row per evaluation and subset), subsets_baselines.csv and
subsets_summary.csv (models: mean +/- s.d. over runs).

Example:
  python subsets.py --confusables $WORK/data/confusables.txt
"""

from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz.distance import Levenshtein
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
from shortcut_baseline import features as string_features  # noqa: E402

SUBSETS = ("all", "equal_length", "equal_width", "equal_length_width", "edit_1", "edit_2", "edit_3+")


def text_widths(names: list[str]) -> np.ndarray:
    """Width in pixels of each name's text as rendered for training (rendering.renderer.render_name)."""
    from matplotlib import font_manager
    from PIL import Image, ImageDraw, ImageFont

    font = ImageFont.truetype(font_manager.findfont("DejaVu Sans", fallback_to_default=False), int(32 * 0.8))
    draw = ImageDraw.Draw(Image.new("L", (1, 1)))

    @lru_cache(maxsize=None)
    def width(name: str) -> int:
        box = draw.textbbox((0, 0), name, font=font)
        return box[2] - box[0]

    return np.array([width(n) for n in names])


def load_confusables(path: Path) -> dict[str, str]:
    table = {}
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        source, target = (field.strip() for field in line.split(";")[:2])
        table[chr(int(source, 16))] = "".join(chr(int(h, 16)) for h in target.split())
    return table


def skeleton(text: str, table: dict[str, str]) -> str:
    """UTS #39 skeleton: NFD, map every character to its confusable prototype, NFD again."""
    text = unicodedata.normalize("NFD", text)
    return unicodedata.normalize("NFD", "".join(table.get(ch, ch) for ch in text))


class Pairs:
    """A pair file with the attributes that define the subsets."""

    def __init__(self, path: Path, width_tolerance: int) -> None:
        self.path = path
        self.rows = common.read_rows(path)
        cand = [r[0] for r in self.rows]
        ref = [r[1] for r in self.rows]
        self.labels = np.array([r[2] for r in self.rows], dtype=np.int64)
        same_length = np.array([len(a) == len(b) for a, b in zip(cand, ref)])
        same_width = np.abs(text_widths(cand) - text_widths(ref)) <= width_tolerance
        edit = np.array([Levenshtein.distance(a, b) for a, b in zip(cand, ref)])
        self.masks = {"all": np.ones(len(self.rows), dtype=bool), "equal_length": same_length,
                      "equal_width": same_width, "equal_length_width": same_length & same_width,
                      "edit_1": edit == 1, "edit_2": edit == 2, "edit_3+": edit >= 3}

    def score(self, scores: np.ndarray) -> list[dict]:
        out = []
        for subset, mask in self.masks.items():
            y, s = self.labels[mask], scores[mask]
            both = 0 < y.sum() < len(y)
            out.append({"subset": subset, "n": int(mask.sum()), "positives": int(y.sum()),
                        "roc_auc": float(roc_auc_score(y, s)) if both else float("nan"),
                        "tpr_at_fpr_0.01": common.tpr_at_fpr(y, s, 0.01) if both else float("nan")})
        return out


def width_features(rows) -> np.ndarray:
    return np.column_stack([text_widths([r[0] for r in rows]), text_widths([r[1] for r in rows])]).astype(np.float32)


def baseline_scores(train_rows, test_rows, confusables: dict[str, str] | None, max_train: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(0)
    if len(train_rows) > max_train:
        train_rows = [train_rows[i] for i in rng.choice(len(train_rows), max_train, replace=False)]
    y = np.array([r[2] for r in train_rows])
    out = {}
    for name, make in (("lengths", lambda rows: string_features(rows, "lengths")),
                       ("widths", width_features),
                       ("string_edit", lambda rows: string_features(rows, "character_set_edit"))):
        model = HistGradientBoostingClassifier(random_state=0).fit(make(train_rows), y)
        out[name] = model.predict_proba(make(test_rows))[:, 1]
    if confusables is not None:
        dist = []
        for cand, ref, _ in test_rows:
            a, b = skeleton(cand, confusables), skeleton(ref, confusables)
            dist.append(Levenshtein.distance(a, b) / max(len(a), len(b), 1))
        out["skeleton"] = -np.array(dist)  # closer skeletons = more likely a spoof
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--confusables", type=Path, default=None, help="Unicode confusables.txt (skeleton baseline)")
    ap.add_argument("--width-tolerance", type=int, default=2, help="pixels, for the equal_width subsets")
    ap.add_argument("--process-train", default="woodbridge_p0", help="training data of the baselines for process names")
    ap.add_argument("--max-train", type=int, default=300_000)
    ap.add_argument("--work", default=None)
    args = ap.parse_args()
    work = common.work_root(args.work)
    res = work / "results"
    res.mkdir(parents=True, exist_ok=True)
    confusables = load_confusables(args.confusables) if args.confusables else None

    pairs: dict[Path, Pairs] = {}

    def get_pairs(path: Path) -> Pairs:
        if path not in pairs:
            print(f"subsets of {path}", flush=True)
            pairs[path] = Pairs(path, args.width_tolerance)
            for subset, mask in pairs[path].masks.items():
                print(f"  {subset:20s} {mask.sum():8d} pairs, {pairs[path].labels[mask].sum():8d} positives")
        return pairs[path]

    rows = []
    for metrics_path in sorted((work / "runs").glob("*/*/seed*/evals/*/metrics.json")):
        evaluation = metrics_path.parent.name
        run = metrics_path.parents[2]
        if evaluation.startswith("font_") or "__" in run.parent.name:
            continue  # font evaluations re-render the same pairs; sweep variants are not compared here
        recorded = Path(json.loads(metrics_path.read_text())["pairs"])
        data_path = work / "data" / recorded.parent.name / recorded.name  # also if WORK has moved
        p = get_pairs(data_path)
        labels = np.load(metrics_path.parent / "labels.npy").astype(np.int64)
        scores = np.load(metrics_path.parent / "scores.npy").astype(np.float64)
        if not np.array_equal(labels, p.labels):
            raise SystemExit(f"labels of {metrics_path.parent} do not match {data_path}")
        data = run.parent.parent.name
        family, _, partition = data.rpartition("_p")
        for r in p.score(scores):
            rows.append({"family": family, "partition": int(partition), "model": run.parent.name,
                         "seed": int(run.name.removeprefix("seed")), "evaluation": evaluation,
                         "pairs": str(data_path.relative_to(work)), **r})
    if not rows:
        raise SystemExit(f"no evaluations under {work / 'runs'}")
    runs = pd.DataFrame(rows)
    runs.to_csv(res / "subsets_runs.csv", index=False)

    # Non-visual baselines on every pair file that a model was evaluated on
    base = []
    for path, p in pairs.items():
        data = path.parent.name
        train_data = args.process_train if data == "process" else data
        train = work / "data" / train_data / "train.pkl"
        if not train.exists():
            continue
        print(f"baselines on {path}, trained on {train_data}", flush=True)
        for method, scores in baseline_scores(common.read_rows(train), p.rows, confusables, args.max_train).items():
            for r in p.score(scores):
                base.append({"pairs": str(path.relative_to(work)), "trained_on": train_data, "method": method, **r})
    baselines = pd.DataFrame(base)
    baselines.to_csv(res / "subsets_baselines.csv", index=False)

    g = runs.groupby(["family", "evaluation", "model", "subset"])
    summary = g[["roc_auc", "tpr_at_fpr_0.01"]].agg(["mean", "std"])
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    summary = summary.join(g.size().rename("runs")).join(g[["n", "positives"]].mean()).reset_index()
    summary.to_csv(res / "subsets_summary.csv", index=False)

    print("\nTest ROC-AUC on subsets (mean +/- s.d. over runs; pair counts in subsets_summary.csv)")
    for (family, evaluation), f in summary.groupby(["family", "evaluation"]):
        print(f"\n{family} / {evaluation}")
        for model, m in f.groupby("model"):
            m = m.set_index("subset")
            cells = [f"{s} {m.loc[s, 'roc_auc_mean']:.4f}+/-{m.loc[s, 'roc_auc_std']:.4f}" for s in SUBSETS if s in m.index]
            print(f"  {model:26s} " + "  ".join(cells))
    if not baselines.empty:
        print("\nNon-visual baselines on the same subsets (ROC-AUC)")
        for (pairs_name, method), b in baselines.groupby(["pairs", "method"]):
            b = b.set_index("subset")
            cells = [f"{s} {b.loc[s, 'roc_auc']:.4f}" for s in SUBSETS if s in b.index]
            print(f"  {pairs_name:40s} {method:12s} " + "  ".join(cells))
    print(f"\nwrote {res / 'subsets_runs.csv'}, subsets_baselines.csv, subsets_summary.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
