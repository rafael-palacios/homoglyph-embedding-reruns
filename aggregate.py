#!/usr/bin/env python3
"""Summarize every finished run and evaluation under $WORK/runs into $WORK/results/.

  runs.csv                one row per run and evaluation (all metrics)
  main_summary.csv/.tex   test metrics, mean +/- s.d. over all partitions and seeds, per dataset and model
  by_partition.csv        the same per partition (mean over seeds), to separate data and seed variability
  paired_tests.csv        paired comparisons between models on the same partition and seed: mean ROC-AUC
                          difference, runs won, paired t-test and Wilcoxon test over runs, and the number of
                          runs in which a per-run paired DeLong test is significant in either direction
  delong_per_run.csv      the per-run DeLong tests
  slice_sweep.csv/.pdf    test and validation ROC-AUC against slice width (selected models, p0)
  fonts.csv               ROC-AUC on the test split rendered in unseen fonts
  process.csv             ROC-AUC on Woodbridge's process-name test split
  controlled.csv          synthetic models on the shortcut-controlled test split, and models trained on it
  det_curves.pdf          DET curves of the four models on each dataset (p0, first seed)
  report.txt              the text printed by this script

Example:
  python aggregate.py
"""

from __future__ import annotations

import argparse
import io
import json
import re
import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata, ttest_rel, wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

LABELS = {"interaction_cnn": "Interaction CNN", "cross_attention_2block": "Two-block cross-attention",
          "whole_image_cnn": "Whole-image CNN", "conv1d": "Conv1D"}
ORDER = list(LABELS)
PAIRS = [("interaction_cnn", "whole_image_cnn"), ("cross_attention_2block", "whole_image_cnn"),
         ("interaction_cnn", "cross_attention_2block"), ("whole_image_cnn", "conv1d"),
         ("interaction_cnn", "conv1d"), ("cross_attention_2block", "conv1d")]
METRICS = ["roc_auc", "tpr_at_fpr_0.001", "tpr_at_fpr_0.01", "accuracy_0_5", "f1_0_5", "mcc_0_5"]


def delong_paired(y: np.ndarray, s1: np.ndarray, s2: np.ndarray):
    """Paired DeLong test for two ROC-AUCs on the same samples (fast algorithm of Sun and Xu, 2014)."""
    pos, neg = y == 1, y == 0
    m, n = int(pos.sum()), int(neg.sum())
    aucs, v_pos, v_neg = [], [], []
    for s in (s1, s2):
        x, z = s[pos], s[neg]
        tx, tz = rankdata(x), rankdata(z)
        t = rankdata(np.concatenate([x, z]))
        aucs.append((t[:m].sum() - m * (m + 1) / 2) / (m * n))
        v_pos.append((t[:m] - tx) / n)
        v_neg.append(1.0 - (t[m:] - tz) / m)
    cov = np.cov(np.vstack(v_pos)) / m + np.cov(np.vstack(v_neg)) / n
    diff = aucs[0] - aucs[1]
    se = float(np.sqrt(max(cov[0, 0] + cov[1, 1] - 2 * cov[0, 1], 0.0)))
    p = float(2 * norm.sf(abs(diff) / se)) if se > 0 else float("nan")
    return aucs[0], aucs[1], diff, p


def collect(work: Path) -> pd.DataFrame:
    rows = []
    for path in sorted((work / "runs").glob("*/*/seed*/evals/*/metrics.json")):
        evaluation = path.parent.name
        run = path.parents[2]
        data, folder = run.parent.parent.name, run.parent.name
        match = re.fullmatch(r"(.+)_p(\d+)", data)
        family, partition = (match.group(1), int(match.group(2))) if match else (data, -1)
        model, _, variant = folder.partition("__")
        m = json.loads(path.read_text())
        t = json.loads((run / "train" / "metrics.json").read_text())
        cfg = t["resolved_config"]
        row = {
            "family": family, "partition": partition, "model": model, "variant": variant or "paper",
            "slice_width": cfg.get("slice_width"), "stride": cfg.get("stride"),
            "seed": int(run.name.removeprefix("seed")), "evaluation": evaluation, "run": str(run),
            "best_validation_roc_auc": t["best_validation_roc_auc"], "best_epoch": t["best_epoch"],
            "parameters": t["trainable_parameter_count"], "training_hours": t["training_seconds"] / 3600,
            "roc_auc": m["roc_auc"], "tpr_at_fpr_0.001": m["tpr_at_fpr_0.001"], "tpr_at_fpr_0.01": m["tpr_at_fpr_0.01"],
            "accuracy_0_5": m["at_0_5"]["accuracy"], "f1_0_5": m["at_0_5"]["f1"], "mcc_0_5": m["at_0_5"]["mcc"],
            "font_coverage": m.get("font_coverage"),
            "roc_auc_covered": m.get("covered_pairs", {}).get("roc_auc"),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def mean_sd(frame: pd.DataFrame, keys: list[str], cols: list[str]) -> pd.DataFrame:
    g = frame.groupby(keys)[cols]
    out = g.mean().add_suffix("_mean").join(g.std().add_suffix("_sd")).join(g.size().rename("runs"))
    return out.reset_index()


def main_tables(df: pd.DataFrame, res: Path) -> None:
    main = df[(df.evaluation == "test") & (df.variant == "paper")]
    if main.empty:
        return
    summary = mean_sd(main, ["family", "model"], METRICS + ["best_validation_roc_auc", "training_hours"])
    summary["order"] = summary.model.map(ORDER.index)
    summary = summary.sort_values(["family", "order"]).drop(columns="order")
    summary.to_csv(res / "main_summary.csv", index=False)
    mean_sd(main, ["family", "partition", "model"], ["roc_auc", "accuracy_0_5"]).to_csv(res / "by_partition.csv", index=False)

    tex = ["% Dataset & Model & ROC-AUC & TPR@FPR=0.1% & TPR@FPR=1% & Accuracy & F1 & MCC (mean $\\pm$ s.d. over runs) & runs"]
    print("Test metrics, mean +/- s.d. over partitions and seeds")
    for _, r in summary.iterrows():
        cells = [f"{r[f'{c}_mean']:.4f} $\\pm$ {r[f'{c}_sd']:.4f}" for c in METRICS]
        tex.append(f"{r.family} & {LABELS.get(r.model, r.model)} & " + " & ".join(cells) + f" & {int(r.runs)} \\\\")
        print(f"  {r.family:15s} {LABELS.get(r.model, r.model):26s} ROC-AUC {r.roc_auc_mean:.5f} +/- {r.roc_auc_sd:.5f}  "
              f"TPR@0.1%FPR {r['tpr_at_fpr_0.001_mean']:.4f}  acc {r.accuracy_0_5_mean:.4f}  (n={int(r.runs)})")
    (res / "main_table.tex").write_text("\n".join(tex) + "\n")


def paired_tests(df: pd.DataFrame, res: Path) -> None:
    main = df[(df.evaluation == "test") & (df.variant == "paper")]
    per_run, summary = [], []
    for family in sorted(main.family.unique()):
        f = main[main.family == family].set_index(["partition", "seed", "model"])
        for a, b in PAIRS:
            keys = sorted({k[:2] for k in f.index if k[2] == a} & {k[:2] for k in f.index if k[2] == b})
            if not keys:
                continue
            for p, s in keys:
                ra, rb = Path(f.loc[(p, s, a), "run"]), Path(f.loc[(p, s, b), "run"])
                y = np.load(ra / "evals" / "test" / "labels.npy").astype(int)
                sa = np.load(ra / "evals" / "test" / "scores.npy").astype(float)
                sb = np.load(rb / "evals" / "test" / "scores.npy").astype(float)
                auc_a, auc_b, diff, pval = delong_paired(y, sa, sb)
                per_run.append(dict(family=family, model_a=a, model_b=b, partition=p, seed=s,
                                    auc_a=auc_a, auc_b=auc_b, diff=diff, delong_p=pval))
            d = pd.DataFrame([r for r in per_run if r["family"] == family and r["model_a"] == a and r["model_b"] == b])
            n = len(d)
            t_p = ttest_rel(d.auc_a, d.auc_b).pvalue if n > 1 else float("nan")
            w_p = wilcoxon(d["diff"]).pvalue if n > 5 and (d["diff"] != 0).any() else float("nan")
            summary.append(dict(
                family=family, model_a=a, model_b=b, runs=n,
                mean_diff=d["diff"].mean(), sd_diff=d["diff"].std(),
                ci95_low=d["diff"].mean() - 1.96 * d["diff"].std() / np.sqrt(n) if n > 1 else float("nan"),
                ci95_high=d["diff"].mean() + 1.96 * d["diff"].std() / np.sqrt(n) if n > 1 else float("nan"),
                runs_a_better=int((d["diff"] > 0).sum()),
                delong_sig_a_better=int(((d.delong_p < 0.05) & (d["diff"] > 0)).sum()),
                delong_sig_b_better=int(((d.delong_p < 0.05) & (d["diff"] < 0)).sum()),
                paired_t_p=t_p, wilcoxon_p=w_p,
            ))
    if not summary:
        return
    pd.DataFrame(per_run).to_csv(res / "delong_per_run.csv", index=False)
    s = pd.DataFrame(summary)
    s.to_csv(res / "paired_tests.csv", index=False)
    print("\nPaired comparisons on the test split (A - B in ROC-AUC, same partition and seed)")
    for _, r in s.iterrows():
        print(f"  {r.family:15s} {LABELS[r.model_a]} vs {LABELS[r.model_b]}: diff {r.mean_diff:+.5f} "
              f"[{r.ci95_low:+.5f}, {r.ci95_high:+.5f}], A better in {r.runs_a_better}/{r.runs} runs, "
              f"DeLong p<0.05 for A/B in {r.delong_sig_a_better}/{r.delong_sig_b_better} runs, "
              f"paired t p={r.paired_t_p:.3g}, Wilcoxon p={r.wilcoxon_p:.3g}")


def slice_sweep(df: pd.DataFrame, res: Path) -> None:
    sweep_models = df[(df.evaluation == "test") & (df.variant != "paper")][["family", "model"]].drop_duplicates()
    if sweep_models.empty:
        return
    seeds = set(df[(df.evaluation == "test") & (df.variant != "paper")].seed)
    keep = (df.evaluation == "test") & (df.partition == 0) & df.seed.isin(seeds)
    keep &= df.set_index(["family", "model"]).index.isin(sweep_models.set_index(["family", "model"]).index)
    s = mean_sd(df[keep], ["family", "model", "slice_width", "stride"], ["roc_auc", "best_validation_roc_auc"])
    s.to_csv(res / "slice_sweep.csv", index=False)
    print("\nSlice-width sensitivity (p0; test ROC-AUC, mean +/- s.d. over seeds)")
    for _, r in s.iterrows():
        print(f"  {r.family:15s} {LABELS[r.model]:26s} width {int(r.slice_width):2d} stride {int(r.stride):2d}: "
              f"{r.roc_auc_mean:.5f} +/- {r.roc_auc_sd:.5f} (validation {r.best_validation_roc_auc_mean:.5f}, n={int(r.runs)})")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    families = list(s.family.unique())
    fig, axes = plt.subplots(1, len(families), figsize=(4.5 * len(families), 3.6), squeeze=False)
    for ax, family in zip(axes[0], families):
        g = s[s.family == family].sort_values("slice_width")
        ax.errorbar(g.slice_width, g.roc_auc_mean, yerr=g.roc_auc_sd, marker="o", capsize=3, label="test")
        ax.errorbar(g.slice_width, g.best_validation_roc_auc_mean, yerr=g.best_validation_roc_auc_sd,
                    marker="s", capsize=3, ls="--", label="validation")
        ax.set_xscale("log", base=2)
        ax.set_xticks(g.slice_width, [str(int(w)) for w in g.slice_width])
        ax.set_xlabel("Slice width (pixels)")
        ax.set_ylabel("ROC-AUC")
        ax.set_title(f"{family}: {LABELS[g.model.iloc[0]]}")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(res / "slice_sweep.pdf")
    fig.savefig(res / "slice_sweep.png", dpi=200)


def other_evaluations(df: pd.DataFrame, res: Path) -> None:
    base = df[df.variant == "paper"]
    font_runs = set(base[base.evaluation.str.startswith("font_")].run)
    fonts = base[base.run.isin(font_runs) & (base.evaluation.str.startswith("font_") | (base.evaluation == "test"))]
    if font_runs:
        f = mean_sd(fonts, ["family", "model", "evaluation"], ["roc_auc", "roc_auc_covered", "font_coverage"])
        f.to_csv(res / "fonts.csv", index=False)
        print("\nUnseen fonts (p0; 'test' is the training font DejaVu Sans; ROC-AUC on all pairs / on pairs fully covered by the font)")
        for _, r in f.iterrows():
            cov = "" if pd.isna(r.font_coverage_mean) else f" / {r.roc_auc_covered_mean:.5f} (coverage {r.font_coverage_mean:.3f})"
            print(f"  {r.family:15s} {LABELS[r.model]:26s} {r.evaluation:28s} {r.roc_auc_mean:.5f} +/- {r.roc_auc_sd:.5f}{cov}")

    proc = base[base.evaluation == "process"]
    if not proc.empty:
        p = mean_sd(proc, ["family", "model"], ["roc_auc", "tpr_at_fpr_0.01", "accuracy_0_5"])
        p.to_csv(res / "process.csv", index=False)
        print("\nProcess-name test split (independent data; models trained on each dataset's p0)")
        for _, r in p.iterrows():
            print(f"  trained on {r.family:15s} {LABELS[r.model]:26s} ROC-AUC {r.roc_auc_mean:.5f} +/- {r.roc_auc_sd:.5f}")

    ctrl = base[((base.family == "synthetic") & base.evaluation.isin(["test", "ctrl_test"]))
                | ((base.family == "synthetic_ctrl") & (base.evaluation == "test"))]
    if (ctrl.evaluation == "ctrl_test").any() or (ctrl.family == "synthetic_ctrl").any():
        c = ctrl.assign(setting=np.where(ctrl.family == "synthetic_ctrl", "trained and tested on controlled",
                                         np.where(ctrl.evaluation == "ctrl_test", "trained on original, tested on controlled",
                                                  "trained and tested on original")))
        c = mean_sd(c, ["setting", "model"], ["roc_auc", "tpr_at_fpr_0.01", "accuracy_0_5"])
        c.to_csv(res / "controlled.csv", index=False)
        print("\nShortcut control (synthetic dataset)")
        for _, r in c.iterrows():
            print(f"  {r.setting:42s} {LABELS[r.model]:26s} ROC-AUC {r.roc_auc_mean:.5f} +/- {r.roc_auc_sd:.5f} (n={int(r.runs)})")

    shortcut = res / "shortcut_baseline.csv"
    if shortcut.exists():
        print("\nNon-visual baselines (shortcut_baseline.py)")
        for _, r in pd.read_csv(shortcut).iterrows():
            print(f"  {r.data:22s} {r.features:20s} ROC-AUC {r.test_roc_auc:.4f}")


def det_curves(df: pd.DataFrame, res: Path) -> None:
    main = df[(df.evaluation == "test") & (df.variant == "paper") & (df.partition == 0)]
    families = [f for f in ("woodbridge", "synthetic", "synthetic_ctrl") if f in set(main.family)]
    if not families:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import det_curve

    fig, axes = plt.subplots(1, len(families), figsize=(4.5 * len(families), 4.2), squeeze=False)
    ticks = [0.0005, 0.001, 0.01, 0.05, 0.2, 0.5]
    labels = ["0.05%", "0.1%", "1%", "5%", "20%", "50%"]
    for ax, family in zip(axes[0], families):
        f = main[main.family == family]
        seed = sorted(f.seed)[0]
        for model in ORDER:
            r = f[(f.model == model) & (f.seed == seed)]
            if r.empty:
                continue
            d = Path(r.run.iloc[0]) / "evals" / "test"
            fpr, fnr, _ = det_curve(np.load(d / "labels.npy"), np.load(d / "scores.npy"))
            ax.plot(norm.ppf(np.clip(fpr, 1e-6, 1 - 1e-6)), norm.ppf(np.clip(fnr, 1e-6, 1 - 1e-6)), label=LABELS[model])
        ax.set_xticks(norm.ppf(ticks[1:]), labels[1:])
        ax.set_yticks(norm.ppf(ticks), labels)
        ax.set_xlim(norm.ppf(0.0003), norm.ppf(0.5))
        ax.set_ylim(norm.ppf(0.0003), norm.ppf(0.5))
        ax.axvline(norm.ppf(0.001), color="gray", lw=0.8, ls=":")
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("False negative rate")
        ax.set_title(f"{family} (p0, seed {seed})")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(res / "det_curves.pdf")
    fig.savefig(res / "det_curves.png", dpi=200)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", default=None)
    args = ap.parse_args()
    work = common.work_root(args.work)
    res = work / "results"
    res.mkdir(parents=True, exist_ok=True)
    df = collect(work)
    if df.empty:
        raise SystemExit(f"no finished runs under {work / 'runs'}")
    df.assign(run=[str(Path(r).relative_to(work)) for r in df.run]).to_csv(res / "runs.csv", index=False)

    buffer = io.StringIO()
    with redirect_stdout(buffer):
        print(f"{df[df.evaluation == 'test'].shape[0]} finished runs, {len(df)} evaluations")
        main_tables(df, res)
        paired_tests(df, res)
        slice_sweep(df, res)
        other_evaluations(df, res)
        det_curves(df, res)
    print(buffer.getvalue())
    (res / "report.txt").write_text(buffer.getvalue())
    print(f"wrote {res}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
