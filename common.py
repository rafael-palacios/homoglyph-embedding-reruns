"""Shared helpers for the MAKE rerun package.

The package reuses the paper's own training and model code (``synth_dataset/architecture_search``
in github.com/setliu22/Itau, branch ``agent-update-domains-spoof-no-com-metrics``). Nothing in that
repository is edited; this module only

- puts the repository on ``sys.path``,
- replaces the hard-coded ``/home/setliu22/...`` path used to fingerprint the renderer,
- relaxes the slice-width check (the code only accepts 6, 16 and 32 pixels) for the sensitivity sweep,
- evaluates a trained checkpoint on any pair file, optionally rendered with another font.
"""

from __future__ import annotations

import json
import os
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np

PACKAGE_DIR = Path(__file__).resolve().parent
_SETUP_DONE = False


def itau_root(value: str | None = None) -> Path:
    path = Path(value or os.environ.get("ITAU", "")).expanduser().resolve()
    if not (path / "synth_dataset" / "architecture_search" / "training.py").exists():
        raise SystemExit(
            f"ITAU={path} is not a checkout of github.com/setliu22/Itau on branch "
            "agent-update-domains-spoof-no-com-metrics (synth_dataset/architecture_search is missing)."
        )
    return path


def work_root(value: str | None = None) -> Path:
    path = value or os.environ.get("WORK")
    if not path:
        raise SystemExit("Set WORK to the working folder, e.g. export WORK=$HOME/make")
    return Path(path).expanduser().resolve()


def setup(itau: Path, work: Path) -> None:
    """Make the paper's code importable and apply the two runtime patches."""

    global _SETUP_DONE
    if _SETUP_DONE:
        return
    # rendering/ and models/ live at the repository root; architecture_search/ in synth_dataset/
    for path in (itau, itau / "synth_dataset"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    os.environ.setdefault("ARCH_SEARCH_RENDER_CACHE_ROOT", str(work / "render_cache"))

    from architecture_search import modeling, render_cache, training

    render_cache.EXTERNAL_REPO = itau
    original = modeling.validate_resolved_config

    def validate_relaxed(config: dict[str, Any]) -> None:
        if config.get("architecture") != "whole_image_cnn":
            width, stride = int(config["slice_width"]), int(config["stride"])
            if width < 2 or stride not in {width, width // 2}:
                raise ValueError("stride must equal slice_width or slice_width // 2")
            # Let the original check every other rule with a slice width it accepts
            config = {**config, "slice_width": 6, "stride": 6}
        original(config)

    modeling.validate_resolved_config = validate_relaxed
    training.validate_resolved_config = validate_relaxed
    _SETUP_DONE = True


def device():
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def read_rows(path: Path) -> list[tuple[str, str, int]]:
    with path.open("rb") as handle:
        return pickle.load(handle)


def write_rows(rows: list[tuple[str, str, int]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("wb") as handle:
        pickle.dump(rows, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


def ensure_render_cache(paths: list[Path], workers: int) -> None:
    """Build the paper's packed DejaVu Sans render cache for each pair file (once; file-locked)."""

    from architecture_search.render_cache import build_render_cache

    for path in paths:
        build_render_cache(path, workers=workers)


class FontRenderer:
    """Stands in for the render cache so that a dataset renders its names with another font."""

    def __init__(self, rows: list[tuple[str, str, int]], font_path: str, height: int, background: str) -> None:
        self.rows = rows
        self.font_path = font_path
        self.height = height
        self.background = background

    def image(self, row_index: int, side: int) -> np.ndarray:
        from rendering.renderer import render_name

        return render_name(
            str(self.rows[row_index][side]),
            height=self.height,
            font_path=self.font_path,
            background=self.background,
        )


def font_coverage(rows: list[tuple[str, str, int]], font_path: str) -> np.ndarray:
    """True for pairs whose characters all have a glyph in the font (others render as boxes)."""

    from fontTools.ttLib import TTFont

    cmap = set(TTFont(font_path, fontNumber=0, lazy=True).getBestCmap())
    return np.array([all(ord(ch) in cmap for ch in a + b) for a, b, _ in rows], dtype=bool)


def tpr_at_fpr(labels: np.ndarray, scores: np.ndarray, target: float) -> float:
    from sklearn.metrics import roc_curve

    fpr, tpr, _ = roc_curve(labels, scores)
    return float(tpr[fpr <= target].max())


def score_metrics(labels: np.ndarray, scores: np.ndarray, validation_threshold: float | None) -> dict[str, Any]:
    from architecture_search.metrics import threshold_metrics
    from sklearn.metrics import roc_auc_score

    out: dict[str, Any] = {
        "n": int(len(labels)),
        "positives": int(labels.sum()),
        "roc_auc": float(roc_auc_score(labels, scores)) if 0 < labels.sum() < len(labels) else float("nan"),
        "tpr_at_fpr_0.001": tpr_at_fpr(labels, scores, 1e-3) if 0 < labels.sum() < len(labels) else float("nan"),
        "tpr_at_fpr_0.01": tpr_at_fpr(labels, scores, 1e-2) if 0 < labels.sum() < len(labels) else float("nan"),
        "at_0_5": threshold_metrics(labels, scores, 0.5),
    }
    if validation_threshold is not None:
        out["at_validation_threshold"] = threshold_metrics(labels, scores, float(validation_threshold))
    return out


def evaluate(
    run_dir: Path,
    pairs: Path,
    out_dir: Path,
    *,
    font: str | None = None,
    num_workers: int = 4,
) -> dict[str, Any]:
    """Score the best checkpoint of ``run_dir`` on ``pairs`` and write metrics.json and scores.npy."""

    import torch
    import torch.nn as nn
    from architecture_search.data import (
        RenderedNamePairDataset,
        WholeImageNamePairDataset,
        collate_pairs,
        collate_whole_images,
    )
    from architecture_search.modeling import PairClassifier
    from architecture_search.training import _load_checkpoint, run_epoch

    dev = device()
    checkpoint = _load_checkpoint(run_dir / "train" / "best.pt", dev)
    config = checkpoint["resolved_config"]
    train_metrics = json.loads((run_dir / "train" / "metrics.json").read_text())
    validation_threshold = float(train_metrics["validation"]["best_f1_threshold"])

    common = dict(height=int(config["image_height"]), background=str(config["background"]),
                  remove_padding=bool(config["remove_padding"]))
    if config["architecture"] == "whole_image_cnn":
        dataset = WholeImageNamePairDataset(pairs, **common)
        collate = collate_whole_images
    else:
        dataset = RenderedNamePairDataset(pairs, slice_width=int(config["slice_width"]),
                                          stride=int(config["stride"]), **common)
        collate = collate_pairs
    if font:
        dataset.render_cache = FontRenderer(dataset.rows, font, common["height"], common["background"])

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=int(config["batch_size"]),
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )
    model = PairClassifier(config).to(dev)
    model.load_state_dict(checkpoint["model"])
    output = run_epoch(model, loader, nn.BCEWithLogitsLoss(), dev, None)
    labels = output.labels.astype(np.int64)
    scores = output.probabilities.astype(np.float64)

    result: dict[str, Any] = {
        "run_dir": str(run_dir),
        "pairs": str(pairs),
        "font": font or "DejaVu Sans (training font)",
        "architecture": config["architecture"],
        "seed": int(config["seed"]),
        "best_epoch": int(checkpoint["best_epoch"]),
        "validation_threshold": validation_threshold,
        **score_metrics(labels, scores, validation_threshold),
    }
    if font:
        covered = font_coverage(dataset.rows, font)
        result["font_coverage"] = float(covered.mean())
        result["covered_pairs"] = score_metrics(labels[covered], scores[covered], validation_threshold)

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "scores.npy", scores.astype(np.float32))
    np.save(out_dir / "labels.npy", labels.astype(np.int8))
    tmp = out_dir / "metrics.json.tmp"
    tmp.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, out_dir / "metrics.json")
    return result
