#!/usr/bin/env python3
"""Train one frozen configuration with one seed on one partition, then score it on that partition's test split.

Training is the paper's final protocol (architecture_search.training.train_validation_trial): 25 epochs,
no early stopping, the checkpoint with the best validation ROC-AUC is kept. Only the seed, the data
partition and, for the sweep, the slice width and stride change. A finished run is skipped. Resuming an
interrupted run from its latest checkpoint fails with some PyTorch versions ("RNG state must be a
torch.ByteTensor"); delete the run folder of an interrupted run so that it starts again from scratch.

Usage:
  python run_one.py --data $WORK/data/woodbridge_p0 --config configs/woodbridge_interaction_cnn.yaml \
      --seed 7 --out $WORK/runs/woodbridge/p0/interaction_cnn/seed7 [--slice-width 8 --stride 4]

Writes <out>/train/ (checkpoints, history.csv, metrics.json) and <out>/evals/test/ (metrics.json,
scores.npy, labels.npy).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True, help="folder with train.pkl, validation.pkl, test.pkl")
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--slice-width", type=int, default=None)
    ap.add_argument("--stride", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--cache-workers", type=int, default=8, help="processes used to build a missing render cache")
    ap.add_argument("--itau", default=None, help="Itau checkout (default: $ITAU)")
    ap.add_argument("--work", default=None, help="working folder (default: $WORK)")
    args = ap.parse_args()

    common.setup(common.itau_root(args.itau), common.work_root(args.work))
    from architecture_search.training import train_validation_trial

    if (args.out / "evals" / "test" / "metrics.json").exists():
        print(f"already finished: {args.out}")
        return 0

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    config["seed"] = int(args.seed)
    config["max_epochs"] = int(args.epochs)
    config["early_stopping_patience"] = int(args.epochs)
    config["num_workers"] = int(args.num_workers)
    config["dataset"] = args.data.name
    if args.slice_width is not None:
        config["slice_width"] = int(args.slice_width)
        config["stride"] = int(args.stride if args.stride is not None else args.slice_width)
        config["stride_mode"] = "nonoverlap" if config["stride"] == config["slice_width"] else "half_overlap"

    splits = {name: (args.data / f"{name}.pkl").resolve() for name in ("train", "validation", "test")}
    for name, path in splits.items():
        if not path.exists():
            raise SystemExit(f"missing {name} split: {path}")
    common.ensure_render_cache(list(splits.values()), workers=args.cache_workers)

    result = train_validation_trial(
        config=config,
        train_path=splits["train"],
        validation_path=splits["validation"],
        run_dir=args.out / "train",
        device=common.device(),
        trial=None,
    )
    if int(result["epochs_completed"]) != int(args.epochs):
        raise RuntimeError(f"expected {args.epochs} epochs, got {result['epochs_completed']}")

    test = common.evaluate(args.out, splits["test"], args.out / "evals" / "test", num_workers=args.num_workers)
    print(
        f"{args.out}: best epoch {result['best_epoch']}, validation ROC-AUC {result['best_validation_roc_auc']:.6f}, "
        f"test ROC-AUC {test['roc_auc']:.6f}, test accuracy@0.5 {test['at_0_5']['accuracy']:.4f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
