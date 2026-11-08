#!/usr/bin/env python3
"""Score a finished run on another pair file, optionally rendered with another font.

Used for the generalization tests; no training happens here:
  - unseen fonts:        --pairs <test.pkl> --font /path/to/Font.ttf --name font_<font>
  - independent data:    --pairs $WORK/data/process/test.pkl --name process
  - shortcut control:    --pairs $WORK/data/synthetic_ctrl_p0/test.pkl --name ctrl_test

With --font, metrics are also reported for the pairs whose characters all exist in that font
(characters missing from a font are drawn as boxes).

Writes <run>/evals/<name>/metrics.json, scores.npy and labels.npy.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True, help="run folder written by run_one.py")
    ap.add_argument("--pairs", type=Path, required=True)
    ap.add_argument("--name", required=True, help="name of the evaluation folder under <run>/evals/")
    ap.add_argument("--font", default=None, help="font file used to render the pairs (default: training font)")
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--itau", default=None)
    ap.add_argument("--work", default=None)
    args = ap.parse_args()

    out = args.run / "evals" / args.name
    if (out / "metrics.json").exists():
        print(f"already evaluated: {out}")
        return 0
    if not (args.run / "train" / "best.pt").exists():
        raise SystemExit(f"no trained model in {args.run}")
    common.setup(common.itau_root(args.itau), common.work_root(args.work))
    result = common.evaluate(args.run, args.pairs.resolve(), out, font=args.font, num_workers=args.num_workers)
    extra = f", font coverage {result['font_coverage']:.3f}" if "font_coverage" in result else ""
    print(f"{out}: ROC-AUC {result['roc_auc']:.6f}{extra}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
