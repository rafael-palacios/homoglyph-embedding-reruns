#!/usr/bin/env python3
"""Write the job list of one stage, one shell command per line, for run_gpu.sh.

Stages (see README.md):
  main       4 models x 2 datasets x partitions x seeds
  sweep      slice width sweep of each dataset's selected model, p0
  ctrl       the 4 synthetic models trained on the shortcut-controlled data
  fonts      main p0 runs scored on their test split in unseen fonts
  process    main p0 runs scored on Woodbridge's process-name test set (independent data)
  ctrl_eval  main synthetic runs scored on the controlled test split

Every command skips work that is already finished, so a job list can be re-run after an interruption.

Example:
  python make_jobs.py main --partitions 0 1 2 --seeds 7 1 2 3 4 > $WORK/jobs/main.txt
"""

from __future__ import annotations

import argparse
import os
import shlex
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODELS = ("interaction_cnn", "cross_attention_2block", "whole_image_cnn", "conv1d")
# The model selected for each dataset in the paper, and its stride mode, for the sweep
SELECTED = {"woodbridge": ("interaction_cnn", "half"), "synthetic": ("cross_attention_2block", "full")}
PAPER_WIDTH = {"woodbridge": 16, "synthetic": 6}


def default_fonts(itau: str) -> list[str]:
    import matplotlib

    ttf = Path(matplotlib.__file__).parent / "mpl-data" / "fonts" / "ttf"
    return [str(ttf / "DejaVuSerif.ttf"), str(ttf / "DejaVuSansMono.ttf"), str(ttf / "STIXGeneral.ttf"),
            str(Path(itau) / "synth_dataset" / "fonts" / "unifont-17.0.04.otf")]


def font_name(path: str) -> str:
    return Path(path).stem.lower().replace("-", "_").replace(".", "_")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["main", "sweep", "ctrl", "fonts", "process", "ctrl_eval"])
    ap.add_argument("--partitions", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--seeds", type=int, nargs="+", default=[7, 1, 2, 3, 4])
    ap.add_argument("--widths", type=int, nargs="+", default=[4, 6, 8, 12, 16, 24, 32])
    ap.add_argument("--fonts", nargs="+", default=None, help="font files for the fonts stage")
    ap.add_argument("--work", default=os.environ.get("WORK"))
    ap.add_argument("--itau", default=os.environ.get("ITAU"))
    args = ap.parse_args()
    if not args.work or not args.itau:
        raise SystemExit("set WORK and ITAU (or pass --work and --itau)")
    work = Path(args.work).resolve()
    py = shlex.quote(sys.executable)

    def train(data: str, model_cfg: str, folder: str, seed: int, extra: str = "") -> str:
        return (f"{py} {HERE / 'run_one.py'} --data {work / 'data' / data} --config {HERE / 'configs' / model_cfg}.yaml "
                f"--seed {seed} --out {work / 'runs' / data / folder / f'seed{seed}'}{extra}")

    def evaluate(data: str, folder: str, seed: int, pairs: Path, name: str, font: str | None = None) -> str:
        run = work / "runs" / data / folder / f"seed{seed}"
        f = f" --font {shlex.quote(font)}" if font else ""
        return f"{py} {HERE / 'evaluate.py'} --run {run} --pairs {pairs} --name {name}{f}"

    lines = []
    if args.stage == "main":
        # Interleave so that early results cover every model and dataset
        for p in args.partitions:
            for seed in args.seeds:
                for ds in ("woodbridge", "synthetic"):
                    for m in MODELS:
                        lines.append(train(f"{ds}_p{p}", f"{ds}_{m}", m, seed))
    elif args.stage == "sweep":
        for seed in args.seeds:
            for ds, (m, mode) in SELECTED.items():
                for w in args.widths:
                    stride = w // 2 if mode == "half" else w
                    if w == PAPER_WIDTH[ds]:
                        continue  # the paper's width is the main run of the same seed
                    lines.append(train(f"{ds}_p0", f"{ds}_{m}", f"{m}__sw{w}_st{stride}", seed,
                                       f" --slice-width {w} --stride {stride}"))
    elif args.stage == "ctrl":
        for p in args.partitions:
            for seed in args.seeds:
                for m in MODELS:
                    lines.append(train(f"synthetic_ctrl_p{p}", f"synthetic_{m}", m, seed))
    elif args.stage == "fonts":
        fonts = args.fonts or default_fonts(args.itau)
        for font in fonts:
            if not Path(font).exists():
                raise SystemExit(f"font not found: {font}")
        for seed in args.seeds:
            for ds in ("woodbridge", "synthetic"):
                for m in MODELS:
                    for font in fonts:
                        lines.append(evaluate(f"{ds}_p0", m, seed, work / "data" / f"{ds}_p0" / "test.pkl",
                                              f"font_{font_name(font)}", font))
    elif args.stage == "process":
        for seed in args.seeds:
            for ds in ("woodbridge", "synthetic"):
                for m in MODELS:
                    lines.append(evaluate(f"{ds}_p0", m, seed, work / "data" / "process" / "test.pkl", "process"))
    elif args.stage == "ctrl_eval":
        for p in args.partitions:
            for seed in args.seeds:
                for m in MODELS:
                    lines.append(evaluate(f"synthetic_p{p}", m, seed,
                                          work / "data" / f"synthetic_ctrl_p{p}" / "test.pkl", "ctrl_test"))
    print("\n".join(lines))
    print(f"{len(lines)} jobs", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
