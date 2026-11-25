#!/usr/bin/env python3
"""Write the job list of one stage, one shell command per line, for run_gpu.sh.

Stages (see README.md):
  main       4 models x 2 datasets x partitions x seeds
  sweep      slice width sweep of each dataset's selected model, p0
  ctrl       the 4 synthetic models trained on the shortcut-controlled data
  fonts      main p0 runs scored on their test split in unseen fonts
  process    main p0 runs scored on Woodbridge's process-name test set (independent data)
  ctrl_eval  main synthetic runs scored on the controlled test split
  mechanism  Woodbridge controls for the pairwise comparison: BiLSTM and Transformer (independent sequence
             encoders), one-block cross-attention, and a whole-image CNN widened to the size of the
             two-block cross-attention model; partitions x seeds
  mechanism_eval  the mechanism models of p0 scored in unseen fonts and on the process-name test set
  woodbridge_ctrl_eval  Woodbridge models (main and mechanism; --models to choose) scored on the
             shortcut-controlled Woodbridge test split (make_controlled.py --source woodbridge)
  woodbridge_ctrl  the 4 Woodbridge models trained on the shortcut-controlled Woodbridge data
  woodbridge_ctrl_orig_eval  those models scored on the original Woodbridge test split

Round 4 stages (default seeds for these: 7 1 2, as for the controlled runs and the sweep):
  r4_ctrl_mechanism  mechanism models (--models, default all four) trained on the shortcut-controlled
             Woodbridge data; partitions x seeds
  r4_width8  Woodbridge sliced models at a common slice width of 8 px, stride 4 (--models, default BiLSTM,
             Transformer and both cross-attention models; the Interaction CNN at 8/4 is in the sweep), p0
  r4_ctrl_generalization  controlled-data models (--ctrl-sets, --models) of p0 scored on their controlled
             test split in unseen fonts and on the process-name test set
  r4_ctrl_mechanism_orig_eval  r4_ctrl_mechanism models (--models) scored on the original test split
  r4_ctrl_orig_fonts  controlled-data Woodbridge models (--models, default all eight) of p0 scored on the
             ORIGINAL test split in unseen fonts (evaluation name orig_font_<font>), for a like-for-like
             comparison with the original-data models of the fonts stage

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
# Woodbridge models added to separate the pairwise comparison from sequence encoding and capacity
MECHANISM = ("bilstm", "transformer", "cross_attention_1block", "whole_image_cnn_wide")
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
    ap.add_argument("stage", choices=["main", "sweep", "ctrl", "fonts", "process", "ctrl_eval", "mechanism", "mechanism_eval",
                                      "woodbridge_ctrl_eval", "woodbridge_ctrl", "woodbridge_ctrl_orig_eval",
                                      "r4_ctrl_mechanism", "r4_width8", "r4_ctrl_generalization",
                                      "r4_ctrl_mechanism_orig_eval", "r4_ctrl_orig_fonts"])
    ap.add_argument("--partitions", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--seeds", type=int, nargs="+", default=None, help="default 7 1 2 3 4; 7 1 2 for r4_ stages")
    ap.add_argument("--widths", type=int, nargs="+", default=[4, 6, 8, 12, 16, 24, 32])
    ap.add_argument("--fonts", nargs="+", default=None, help="font files for the fonts stage")
    ap.add_argument("--models", nargs="+", default=None,
                    help="models of woodbridge_ctrl_eval (default all) and of the r4_ stages")
    ap.add_argument("--ctrl-sets", nargs="+", default=["woodbridge", "synthetic"],
                    help="controlled datasets of r4_ctrl_generalization")
    ap.add_argument("--no-process", action="store_true", help="r4_ctrl_generalization: fonts only")
    ap.add_argument("--work", default=os.environ.get("WORK"))
    ap.add_argument("--itau", default=os.environ.get("ITAU"))
    args = ap.parse_args()
    if not args.work or not args.itau:
        raise SystemExit("set WORK and ITAU (or pass --work and --itau)")
    work = Path(args.work).resolve()
    py = shlex.quote(sys.executable)
    r4 = args.stage.startswith("r4_")
    if args.seeds is None:
        args.seeds = [7, 1, 2] if r4 else [7, 1, 2, 3, 4]
    if args.models is None:
        args.models = {
            "r4_ctrl_mechanism": list(MECHANISM),
            "r4_ctrl_mechanism_orig_eval": list(MECHANISM),
            "r4_width8": ["bilstm", "transformer", "cross_attention_1block", "cross_attention_2block"],
            "r4_ctrl_generalization": list(MODELS),
            "r4_ctrl_orig_fonts": list(MODELS + MECHANISM),
        }.get(args.stage, list(MODELS + MECHANISM))

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
    elif args.stage == "mechanism":
        for p in args.partitions:
            for seed in args.seeds:
                for m in MECHANISM:
                    lines.append(train(f"woodbridge_p{p}", f"woodbridge_{m}", m, seed))
    elif args.stage == "mechanism_eval":
        fonts = args.fonts or default_fonts(args.itau)
        for seed in args.seeds:
            for m in MECHANISM:
                for font in fonts:
                    lines.append(evaluate("woodbridge_p0", m, seed, work / "data" / "woodbridge_p0" / "test.pkl",
                                          f"font_{font_name(font)}", font))
                lines.append(evaluate("woodbridge_p0", m, seed, work / "data" / "process" / "test.pkl", "process"))
    elif args.stage == "woodbridge_ctrl_eval":
        for p in args.partitions:
            for seed in args.seeds:
                for m in args.models:
                    lines.append(evaluate(f"woodbridge_p{p}", m, seed,
                                          work / "data" / f"woodbridge_ctrl_p{p}" / "test.pkl", "ctrl_test"))
    elif args.stage == "woodbridge_ctrl":
        for p in args.partitions:
            for seed in args.seeds:
                for m in MODELS:
                    lines.append(train(f"woodbridge_ctrl_p{p}", f"woodbridge_{m}", m, seed))
    elif args.stage == "woodbridge_ctrl_orig_eval":
        for p in args.partitions:
            for seed in args.seeds:
                for m in MODELS:
                    lines.append(evaluate(f"woodbridge_ctrl_p{p}", m, seed,
                                          work / "data" / f"woodbridge_p{p}" / "test.pkl", "orig_test"))
    elif args.stage == "r4_ctrl_mechanism":
        for p in args.partitions:
            for seed in args.seeds:
                for m in args.models:
                    lines.append(train(f"woodbridge_ctrl_p{p}", f"woodbridge_{m}", m, seed))
    elif args.stage == "r4_ctrl_mechanism_orig_eval":
        for p in args.partitions:
            for seed in args.seeds:
                for m in args.models:
                    lines.append(evaluate(f"woodbridge_ctrl_p{p}", m, seed,
                                          work / "data" / f"woodbridge_p{p}" / "test.pkl", "orig_test"))
    elif args.stage == "r4_width8":
        # Same folder naming as the sweep (interaction_cnn__sw8_st4), so that all models at 8/4 line up
        for seed in args.seeds:
            for m in args.models:
                if "whole_image" in m:
                    raise SystemExit(f"{m} has no slices")
                lines.append(train("woodbridge_p0", f"woodbridge_{m}", f"{m}__sw8_st4", seed,
                                   " --slice-width 8 --stride 4"))
    elif args.stage == "r4_ctrl_generalization":
        fonts = args.fonts or default_fonts(args.itau)
        for font in fonts:
            if not Path(font).exists():
                raise SystemExit(f"font not found: {font}")
        for seed in args.seeds:
            for ds in args.ctrl_sets:
                data = f"{ds}_ctrl_p0"
                for m in args.models:
                    for font in fonts:
                        lines.append(evaluate(data, m, seed, work / "data" / data / "test.pkl",
                                              f"font_{font_name(font)}", font))
                    if not args.no_process:
                        lines.append(evaluate(data, m, seed, work / "data" / "process" / "test.pkl", "process"))
    elif args.stage == "r4_ctrl_orig_fonts":
        fonts = args.fonts or default_fonts(args.itau)
        for font in fonts:
            if not Path(font).exists():
                raise SystemExit(f"font not found: {font}")
        for seed in args.seeds:
            for m in args.models:
                for font in fonts:
                    lines.append(evaluate("woodbridge_ctrl_p0", m, seed, work / "data" / "woodbridge_p0" / "test.pkl",
                                          f"orig_font_{font_name(font)}", font))
    print("\n".join(lines))
    print(f"{len(lines)} jobs", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
