#!/usr/bin/env python3
"""Find (and optionally delete) the runs and evaluations of a job list that a machine reset interrupted.

An interrupted training run cannot be resumed (see run_one.py), so its folder must be deleted before the
job list is started again; finished runs are then skipped. Only the folders named in the given job list(s)
are checked, so a run that the OTHER machine is training on the shared disk is never touched, as long as
the two machines use different job lists.

A folder counts as interrupted when it exists but has no metrics.json:
  training run (run_one.py --out X):          X exists, X/evals/test/metrics.json missing
  evaluation   (evaluate.py --run X --name N): X/evals/N exists, X/evals/N/metrics.json missing
As a second safeguard, folders with any file modified in the last --min-idle minutes are reported as
"possibly still running" and left alone.

Usage (dry run first, then delete):
  python cleanup_interrupted.py $WORK/jobs/r4_small.txt
  python cleanup_interrupted.py $WORK/jobs/r4_small.txt --delete
"""

from __future__ import annotations

import argparse
import shlex
import shutil
import time
from pathlib import Path


def arg(tokens: list[str], name: str) -> str | None:
    return tokens[tokens.index(name) + 1] if name in tokens else None


def newest_mtime(folder: Path) -> float:
    newest = folder.stat().st_mtime
    for p in folder.rglob("*"):
        try:
            newest = max(newest, p.stat().st_mtime)
        except FileNotFoundError:
            pass
    return newest


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("jobs", type=Path, nargs="+", help="job list(s) of THIS machine")
    ap.add_argument("--delete", action="store_true", help="delete the interrupted folders (default: only list them)")
    ap.add_argument("--min-idle", type=float, default=30, help="minutes without file changes before a folder "
                    "counts as interrupted (default 30)")
    args = ap.parse_args()

    candidates: list[tuple[str, Path]] = []
    finished = pending = 0
    for jobfile in args.jobs:
        for line in jobfile.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            tokens = shlex.split(line)
            if any(t.endswith("run_one.py") for t in tokens):
                out = Path(arg(tokens, "--out"))
                done, folder, kind = out / "evals" / "test" / "metrics.json", out, "training"
            elif any(t.endswith("evaluate.py") for t in tokens):
                folder = Path(arg(tokens, "--run")) / "evals" / arg(tokens, "--name")
                done, kind = folder / "metrics.json", "evaluation"
            else:
                print(f"skipped (not a run_one.py/evaluate.py command): {line[:100]}")
                continue
            if done.exists():
                finished += 1
            elif folder.exists():
                candidates.append((kind, folder))
            else:
                pending += 1

    now = time.time()
    interrupted, active = [], []
    for kind, folder in candidates:
        idle = (now - newest_mtime(folder)) / 60
        (interrupted if idle >= args.min_idle else active).append((kind, folder, idle))

    print(f"finished: {finished}, not started: {pending}, interrupted: {len(interrupted)}, "
          f"possibly still running: {len(active)}")
    for kind, folder, idle in active:
        print(f"  RUNNING?    {kind:10s} {folder}  (last change {idle:.0f} min ago; left alone)")
    for kind, folder, idle in interrupted:
        if args.delete:
            shutil.rmtree(folder)
            print(f"  deleted     {kind:10s} {folder}")
        else:
            print(f"  interrupted {kind:10s} {folder}  (last change {idle:.0f} min ago)")
    if interrupted and not args.delete:
        print("dry run: nothing deleted; add --delete to remove the interrupted folders")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
