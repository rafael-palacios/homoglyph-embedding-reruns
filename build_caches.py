#!/usr/bin/env python3
"""Pre-render every name of the given datasets into the paper's packed render cache (CPU only).

Training reads rendered images from this cache, as in the paper (DejaVu Sans, 32 px, white on black).
run_one.py builds a missing cache itself, but doing it here first keeps GPUs from waiting on it.
Each training split needs a few GB under $WORK/render_cache.

Example:
  python build_caches.py --data woodbridge_p0 synthetic_p0 --workers 32
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", nargs="+", required=True, help="dataset folders under $WORK/data")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--itau", default=None)
    ap.add_argument("--work", default=None)
    args = ap.parse_args()
    work = common.work_root(args.work)
    common.setup(common.itau_root(args.itau), work)
    from architecture_search.render_cache import verify_render_cache

    for name in args.data:
        paths = sorted((work / "data" / name).glob("*.pkl"))
        if not paths:
            raise SystemExit(f"no pair files in {work / 'data' / name}")
        common.ensure_render_cache(paths, workers=args.workers)
        for path in paths:
            verify_render_cache(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
