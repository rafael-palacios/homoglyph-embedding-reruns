#!/usr/bin/env python3
"""Generate the synthetic substitution dataset ("mix65") for one partition.

Runs the paper's generator (synth_dataset/generate_validation/build_mix65_full_splits.py, unchanged) on
base/p<k>/ from make_partitions.py, with generation seed 42 + k (42 is the paper's seed), and converts its
output into data/synthetic_p<k>/. All partitions, p0 included, are generated this way: the paper's
generated files are not in the repository.

The generator needs the OCR and multi-character lookup tables as parquet; the repository stores them as
CSV, so they are converted once into $WORK/lookups/.

Negative candidates are drawn from the reference names of the Woodbridge benchmark (Woodbridge et al.,
IEEE SPW 2018; data/domains_spoof.pkl in github.com/endgameinc/homoglyph), as the manuscript states:
the generator's own fallback when no extra name list is given, i.e. the 91,560 names of the base files of
p0 (69,723 train + 3,488 validation + 18,349 test). The paper's run had read an unpublished subset of
87,211 of these names (DONOTDELETE/unique_real_names_no_single_char_hyphen_prefix.parquet), which is in
none of the project's repositories. With the full set, the regenerated p0 reproduces every count in the
paper's generated_datasets/mix65/generation_metrics.json (split and class sizes, real-name pools, 799,076
distinct positive candidates, 45 negative length buckets); the negatives themselves differ.

--negative-names swap-lookup uses instead the 87,337 names of lookup_tables/archive/
best_legit_adjacent_swap_lookup.csv (commit 4f93925). It is not recommended: it has no names of 1-3
characters, so p0 gets 42 negative length buckets instead of the paper's 45.

--from-parquet DIR converts existing mix65 files (train.parquet, validation.parquet, test.parquet) instead.

Example:
  python prepare_synthetic.py --partition 0 --generate
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

FILES = {"train": "train.parquet", "validation": "validation.parquet", "test": "test.parquet"}
LOOKUPS = ("ocr_q25_lookup", "multichar_forward_q25_lookup")
REAL_NAMES = Path("lookup_tables") / "archive" / "best_legit_adjacent_swap_lookup.csv"


def to_pickles(parquet_dir: Path, out: Path) -> dict:
    sizes = {}
    for split, name in FILES.items():
        frame = pd.read_parquet(parquet_dir / name, columns=["fraudulent_name", "real_name", "label"])
        rows = [(str(a), str(b), int(float(c))) for a, b, c in frame.itertuples(index=False, name=None)]
        common.write_rows(rows, out / f"{split}.pkl")
        sizes[split] = {"rows": len(rows), "positives": int(sum(r[2] for r in rows))}
    return sizes


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--partition", type=int, required=True)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--from-parquet", type=Path)
    mode.add_argument("--generate", action="store_true")
    ap.add_argument("--negative-names", choices=("woodbridge", "swap-lookup"), default="woodbridge",
                    help="pool of names negatives are drawn from (default: all Woodbridge reference names)")
    ap.add_argument("--itau", default=None)
    ap.add_argument("--work", default=None)
    args = ap.parse_args()
    itau, work = common.itau_root(args.itau), common.work_root(args.work)
    synth = itau / "synth_dataset"
    out = work / "data" / f"synthetic_p{args.partition}"

    if args.from_parquet:
        source = args.from_parquet.resolve()
    else:
        base = work / "base" / f"p{args.partition}"
        if not (base / "train").exists():
            raise SystemExit(f"{base} is missing: run make_partitions.py first")
        lookups = work / "lookups"
        lookups.mkdir(parents=True, exist_ok=True)
        for name in LOOKUPS:
            target = lookups / f"{name}.parquet"
            if not target.exists():
                pd.read_csv(synth / "lookup_tables" / "in_use" / f"{name}.csv").to_parquet(target, index=False)
        if args.negative_names == "swap-lookup":
            real_names = lookups / "real_names_adjacent_swap.parquet"
            if not real_names.exists():
                if not (synth / REAL_NAMES).exists():
                    raise SystemExit(f"{synth / REAL_NAMES} is missing: check out the paper's commit 4f93925")
                names = pd.read_csv(synth / REAL_NAMES, usecols=["real_name"], dtype=str, keep_default_na=False)
                names.to_parquet(real_names, index=False)
        else:
            # A path that does not exist makes the generator fall back to the base files' reference names
            real_names = lookups / "no_extra_real_names.parquet"
            if real_names.exists():
                raise SystemExit(f"{real_names} must not exist")
        source = work / "synthetic_generated" / f"p{args.partition}"
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(
            [str(synth / "generate_validation"), str(synth / "scripts"), env.get("PYTHONPATH", "")]
        )
        cmd = [
            sys.executable,
            str(Path(__file__).resolve().parent / "run_generator.py"),  # the paper's generator, compact variant table
            "--input-dir", str(base),
            "--output-dir", str(source),
            "--lookup-dir", str(lookups),
            "--old-exact-lookup", str(synth / "lookup_tables" / "archive" / "dejavu_sans_exact_lookalike_lookup.csv"),
            "--unique-real-names", str(real_names),
            "--seed", str(42 + args.partition),
        ]
        print(" ".join(cmd), flush=True)
        subprocess.run(cmd, cwd=synth, env=env, check=True)

    sizes = to_pickles(source, out)
    (out / "source.json").write_text(json.dumps({"source": str(source), "negative_names": None if args.from_parquet else args.negative_names, "sizes": sizes}, indent=2) + "\n")
    print(f"wrote {out}: {sizes}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
