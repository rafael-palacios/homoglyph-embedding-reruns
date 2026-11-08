#!/usr/bin/env python3
"""Build the Woodbridge partitions, the base files for the synthetic generator and the process-name test set.

Partition 0 is the paper's split: the train/validate/test lists of Woodbridge's domains_spoof.pkl with
".com" removed (976,122 / 51,380 / 256,886 pairs).

In Woodbridge's data every reference name has 9 spoofs (positives) and 5 negatives, and the 5 negatives
are the 5 reference names of the same split nearest to it in Levenshtein distance, with ties broken at
random (checked on the paper's test split: all negatives of 500 sampled references satisfy this).
Partitions 1, 2, ... therefore re-assign the reference names to train/validation/test at random, with
the same number of reference names per split as partition 0, keep each reference with its 9 spoofs,
and rebuild the negatives within each split with the same nearest-neighbour rule. Re-splitting the
original pairs directly would leak names: a test negative would pair a test reference with a name that
is a training reference.

Outputs (under --work):
  data/woodbridge_p<k>/{train,validation,test}.pkl   pair lists (fraudulent_name, real_name, label)
  base/p<k>/{train,validate,test}                    parquet inputs of build_mix65_full_splits.py;
                                                     validate is the paper's 9,999-pair subsample for
                                                     k = 0 and a stratified subsample of the same size
                                                     and class balance otherwise
  data/process/test.pkl                              Woodbridge process-name test split (--process-pkl)
  data/partitions.json                               sizes, class balance and overlap checks

domains_spoof.pkl stores (real_name, fraudulent_name, label); the paper's files use
(fraudulent_name, real_name, label). The script checks this against the paper's validation subsample.

Example:
  python make_partitions.py --work $WORK --partitions 0 1 2 \
      --domains-pkl $ITAU/../fine-grained-homoglyph-detection/data/raw/domains_spoof.pkl \
      --process-pkl $ITAU/../fine-grained-homoglyph-detection/data/raw/process_spoof.pkl \
      --val10k $ITAU/synth_dataset/base_datasets/validate_pairs_ref_10k.parquet
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

SPLITS = ("train", "validation", "test")
PICKLE_KEYS = {"train": "train", "validation": "validate", "test": "test"}


class _DataOnlyUnpickler(pickle.Unpickler):
    """The Woodbridge pickles contain only dicts, lists, tuples, strings and floats."""

    def find_class(self, module, name):
        raise pickle.UnpicklingError(f"Refusing to load object {module}.{name}: expected plain data only")


def load_pickle(path: Path) -> dict:
    with path.open("rb") as handle:
        return _DataOnlyUnpickler(handle, encoding="latin1").load()


def strip_com(name: str) -> str:
    return name[:-4] if name.lower().endswith(".com") else name


def convert(rows) -> list[tuple[str, str, int]]:
    return [(strip_com(str(r[1])), strip_com(str(r[0])), int(float(r[2]))) for r in rows]


def name_key(name: str) -> str:
    return unicodedata.normalize("NFKC", name).casefold()


def nearest_negatives(pool: list[str], rng: np.random.Generator, k: int = 5, chunk: int = 1000) -> list:
    """Pair each reference with the k other references of the pool nearest in Levenshtein distance."""

    from rapidfuzz.distance import Levenshtein
    from rapidfuzz.process import cdist

    out = []
    for start in range(0, len(pool), chunk):
        queries = pool[start:start + chunk]
        d = cdist(queries, pool, scorer=Levenshtein.distance, dtype=np.int32, workers=-1).astype(np.float32)
        d[np.arange(len(queries)), np.arange(start, start + len(queries))] = np.inf
        d += 0.5 * rng.random(d.shape, dtype=np.float32)  # random tie-breaking within a distance
        nearest = np.argpartition(d, k, axis=1)[:, :k]
        for ref, row in zip(queries, nearest):
            out.extend((pool[j], ref, 0) for j in row)
    return out


def regroup(splits: dict[str, list], seed: int) -> dict[str, list]:
    """Re-assign reference names to splits and rebuild the negatives within each split."""

    spoofs: dict[str, list] = defaultdict(list)
    for split in SPLITS:
        for row in splits[split]:
            if row[2] == 1:
                spoofs[row[1]].append(row)
    references = {split: {r[1] for r in splits[split]} for split in SPLITS}
    if set(spoofs) != set().union(*references.values()):
        raise SystemExit("some reference names have no spoofs; the nearest-neighbour rule does not apply")
    rng = np.random.default_rng(seed)
    names = sorted(spoofs)
    rng.shuffle(names)
    n_test, n_val = len(references["test"]), len(references["validation"])
    pools = {"test": names[:n_test], "validation": names[n_test:n_test + n_val], "train": names[n_test + n_val:]}
    out = {}
    for split, pool in pools.items():
        rows = [row for ref in pool for row in spoofs[ref]] + nearest_negatives(sorted(pool), rng)
        order = rng.permutation(len(rows))
        out[split] = [rows[i] for i in order]
        print(f"  {split}: {len(pool):,} reference names, {len(rows):,} pairs", flush=True)
    return out


def stratified_subsample(rows: list, positives: int, negatives: int, seed: int) -> list:
    rng = np.random.default_rng(seed)
    pos = [r for r in rows if r[2] == 1]
    neg = [r for r in rows if r[2] == 0]
    chosen = [pos[i] for i in rng.choice(len(pos), positives, replace=False)]
    chosen += [neg[i] for i in rng.choice(len(neg), negatives, replace=False)]
    return [chosen[i] for i in rng.permutation(len(chosen))]


def to_frame(rows: list) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "fraudulent_name": [r[0] for r in rows],
            "real_name": [r[1] for r in rows],
            "label": [float(r[2]) for r in rows],
        }
    )


def describe(splits: dict[str, list]) -> dict:
    from rapidfuzz.distance import Levenshtein

    refs = {split: {name_key(r[1]) for r in rows} for split, rows in splits.items()}
    train_strings = {name_key(s) for r in splits["train"] for s in r[:2]}
    test_rows = splits["test"]
    return {
        "sizes": {split: len(rows) for split, rows in splits.items()},
        "positives": {split: int(sum(r[2] for r in rows)) for split, rows in splits.items()},
        "reference_name_overlap": {
            "train_validation": len(refs["train"] & refs["validation"]),
            "train_test": len(refs["train"] & refs["test"]),
            "validation_test": len(refs["validation"] & refs["test"]),
        },
        "test_pairs_without_any_training_string": float(
            np.mean([name_key(r[0]) not in train_strings and name_key(r[1]) not in train_strings for r in test_rows])
        ),
        "mean_levenshtein_of_negatives": {
            split: float(np.mean([Levenshtein.distance(r[0], r[1]) for r in rows if r[2] == 0]))
            for split, rows in splits.items()
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", default=None)
    ap.add_argument("--domains-pkl", type=Path, required=True)
    ap.add_argument("--process-pkl", type=Path, default=None)
    ap.add_argument("--val10k", type=Path, required=True, help="synth_dataset/base_datasets/validate_pairs_ref_10k.parquet")
    ap.add_argument("--partitions", type=int, nargs="+", default=[0, 1, 2])
    args = ap.parse_args()
    work = common.work_root(args.work)

    raw = load_pickle(args.domains_pkl)
    paper = {split: convert(raw[PICKLE_KEYS[split]]) for split in SPLITS}
    expected = {"train": 976_122, "validation": 51_380, "test": 256_886}
    if {s: len(r) for s, r in paper.items()} != expected:
        raise SystemExit(f"unexpected split sizes in {args.domains_pkl}: { {s: len(r) for s, r in paper.items()} }")

    val10k = pd.read_parquet(args.val10k)[["fraudulent_name", "real_name", "label"]]
    val10k_keys = set(zip(val10k.fraudulent_name, val10k.real_name, val10k.label.astype(int)))
    matched = len(val10k_keys & set(paper["validation"]))
    print(f"paper validation subsample: {matched:,} / {len(val10k_keys):,} pairs found in the converted validation split")
    if matched < 0.99 * len(val10k_keys):
        raise SystemExit("column convention does not match the paper's files")
    val10k_pos = int(val10k.label.sum())
    val10k_neg = len(val10k) - val10k_pos

    report = {}
    for k in args.partitions:
        splits = paper if k == 0 else regroup(paper, seed=1000 + k)
        out = work / "data" / f"woodbridge_p{k}"
        for split, rows in splits.items():
            common.write_rows(rows, out / f"{split}.pkl")

        base = work / "base" / f"p{k}"
        base.mkdir(parents=True, exist_ok=True)
        to_frame(splits["train"]).to_parquet(base / "train", index=False)
        to_frame(splits["test"]).to_parquet(base / "test", index=False)
        if k == 0:
            val10k.to_parquet(base / "validate", index=False)
        else:
            sub = stratified_subsample(splits["validation"], val10k_pos, val10k_neg, seed=2000 + k)
            to_frame(sub).to_parquet(base / "validate", index=False)

        report[f"p{k}"] = describe(splits)
        print(f"p{k}: {json.dumps(report[f'p{k}'])}", flush=True)

    if args.process_pkl:
        process = load_pickle(args.process_pkl)
        rows = convert(process["test"])
        common.write_rows(rows, work / "data" / "process" / "test.pkl")
        report["process_test"] = {"size": len(rows), "positives": int(sum(r[2] for r in rows))}
        print(f"process test: {report['process_test']}")

    path = work / "data" / "partitions.json"
    previous = json.loads(path.read_text()) if path.exists() else {}
    path.write_text(json.dumps({**previous, **report}, indent=2) + "\n")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
