#!/usr/bin/env python3
"""Build a shortcut-controlled version of a data partition (character-set shortcut).

In the synthetic dataset, positive candidates carry the non-ASCII substitutions, while negative candidates
are unmodified legitimate names. A model can therefore separate the classes by looking only at the
character set of the candidate. The Woodbridge benchmark has the same cue to a lesser degree (its negatives
are also legitimate names). This script removes that cue: every negative candidate B (a legitimate
name paired with a different reference A) is replaced by a homoglyph-substituted copy B' of itself. B'
still imitates B, not A, so the label stays 0.

The substitutions copy the positives:
  - the number of non-ASCII characters added to each negative candidate is drawn from the distribution of
    non-ASCII characters in the positive candidates of the training split whose reference has the same
    length (zero included), so both classes end up with the same distribution at every name length;
  - the positions to substitute are drawn with the rate at which the generator substituted each source
    character in the positive pairs of the training split (substitutions per occurrence), and each
    replacement with the frequency observed for that source character, so the negatives also get the
    positives' mix of Latin, non-Latin and NFKC-changing replacements.

Writes data/<source>_ctrl_p<k>/{train,validation,test}.pkl and control_report.json, which compares
the non-ASCII distributions of both classes before and after. The synthetic models are also retrained on
the controlled data; for the Woodbridge benchmark only the controlled test split is used, to score the
models trained on the original data.

Example:
  python make_controlled.py --partition 0                       # synthetic_ctrl_p0
  python make_controlled.py --partition 0 --source woodbridge   # woodbridge_ctrl_p0
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

SPLITS = ("train", "validation", "test")


def non_ascii(text: str) -> int:
    return sum(ord(ch) > 127 for ch in text)


def substitution_table(rows) -> dict[str, tuple[float, list[str], np.ndarray]]:
    """Source character -> (substitutions per occurrence, replacements, replacement probabilities)."""
    counts: dict[str, Counter] = defaultdict(Counter)
    occurrences: Counter = Counter()
    for cand, ref, label in rows:
        if label == 1 and len(cand) == len(ref):
            occurrences.update(ref)
            for a, b in zip(ref, cand):
                if a != b and ord(b) > 127:
                    counts[a][b] += 1
    table = {}
    for source, counter in counts.items():
        replacements = sorted(counter)
        weights = np.array([counter[r] for r in replacements], dtype=float)
        table[source] = (weights.sum() / occurrences[source], replacements, weights / weights.sum())
    return table


def substitute(name: str, k: int, table, rng: np.random.Generator) -> tuple[str, int]:
    positions = [i for i, ch in enumerate(name) if ch in table]
    if k <= 0 or not positions:
        return name, 0
    rates = np.array([table[name[i]][0] for i in positions])
    chosen = rng.choice(positions, size=min(k, len(positions)), replace=False, p=rates / rates.sum())
    chars = list(name)
    for i in chosen:
        _, replacements, probs = table[chars[i]]
        chars[i] = replacements[rng.choice(len(replacements), p=probs)]
    return "".join(chars), len(chosen)


def length_bucket(name: str) -> int:
    return min(len(name), 20)


def counts_by_length(rows) -> dict[int, np.ndarray]:
    """Non-ASCII characters per positive candidate, grouped by the length of its reference."""
    groups: dict[int, list[int]] = defaultdict(list)
    for cand, ref, label in rows:
        if label == 1:
            groups[length_bucket(ref)].append(non_ascii(cand))
    lengths = sorted(groups)
    # Lengths with few positives borrow from the nearest length with enough of them
    filled = {}
    for length in range(1, 21):
        nearest = min((g for g in lengths if len(groups[g]) >= 50), key=lambda g: abs(g - length), default=None)
        filled[length] = np.array(groups[nearest] if nearest is not None else [0])
    return filled


def histogram(values) -> dict[str, float]:
    counts = Counter(min(v, 4) for v in values)
    total = sum(counts.values())
    return {("4+" if k == 4 else str(k)): round(counts[k] / total, 4) for k in sorted(counts)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--partition", type=int, required=True)
    ap.add_argument("--source", default="synthetic", choices=["synthetic", "woodbridge"])
    ap.add_argument("--work", default=None)
    args = ap.parse_args()
    work = common.work_root(args.work)
    src = work / "data" / f"{args.source}_p{args.partition}"
    dst = work / "data" / f"{args.source}_ctrl_p{args.partition}"
    data = {split: common.read_rows(src / f"{split}.pkl") for split in SPLITS}

    table = substitution_table(data["train"])
    positive_counts = counts_by_length(data["train"])
    print(f"{len(table)} source characters with substitutions; non-ASCII per positive candidate: "
          f"{histogram(non_ascii(c) for c, _, y in data['train'] if y == 1)}")

    rng = np.random.default_rng((3000 if args.source == "synthetic" else 4000) + args.partition)
    report = {"substitution_sources": len(table), "splits": {}}
    for split in SPLITS:
        rows, out, shortfall = data[split], [], 0
        for cand, ref, label in rows:
            if label == 0:
                k = int(rng.choice(positive_counts[max(length_bucket(cand), 1)]))
                new, done = substitute(cand, k, table, rng)
                shortfall += k - done
                if new != ref:
                    cand = new
            out.append((cand, ref, label))
        common.write_rows(out, dst / f"{split}.pkl")
        report["splits"][split] = {
            "rows": len(out),
            "requested_substitutions_not_possible": int(shortfall),
            "non_ascii_before": {
                "positive": histogram(non_ascii(c) for c, _, y in rows if y == 1),
                "negative": histogram(non_ascii(c) for c, _, y in rows if y == 0),
            },
            "non_ascii_after": {
                "positive": histogram(non_ascii(c) for c, _, y in out if y == 1),
                "negative": histogram(non_ascii(c) for c, _, y in out if y == 0),
            },
        }
        print(f"{split}: {json.dumps(report['splits'][split]['non_ascii_after'])}", flush=True)

    (dst / "control_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
