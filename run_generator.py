#!/usr/bin/env python3
"""Run the paper's synthetic generator with a compact table of positive variants.

build_mix65_full_splits.py first lists every substitution variant of every reference name, for both
preferred distances, and keeps them all in memory, each with a list of edit descriptions: about 0.8 MB
per name, some 72 GB for a partition of 91,560 names, more than a server account may get.

This wrapper changes only how that table is stored. For every name it keeps each variant's string and
the indices of its edits (its signature) once, plus the order of the variants for each preferred
distance. When generate_split reads a variant, its edit descriptions are rebuilt from the signature with
the generator's own candidate_edits and apply_edits, which is how enumerate_positive_variants made them.
The generator's code, random draws and output are otherwise untouched, so the files are identical to
those of an unwrapped run; prepare_synthetic.py calls this script with the generator's arguments.
"""

from __future__ import annotations

import sys
from array import array
from pathlib import Path

import build_final_visual_hardneg_validation as hardneg
import build_mix65_full_splits as generator


class CompactVariants:
    """The variants of one name for one preferred distance, as enumerate_positive_variants orders them."""

    __slots__ = ("name", "entry", "order", "lookups")

    def __init__(self, name, entry, order, lookups):
        self.name, self.entry, self.order, self.lookups = name, entry, order, lookups

    def __len__(self):
        return len(self.order)

    def __iter__(self):
        generated, signatures = self.entry
        edits = None
        for index in self.order:
            if edits is None:
                edits = hardneg.candidate_edits(self.name, **self.lookups)
            signature = signatures[index]
            _, operations = hardneg.apply_edits(self.name, [edits[i] for i in signature])
            yield generated[index], operations, signature


class CompactCache:
    def __init__(self, lookups):
        self.lookups = lookups
        self.entries = {}

    def add(self, name, by_preferred):
        index = {}
        generated, signatures, orders = [], [], {}
        for preferred, variants in by_preferred.items():
            order = array("I")
            for text, _, signature in variants:
                if text not in index:
                    index[text] = len(generated)
                    generated.append(text)
                    signatures.append(signature)
                order.append(index[text])
            orders[preferred] = order
        entry = (tuple(generated), tuple(signatures))
        for preferred, order in orders.items():
            self.entries[(name, preferred)] = CompactVariants(name, entry, order, self.lookups)

    def get(self, key, default=None):
        return self.entries.get(key, default)


def build_compact_cache(names, *, ocr_lookup, exact_lookup, multichar_rules, max_edits):
    lookups = {"ocr_lookup": ocr_lookup, "exact_lookup": exact_lookup, "multichar_rules": multichar_rules}
    cache = CompactCache(lookups)
    for idx, name in enumerate(names):
        cache.add(name, {
            preferred: generator.enumerate_positive_variants(
                name, **lookups, max_edits=max_edits, preferred_distance=preferred
            )
            for preferred in (1, 2)
        })
        if (idx + 1) % 10000 == 0:
            print(f"cached_positive_variants={idx + 1}/{len(names)}", flush=True)
    return cache


if __name__ == "__main__":
    generator.build_positive_variant_cache = build_compact_cache
    sys.argv[0] = str(Path(generator.__file__))
    raise SystemExit(generator.main())
