#!/usr/bin/env python3
"""Apply the frozen OPS/A549 measurement-tier rule to an evidence table.

The evidence table is an external input: it carries recoverability, magnitude
Spearman, variance ratio, top-5% recall and within-screen reliability per
reporter. `--rename` maps the frozen figure-package column names onto the ones
the rule consumes without editing the CSV.
"""
from __future__ import annotations
import argparse
from pathlib import Path
from _common import read_table, write_table
from measurement_sufficiency.analysis.tiers import tier_table

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    # default=None rather than a list literal: with action="append" a supplied
    # value would otherwise be added to the default instead of replacing it, so
    # reporter_id could never be removed.
    parser.add_argument("--group-column", action="append", default=None,
                        help="repeatable; defaults to reporter_id")
    parser.add_argument("--aggregate", choices=("median", "mean", "none"), default="median",
                        help="how to reduce replicate evidence rows within a group")
    parser.add_argument("--tier-column", default="tier",
                        help="name of the emitted tier column (the frozen tables use substitutability_tier)")
    parser.add_argument("--rename", action="append", default=None, metavar="OLD=NEW",
                        help="repeatable column rename applied before tiering")
    parser.add_argument("--expect-counts", metavar="TIER=N,...",
                        help="fail unless the tier counts match exactly, e.g. the published "
                             "quantitative_proxy=30,ranking_proxy=6,measurement_required=3,"
                             "not_identifiable=9,unresolved=4")
    args = parser.parse_args()

    evidence = read_table(args.evidence)
    for mapping in args.rename or []:
        old, _, new = mapping.partition("=")
        if not old or not new:
            raise SystemExit(f"--rename expects OLD=NEW, got {mapping!r}")
        if old not in evidence:
            raise SystemExit(f"--rename source column {old!r} is not in the evidence table")
        evidence = evidence.rename(columns={old: new})

    table = tier_table(
        evidence,
        group_columns=args.group_column or ["reporter_id"],
        aggregate=args.aggregate,
        tier_column=args.tier_column,
    )
    if args.expect_counts:
        expected = {}
        for item in args.expect_counts.split(","):
            name, _, count = item.partition("=")
            if not name or not count.isdigit():
                raise SystemExit(f"--expect-counts expects TIER=N pairs, got {item!r}")
            expected[name.strip()] = int(count)
        observed = table[args.tier_column].value_counts().to_dict()
        if {key: observed.get(key, 0) for key in expected} != expected:
            raise SystemExit(
                f"tier counts do not reproduce the expected decision table: "
                f"expected {expected}, observed {observed}"
            )
        print(f"tier counts reproduce the expected decision table: {expected}")
    write_table(table, args.output)
    return 0

if __name__ == "__main__": raise SystemExit(main())
