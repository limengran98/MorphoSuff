#!/usr/bin/env python3
"""Calculate OPS control-relative response fidelity from a local prediction table.

Scores follow the published Methods: they are computed on the Euclidean norm of
each held-out gene's control-relative endpoint response vector, and strong-hit
recall is the top 5% of genes, not an absolute top-k.
"""
from __future__ import annotations
import argparse
from pathlib import Path
from _common import read_table, write_table
from measurement_sufficiency.adapters import LocalManifestAdapter
from measurement_sufficiency.analysis.response_fidelity import response_fidelity

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--top-fraction",
        type=float,
        default=0.05,
        help="strong-hit recall cut as a fraction of held-out genes (Methods: top 5%%)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="optional absolute rank cut, reported alongside the fractional one; not the tier input",
    )
    parser.add_argument(
        "--across-screens",
        choices=("mean", "separate"),
        default="mean",
        help="average a gene's response over its screens before taking the norm, or score each screen",
    )
    parser.add_argument(
        "--dataset-manifest",
        type=Path,
        help="canonical manifest; when given, the declared screen_matched_controls capability is enforced",
    )
    args = parser.parse_args()
    capabilities = (
        LocalManifestAdapter(args.dataset_manifest).capabilities if args.dataset_manifest else None
    )
    output = response_fidelity(
        read_table(args.predictions),
        top_fraction=args.top_fraction,
        top_k=args.top_k,
        across_screens=args.across_screens,
        capabilities=capabilities,
    )
    for name, table in output.items():
        write_table(table, args.output_dir / f"{name}.csv")
    return 0
if __name__ == "__main__": raise SystemExit(main())
