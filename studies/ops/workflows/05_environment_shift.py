#!/usr/bin/env python3
"""Calculate directed strict-screen environment shift metrics and their association.

Transfer loss follows the published Methods: the held-out-gene knockout-response
Pearson minus the strict whole-screen one, both supplied by the caller from
evaluations already run. The association P value permutes whole reporter clusters
so that directions from one reporter move together.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from _common import read_table, write_table
from measurement_sufficiency.adapters import LocalManifestAdapter
from measurement_sufficiency.analysis.environment_shift import (
    cluster_permutation_spearman,
    environment_shift,
)

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transfer-metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--within-column", default="gene_pearson")
    parser.add_argument("--shifted-column", default="strict_screen_pearson")
    parser.add_argument("--destination-column", default="destination_screen_id")
    parser.add_argument("--source-column", help="optional provenance column; never used in the arithmetic")
    parser.add_argument("--group-column", action="append", default=None,
                        help="repeatable; defaults to reporter_id and model_id")
    parser.add_argument("--output-column", default="gene_minus_strict_screen_pearson",
                        help="name of the emitted transfer-loss column")
    parser.add_argument("--shift-column", action="append", default=None,
                        help="repeatable predictor column to associate with transfer loss")
    parser.add_argument("--association-output", type=Path,
                        help="write one row per shift predictor with rho, P and the permutation settings")
    parser.add_argument("--permutations", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cluster-column", default="reporter_id")
    parser.add_argument("--dataset-manifest", type=Path,
                        help="canonical manifest; when given, the declared repeated_screens capability is enforced")
    parser.add_argument("--expect-directions", type=int,
                        help="fail unless exactly this many directed evaluations are present")
    args = parser.parse_args()

    metrics = read_table(args.transfer_metrics)
    capabilities = (
        LocalManifestAdapter(args.dataset_manifest).capabilities if args.dataset_manifest else None
    )
    shifted = environment_shift(
        metrics,
        within_environment_column=args.within_column,
        shifted_environment_column=args.shifted_column,
        destination_column=args.destination_column,
        group_columns=args.group_column,
        source_column=args.source_column,
        output_column=args.output_column,
        capabilities=capabilities,
    )
    if args.expect_directions is not None and len(shifted) != args.expect_directions:
        raise SystemExit(
            f"expected {args.expect_directions} directed evaluations, found {len(shifted)}"
        )
    write_table(shifted, args.output)

    if args.shift_column:
        joined = shifted.merge(
            metrics.loc[:, [args.destination_column, args.cluster_column, *args.shift_column]],
            on=[args.destination_column, args.cluster_column],
            how="left",
            validate="one_to_one",
        )
        associations = []
        for predictor in args.shift_column:
            result = cluster_permutation_spearman(
                joined,
                x_column=predictor,
                y_column=args.output_column,
                cluster_column=args.cluster_column,
                n_permutations=args.permutations,
                seed=args.seed,
            )
            associations.append({"predictor": predictor, **result})
        if args.association_output:
            write_table(__import__("pandas").DataFrame(associations), args.association_output)
        else:
            print(json.dumps(associations, indent=2))
    return 0

if __name__ == "__main__": raise SystemExit(main())
