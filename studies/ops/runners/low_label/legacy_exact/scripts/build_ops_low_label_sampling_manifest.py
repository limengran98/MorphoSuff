#!/usr/bin/env python3
"""Build the central Target12 low-label sampling assets used by every method."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import ops_low_label_sampling_lib as sampling
import run_ops_reporter_candidate_training as common_runner
from ops_reporter_specialist_lib import PhaseCache


def parse_fractions(value: str) -> tuple[float, ...]:
    result = tuple(float(token) for token in value.split(",") if token.strip())
    if not result or any(not 0.0 < item <= 1.0 for item in result):
        raise argparse.ArgumentTypeError("fractions must be comma-separated values in (0,1]")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="gene_holdout_main", choices=("gene_holdout_main",))
    parser.add_argument("--folds", default="0,1,2,3,4")
    parser.add_argument(
        "--fractions",
        type=parse_fractions,
        default=parse_fractions("0.001,0.01,0.2"),
    )
    parser.add_argument("--seed", type=int, default=sampling.DEFAULT_SEED)
    parser.add_argument("--output-root", type=Path, default=sampling.DEFAULT_ROOT)
    parser.add_argument("--phase-cache", type=Path, default=common_runner.DEFAULT_PHASE_CACHE)
    parser.add_argument("--exact-root", type=Path, default=common_runner.DEFAULT_EXACT_ROOT)
    parser.add_argument("--target-table", type=Path, default=common_runner.DEFAULT_TARGET_TABLE)
    parser.add_argument(
        "--target-feature-dictionary",
        type=Path,
        default=common_runner.DEFAULT_TARGET_FEATURE_DICTIONARY,
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    folds = tuple(int(token) for token in args.folds.split(",") if token.strip())
    if not folds or any(fold not in range(5) for fold in folds):
        raise ValueError("--folds must contain values 0..4")
    for name in (
        "output_root",
        "phase_cache",
        "exact_root",
        "target_table",
        "target_feature_dictionary",
    ):
        setattr(args, name, Path(getattr(args, name)).expanduser().resolve())

    table = pd.read_csv(args.target_table)
    targets = sampling.canonical_target12(table)
    indexed = table.set_index("reporter_slug", drop=False)
    phase_cache = PhaseCache.open(args.phase_cache)
    common_runner.validate_frozen_assets(phase_cache, table, args.exact_root)
    technical = common_runner.frozen_engine.load_technical_core_features(
        args.target_feature_dictionary, table
    )
    built: list[str] = []
    for fold in folds:
        heads = []
        for slug in targets:
            data = common_runner.load_sealed_training_data(
                common_runner.exact_cache_path(args.exact_root, slug),
                slug,
                technical[slug],
                phase_cache,
                args.split,
                fold,
            )
            namespace = argparse.Namespace(
                split=args.split,
                fold=fold,
                max_train_observations_per_head=0,
                max_validation_observations_per_head=0,
                max_test_observations_per_head=0,
            )
            heads.append(
                common_runner.build_sealed_training_partition(
                    namespace, phase_cache, indexed.loc[slug], data
                )
            )
        for fraction in args.fractions:
            manifest_file = sampling.manifest_path(
                args.output_root,
                split=args.split,
                fold=fold,
                fraction=fraction,
            )
            manifest_file.parent.mkdir(parents=True, exist_ok=True)
            selections = {}
            for head in heads:
                train = sampling.exact_budget_subset(
                    slug=head.slug,
                    indices=head.complete_train_indices,
                    phase_rows_all=head.data.phase_rows,
                    is_control_all=head.data.is_control,
                    phase_cache=phase_cache,
                    fraction=fraction,
                    seed=args.seed,
                    partition_name="train",
                )
                validation = sampling.exact_budget_subset(
                    slug=head.slug,
                    indices=head.complete_validation_indices,
                    phase_rows_all=head.data.phase_rows,
                    is_control_all=head.data.is_control,
                    phase_cache=phase_cache,
                    fraction=fraction,
                    seed=args.seed,
                    partition_name="validation",
                )
                selection_path = manifest_file.parent / f"{head.slug}.npz"
                selections[head.slug] = sampling.write_selection(
                    selection_path,
                    train_indices=train,
                    validation_indices=validation,
                    phase_rows=head.data.phase_rows,
                    source_h5_rows=head.data.source_h5_rows,
                )
            payload = {
                "schema_version": sampling.SCHEMA_VERSION,
                "split": args.split,
                "fold": fold,
                "label_fraction": float(fraction),
                "seed": args.seed,
                "target_reporters": list(targets),
                "selection_policy": {
                    "unit": "exact_reporter_observation",
                    "strata": "control_status_x_gene_code",
                    "allocation": "exact_budget_Hamilton",
                    "train_and_validation_subsampled": True,
                    "selection_uses_target_values": False,
                },
                "selections": selections,
            }
            if manifest_file.is_file():
                existing = sampling.load_manifest(manifest_file)
                if existing != payload:
                    raise RuntimeError(
                        f"Existing sampling manifest differs: {manifest_file}"
                    )
            else:
                sampling.write_manifest(manifest_file, payload)
            built.append(str(manifest_file))
            print(
                f"SAMPLING_MANIFEST_PASS fold={fold} fraction={fraction} "
                f"reporters={len(selections)} path={manifest_file}",
                flush=True,
            )
    print(json.dumps({"status": "complete", "manifests": built}, indent=2))


if __name__ == "__main__":
    main()
