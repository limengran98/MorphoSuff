#!/usr/bin/env python3
"""Train one reporter head on a frozen raw-phase embedding.

All paths are explicit.  The program deliberately performs the split and
preprocessing after selecting the reporter's exact-paired cells, and fits every
normalization quantity on training rows only.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from common import (
    PhaseCache,
    aggregate_gene_profiles,
    assert_gene_split_integrity,
    atomic_json,
    atomic_npy,
    bootstrap_gene_metrics,
    embedding_lookup,
    fit_mlp,
    fit_preprocessing,
    load_reporter_data,
    matrix_metrics,
    partition_indices,
    profile_metrics,
    stable_seed,
    technical_core,
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reporter", required=True)
    parser.add_argument("--fold", required=True, type=int)
    parser.add_argument("--embedding-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--phase-cache", required=True, type=Path)
    parser.add_argument("--exact-cache-root", required=True, type=Path)
    parser.add_argument("--target-feature-dictionary", required=True, type=Path)
    parser.add_argument("--split", default="gene_holdout_main")
    parser.add_argument("--hidden", default="256,128")
    parser.add_argument("--batch-size", type=int, default=16384)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--bootstrap-draws", type=int, default=300)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = arguments()
    required = (
        args.embedding_root / "manifest.json",
        args.phase_cache / "manifest.json",
        args.exact_cache_root / f"all_cells_fluor_{args.reporter}.exact.h5",
        args.target_feature_dictionary,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing frozen-head inputs: {missing}")
    phase = PhaseCache.open(args.phase_cache, args.split)
    n_folds = int(phase.manifest.get("n_folds", 5))
    if args.fold < 0 or args.fold >= n_folds:
        raise ValueError(f"fold must be in [0,{n_folds - 1}]")
    feature_names = technical_core(args.target_feature_dictionary, args.reporter)
    source = required[2]
    data = load_reporter_data(source, args.reporter, feature_names)
    metadata = {
        name: np.asarray(values[data.phase_rows]) for name, values in phase.metadata.items()
    }
    fold_values = np.asarray(phase.folds[data.phase_rows], dtype=np.int16)
    train, validation, test, validation_fold = partition_indices(
        fold_values, args.fold, n_folds
    )
    assert_gene_split_integrity(
        train,
        validation,
        test,
        data.is_control,
        metadata["gene_code"],
        data.phase_rows,
    )
    if args.dry_run:
        print(
            json.dumps(
                {
                    "status": "DRY_RUN_PASS",
                    "reporter": args.reporter,
                    "fold": args.fold,
                    "validation_fold": validation_fold,
                    "n_train": len(train),
                    "n_validation": len(validation),
                    "n_test": len(test),
                    "n_endpoints": data.y.shape[1],
                },
                indent=2,
            )
        )
        return

    started = time.time()
    x_all = embedding_lookup(args.embedding_root, data.phase_rows)
    preprocessing = fit_preprocessing(x_all[train], data.y[train])
    x_train = preprocessing.transform_x(x_all[train])
    x_validation = preprocessing.transform_x(x_all[validation])
    x_test = preprocessing.transform_x(x_all[test])
    y_train = preprocessing.transform_y(data.y[train])
    y_validation = preprocessing.transform_y(data.y[validation])
    y_test = preprocessing.transform_y(data.y[test])
    hidden = [int(value) for value in args.hidden.split(",") if value.strip()]
    if not hidden or min(hidden) <= 0:
        raise ValueError("--hidden must contain positive comma-separated widths")
    seed = stable_seed("raw_image_frozen_head", args.reporter, args.fold)
    prediction_scaled, training = fit_mlp(
        x_train,
        y_train,
        x_validation,
        y_validation,
        x_test,
        hidden=hidden,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seed=seed,
        device_name=args.device,
    )
    feature_names_kept = data.target_feature_names[preprocessing.y_keep]
    target_test = ~data.is_control[test]
    truth_raw = data.y[test][:, preprocessing.y_keep]
    prediction_raw = preprocessing.inverse_y(prediction_scaled)
    baseline_raw = np.broadcast_to(
        preprocessing.y_mean, truth_raw[target_test].shape
    )
    cell_summary, cell_table = matrix_metrics(
        truth_raw[target_test],
        prediction_raw[target_test],
        baseline_raw,
        feature_names_kept,
    )
    profiles = aggregate_gene_profiles(
        y_test,
        prediction_scaled,
        metadata["screen_code"][test],
        metadata["gene_code"][test],
        data.is_control[test],
    )
    gene_summary, gene_table = profile_metrics(profiles, feature_names_kept)
    if args.bootstrap_draws:
        gene_summary.update(
            bootstrap_gene_metrics(profiles, args.bootstrap_draws, seed + 71)
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_npy(args.output_dir / "gene_truth_control_relative_scaled.npy", profiles.truth)
    atomic_npy(
        args.output_dir / "gene_prediction_control_relative_scaled.npy",
        profiles.prediction,
    )
    cell_table.to_csv(args.output_dir / "cell_endpoint_metrics.csv", index=False)
    gene_table.to_csv(args.output_dir / "gene_endpoint_metrics.csv", index=False)
    embedding_manifest = json.loads(
        (args.embedding_root / "manifest.json").read_text(encoding="utf-8")
    )
    result = {
        "status": "PASS",
        "completed": True,
        "representation": embedding_manifest.get("encoder", "frozen_raw_phase"),
        "backbone_frozen": True,
        "reporter_slug": args.reporter,
        "fold": args.fold,
        "validation_fold": validation_fold,
        "split": args.split,
        "n_target_features": len(feature_names_kept),
        "n_embedding_features": x_train.shape[1],
        "n_train": len(train),
        "n_validation": len(validation),
        "n_test": len(test),
        "cell_metrics": cell_summary,
        "gene_metrics": gene_summary,
        "head_hyperparameters": training,
        "embedding_fingerprint": embedding_manifest.get("fingerprint"),
        "runtime_seconds": time.time() - started,
    }
    atomic_json(args.output_dir / "training_result.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
