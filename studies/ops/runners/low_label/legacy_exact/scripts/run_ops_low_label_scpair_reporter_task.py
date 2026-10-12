#!/usr/bin/env python3
"""Run one source-faithful scPair reporter under the frozen gene low-label protocol.

This wrapper changes only the labelled training/inner-validation rows of the
already audited reporter-specific scPair OPS adapter.  The selected rows come
from the same immutable Target12 manifest used by every other low-label
method.  The outer held-out-gene fold remains physically unopened until all
three validation-selected scPair states have been frozen.

scPair is reporter-specific in this benchmark, so one process owns exactly one
reporter, fraction and outer fold.  Donor40 labels are absent; aggregating the
twelve reporter jobs later gives the method-level Target12 result without
inventing a cross-reporter scPair architecture.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import ops_low_label_sampling_lib as sampling
import run_ops_paired_source_benchmark_task as paired


SCHEMA = "ops-low-label-scpair-reporter-result-v1"
DEFAULT_SAMPLING = Path(__file__).resolve().parents[1] / "results/ops_low_label_sampling_manifest_v1"


def _pop(argv: list[str], name: str, *, required: bool = False) -> str | None:
    if name not in argv:
        if required:
            raise SystemExit(f"Missing required option {name}")
        return None
    index = argv.index(name)
    if index + 1 >= len(argv):
        raise SystemExit(f"Missing value for {name}")
    value = argv[index + 1]
    del argv[index : index + 2]
    return value


def main() -> None:
    if "--help" in sys.argv or "-h" in sys.argv:
        print(
            "usage: run_ops_low_label_scpair_reporter_task.py --method scpair_ops "
            "--reporter SLUG --label-fraction FRACTION --split gene_holdout_main "
            "--fold {0,1,2,3,4} --output-root PATH "
            "--sampling-manifest-root PATH [delegated OPS asset options]\n\n"
            + (__doc__ or "")
        )
        return
    argv = list(sys.argv)
    raw_fraction = _pop(argv, "--label-fraction", required=True)
    sampling_root = Path(
        _pop(argv, "--sampling-manifest-root") or DEFAULT_SAMPLING
    ).expanduser().resolve()
    assert raw_fraction is not None
    fraction = float(raw_fraction)
    if not 0.0 < fraction <= 1.0:
        raise SystemExit("--label-fraction must be in (0, 1]")

    # The source runner retains ownership of all ordinary arguments.  Its
    # method is fixed here so a plan cannot accidentally substitute APOLLO.
    if "--method" in argv:
        method_index = argv.index("--method")
        if method_index + 1 >= len(argv) or argv[method_index + 1] != "scpair_ops":
            raise SystemExit("This wrapper accepts only --method scpair_ops")
    else:
        argv[1:1] = ["--method", "scpair_ops"]
    sys.argv = argv

    parser_args = paired.parser().parse_args(argv[1:])
    if parser_args.split != "gene_holdout_main":
        raise SystemExit("Frozen scPair low-label alignment uses gene_holdout_main only")
    if parser_args.reporter is None or parser_args.task_id is not None:
        raise SystemExit("One --reporter and no --task-id are required")
    if int(parser_args.seed) != sampling.DEFAULT_SEED:
        raise SystemExit(
            f"Frozen sampling seed must be {sampling.DEFAULT_SEED}, got {parser_args.seed}"
        )

    table = pd.read_csv(parser_args.target_table)
    target12 = sampling.canonical_target12(table)
    if parser_args.reporter not in set(target12):
        raise SystemExit(f"Reporter is outside frozen Target12: {parser_args.reporter}")
    manifest_file = sampling.manifest_path(
        sampling_root,
        split=parser_args.split,
        fold=parser_args.fold,
        fraction=fraction,
    )
    payload = sampling.load_manifest(manifest_file)
    sampling.validate_manifest_binding(
        payload,
        split=parser_args.split,
        fold=parser_args.fold,
        fraction=fraction,
        seed=parser_args.seed,
        target_reporters=target12,
    )

    original_prepare = paired.prepare_normal

    def prepare_low_label(args: Any) -> tuple[Any, Any, Any, dict[str, Any]]:
        phase, head, x_state, seal = original_prepare(args)
        train, validation = sampling.load_and_verify_selection(
            manifest_file,
            payload,
            slug=head.slug,
            phase_rows=head.data.phase_rows,
            source_h5_rows=head.data.source_h5_rows,
        )
        head.complete_train_indices = train.copy()
        head.active_train_indices = train.copy()
        head.partial_train_indices = np.empty(0, dtype=np.int64)
        head.complete_validation_indices = validation.copy()
        head.y_preprocessing = paired.frozen.fit_head_y_preprocessing(
            head.data.y, train
        )
        seal.update(
            {
                "schema_version": SCHEMA,
                "label_fraction": fraction,
                "sampling_manifest": str(manifest_file),
                "sampling_manifest_sha256": sampling.file_sha256(manifest_file),
                "sampling_schema": payload["schema_version"],
                "target_reporter_policy": "panel12_low_label",
                "donor_reporter_policy": "absent",
                "experiment_arm": "target_only",
                "selected_train": int(len(train)),
                "selected_validation": int(len(validation)),
                "outer_test_y_physically_loaded": False,
            }
        )
        return phase, head, x_state, seal

    paired.prepare_normal = prepare_low_label
    paired.main()

    # Contract checks deliberately produce no output marker.
    if "--contract-check-only" in argv:
        return
    output_root = Path(parser_args.output_root).expanduser().resolve()
    marker = output_root / "training_result.json"
    if not marker.is_file():
        raise RuntimeError(f"scPair source runner did not emit {marker}")
    result = json.loads(marker.read_text(encoding="utf-8"))
    result.update(
        {
            "schema_version": SCHEMA,
            "status": "complete",
            "completed": True,
            "method_id": "scpair_ops",
            "split": "gene_holdout_main",
            "fold": int(parser_args.fold),
            "reporter_slug": str(parser_args.reporter),
            "label_fraction": fraction,
            "seed": int(parser_args.seed),
            "target_reporter_policy": "panel12_low_label",
            "donor_reporter_policy": "absent",
            "experiment_arm": "target_only",
            "n_target_reporters": 1,
            "execution_device": "cuda",
            "cpu_fallback_allowed": False,
            "outer_test_used_for_selection": False,
            "outer_test_y_loaded_after_checkpoint_selection": True,
            "sampling_manifest": str(manifest_file),
            "sampling_manifest_sha256": sampling.file_sha256(manifest_file),
        }
    )
    paired.atomic_json(marker, result)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
