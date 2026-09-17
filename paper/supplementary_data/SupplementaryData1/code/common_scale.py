"""Read-only reconstruction of the manuscript's common NTC reference scale.

No model fitting and no writes are performed.  Equation (common-control-scale)
in paper/manuscript.tex uses population variance after physical-screen centering
of all unique out-of-fold non-targeting-control cells.  Fold-standardized
response vectors are converted by ``fold_y_scale / control_scale``.
"""
from __future__ import annotations

import json
import hashlib
from pathlib import Path
import time

import numpy as np


def control_scale(
    reporter: str,
    mm_root: str | Path,
    *,
    baseline: str | Path | None = None,
):
    """Return ``(endpoint_scales, JSON-compatible_audit)`` for one reporter.

    Large raw-target arrays are memory-mapped; only control rows and, if needed,
    duplicate rows are materialized.  Duplicate controls are admitted only when
    their raw targets and physical screens agree exactly.  Repeated targeting
    rows are rejected.  Returned paths identify every input used in this audit.
    """
    start = time.perf_counter()
    if Path(reporter).name != reporter:
        raise ValueError(f"Not a reporter slug: {reporter!r}")
    baseline = (Path(baseline) if baseline is not None else Path(mm_root)
                / "results/ops_reporter_specialists_v1/gene_holdout_main")
    control_ids, control_screens, control_raw = [], [], []
    test_ids, test_flags = [], []
    names = None
    fold_records, paths = [], []
    expected_total = None
    for fold in range(5):
        shared = baseline / f"fold_{fold}" / reporter / "shared"
        these_paths = {
            key: shared / filename
            for key, filename in {
                "truth_raw": "truth_raw.npy",
                "is_control": "test_is_control.npy",
                "phase_row_index": "test_phase_row_index.npy",
                "screen_code": "test_screen_code.npy",
                "manifest": "manifest.json",
                "target_feature_names": "target_feature_names.txt",
            }.items()
        }
        paths.extend(str(path) for path in these_paths.values())
        manifest = json.loads(these_paths["manifest"].read_text())
        fold_names = these_paths["target_feature_names"].read_text().splitlines()
        if names is None:
            names = fold_names
        elif names != fold_names:
            raise ValueError(f"{reporter}: endpoint order differs in fold {fold}")
        raw = np.load(these_paths["truth_raw"], mmap_mode="r", allow_pickle=False)
        flag = np.load(these_paths["is_control"], mmap_mode="r", allow_pickle=False)
        ids = np.load(these_paths["phase_row_index"], mmap_mode="r", allow_pickle=False)
        screens = np.load(these_paths["screen_code"], mmap_mode="r", allow_pickle=False)
        if raw.ndim != 2 or raw.shape[1] != len(names):
            raise ValueError(f"{reporter}: invalid target shape in fold {fold}")
        n = len(raw)
        if any(arr.shape != (n,) for arr in (flag, ids, screens)):
            raise ValueError(f"{reporter}: row shape mismatch in fold {fold}")
        mask = np.asarray(flag, dtype=bool)
        if not np.array_equal(flag, mask):
            raise ValueError(f"{reporter}: non-boolean control flags in fold {fold}")
        if int(manifest["n_test_observations"]) != n:
            raise ValueError(f"{reporter}: manifest observation mismatch in fold {fold}")
        if int(manifest["n_test_controls"]) != int(mask.sum()):
            raise ValueError(f"{reporter}: manifest control mismatch in fold {fold}")
        target_scale = np.asarray(manifest["preprocessing"]["y_scale"], dtype=float)
        if (target_scale.shape != (len(names),)
                or not np.isfinite(target_scale).all() or not (target_scale > 0).all()):
            raise ValueError(f"{reporter}: invalid training target scale in fold {fold}")
        finite_total = int(manifest["n_finite_technical_core_pairs"])
        if expected_total is None:
            expected_total = finite_total
        elif expected_total != finite_total:
            raise ValueError(f"{reporter}: cohort total differs across folds")
        values = np.asarray(raw[mask], dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f"{reporter}: nonfinite raw control targets in fold {fold}")
        control_ids.append(np.asarray(ids[mask]))
        control_screens.append(np.asarray(screens[mask]))
        control_raw.append(values)
        test_ids.append(np.asarray(ids))
        test_flags.append(mask)
        fold_records.append({
            "fold": fold,
            "n_test_rows": n,
            "n_control_rows": int(mask.sum()),
            "y_scale": target_scale.tolist(),
        })

    all_ids = np.concatenate(test_ids)
    all_flags = np.concatenate(test_flags)
    unique_ids, inverse, counts = np.unique(
        all_ids, return_inverse=True, return_counts=True
    )
    duplicate_mask = counts[inverse] > 1
    if np.any(duplicate_mask & ~all_flags):
        raise ValueError(f"{reporter}: targeting OOF phase rows repeat across folds")
    if len(unique_ids) != expected_total:
        raise ValueError(
            f"{reporter}: OOF coverage {len(unique_ids)} != cohort {expected_total}"
        )

    ids = np.concatenate(control_ids)
    screens = np.concatenate(control_screens)
    values = np.concatenate(control_raw, axis=0)
    unique_control_ids, first, inv = np.unique(ids, return_index=True, return_inverse=True)
    raw_control_count = len(ids)
    if not np.array_equal(screens, screens[first][inv]):
        raise ValueError(f"{reporter}: repeated control phase IDs change screen")
    if len(first) != len(ids):
        # Exact comparison: these are the same saved raw observation, not a fit.
        if not np.array_equal(values, values[first][inv]):
            raise ValueError(f"{reporter}: repeated control phase IDs change raw target")
    screens, values = screens[first], values[first]
    if not len(values):
        raise ValueError(f"{reporter}: no unique OOF controls")
    squared_deviations = np.zeros(len(names), dtype=np.float64)
    by_screen = []
    for screen in np.unique(screens):
        block = values[screens == screen]
        centered = block - block.mean(axis=0, dtype=np.float64)
        squared_deviations += np.sum(centered * centered, axis=0, dtype=np.float64)
        by_screen.append({"screen_code": int(screen), "n_unique_controls": len(block)})
    scales = np.sqrt(squared_deviations / len(values))
    if not np.isfinite(scales).all() or not np.all(scales > 1e-8):
        raise ValueError(f"{reporter}: common control scales are nonfinite or <= 1e-8")
    audit = {
        "reporter_slug": reporter,
        "definition": "pooled within-physical-screen population SD of unique OOF NTC cells",
        "ddof": 0,
        "n_endpoints": len(names),
        "endpoint_names": names,
        "n_fold_test_rows": len(all_ids),
        "n_unique_oof_rows": len(unique_ids),
        "expected_finite_cohort_rows": expected_total,
        "n_duplicate_test_rows_removed": len(all_ids) - len(unique_ids),
        "n_fold_control_rows": raw_control_count,
        "n_unique_controls": len(unique_control_ids),
        "n_duplicate_control_rows_removed": raw_control_count - len(unique_control_ids),
        "duplicate_raw_targets_and_screens_equal": True,
        "oof_unique_coverage_exhaustive": True,
        "screen_control_counts": by_screen,
        "scale_min": float(scales.min()),
        "scale_max": float(scales.max()),
        "selected_control_values_sha256": hashlib.sha256(values.tobytes()).hexdigest(),
        "selected_control_ids_sha256": hashlib.sha256(unique_control_ids.tobytes()).hexdigest(),
        "selected_control_screens_sha256": hashlib.sha256(screens.tobytes()).hexdigest(),
        "folds": fold_records,
        "input_paths": paths,
        "elapsed_seconds": time.perf_counter() - start,
    }
    return scales, audit
