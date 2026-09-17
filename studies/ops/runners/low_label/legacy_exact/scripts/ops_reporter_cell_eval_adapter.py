#!/usr/bin/env python3
"""OPS-native perturbation metrics adapted from Arc Institute cell-eval.

The upstream package evaluates perturbation-response transcriptomes stored in
AnnData.  OPS instead has a separate, heterogeneous 20--72 dimensional
phenotype space for every reporter.  This module therefore implements only the
parts whose statistical meaning transfers cleanly:

* perturbation-control delta MSE, MAE and profile Pearson correlation;
* perturbation identity retrieval (``discrimination_score``) within screen;
* a truth-only, stratified bootstrap split-half data ceiling.

All inputs are already train-standardised and control-relative.  Metrics are
computed separately for each reporter, with screen-matched controls, and are
reporting-only.  Differential-expression and clustering metrics are
deliberately not relabelled as phenotype metrics.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ops_reporter_specialist_lib import GeneProfiles


EVALUATOR_ID = "common_ops_reporter_evaluator_v2_cell_eval_adapted"
ADAPTER_SCHEMA_VERSION = "ops-reporter-cell-eval-adapter-v1"
UPSTREAM_RELEASE = "0.8.1"
UPSTREAM_REPOSITORY = "https://github.com/ArcInstitute/cell-eval"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _as_finite_matrix(value: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    _require(result.ndim == 2 and result.shape[0] > 0, f"{name} must be a nonempty matrix")
    _require(result.shape[1] >= 2, f"{name} needs at least two endpoints")
    _require(np.all(np.isfinite(result)), f"{name} contains nonfinite values")
    return result


def _nanmean(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    # The frozen OPS result schema forbids non-finite aggregate metrics.  Keep
    # the exact defined-count beside correlation summaries and use neutral
    # zero only when no profile has a defined correlation.
    return float(np.mean(values[finite])) if np.any(finite) else 0.0


def _row_pearson(truth: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    truth_centered = truth - truth.mean(axis=1, keepdims=True)
    prediction_centered = prediction - prediction.mean(axis=1, keepdims=True)
    numerator = np.sum(truth_centered * prediction_centered, axis=1)
    denominator = np.sqrt(
        np.sum(truth_centered**2, axis=1)
        * np.sum(prediction_centered**2, axis=1)
    )
    result = np.full(len(truth), np.nan, dtype=np.float64)
    valid = denominator > 0
    result[valid] = numerator[valid] / denominator[valid]
    return result


def _row_cosine(truth: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    denominator = np.linalg.norm(truth, axis=1) * np.linalg.norm(prediction, axis=1)
    result = np.full(len(truth), np.nan, dtype=np.float64)
    valid = denominator > 0
    result[valid] = np.sum(truth[valid] * prediction[valid], axis=1) / denominator[valid]
    return result


def _stable_ranks(distances: np.ndarray, correct_indices: np.ndarray) -> np.ndarray:
    """Return zero-based ranks matching NumPy's stable ascending argsort."""

    rows = np.arange(len(distances))
    correct = distances[rows, correct_indices]
    strictly_better = np.sum(distances < correct[:, None], axis=1)
    candidate_indices = np.arange(distances.shape[1])[None, :]
    tied_before = np.sum(
        (distances == correct[:, None])
        & (candidate_indices < correct_indices[:, None]),
        axis=1,
    )
    return np.asarray(strictly_better + tied_before, dtype=np.int64)


def _distance_block(
    prediction: np.ndarray,
    truth: np.ndarray,
    metric: str,
) -> np.ndarray:
    if metric == "l2":
        distances = (
            np.sum(prediction**2, axis=1, keepdims=True)
            + np.sum(truth**2, axis=1)[None, :]
            - 2.0 * prediction @ truth.T
        )
        return np.maximum(distances, 0.0)
    if metric == "cosine":
        prediction_norm = np.linalg.norm(prediction, axis=1, keepdims=True)
        truth_norm = np.linalg.norm(truth, axis=1, keepdims=True)
        prediction_unit = np.divide(
            prediction,
            prediction_norm,
            out=np.zeros_like(prediction),
            where=prediction_norm > 0,
        )
        truth_unit = np.divide(
            truth,
            truth_norm,
            out=np.zeros_like(truth),
            where=truth_norm > 0,
        )
        return 1.0 - prediction_unit @ truth_unit.T
    if metric == "l1":
        # Keep the temporary [query, candidate, endpoint] tensor below 64 MiB.
        bytes_per_query = max(1, truth.shape[0] * truth.shape[1] * 8)
        query_chunk = max(1, min(len(prediction), (64 * 1024**2) // bytes_per_query))
        result = np.empty((len(prediction), len(truth)), dtype=np.float64)
        for start in range(0, len(prediction), query_chunk):
            stop = min(start + query_chunk, len(prediction))
            result[start:stop] = np.sum(
                np.abs(prediction[start:stop, None, :] - truth[None, :, :]),
                axis=2,
            )
        return result
    raise ValueError(f"Unsupported retrieval metric: {metric}")


def _within_screen_retrieval(
    truth: np.ndarray,
    prediction: np.ndarray,
    screen_code: np.ndarray,
) -> dict[str, np.ndarray]:
    n_rows = len(truth)
    output = {
        f"discrimination_{metric}": np.full(n_rows, np.nan, dtype=np.float64)
        for metric in ("l1", "l2", "cosine")
    }
    for metric in ("l1", "l2", "cosine"):
        output[f"top1_{metric}"] = np.full(n_rows, np.nan, dtype=np.float64)
        output[f"top5_{metric}"] = np.full(n_rows, np.nan, dtype=np.float64)
        output[f"rank_{metric}"] = np.full(n_rows, -1, dtype=np.int64)

    for screen in np.unique(screen_code):
        indices = np.flatnonzero(screen_code == screen)
        _require(len(indices) > 0, "Empty screen group")
        for metric in ("l1", "l2", "cosine"):
            distances = _distance_block(prediction[indices], truth[indices], metric)
            local = np.arange(len(indices), dtype=np.int64)
            ranks = _stable_ranks(distances, local)
            # This matches cell-eval: 1 - zero_based_rank / number_of_candidates.
            output[f"discrimination_{metric}"][indices] = 1.0 - ranks / len(indices)
            output[f"top1_{metric}"][indices] = ranks == 0
            output[f"top5_{metric}"][indices] = ranks < min(5, len(indices))
            output[f"rank_{metric}"][indices] = ranks
    return output


def phenotype_perturbation_metrics(
    profiles: GeneProfiles,
) -> tuple[dict[str, float], pd.DataFrame]:
    """Evaluate screen-matched, control-relative phenotype response profiles."""

    truth = _as_finite_matrix(profiles.truth, "profiles.truth")
    prediction = _as_finite_matrix(profiles.prediction, "profiles.prediction")
    _require(truth.shape == prediction.shape, "Truth/prediction profile shapes differ")
    screen_code = np.asarray(profiles.screen_code, dtype=np.int64)
    gene_code = np.asarray(profiles.gene_code, dtype=np.int64)
    _require(screen_code.shape == (len(truth),), "screen_code shape mismatch")
    _require(gene_code.shape == (len(truth),), "gene_code shape mismatch")
    keys = np.stack([screen_code, gene_code], axis=1)
    _require(len(np.unique(keys, axis=0)) == len(keys), "Duplicate screen-gene profiles")

    per_group: dict[str, np.ndarray] = {
        "screen_code": screen_code,
        "gene_code": gene_code,
        "n_cells": np.asarray(profiles.n_cells, dtype=np.int64),
        "n_control_cells": np.asarray(profiles.n_control_cells, dtype=np.int64),
        "mse_delta": np.mean((truth - prediction) ** 2, axis=1),
        "mae_delta": np.mean(np.abs(truth - prediction), axis=1),
        "pearson_delta": _row_pearson(truth, prediction),
        "profile_cosine": _row_cosine(truth, prediction),
        "direction_match": np.mean(np.sign(truth) == np.sign(prediction), axis=1),
        "truth_response_norm": np.linalg.norm(truth, axis=1),
        "prediction_response_norm": np.linalg.norm(prediction, axis=1),
    }
    per_group.update(_within_screen_retrieval(truth, prediction, screen_code))
    table = pd.DataFrame(per_group)

    summary = {
        "cell_eval_mse_delta": float(np.mean(per_group["mse_delta"])),
        "cell_eval_mae_delta": float(np.mean(per_group["mae_delta"])),
        "cell_eval_mae_delta_skill_vs_zero": float(
            1.0
            - np.mean(per_group["mae_delta"])
            / max(float(np.mean(np.abs(truth))), np.finfo(np.float64).tiny)
        ),
        "cell_eval_pearson_delta": _nanmean(per_group["pearson_delta"]),
        "cell_eval_profile_cosine": _nanmean(per_group["profile_cosine"]),
        "cell_eval_direction_match": _nanmean(per_group["direction_match"]),
        "cell_eval_n_profile_pearson_defined": int(
            np.isfinite(per_group["pearson_delta"]).sum()
        ),
        "cell_eval_n_retrieval_queries": int(len(table)),
    }
    for metric in ("l1", "l2", "cosine"):
        summary[f"cell_eval_discrimination_{metric}"] = _nanmean(
            per_group[f"discrimination_{metric}"]
        )
        summary[f"cell_eval_top1_{metric}"] = _nanmean(per_group[f"top1_{metric}"])
        summary[f"cell_eval_top5_{metric}"] = _nanmean(per_group[f"top5_{metric}"])
    return summary, table


def bootstrap_data_ceiling_profiles(
    truth: np.ndarray,
    screen_code: np.ndarray,
    gene_code: np.ndarray,
    is_control: np.ndarray,
    seed: int,
) -> GeneProfiles:
    """Mirror cell-eval's per-perturbation 2n bootstrap split-half ceiling.

    OPS controls are screen-specific, so controls are resampled separately in
    every screen before target profiles are made control-relative.
    """

    values = _as_finite_matrix(truth, "truth")
    screen = np.asarray(screen_code, dtype=np.int64)
    gene = np.asarray(gene_code, dtype=np.int64)
    control = np.asarray(is_control, dtype=bool)
    _require(screen.shape == gene.shape == control.shape == (len(values),), "Metadata shape mismatch")
    _require(np.any(control) and np.any(~control), "Ceiling requires controls and perturbations")
    rng = np.random.default_rng(int(seed))

    control_halves: dict[int, tuple[np.ndarray, np.ndarray, int]] = {}
    for screen_id in np.unique(screen[control]):
        indices = np.flatnonzero(control & (screen == screen_id))
        draws = rng.choice(indices, size=2 * len(indices), replace=True)
        control_halves[int(screen_id)] = (
            values[draws[: len(indices)]].mean(axis=0),
            values[draws[len(indices) :]].mean(axis=0),
            len(indices),
        )

    target_indices = np.flatnonzero(~control)
    packed_keys = (screen[target_indices] << 32) ^ (gene[target_indices] & 0xFFFFFFFF)
    order = np.argsort(packed_keys, kind="stable")
    sorted_keys = packed_keys[order]
    unique_keys, starts, counts = np.unique(
        sorted_keys, return_index=True, return_counts=True
    )
    first: list[np.ndarray] = []
    second: list[np.ndarray] = []
    screens: list[int] = []
    genes: list[int] = []
    control_counts: list[int] = []
    for key, start, count in zip(unique_keys, starts, counts, strict=True):
        indices = target_indices[order[start : start + count]]
        screen_id = int(np.int64(key) >> 32)
        gene_id = int(np.int64(key) & np.int64(0xFFFFFFFF))
        _require(screen_id in control_halves, f"Missing controls for screen {screen_id}")
        draws = rng.choice(indices, size=2 * len(indices), replace=True)
        control_first, control_second, n_controls = control_halves[screen_id]
        first.append(values[draws[: len(indices)]].mean(axis=0) - control_first)
        second.append(values[draws[len(indices) :]].mean(axis=0) - control_second)
        screens.append(screen_id)
        genes.append(gene_id)
        control_counts.append(n_controls)
    _require(first, "No target groups for ceiling")
    return GeneProfiles(
        truth=np.asarray(first, dtype=np.float32),
        prediction=np.asarray(second, dtype=np.float32),
        screen_code=np.asarray(screens, dtype=np.int64),
        gene_code=np.asarray(genes, dtype=np.int64),
        n_cells=np.asarray(counts, dtype=np.int64),
        n_control_cells=np.asarray(control_counts, dtype=np.int64),
    )


def adapter_manifest() -> dict[str, Any]:
    implementation_path = Path(__file__).resolve()
    implementation_sha256 = hashlib.sha256(implementation_path.read_bytes()).hexdigest()
    return {
        "schema_version": ADAPTER_SCHEMA_VERSION,
        "evaluator_id": EVALUATOR_ID,
        "upstream": {
            "repository": UPSTREAM_REPOSITORY,
            "release": UPSTREAM_RELEASE,
            "relationship": "OPS-native equation-level adaptation; upstream package is not imported",
        },
        "implementation": {
            "path": str(implementation_path),
            "sha256": implementation_sha256,
        },
        "included": [
            "mse_delta",
            "mae_delta",
            "pearson_delta",
            "discrimination_score_l1",
            "discrimination_score_l2",
            "discrimination_score_cosine",
            "stratified_bootstrap_data_ceiling",
            "phenotype_direction_match",
        ],
        "adaptations": {
            "feature_space": "reporter-specific train-standardized phenotype endpoints",
            "perturbation_unit": "screen_by_gene",
            "control_policy": "screen-matched controls",
            "delta_definition": "signed_perturbation_mean_minus_screen_matched_control_mean",
            "retrieval_candidate_set": "within reporter and screen",
            "discrimination_formula": "1 - zero_based_rank / n_candidates (cell-eval v0.8.1 code convention)",
            "aggregation": "perturbation macro then reporter-equal downstream",
            "selection_role": "reporting_only_post_checkpoint_freeze",
        },
        "excluded": {
            "differential_expression_suite": "phenotype endpoints are not genes or counts",
            "pearson_edistance": "quadratic cell-pair cost is unsafe at OPS scale; requires a separately frozen stratified-subsample protocol",
            "clustering_agreement": "upstream predicted-resolution sweep is exploratory and is not admitted to the formal score",
        },
    }


def validate_manifest(payload: Mapping[str, Any]) -> None:
    _require(payload.get("schema_version") == ADAPTER_SCHEMA_VERSION, "Adapter schema mismatch")
    _require(payload.get("evaluator_id") == EVALUATOR_ID, "Evaluator ID mismatch")
    adaptations = payload.get("adaptations")
    _require(isinstance(adaptations, Mapping), "Missing adapter adaptations")
    _require(
        adaptations.get("selection_role") == "reporting_only_post_checkpoint_freeze",
        "Adapter metrics may not become selection inputs",
    )


__all__ = [
    "ADAPTER_SCHEMA_VERSION",
    "EVALUATOR_ID",
    "adapter_manifest",
    "bootstrap_data_ceiling_profiles",
    "phenotype_perturbation_metrics",
    "validate_manifest",
]
