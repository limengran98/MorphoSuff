#!/usr/bin/env python3
"""One ridge estimator for both PERISCOPE levels, solved from sufficient statistics.

The guide level and the single-cell level differ in what a row is and in nothing
else, so they must not differ in how the model is fitted or how its penalty is
chosen. Both were originally written with a fixed ``alpha=1.0``, and on this design
that is not a weak prior but an unregularised solve: the 2,508 CellProfiler features
carry 155 correlation eigenvalues below 1e-8, the smallest at machine zero, while
``alpha=1`` against a Gram whose diagonal is the row count is a ratio near 1e-6.
scikit-learn said as much at the guide level, warning ``Ill-conditioned matrix
(rcond=2.2e-09)`` on every fold; at the single-cell level one fold in five returned
predicted response magnitudes 52 times the observed spread.

Two things follow, and both live here so that neither level can drift from the other.

**The fit is a solve on moments.** ``X'X`` and ``X'Y`` are additive over rows, so one
pass per block yields every fold's training statistics as the totals minus the block
held out. Centring, scaling and the solve are then algebra on a 2,508 by 2,508
matrix. ``ridge_via_moments`` is checked against ``sklearn.linear_model.Ridge`` in the
test suite rather than against its own output.

**The penalty is selected inside the training set.** For each outer fold, one of the
remaining folds serves as an inner validation set; the outer fold enters neither inner
half. The grid is a ratio to the row count, so it does not depend on how many rows the
level happens to have, and a choice at either end of the grid is raised as an error
rather than returned as a result.
"""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

def gene_folds(genes: np.ndarray, *, n_folds: int, seed: int) -> dict[str, int]:
    """Assign each perturbation to one held-out fold, controls excluded from holdout."""
    rng = np.random.default_rng(seed)
    ordered = np.sort(np.unique(genes))
    shuffled = rng.permutation(ordered)
    return {gene: index % n_folds for index, gene in enumerate(shuffled)}


def screen_responses(
    frame: pd.DataFrame, endpoints: list[str], *, column: str
) -> pd.DataFrame:
    """Gene by endpoint control-relative responses, one block per screen, kept wide.

    Subtracting the same-screen non-targeting mean is the definition of the response;
    the plate is the screen here because a plate carries its own non-targeting guides.
    """
    blocks = []
    for screen, group in frame.groupby("screen_id", sort=True):
        controls = group.loc[group["is_control"], endpoints]
        if controls.empty:
            raise ValueError(f"screen {screen} has no non-targeting guides to subtract")
        baseline = controls.mean()
        perturbed = group.loc[~group["is_control"]]
        means = perturbed.groupby("perturbation_id")[endpoints].mean()
        block = (means - baseline).reset_index()
        block.insert(0, "screen_id", screen)
        blocks.append(block)
    wide = pd.concat(blocks, ignore_index=True)
    return wide.melt(
        id_vars=["screen_id", "perturbation_id"], var_name="endpoint_id", value_name=column
    )


def block_moments(
    x: np.ndarray, y: np.ndarray, block_of_row: np.ndarray, n_blocks: int, *, chunk: int = 50_000
) -> list[dict[str, np.ndarray | int]]:
    """Uncentred sufficient statistics per block, accumulated in float64.

    ``X'X`` and ``X'Y`` are additive over rows, so a fold's training statistics are the
    totals minus that fold's own block. One pass therefore serves every fold.
    """
    n_features, n_endpoints = x.shape[1], y.shape[1]
    stats = []
    for block in range(n_blocks):
        rows = np.flatnonzero(block_of_row == block)
        gram = np.zeros((n_features, n_features), dtype=np.float64)
        cross = np.zeros((n_features, n_endpoints), dtype=np.float64)
        sum_x = np.zeros(n_features, dtype=np.float64)
        sum_y = np.zeros(n_endpoints, dtype=np.float64)
        sumsq_y = np.zeros(n_endpoints, dtype=np.float64)
        for start in range(0, len(rows), chunk):
            part = rows[start : start + chunk]
            xb = x[part].astype(np.float64)
            yb = y[part].astype(np.float64)
            sum_x += xb.sum(axis=0)
            sum_y += yb.sum(axis=0)
            sumsq_y += (yb * yb).sum(axis=0)
            gram += xb.T @ xb
            cross += xb.T @ yb
            del xb, yb
        stats.append({
            "n": int(len(rows)), "sum_x": sum_x, "sum_y": sum_y,
            "sumsq_y": sumsq_y, "gram": gram, "cross": cross,
        })
    return stats


ALPHA_RATIOS = (1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0)


def rank_deficiency(gram: np.ndarray, sum_x: np.ndarray, n: int, *, threshold: float = 1e-8) -> tuple[int, float]:
    """How many correlation eigenvalues sit at numerical zero, and the smallest one.

    Reported per fold because it is the quantity that decides whether a penalty is
    doing anything: a ratio below the smallest informative eigenvalue is not a weak
    prior, it is an unregularised solve of a singular system.
    """
    mean = sum_x / n
    centred = gram - n * np.outer(mean, mean)
    scale = np.sqrt(np.clip(np.diag(centred) / n, 0.0, None))
    scale[scale <= 0.0] = 1.0
    eigenvalues = np.linalg.eigvalsh(centred / np.outer(scale, scale) / n)
    return int((eigenvalues < threshold).sum()), float(eigenvalues[0])


def combine_moments(total: dict, *subtract: dict) -> dict:
    out = {key: value for key, value in total.items()}
    for block in subtract:
        for key in out:
            out[key] = out[key] - block[key]
    return out


def validation_error(
    x: np.ndarray,
    y: np.ndarray,
    rows: np.ndarray,
    fitted: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    endpoint_scale: np.ndarray,
    *,
    chunk: int = 50_000,
) -> float:
    """Mean squared error on held-in validation rows, per endpoint-standardised unit.

    The endpoints are in native CellProfiler units and one of them holds most of the
    raw spread, so an unstandardised error would select the penalty that best fits
    that single endpoint. The scale comes from the inner training half only.
    """
    mean, scale, coefficients, intercept = fitted
    total = 0.0
    for start in range(0, len(rows), chunk):
        part = rows[start : start + chunk]
        predicted = ((x[part].astype(np.float64) - mean) / scale) @ coefficients + intercept
        residual = (y[part].astype(np.float64) - predicted) / endpoint_scale
        total += float((residual * residual).sum())
    return total / (len(rows) * y.shape[1])


def select_alpha(
    x: np.ndarray,
    y: np.ndarray,
    block_of_row: np.ndarray,
    stats: list[dict],
    total: dict,
    *,
    outer_fold: int,
    inner_fold: int,
    ratios: tuple[float, ...] = ALPHA_RATIOS,
) -> tuple[float, list[tuple[float, float]]]:
    """Choose the penalty on an inner validation fold that the outer fold never touches."""
    inner_train = combine_moments(total, stats[outer_fold], stats[inner_fold])
    validation_rows = np.flatnonzero(block_of_row == inner_fold)
    n = inner_train["n"]
    y_mean = inner_train["sum_y"] / n
    y_scale = np.sqrt(np.clip(inner_train["sumsq_y"] / n - y_mean**2, 0.0, None))
    y_scale[y_scale <= 0.0] = 1.0

    scored = []
    for ratio in ratios:
        fitted = ridge_via_moments(
            n, inner_train["sum_x"], inner_train["sum_y"],
            inner_train["gram"], inner_train["cross"], alpha=ratio * n,
        )
        scored.append((ratio, validation_error(x, y, validation_rows, fitted, y_scale)))
    # A non-finite score is a broken evaluation, not a bad penalty, and ``min`` does not
    # know the difference: over NaN scores it returns the first element, which is the
    # smallest ratio, which the boundary check below then reports as "the grid does not
    # bracket the optimum". That is a true statement about a meaningless list, and it
    # sent a real diagnosis down the wrong path once. The usual cause is a missing value
    # in the target block, which no amount of regularisation can fix.
    broken = [ratio for ratio, error in scored if not np.isfinite(error)]
    if broken:
        raise SystemExit(
            f"the validation error is not finite at {len(broken)} of {len(scored)} "
            f"penalties, starting at alpha/n = {broken[0]:g}. A penalty cannot be "
            "selected from scores that are not numbers; the usual cause is a missing "
            "value somewhere in the target block, which enters sumsq_y and the cross "
            "moments and makes every endpoint's error non-finite at once."
        )
    best = min(scored, key=lambda item: item[1])[0]
    if best in (ratios[0], ratios[-1]):
        raise SystemExit(
            f"the selected penalty ratio {best:g} is at the edge of the grid "
            f"{ratios[0]:g} to {ratios[-1]:g}, so the grid does not bracket the "
            f"optimum and the choice is an artefact of where it was cut off"
        )
    return best, scored


def ridge_via_moments(
    n: int,
    sum_x: np.ndarray,
    sum_y: np.ndarray,
    gram: np.ndarray,
    cross: np.ndarray,
    *,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Ridge on standardised features, solved from uncentred moments.

    Equivalent to standardising the training block and calling
    ``Ridge(alpha=alpha, fit_intercept=True)`` on the result, which the test suite
    checks against scikit-learn directly. Returns (mean, scale, coefficients,
    intercept); a prediction is ``((x - mean) / scale) @ coefficients + intercept``.
    """
    if n < 2:
        raise ValueError("a ridge needs at least two training rows")
    mean = sum_x / n
    centred_gram = gram - n * np.outer(mean, mean)
    variance = np.diag(centred_gram) / n
    scale = np.sqrt(np.clip(variance, 0.0, None))
    # A feature with no spread in the training block carries no information; dividing
    # by one leaves it at zero after centring rather than producing an infinity.
    scale[scale <= 0.0] = 1.0
    y_mean = sum_y / n
    centred_cross = cross - n * np.outer(mean, y_mean)
    scaled_gram = centred_gram / np.outer(scale, scale)
    scaled_cross = centred_cross / scale[:, None]
    scaled_gram.flat[:: scaled_gram.shape[0] + 1] += alpha
    coefficients = np.linalg.solve(scaled_gram, scaled_cross)
    return mean, scale, coefficients, y_mean


def predict_rows(
    x: np.ndarray,
    rows: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
    coefficients: np.ndarray,
    intercept: np.ndarray,
    *,
    chunk: int = 50_000,
) -> np.ndarray:
    out = np.empty((len(rows), coefficients.shape[1]), dtype=np.float32)
    for start in range(0, len(rows), chunk):
        part = rows[start : start + chunk]
        xb = (x[part].astype(np.float64) - mean) / scale
        out[start : start + len(part)] = (xb @ coefficients + intercept).astype(np.float32)
        del xb
    return out


def recoverability_by_fold(
    responses: pd.DataFrame, *, min_spread_ratio: float = 1e-3
) -> pd.DataFrame:
    """Correlation of observed and predicted responses, three ways, per fold.

    ``recoverability_r`` is meant to describe a reporter's endpoint block. Pooling all
    (gene, endpoint) pairs in native units does not describe the block when the block
    does not share a scale: it describes whichever endpoint has the largest units.
    That is not a hypothetical here. On the single-cell PERISCOPE fit,
    ``Nuclei_ObjectSkeleton_TotalObjectSkeletonLength_mito_skel`` holds between 60% and
    98% of the native squared spread depending on the fold, and its own correlation is
    0.13 to 0.17, while the median endpoint correlates at 0.83. The pooled native
    number moved from 0.121 to 0.439 across five folds while every other evidence
    column stayed inside a hundredth. It was reporting one coordinate.

    ``response_magnitudes`` already standardises before taking a norm, for exactly this
    reason and with the reason written down. This function applies the same rule to the
    same block, so the two stop disagreeing about what a reporter's response is. The
    tier rule's ``recoverability_min`` was calibrated on OPS reporter blocks, which are
    standardised, so the standardised pooling is the comparable quantity and is the one
    fed to the rule; the other two are emitted beside it rather than discarded.

    The divisor is the observed spread of each endpoint within the fold. The same
    divisor is applied to the observed and the predicted vector, so it cannot inflate
    their agreement inside an endpoint; what it changes is how endpoints are weighted
    against each other, which is the point.
    """
    needed = {"fold", "endpoint_id", "response_true", "response_pred"}
    missing = needed.difference(responses)
    if missing:
        raise ValueError(f"responses lacks columns: {', '.join(sorted(missing))}")

    spread = (
        responses.groupby(["fold", "endpoint_id"], observed=True)["response_true"]
        .std()
        .rename("sd_true")
        .reset_index()
    )
    floor = (
        spread.groupby("fold")["sd_true"].median().rename("floor") * min_spread_ratio
    ).reset_index()
    spread = spread.merge(floor, on="fold", how="left")
    spread["keeps"] = spread["sd_true"].gt(0.0) & spread["sd_true"].gt(spread["floor"])

    frame = responses.merge(
        spread[["fold", "endpoint_id", "sd_true", "keeps"]], on=["fold", "endpoint_id"], how="left"
    )
    kept = frame.loc[frame["keeps"]].copy()
    kept["standardized_true"] = kept["response_true"] / kept["sd_true"]
    kept["standardized_pred"] = kept["response_pred"] / kept["sd_true"]

    # Select only the measured response columns before ``apply``.  This avoids
    # relying on pandas' ``include_groups`` keyword (added after the pinned
    # workstation version) and ensures that grouping keys cannot enter the
    # correlation calculation on any supported pandas release.
    per_endpoint = (
        kept.groupby(["fold", "endpoint_id"], observed=True)[
            ["response_true", "response_pred"]
        ]
        .apply(lambda g: g["response_true"].corr(g["response_pred"]))
        .rename("r")
        .reset_index()
    )

    rows = []
    for fold, block in kept.groupby("fold", observed=True):
        endpoints = per_endpoint.loc[per_endpoint["fold"] == fold, "r"].dropna()
        squared = spread.loc[spread["fold"].eq(fold) & spread["keeps"], "sd_true"] ** 2
        rows.append({
            "fold": fold,
            "recoverability_r": float(
                block["standardized_true"].corr(block["standardized_pred"])
            ),
            "recoverability_r_pooled_native": float(
                block["response_true"].corr(block["response_pred"])
            ),
            "recoverability_r_median_endpoint": float(endpoints.median()),
            "largest_native_endpoint_share": float(squared.max() / squared.sum()),
            "n_endpoints": int(len(endpoints)),
        })
    return pd.DataFrame(rows).sort_values("fold").reset_index(drop=True)


def permute_rows_within(
    x: np.ndarray, groups: np.ndarray, *, seed: int
) -> dict[str, int | float]:
    """Break the link between a row's input and its identity, inside each group.

    This is the negative control the panel needs. Every evidence column in this study
    is computed from a fitted model, and a panel of rows that all fail says nothing
    about whether the criteria can distinguish an informative input from an
    uninformative one unless an uninformative input is also run through them. Here the
    input block is shuffled while the target block, the perturbation labels and the
    control flags stay where they are, so the only thing destroyed is the association
    the model is asked to learn.

    The shuffle is inside each group, normally the screen. A shuffle across screens
    would additionally destroy the plate structure, and the response is defined
    relative to same-plate controls, so the collapse could then be attributed to the
    baseline rather than to the association. Inside a screen there is no such escape.

    ``x`` is modified in place, which matters at this scale: the guide block is 7 GB.
    Returns what was actually shuffled, because "the rows were permuted" is not
    checkable and "99.99% of rows moved" is.
    """
    if len(groups) != len(x):
        raise ValueError(f"groups has {len(groups)} entries for {len(x)} rows")
    rng = np.random.default_rng(seed)
    moved, total = 0, 0
    unique = np.unique(groups)
    for group in unique:
        rows = np.flatnonzero(groups == group)
        if len(rows) < 2:
            continue
        order = rng.permutation(len(rows))
        # The right-hand side is materialised before assignment, so this is safe in
        # place. One screen of the guide block is about 780 MB.
        x[rows] = x[rows[order]]
        moved += int((order != np.arange(len(rows))).sum())
        total += len(rows)
    return {
        "n_groups": int(len(unique)),
        "n_rows": int(total),
        "n_rows_moved": int(moved),
        "fraction_moved": float(moved / total) if total else 0.0,
    }


def impute_from_controls(x: np.ndarray, is_control: np.ndarray) -> tuple[np.ndarray, int, int]:
    """Fill missing inputs with the control-cell median, in place.

    Controls sit in the training set of every fold, so this constant is train-only for
    every fold. Returns the fill vector, the number of columns touched and the number
    of filled cells so that both are reported rather than assumed to be negligible.
    """
    missing = np.isnan(x)
    columns = np.flatnonzero(missing.any(axis=0))
    if columns.size == 0:
        return np.zeros(x.shape[1], dtype=np.float32), 0, 0
    fill = np.zeros(x.shape[1], dtype=np.float32)
    control_rows = np.flatnonzero(is_control)
    for column in columns:
        values = x[control_rows, column]
        values = values[np.isfinite(values)]
        if values.size == 0:
            raise SystemExit(
                f"feature column {column} is missing in every control cell, so no "
                f"train-only fill value exists for it"
            )
        fill[column] = np.median(values)
        rows = np.flatnonzero(missing[:, column])
        x[rows, column] = fill[column]
    return fill, int(columns.size), int(missing.sum())


def fit_folds(
    observations: pd.DataFrame,
    x: np.ndarray,
    y: np.ndarray,
    endpoints: list[str],
    *,
    n_folds: int,
    seed: int,
    alpha: float | None,
    reporter_id: str,
    unit: str = "cells",
    alpha_ratio: float | None = None,
    ratios: tuple[float, ...] = ALPHA_RATIOS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out genes, predict every held-out row, then aggregate to gene responses.

    The same driver serves both levels: at the guide level a row is a (gene, sgRNA,
    plate) profile, at the single-cell level a row is a cell, and nothing else
    differs. Keeping one driver is what stops the two from drifting into different
    estimators and being compared as if they were the same one.

    ``alpha`` of ``None`` selects the penalty per fold by nested cross-validation.
    A number fixes it, which is how the first run of both scripts behaved and is
    kept so that the defect it produced can be reproduced rather than only
    described.

    ``alpha_ratio`` fixes the penalty as a ratio to the training row count, which is
    the unit :func:`select_alpha` searches in. It exists for the permuted-input
    negative control. Selection cannot be used there: with the association destroyed
    the best penalty is the largest one on the grid, which :func:`select_alpha`
    correctly refuses to return, and a control that re-selects its own penalty would
    differ from the run it is a control for in two things rather than one. Fixing the
    ratio at what the observed run selected leaves the estimator identical and the
    permutation as the only difference.
    """
    if alpha is not None and alpha_ratio is not None:
        raise ValueError("pass alpha or alpha_ratio, not both")
    is_control = observations["is_control"].to_numpy()
    genes = observations["perturbation_id"].to_numpy()
    assignment = gene_folds(genes[~is_control], n_folds=n_folds, seed=seed)
    fold_of_row = np.array([assignment.get(gene, -1) for gene in genes])
    # Controls are their own block and are never held out, so they train every fold.
    block_of_row = np.where(is_control, n_folds, fold_of_row)
    if (block_of_row < 0).any():
        raise SystemExit("a non-control cell received no fold assignment")

    started = time.time()
    stats = block_moments(x, y, block_of_row, n_folds + 1)
    print(f"  sufficient statistics for {n_folds + 1} blocks in {time.time() - started:.0f}s", flush=True)

    total = {key: sum(s[key] for s in stats) for key in stats[0]}

    identifiers = observations[["screen_id", "perturbation_id", "is_control"]]
    collected = []
    penalties = []
    for fold in range(n_folds):
        train = combine_moments(total, stats[fold])
        deficient, smallest = rank_deficiency(train["gram"], train["sum_x"], train["n"])
        if alpha_ratio is not None:
            chosen, ratio, scored = alpha_ratio * train["n"], alpha_ratio, []
            print(
                f"  fold {fold}: {deficient} correlation eigenvalues below 1e-8 "
                f"(smallest {smallest:.2e}); alpha/n fixed at {ratio:g}, "
                f"alpha = {chosen:,.0f}",
                flush=True,
            )
        elif alpha is None:
            inner_fold = (fold + 1) % n_folds
            ratio, scored = select_alpha(
                x, y, block_of_row, stats, total,
                outer_fold=fold, inner_fold=inner_fold, ratios=ratios,
            )
            chosen = ratio * train["n"]
            print(
                f"  fold {fold}: {deficient} correlation eigenvalues below 1e-8 "
                f"(smallest {smallest:.2e}); inner fold {inner_fold} selects "
                f"alpha/n = {ratio:g}, alpha = {chosen:,.0f}",
                flush=True,
            )
        else:
            chosen, ratio, scored = alpha, alpha / train["n"], []
            print(
                f"  fold {fold}: {deficient} correlation eigenvalues below 1e-8 "
                f"(smallest {smallest:.2e}); alpha fixed at {chosen:g}, "
                f"alpha/n = {ratio:.2e}",
                flush=True,
            )
        penalties.append({
            "fold": fold, "n_train": train["n"], "alpha": chosen, "alpha_over_n": ratio,
            "eigenvalues_below_1e-8": deficient, "smallest_eigenvalue": smallest,
            "grid": json.dumps([[r, e] for r, e in scored]),
        })
        mean, scale, coefficients, intercept = ridge_via_moments(
            train["n"], train["sum_x"], train["sum_y"], train["gram"], train["cross"],
            alpha=chosen,
        )
        test_rows = np.flatnonzero((block_of_row == fold) | is_control)
        predicted = predict_rows(x, test_rows, mean, scale, coefficients, intercept)

        block = identifiers.iloc[test_rows].reset_index(drop=True)
        observed_block = pd.concat(
            [block, pd.DataFrame(y[test_rows], columns=endpoints)], axis=1
        )
        predicted_block = pd.concat(
            [block, pd.DataFrame(predicted, columns=endpoints)], axis=1
        )
        observed = screen_responses(observed_block, endpoints, column="response_true")
        modelled = screen_responses(predicted_block, endpoints, column="response_pred")
        merged = observed.merge(modelled, on=["screen_id", "perturbation_id", "endpoint_id"])
        merged = merged.loc[merged["perturbation_id"].map(assignment).eq(fold)]
        merged["fold"] = fold
        collected.append(merged)
        print(
            f"  fold {fold}: {len(test_rows):,} {unit} predicted, "
            f"{len(merged):,} gene-endpoint responses",
            flush=True,
        )
        del predicted, observed_block, predicted_block, observed, modelled

    responses = pd.concat(collected, ignore_index=True)
    responses["reporter_id"] = reporter_id
    responses["split_name"] = "gene"
    responses["model_id"] = "ridge"
    return responses, pd.DataFrame(penalties)
