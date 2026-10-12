"""Directed environment-shift transfer loss and its cluster-permutation test.

The published Methods define transfer loss as a difference between two measured
quantities supplied by the caller:

    "Transfer loss was the held-out-gene minus strict whole-screen knockout-response
     Pearson.  Associations were quantified across 81 directions using Spearman
     correlation.  P values were obtained by permuting reporter clusters so that
     directions from the same reporter remained together."

Both terms therefore come from evaluations the caller has already run: the
within-environment score from the gene-holdout split, and the shifted-environment
score from the strict whole-screen split that removed that screen.  This module
does not attempt to infer a baseline from the transfer results themselves; doing
so produces a quantity that is centred within each source by construction and so
can never show a systematic loss.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from ..adapters import AdapterCapabilities


_DEFAULT_GROUPS = ["reporter_id", "model_id"]


def environment_shift(
    metrics: pd.DataFrame,
    *,
    within_environment_column: str = "gene_pearson",
    shifted_environment_column: str = "strict_screen_pearson",
    destination_column: str = "destination_screen_id",
    group_columns: list[str] | None = None,
    source_column: str | None = None,
    output_column: str = "transfer_loss",
    capabilities: AdapterCapabilities | None = None,
) -> pd.DataFrame:
    """Compute directed transfer loss from two explicitly supplied metrics.

    One row per directed evaluation, keyed by ``group_columns`` plus the held-out
    destination screen.  Positive output means the reporter is worse under
    environment shift than within its own environment.
    """
    if capabilities is not None:
        capabilities.require("repeated_screens", analysis="environment shift")
    groups = list(group_columns or _DEFAULT_GROUPS)
    needed = groups + [destination_column, within_environment_column, shifted_environment_column]
    if source_column is not None:
        needed.append(source_column)
    missing = set(needed).difference(metrics)
    if missing:
        raise ValueError(f"transfer metrics lacks columns: {', '.join(sorted(missing))}")

    key = groups + [destination_column]
    duplicated = metrics.duplicated(key, keep=False)
    if duplicated.any():
        sample = metrics.loc[duplicated, key].drop_duplicates().head(3).to_dict("records")
        raise ValueError(
            "each directed evaluation must appear once; duplicate "
            f"{key} rows: {sample}. The manuscript's unit is one held-out destination "
            "screen per reporter and model."
        )
    for column in (within_environment_column, shifted_environment_column):
        values = pd.to_numeric(metrics[column], errors="coerce")
        if values.isna().any():
            raise ValueError(f"{column} must be numeric and non-null for every direction")
        if not values.between(-1.0, 1.0).all():
            raise ValueError(
                f"{column} lies outside [-1, 1]; transfer loss is a difference of two "
                "correlation coefficients, so a variance ratio, an R^2 or an "
                "already-differenced column cannot be passed here"
            )
    if source_column is not None and metrics[source_column].eq(metrics[destination_column]).any():
        raise ValueError("a directed transfer cannot have the same source and destination screen")

    columns = groups + [destination_column]
    if source_column is not None:
        columns.append(source_column)
    out = metrics.loc[:, columns + [within_environment_column, shifted_environment_column]].copy()
    # Sign convention: positive means worse at the destination than within the
    # reporter's own environment.
    out[output_column] = (
        pd.to_numeric(out[within_environment_column]) - pd.to_numeric(out[shifted_environment_column])
    )
    return out


def cluster_permutation_spearman(
    table: pd.DataFrame,
    *,
    x_column: str,
    y_column: str,
    cluster_column: str = "reporter_id",
    n_permutations: int = 3000,
    seed: int = 0,
    alternative: Literal["two-sided", "greater", "less"] = "two-sided",
) -> dict[str, float]:
    """Spearman association with a P value from permuting whole clusters.

    Directions from one reporter are not independent, so a row-wise shuffle
    understates the P value.  Permuting the order of whole reporter blocks keeps
    each reporter's directions together, which is the null the Methods describe.
    Block reassignment is positional, so unequal cluster sizes are handled without
    dropping or padding a reporter.
    """
    missing = {x_column, y_column, cluster_column}.difference(table)
    if missing:
        raise ValueError(f"table lacks columns: {', '.join(sorted(missing))}")
    if n_permutations < 1:
        raise ValueError("n_permutations must be positive")
    frame = table.loc[:, [x_column, y_column, cluster_column]].dropna()
    if len(frame) < 3:
        raise ValueError("cluster permutation needs at least three complete rows")
    clusters = pd.unique(frame[cluster_column])
    if len(clusters) < 2:
        raise ValueError(
            "cluster permutation needs at least two clusters; with one cluster the "
            "null and the observed arrangement are identical"
        )

    x = frame[x_column].to_numpy(dtype=float)
    y = frame[y_column].to_numpy(dtype=float)
    observed = float(pd.Series(x).corr(pd.Series(y), method="spearman"))
    blocks = [np.flatnonzero(frame[cluster_column].to_numpy() == name) for name in clusters]

    rng = np.random.default_rng(seed)
    exceed = 0
    for _ in range(int(n_permutations)):
        order = rng.permutation(len(blocks))
        permuted = np.concatenate([blocks[index] for index in order])
        candidate = float(pd.Series(x).corr(pd.Series(y[permuted]), method="spearman"))
        if not np.isfinite(candidate):
            continue
        if alternative == "two-sided":
            exceed += abs(candidate) >= abs(observed)
        elif alternative == "greater":
            exceed += candidate >= observed
        else:
            exceed += candidate <= observed
    # The +1 in both terms keeps the P value from ever being exactly zero, which
    # a finite permutation set cannot justify.
    p_value = (1 + int(exceed)) / (1 + int(n_permutations))
    return {
        "spearman_rho": observed,
        "n": int(len(frame)),
        "n_clusters": int(len(clusters)),
        "n_permutations": int(n_permutations),
        "p_value": float(p_value),
        "alternative": alternative,
        "seed": int(seed),
    }
