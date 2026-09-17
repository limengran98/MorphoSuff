"""Frozen group split manifests and leakage checks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .schemas import coerce_boolean_column


class LeakageError(ValueError):
    """Raised when a held-out entity is present in training."""


#: The canonical split names. ``strict_whole_screen`` is the value physically
#: written into the ``split_name`` column of every materialised assignment and
#: prediction table already published, so it wins over the config-file and CLI
#: spellings, which are cheap to change.
SPLIT_NAMES: tuple[str, ...] = ("field", "gene", "strict_whole_screen")

#: Historical spellings, kept so an existing manifest, config or command line
#: still resolves instead of silently naming a split that no consumer matches.
SPLIT_NAME_ALIASES: dict[str, str] = {
    "whole-screen": "strict_whole_screen",
    "whole_screen": "strict_whole_screen",
    "strict-screen": "strict_whole_screen",
    "strict_screen": "strict_whole_screen",
    "strict whole screen": "strict_whole_screen",
}


def canonical_split_name(value: str) -> str:
    """Resolve any accepted spelling of a split name to its canonical form."""
    text = str(value).strip()
    if text in SPLIT_NAMES:
        return text
    resolved = SPLIT_NAME_ALIASES.get(text.casefold())
    if resolved is None:
        raise ValueError(
            f"unknown split name {value!r}; expected one of {SPLIT_NAMES} "
            f"or an accepted alias {sorted(SPLIT_NAME_ALIASES)}"
        )
    return resolved


@dataclass(frozen=True)
class SplitManifest:
    assignments: pd.DataFrame
    split_name: str
    heldout_column: str
    #: Column naming rows that are exempt from the held-out-group check because
    #: they are an assay reference rather than an evaluation unit. See
    #: :func:`check_split_leakage`.
    exempt_column: str | None = None

    def validate(self, observations: pd.DataFrame) -> None:
        required = {"observation_id", "fold", "role"}
        if not required.issubset(self.assignments):
            raise LeakageError("split manifest requires observation_id, fold, and role")
        if self.assignments.duplicated(["observation_id", "fold"]).any():
            raise LeakageError("split manifest assigns an observation more than once per fold")
        if not set(self.assignments.role).issubset({"train", "test"}):
            raise LeakageError("split role must be train or test")
        if not set(self.assignments.observation_id).issubset(set(observations.observation_id)):
            raise LeakageError("split manifest includes unknown observation_id")
        check_split_leakage(
            observations, self.assignments, self.heldout_column, exempt_column=self.exempt_column
        )


def _balanced_group_folds(groups: pd.Series, n_folds: int, seed: int) -> dict[object, int]:
    unique = np.array(sorted(pd.unique(groups).tolist(), key=str), dtype=object)
    if n_folds < 2 or len(unique) < n_folds:
        raise ValueError("n_folds must be at least 2 and no greater than unique group count")
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    return {group: i % n_folds for i, group in enumerate(unique)}


def make_group_splits(observations: pd.DataFrame, group_column: str, split_name: str, n_folds: int = 5, seed: int = 0) -> SplitManifest:
    if group_column not in observations:
        raise KeyError(f"observations lacks split group column: {group_column}")
    if observations[group_column].isna().any():
        raise ValueError(f"cannot split null {group_column} values")
    mapping = _balanced_group_folds(observations[group_column], n_folds, seed)
    assigned = observations[["observation_id", group_column]].copy()
    assigned["heldout_fold"] = assigned[group_column].map(mapping)
    rows = []
    for fold in range(n_folds):
        fold_rows = assigned[["observation_id"]].copy()
        fold_rows["fold"] = fold
        fold_rows["role"] = np.where(assigned.heldout_fold.eq(fold), "test", "train")
        rows.append(fold_rows)
    manifest = SplitManifest(pd.concat(rows, ignore_index=True), split_name, group_column)
    manifest.validate(observations)
    return manifest


def make_field_splits(observations: pd.DataFrame, **kwargs: object) -> SplitManifest:
    return make_group_splits(observations, "field_id", "field", **kwargs)


def make_gene_splits(
    observations: pd.DataFrame,
    group_column: str = "perturbation_id",
    *,
    control_column: str | None = None,
    n_folds: int = 5,
    seed: int = 0,
) -> SplitManifest:
    """Hold out perturbations, optionally keeping declared controls in every fold.

    With ``control_column``, control rows are not routed through the group
    assignment. Sending them through it would place the whole control population
    in one fold's held-out role and leave the other folds with no same-screen
    baseline, which is what control-relative response fidelity needs at scoring
    time. Controls are instead spread round-robin over a stable ordering, and the
    leakage check exempts them; they must also be excluded from accuracy metrics,
    which :mod:`measurement_sufficiency.metrics` does by default.
    """
    if control_column is None:
        return make_group_splits(observations, group_column, "gene", n_folds=n_folds, seed=seed)
    if control_column not in observations:
        raise KeyError(f"observations lacks control column: {control_column}")
    is_control = coerce_boolean_column(
        observations[control_column], column=control_column, table="observations"
    )
    targets = observations.loc[~is_control]
    if targets.empty:
        raise ValueError("gene splits need at least one non-control observation")
    mapping = _balanced_group_folds(targets[group_column], n_folds, seed)
    assigned = observations[["observation_id"]].copy()
    heldout = observations[group_column].map(mapping)
    # Controls are spread round-robin over a stable ordering rather than grouped.
    # Grouping them by field would fail whenever a screen has fewer distinct
    # control fields than folds, and the property that matters here is only that
    # every fold's held-out role contains some control rows to form a baseline.
    control_ids = sorted(map(str, observations.loc[is_control, "observation_id"]))
    if len(control_ids) < n_folds:
        raise ValueError(
            f"{len(control_ids)} control observations cannot cover {n_folds} folds; "
            "every fold needs a same-screen baseline in its held-out role"
        )
    control_folds = {value: index % n_folds for index, value in enumerate(control_ids)}
    heldout = heldout.where(
        ~is_control, observations["observation_id"].astype(str).map(control_folds)
    )
    assigned["heldout_fold"] = heldout
    if assigned.heldout_fold.isna().any():
        raise ValueError("every observation must receive a fold")
    rows = []
    for fold in range(n_folds):
        fold_rows = assigned[["observation_id"]].copy()
        fold_rows["fold"] = fold
        fold_rows["role"] = np.where(assigned.heldout_fold.eq(fold), "test", "train")
        rows.append(fold_rows)
    manifest = SplitManifest(
        pd.concat(rows, ignore_index=True), "gene", group_column, exempt_column=control_column
    )
    manifest.validate(observations)
    return manifest


def make_whole_screen_splits(observations: pd.DataFrame, **kwargs: object) -> SplitManifest:
    return make_group_splits(observations, "screen_id", "strict_whole_screen", **kwargs)


def check_split_leakage(
    observations: pd.DataFrame,
    assignments: pd.DataFrame,
    heldout_column: str,
    *,
    exempt_column: str | None = None,
) -> None:
    """Fail when a held-out group also appears in training.

    ``exempt_column`` names a boolean column marking rows that are an assay
    reference rather than an evaluation unit, typically ``is_control``. In a
    gene-holdout design the control population has no "held-out control gene" to
    predict; its expected response is zero by construction and it is present in
    every fold so that a control-relative response has a same-screen baseline at
    scoring time. Those rows are therefore excluded from the group comparison.

    The exemption fails closed: naming a column that is absent raises rather than
    silently skipping the check, and control rows must still be excluded from
    every accuracy metric (see :mod:`measurement_sufficiency.metrics`).
    """
    columns = ["observation_id", heldout_column]
    if exempt_column is not None:
        if exempt_column not in observations:
            raise LeakageError(
                f"leakage-exemption column {exempt_column!r} was requested but is absent "
                "from observations; an exemption is never inferred"
            )
        columns.append(exempt_column)
    joined = assignments.merge(observations[columns], on="observation_id", validate="many_to_one")
    if exempt_column is not None:
        joined = joined.loc[~coerce_boolean_column(
            joined[exempt_column], column=exempt_column, table="observations"
        )]
    for fold_id, fold in joined.groupby("fold", sort=False):
        train = set(fold.loc[fold.role.eq("train"), heldout_column])
        test = set(fold.loc[fold.role.eq("test"), heldout_column])
        overlap = train.intersection(test)
        if overlap:
            sample = sorted(map(str, overlap))[0]
            raise LeakageError(f"{heldout_column} leakage in fold {fold_id}: {sample}")
