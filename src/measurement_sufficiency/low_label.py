"""Deterministic, reusable exact-budget low-label sampling manifests."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import pandas as pd


class LowLabelManifestError(ValueError):
    """Raised when sampling could leak held-out observations or budgets are invalid."""


def exact_budget(n_train: int, fraction: float) -> int:
    """Return a documented exact integer budget, with 100% anchored to all train rows.

    Floored at one label. ``round(n_train * fraction)`` reaches zero whenever the
    product is at most 0.5, which at the frozen 0.1% arm is any reporter with 500
    or fewer training cells; the manifest then selected nothing and validate()
    accepted it as a low-label condition.
    """
    if n_train < 1:
        raise LowLabelManifestError("exact budget requires at least one training observation")
    if not 0.0 < fraction <= 1.0:
        raise LowLabelManifestError("fraction must be in (0, 1]")
    if fraction == 1.0:
        return n_train
    return min(n_train, max(1, int(round(n_train * fraction))))


def _stable_rank(observation_id: object, seed: int) -> str:
    return hashlib.sha256(f"{seed}|{observation_id}".encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class LowLabelManifest:
    """Selections shared across methods, with all train rows retained for auditability."""

    assignments: pd.DataFrame
    seed: int

    def selected_ids(self, fraction: float) -> set[object]:
        subset = self.assignments.loc[(self.assignments.fraction == fraction) & self.assignments.selected, "observation_id"]
        return set(subset)

    def validate(self) -> None:
        required = {"observation_id", "fraction", "selected", "budget", "seed", "source_role"}
        missing = required.difference(self.assignments.columns)
        if missing:
            raise LowLabelManifestError(f"manifest lacks columns: {', '.join(sorted(missing))}")
        if not self.assignments.source_role.eq("train").all():
            raise LowLabelManifestError("low-label manifest may contain only train observations")
        for fraction, group in self.assignments.groupby("fraction", sort=False):
            if group.observation_id.duplicated().any():
                raise LowLabelManifestError("manifest duplicates observation IDs within a budget")
            if int(group.budget.iloc[0]) < 1:
                raise LowLabelManifestError(
                    f"budget of zero labels for fraction={fraction} is not a training condition"
                )
            if int(group.selected.sum()) != int(group.budget.iloc[0]):
                raise LowLabelManifestError(f"selection count does not equal exact budget for fraction={fraction}")
            if fraction == 1.0 and not group.selected.all():
                raise LowLabelManifestError("100% low-label anchor must select every training observation")


def build_low_label_manifest(
    assignments: pd.DataFrame,
    *,
    fold: int,
    fractions: list[float] | tuple[float, ...],
    seed: int = 0,
) -> LowLabelManifest:
    """Create nested deterministic samples exclusively from a fold's train role.

    Hash ranking (rather than incidental input order) makes the same frozen
    manifest reusable by every eligible model and stable across CSV ordering.
    """
    required = {"observation_id", "fold", "role"}
    missing = required.difference(assignments.columns)
    if missing:
        raise LowLabelManifestError(f"split assignments lack columns: {', '.join(sorted(missing))}")
    all_fold = assignments.loc[assignments.fold == fold]
    if all_fold.duplicated("observation_id").any():
        raise LowLabelManifestError("fold assignment must have one role per observation")
    train = all_fold.loc[all_fold.role == "train", "observation_id"]
    if train.empty:
        raise LowLabelManifestError(f"fold {fold} has no train observations")
    if train.isna().any() or train.duplicated().any():
        raise LowLabelManifestError("train observation IDs must be unique and non-null")
    unique_fractions = sorted(set(float(value) for value in fractions))
    if not unique_fractions:
        raise LowLabelManifestError("at least one fraction is required")
    for fraction in unique_fractions:
        exact_budget(len(train), fraction)
    ranked = sorted(train.tolist(), key=lambda oid: _stable_rank(oid, seed))
    rows: list[dict[str, object]] = []
    for fraction in unique_fractions:
        budget = exact_budget(len(ranked), fraction)
        selected = set(ranked[:budget])
        rows.extend(
            {"observation_id": oid, "fraction": fraction, "selected": oid in selected, "budget": budget, "seed": seed, "source_role": "train"}
            for oid in ranked
        )
    manifest = LowLabelManifest(pd.DataFrame(rows), seed)
    manifest.validate()
    return manifest
