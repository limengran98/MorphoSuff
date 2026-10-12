"""Evidence-led measurement-sufficiency summaries and frozen tier rules."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd


class SufficiencyTier(str, Enum):
    QUANTITATIVE_PROXY = "quantitative_proxy"
    RANKING_PROXY = "ranking_proxy"
    MEASUREMENT_REQUIRED = "measurement_required"
    NOT_IDENTIFIABLE = "not_identifiable"
    UNRESOLVED = "unresolved"


@dataclass(frozen=True)
class TierThresholds:
    """Caller-declared thresholds for the portable rule.

    Deliberately without defaults. The published OPS/A549 rule is
    :class:`measurement_sufficiency.analysis.tiers.OPSTierRules`; it is a
    different rule over different evidence, not a calibration of this one, and
    the previous defaults here read as a second OPS calibration that disagreed
    with it on every shared threshold. ``docs/adapt_new_dataset.md`` requires a
    new biological system to calibrate its own numbers prospectively.
    """

    quantitative_r: float
    ranking_spearman: float
    minimum_reliability: float
    minimum_amplitude_ratio: float
    maximum_environment_drop: float
    maximum_ambiguity: float


REQUIRED_EVIDENCE = ("recoverability_r", "ranking_spearman", "reliability", "amplitude_ratio", "environment_drop", "ambiguity")


def assign_tier(evidence: pd.Series | dict[str, float], thresholds: TierThresholds) -> SufficiencyTier:
    """Assign a tier without conflating missing evidence with biological failure."""
    values = pd.Series(evidence)
    identifiable = values.get("identifiable", True)
    if pd.notna(identifiable) and not bool(identifiable):
        return SufficiencyTier.NOT_IDENTIFIABLE
    if any(name not in values or pd.isna(values[name]) for name in REQUIRED_EVIDENCE):
        return SufficiencyTier.UNRESOLVED
    stable = (values.reliability >= thresholds.minimum_reliability and values.amplitude_ratio >= thresholds.minimum_amplitude_ratio and values.environment_drop <= thresholds.maximum_environment_drop and values.ambiguity <= thresholds.maximum_ambiguity)
    if not stable:
        return SufficiencyTier.MEASUREMENT_REQUIRED
    if values.recoverability_r >= thresholds.quantitative_r:
        return SufficiencyTier.QUANTITATIVE_PROXY
    if values.ranking_spearman >= thresholds.ranking_spearman:
        return SufficiencyTier.RANKING_PROXY
    return SufficiencyTier.MEASUREMENT_REQUIRED


def summarize_evidence(records: pd.DataFrame, *, group_columns: list[str], thresholds: TierThresholds, value_columns: list[str] | None = None) -> pd.DataFrame:
    """Aggregate replicate evidence; caller retains the biological resampling unit."""
    columns = value_columns or list(REQUIRED_EVIDENCE)
    missing = set(group_columns + columns).difference(records.columns)
    if missing:
        raise ValueError(f"evidence records lacks columns: {', '.join(sorted(missing))}")
    summary = records.groupby(group_columns, dropna=False)[columns].median().reset_index()
    # Built from strings rather than mapped over a Series of enum members: with a
    # pyarrow backend installed, pandas materialises this str-subclass enum as an
    # Arrow string array and a later ``.map(lambda x: x.value)`` raises on a plain
    # str. See the same note in analysis/tiers.py::tier_table.
    summary["tier"] = [assign_tier(row, thresholds).value for _, row in summary.iterrows()]
    return summary


def bootstrap_biological_units(records: pd.DataFrame, *, unit_column: str, statistic: str, n_boot: int = 1000, seed: int = 0) -> np.ndarray:
    """Resample biological units, expressly not model identities, for uncertainty."""
    if unit_column not in records or statistic not in records:
        raise ValueError("bootstrap columns are absent")
    units = pd.unique(records[unit_column])
    if len(units) == 0:
        raise ValueError("no biological units to resample")
    rng = np.random.default_rng(seed)
    unit_means = records.groupby(unit_column)[statistic].mean()
    return np.array([unit_means.reindex(rng.choice(units, len(units), replace=True)).mean() for _ in range(n_boot)])
