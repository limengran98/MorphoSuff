"""The single, OPS/A549-specific manuscript tier rule."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal, Mapping

import pandas as pd


class OPSTier(str, Enum):
    QUANTITATIVE_PROXY = "quantitative_proxy"
    RANKING_PROXY = "ranking_proxy"
    MEASUREMENT_REQUIRED = "measurement_required"
    NOT_IDENTIFIABLE = "not_identifiable"
    UNRESOLVED = "unresolved"


@dataclass(frozen=True)
class OPSTierRules:
    """The runtime source of truth for the frozen OPS/A549 tier thresholds.

    ``configs/ops/protocols/measurement_tiers.yaml`` is a mirror of these values,
    not their source: the package declares only numpy and pandas, and ``configs/``
    is not installed with the wheel, so reading the YAML at runtime would add a
    core dependency and packaging surface for seven floats.
    ``tests/known_answers/test_tier_rule_contract.py`` asserts that the YAML, this
    dataclass and the published Methods agree, so they cannot drift apart.
    """

    recoverability_min: float = 0.70
    magnitude_spearman_min: float = 0.70
    variance_ratio_min: float = 0.50
    variance_ratio_max: float = 1.50
    top5pct_quantitative_min: float = 0.60
    top5pct_ranking_min: float = 0.50
    reliability_min: float = 0.30
    measurement_recoverability_max_exclusive: float = 0.60


OPS_TIER_RULES = OPSTierRules()
#: ``top5pct_recall`` is recall at the top 5% of held-out genes, produced by
#: ``analysis.response_fidelity.top_fraction_recall``. It is deliberately not the
#: absolute top-k quantity, which the Methods do not use for this rule.
_REQUIRED = ("recoverability_r", "magnitude_spearman", "variance_ratio", "top5pct_recall", "reliability")


def assign_ops_tier(evidence: Mapping[str, object] | pd.Series, rules: OPSTierRules = OPS_TIER_RULES) -> OPSTier:
    """Apply the paper rule in its declared precedence order.

    Domain/environment sensitivity is deliberately not accepted as a tier input:
    the protocol labels it a separate flag, not an additional hidden threshold.

    Missing reliability yields ``NOT_IDENTIFIABLE`` rather than ``UNRESOLVED``.
    That is the published rule, not an oversight: Methods states "Not identifiable
    applied when reliability was <0.30 **or unavailable**", and
    ``configs/ops/protocols/measurement_tiers.yaml`` records the same under
    ``not_identifiable: or: reliability_missing``. Do not "fix" it.

    Precedence between ranking proxy and measurement required. A reporter can
    satisfy both: PSMB7 has recoverability 0.596, below the 0.60 measurement gate,
    while its magnitude Spearman of 0.799 and top-5% recall of 0.80 clear the
    ranking gate. The published decision table resolves it as a ranking proxy, so
    the ranking test runs first. This function previously tested measurement
    required first and returned 5 ranking proxies and 4 measurements required
    against the published 6 and 3, which meant a reader running the released code
    could not reproduce Fig. 6c. Methods and the YAML now state the order.
    ``tests/known_answers/test_tier_rule_contract.py`` pins the published counts.
    """
    values = pd.Series(evidence)
    reliability = values.get("reliability")
    if pd.isna(reliability) or float(reliability) < rules.reliability_min:
        return OPSTier.NOT_IDENTIFIABLE
    if any(name not in values or pd.isna(values[name]) for name in _REQUIRED):
        return OPSTier.UNRESOLVED
    quantitative = (
        float(values.recoverability_r) >= rules.recoverability_min
        and float(values.magnitude_spearman) >= rules.magnitude_spearman_min
        and rules.variance_ratio_min <= float(values.variance_ratio) <= rules.variance_ratio_max
        and float(values.top5pct_recall) >= rules.top5pct_quantitative_min
    )
    if quantitative:
        return OPSTier.QUANTITATIVE_PROXY
    if float(values.magnitude_spearman) >= rules.magnitude_spearman_min and float(values.top5pct_recall) >= rules.top5pct_ranking_min:
        return OPSTier.RANKING_PROXY
    if float(values.recoverability_r) < rules.measurement_recoverability_max_exclusive:
        return OPSTier.MEASUREMENT_REQUIRED
    return OPSTier.UNRESOLVED


def tier_table(
    evidence: pd.DataFrame,
    *,
    group_columns: list[str],
    aggregate: Literal["median", "mean", "none"] = "median",
    rules: OPSTierRules = OPS_TIER_RULES,
    tier_column: str = "tier",
) -> pd.DataFrame:
    """Assign one tier per group, reducing replicate evidence first.

    ``group_columns`` was previously accepted and ignored, so a per-fold or
    per-model evidence table produced one "tier" per input row rather than one per
    reporter. The default median matches the manuscript's unweighted median over
    the ten methods. An all-null column stays null under the median, which
    preserves the reliability-missing precedence in :func:`assign_ops_tier`.
    """
    if aggregate not in {"median", "mean", "none"}:
        raise ValueError("aggregate must be 'median', 'mean' or 'none'")
    if missing := set(group_columns + list(_REQUIRED)).difference(evidence):
        raise ValueError(f"evidence lacks columns: {', '.join(sorted(missing))}")

    if aggregate == "none":
        duplicated = evidence.duplicated(group_columns, keep=False)
        if duplicated.any():
            sample = evidence.loc[duplicated, group_columns].drop_duplicates().head(3).to_dict("records")
            raise ValueError(
                f"aggregate='none' requires one row per group, but these repeat: {sample}"
            )
        result = evidence.copy()
        result["n_records"] = 1
    else:
        reduced = (
            evidence.groupby(group_columns, dropna=False)[list(_REQUIRED)]
            .agg(aggregate)
            .reset_index()
        )
        counts = (
            evidence.groupby(group_columns, dropna=False)
            .size()
            .rename("n_records")
            .reset_index()
        )
        result = reduced.merge(counts, on=group_columns, validate="one_to_one")

    # Take .value inside the row function rather than mapping over the result.
    # OPSTier subclasses str, so a Series built from tier members is only an
    # object-dtype Series of enums when pandas has no better inference available.
    # With a pyarrow backend installed, pandas materialises the same values as an
    # Arrow string array and the enum identity is gone, after which a subsequent
    # ``.map(lambda tier: tier.value)`` raises AttributeError on a plain str. The
    # tier text must not depend on which optional dependencies happen to be
    # present, so build the column from strings directly.
    result[tier_column] = [
        assign_ops_tier(row, rules).value for _, row in result.iterrows()
    ]
    return result
