"""Conditional target ambiguity, separated from prediction error."""

from __future__ import annotations

import numpy as np
import pandas as pd


def conditional_ambiguity(targets: pd.DataFrame, *, condition_columns: list[str], value_column: str = "value", group_columns: list[str] | None = None) -> pd.DataFrame:
    """Estimate irreducible conditional spread as E[var(Y | declared condition)]."""
    if not condition_columns:
        raise ValueError("conditional ambiguity requires declared condition columns")
    groups = group_columns or ["reporter_id", "endpoint_id"]
    needed = set(groups + condition_columns + [value_column])
    if missing := needed.difference(targets):
        raise ValueError(f"targets lacks columns: {', '.join(sorted(missing))}")
    rows = []
    for key, group in targets.groupby(groups, dropna=False, sort=False):
        key = key if isinstance(key, tuple) else (key,)
        conditional = group.groupby(condition_columns, dropna=False)[value_column].agg(["var", "count"])
        usable = conditional[conditional["count"] >= 2]
        variance = np.average(usable["var"], weights=usable["count"]) if len(usable) else float("nan")
        total = group[value_column].var(ddof=1)
        normalized = float("nan") if not np.isfinite(total) or total <= 0 else float(variance / total)
        rows.append(dict(zip(groups, key), n_rows=len(group), conditional_variance=variance, ambiguity=normalized))
    return pd.DataFrame(rows)
