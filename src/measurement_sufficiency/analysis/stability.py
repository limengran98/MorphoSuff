"""Within-screen and guide-half stability summaries."""

from __future__ import annotations

import pandas as pd


def _correlation(frame: pd.DataFrame, left: str, right: str) -> float:
    if len(frame) < 2 or frame[left].nunique() < 2 or frame[right].nunique() < 2:
        return float("nan")
    return float(frame[left].corr(frame[right]))


def within_screen_reliability(replicates: pd.DataFrame, *, value_a: str = "value_a", value_b: str = "value_b", group_columns: list[str] | None = None) -> pd.DataFrame:
    """Pearson reliability between independently declared within-screen repeats."""
    groups = group_columns or ["reporter_id", "screen_id"]
    if missing := set(groups + [value_a, value_b]).difference(replicates):
        raise ValueError(f"replicates lacks columns: {', '.join(sorted(missing))}")
    return _summarize(replicates, groups, value_a, value_b, "within_screen_reliability")


def guide_half_consistency(halves: pd.DataFrame, *, group_columns: list[str] | None = None) -> pd.DataFrame:
    """Correlation of predeclared guide partitions, never data-driven halves."""
    groups = group_columns or ["reporter_id", "screen_id"]
    return _summarize(halves, groups, "half_a", "half_b", "guide_half_consistency")


def _summarize(frame: pd.DataFrame, groups: list[str], left: str, right: str, name: str) -> pd.DataFrame:
    rows = []
    for key, group in frame.groupby(groups, dropna=False, sort=False):
        key = key if isinstance(key, tuple) else (key,)
        rows.append(dict(zip(groups, key), n_pairs=len(group), **{name: _correlation(group, left, right)}))
    return pd.DataFrame(rows)


def stability_summary(replicates: pd.DataFrame, guide_halves: pd.DataFrame | None = None) -> pd.DataFrame:
    """Combine stability components by declared reporter/screen keys."""
    reliability = within_screen_reliability(replicates)
    if guide_halves is None:
        return reliability
    guide = guide_half_consistency(guide_halves)
    return reliability.merge(guide, on=["reporter_id", "screen_id"], how="outer", validate="one_to_one")
