#!/usr/bin/env python3
"""Load and verify the frozen Figure 2 scores and reporter-level inference.

Frozen source IDs and current manuscript panels, per `CONTRIBUTING.md`:

    views b,c -> panel b   reporter (52 reporters; 34 under strict whole-screen)
    views d,e -> panel c   physical screen (73 / 73 / 66 screens)
    views f,g -> panel d   reporter x screen assay (99 / 99 / 81 assays)

The local `source_data/frozen_figure2_package` contains one `*_by_method.csv` per
source view, with ten methods per unit per test. Rendering verifies six frozen descriptive
medians. The optional `--verify-inference` check independently reproduces the 54
prespecified reporter-level comparisons in `source_data/statistics` using SciPy.
Both checks are read-only; neither updates scores, statistics or rendered outputs.
The separate rebuild_benchmark_sources.py regenerates these tables from the
portable partition source using arithmetic means, without fitting any predictor.

Aggregation is unchanged from the manuscript: the cross-model consensus is the
unweighted median over the ten aligned methods, as defined in Methods.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

# This package is self-contained: every numerical table used below ships beside
# this module in the released source bundle.
FIGURE2 = Path(__file__).resolve().parents[1]
PAPER = Path(__file__).resolve().parents[4]
PKG = FIGURE2 / "source_data" / "frozen_figure2_package"
INFERENCE = FIGURE2 / "source_data" / "statistics" / "reporter_level_primary_inference.csv"

PANELS = {
    "b": ("B_reporter_ko", "reporter_ko_by_method.csv", "ko_pearson", "reporter_slug"),
    "c": ("C_reporter_cell", "reporter_cell_by_method.csv", "cell_pearson", "reporter_slug"),
    "d": ("D_screen_ko", "screen_ko_by_method.csv", "ko_pearson", "screen_id"),
    "e": ("E_screen_cell", "screen_cell_by_method.csv", "cell_pearson", "screen_id"),
    "f": ("F_assay_ko", "assay_ko_by_method.csv", "ko_pearson", "assay_id"),
    "g": ("G_assay_cell", "assay_cell_by_method.csv", "cell_pearson", "assay_id"),
}

TESTS = ["Field", "Gene", "Whole-screen"]

# Corrected descriptive medians provide rendering-integrity checks. Their ordering
# is not evidence of paired superiority; verify_inference() checks paired reporters.
EXPECTED = {
    ("b", "Field"): ("MLP", 0.715),
    ("b", "Gene"): ("MLP", 0.768),
    ("b", "Whole-screen"): ("Ridge", 0.730),
    ("c", "Field"): ("MLP", 0.763),
    ("c", "Gene"): ("MLP", 0.760),
    ("c", "Whole-screen"): ("Ridge", 0.621),
}


def _require_package() -> None:
    if not PKG.is_dir():
        raise SystemExit(
            f"frozen Figure 2 source package not found at {PKG}.\n"
            "It ships with the figure package; without it no panel can be drawn from "
            "verified source tables."
        )


def load(panel: str) -> pd.DataFrame:
    """One panel's long table, with the metric and unit columns renamed generically."""
    _require_package()
    folder, filename, metric, unit = PANELS[panel]
    frame = pd.read_csv(PKG / folder / "source_data" / filename)
    return frame.rename(columns={metric: "score", unit: "unit"})


def roster() -> pd.DataFrame:
    """The ten-method roster with its frozen colours and display order."""
    _require_package()
    return pd.read_csv(
        PKG / "B_reporter_ko" / "source_data" / "method_roster.csv"
    ).sort_values("method_order")


def verify() -> list[str]:
    """Reproduce six frozen descriptive medians. Returns the report."""
    lines = []
    for (panel, test), (want_method, want_value) in EXPECTED.items():
        frame = load(panel)
        medians = (frame[frame["split_label"] == test]
                   .groupby("method")["score"].median().sort_values(ascending=False))
        got_method, got_value = medians.index[0], medians.iloc[0]
        assert got_method == want_method, (
            f"panel {panel} / {test}: leading method is {got_method}, "
            f"frozen check expects {want_method}")
        assert round(got_value, 3) == want_value, (
            f"panel {panel} / {test}: top median {got_value:.4f} != frozen {want_value}")
        runner_up = f"{medians.index[1]} {medians.iloc[1]:.4f}"
        lines.append(f"  {panel} {test:<13} {got_method:<8} {got_value:.4f} == {want_value}  "
                     f"(runner-up {runner_up})")
    return lines


def _holm_adjust(p_values: list[float]) -> list[float]:
    """Holm step-down adjustment over the complete supplied comparison family."""
    order = sorted(range(len(p_values)), key=p_values.__getitem__)
    adjusted = [0.0] * len(p_values)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(p_values) - rank) * p_values[index])
        adjusted[index] = min(running, 1.0)
    return adjusted


def compute_inference() -> pd.DataFrame:
    """Compute the prespecified 54 tests from the current reporter tables.

    The paired effect is median(method - MLP) within reporter, not a difference
    between marginal medians. Multiplicity spans 9 comparators x 3 tests x 2
    reporter-level metrics. Obsolete alternative-family correction columns are
    excluded from the corrected primary table and survive only in the local archive.
    SciPy is needed only when this optional function is called.
    """
    import numpy as np

    try:
        from scipy.stats import wilcoxon
    except ImportError as error:
        raise SystemExit(
            "--verify-inference requires SciPy (verified with SciPy 1.11.4); "
            "the rendering-only verification does not require it."
        ) from error

    methods = roster()["method"].tolist()
    assert len(methods) == len(set(methods)) == 10 and "MLP" in methods
    rows = []
    for panel in ("b", "c"):
        frame = load(panel)
        for test in TESTS:
            wide = frame.loc[frame["split_label"].eq(test)].pivot(
                index="unit", columns="method", values="score"
            )
            assert set(wide.columns) == set(methods), f"method roster changed: {panel}/{test}"
            for method in methods:
                if method == "MLP":
                    continue
                paired = wide[[method, "MLP"]].dropna()
                delta = paired[method].to_numpy(float) - paired["MLP"].to_numpy(float)
                expected_n = 34 if test == "Whole-screen" else 52
                assert len(delta) == expected_n, f"paired reporter count changed: {panel}/{test}"
                assert np.isfinite(paired.to_numpy(float)).all(), "non-finite paired scores"
                p_raw = (float(wilcoxon(
                    delta, alternative="two-sided", zero_method="wilcox", method="auto"
                ).pvalue) if np.any(delta != 0) else 1.0)
                rows.append({
                    "panel": panel, "split": test, "method": method,
                    "panel_title": "Reporter · KO response" if panel == "b" else "Reporter · cell phenotype",
                    "aggregation_unit": "reporters", "reference": "MLP",
                    "evaluation_metric": PANELS[panel][2],
                    "n_paired_units": len(delta),
                    "median_method": float(np.median(paired[method])),
                    "median_reference": float(np.median(paired["MLP"])),
                    "median_paired_delta": float(np.median(delta)),
                    "p_raw_wilcoxon": p_raw,
                    "inference_status": "primary paired reporter-level inference",
                    "displayed_in_main_figure": False, "display_symbol": "",
                })

    computed = pd.DataFrame(rows)
    computed["p_holm_reporter54"] = _holm_adjust(computed["p_raw_wilcoxon"].tolist())
    effect_gate = "effect_gate_abs_median_delta_ge_0_02"
    computed[effect_gate] = computed["median_paired_delta"].abs().ge(0.02)
    computed["meets_reporter_inference_gate"] = (
        computed["p_holm_reporter54"].lt(0.05) & computed[effect_gate]
    )
    assert len(computed) == 54
    return computed


def verify_inference() -> list[str]:
    """Reproduce the corrected primary table; never rewrite its frozen record."""
    import numpy as np

    archived = pd.read_csv(INFERENCE)
    keys = ["panel", "split", "method"]
    effect_gate = "effect_gate_abs_median_delta_ge_0_02"
    assert not archived.duplicated(keys).any() and len(archived) == 54
    assert archived["reference"].eq("MLP").all()
    assert archived["aggregation_unit"].eq("reporters").all()
    assert archived["inference_status"].eq("primary paired reporter-level inference").all()
    assert not archived["displayed_in_main_figure"].any()
    computed = compute_inference()
    computed = computed.set_index(keys).sort_index()
    archived = archived.set_index(keys).sort_index()
    assert computed.index.equals(archived.index), "primary comparison identities changed"
    for column in ("evaluation_metric", "n_paired_units", effect_gate,
                   "meets_reporter_inference_gate"):
        assert computed[column].equals(archived[column]), f"archived {column} differs"
    for column in ("median_method", "median_reference", "median_paired_delta",
                   "p_raw_wilcoxon", "p_holm_reporter54"):
        np.testing.assert_allclose(
            computed[column], archived[column], rtol=1e-9, atol=1e-14,
            err_msg=f"archived {column} differs from frozen reporter scores",
        )

    passed = computed.index[computed["meets_reporter_inference_gate"]]
    expected_passes = {
        (panel, test, method) for panel in ("b", "c")
        for test in ("Field", "Gene") for method in ("Ridge", "scButterfly", "MIDAS")
    } - {("b", "Gene", "Ridge")}
    assert set(passed) == expected_passes, "the corrected 11 dual-gate results changed"
    assert computed.loc[passed, "median_paired_delta"].lt(0).all()
    lines = ["all 54 primary comparisons reproduced; 11 meet both frozen criteria"]
    for method in ("Ridge", "scButterfly", "MIDAS"):
        result = computed.loc[("b", "Gene", method)]
        lines.append(
            f"  Gene KO: MLP minus {method:<11} paired median "
            f"{float(-result['median_paired_delta']):.6f}; "
            f"Holm P={float(result['p_holm_reporter54']):.6g}"
        )
    transfer = computed.loc[("b", "Whole-screen", "Ridge")]
    assert transfer["median_method"] > transfer["median_reference"]
    assert transfer["median_paired_delta"] < 0, "marginal medians are not paired effects"
    lines.append(
        f"  Whole-screen KO: Ridge minus MLP paired median "
        f"{float(transfer['median_paired_delta']):.6f}; "
        f"Holm P={float(transfer['p_holm_reporter54']):.6g}"
    )
    return lines


def consensus(panel: str) -> pd.DataFrame:
    """Unweighted ten-method median per unit per test, with the inter-method spread.

    `lo`/`hi` are the min and max across the ten methods and `q25`/`q75` the
    interquartile spread; the redesigned d,e panels draw the first as a faint band so
    single-method failures stay visible and the second as the solid band.
    """
    frame = load(panel)
    out = (frame.groupby(["split_label", "unit"])["score"]
           .agg(consensus="median", lo="min", hi="max",
                q25=lambda s: s.quantile(0.25),
                q75=lambda s: s.quantile(0.75),
                n_methods="size").reset_index())
    assert (out["n_methods"] == 10).all(), "a unit is missing one of the ten methods"
    return out


def assay_axes(panel: str) -> pd.DataFrame:
    """Assay-level consensus carrying its reporter and screen identity."""
    frame = load(panel)
    keys = ["split_label", "unit", "reporter_slug", "screen_id"]
    out = (frame.groupby(keys)["score"]
           .agg(consensus="median", lo="min", hi="max", n_methods="size").reset_index())
    assert (out["n_methods"] == 10).all()
    return out


def assay_unit_note(panel: str, test: str) -> str:
    """Honest count for one assay sub-column.

    Panel g's field and gene columns carry no assay-level information: every assay of
    a reporter holds an identical cell-level score, because those views reweight the
    frozen reporter-cell score by physical membership rather than evaluating
    destination cells directly. Ninety-nine marks would be drawn from 52 values, so
    the column is annotated with what it actually resolves, not with the assay count.
    The distinction is detected from the data rather than assumed.
    """
    part = assay_axes(panel)
    part = part[part["split_label"] == test]
    n_assays = len(part)
    n_distinct = part[["reporter_slug", "consensus"]].drop_duplicates().shape[0]
    n_reporters = part["reporter_slug"].nunique()
    if n_distinct == n_reporters < n_assays:
        return f"{n_reporters} reporter-level"
    assert n_distinct == n_assays, (
        f"{panel}/{test}: {n_assays} assays resolve to {n_distinct} values, which is "
        "neither fully assay-level nor fully reporter-level; the annotation would lie")
    return f"{n_assays} assays"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verify-inference", action="store_true",
        help="also reproduce all 54 primary paired comparisons (requires SciPy)",
    )
    args = parser.parse_args()
    print("Verifying frozen Figure 2 descriptive medians:")
    for line in verify():
        print(line)
    print("\nall six descriptive medians reproduced from the frozen CSVs\n")
    for panel in ("d", "e"):
        con = consensus(panel)
        print(f"panel {panel}: units per test =",
              dict(con.groupby("split_label")["unit"].nunique()))
    for panel in ("f", "g"):
        ax = assay_axes(panel)
        print(f"panel {panel}: assays per test =",
              dict(ax.groupby("split_label")["unit"].nunique()),
              "| reporters =", ax["reporter_slug"].nunique(),
              "| screens =", ax["screen_id"].nunique())
    if args.verify_inference:
        print("\nVerifying archived reporter-level inference:")
        for line in verify_inference():
            print(line)
