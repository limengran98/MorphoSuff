#!/usr/bin/env python3
"""Ask whether the published tier counts survive a small move in every threshold.

Figure 6c reports 30 quantitative proxies, 6 ranking proxies, 3 measurements
required, 9 not identifiable and 4 unresolved. Those five numbers are the paper's
payoff and they come from seven analysis-defined cut points. A reader cannot tell
from the counts alone whether a reporter sits far from its cut or one thousandth
above it. This perturbs each cut in a neighbourhood and reports what moves.

Two schemes, because they answer different questions. `absolute` shifts every cut
by the same number of correlation units, which is the scale a reader thinks in.
`relative` shifts each cut by a fraction of its own value, which keeps the 0.30
reliability floor from moving proportionally further than the 0.70 gates.

Precedence
----------
The published table resolves a reporter that satisfies both the ranking-proxy and
the measurement-required rule as a ranking proxy. PSMB7 is the only such reporter:
recoverability 0.596, magnitude Spearman 0.799, top-5% recall 0.80, reliability
0.431. `measurement_sufficiency.analysis.tiers.assign_ops_tier` resolves it the
other way and so returns 5 ranking proxies and 4 measurements required, not the
published 6 and 3. Methods and `configs/ops/protocols/measurement_tiers.yaml` both
list the rules in the order used here but state no precedence, so neither reading
is contradicted by the text. This script implements the published order, and
`--expect-baseline` fails if that stops reproducing the figure. The disagreement
with the library primitive is a real defect and is not silently fixed here.
"""
from __future__ import annotations

import argparse
import dataclasses
import itertools
import json
from pathlib import Path

import pandas as pd

from measurement_sufficiency.analysis.tiers import OPS_TIER_RULES, OPSTierRules, OPSTier

#: The frozen figure-package column names, mapped onto the rule's own vocabulary.
RENAME = {"consensus10_gene_pearson": "recoverability_r", "magnitude_variance_ratio": "variance_ratio"}

#: Every threshold the rule reads, and the direction that makes the rule stricter.
#: The variance band is two-sided, so tightening moves its edges towards 1.0 while
#: loosening moves them apart. A single signed offset cannot express that, which is
#: why the band carries its own sign pair.
GATES: dict[str, int] = {
    "recoverability_min": +1,
    "magnitude_spearman_min": +1,
    "top5pct_quantitative_min": +1,
    "top5pct_ranking_min": +1,
    "reliability_min": +1,
    "measurement_recoverability_max_exclusive": +1,
    "variance_ratio_min": +1,
    "variance_ratio_max": -1,
}

TIER_ORDER = [
    OPSTier.QUANTITATIVE_PROXY.value,
    OPSTier.RANKING_PROXY.value,
    OPSTier.MEASUREMENT_REQUIRED.value,
    OPSTier.NOT_IDENTIFIABLE.value,
    OPSTier.UNRESOLVED.value,
]

PUBLISHED_COUNTS = {
    "quantitative_proxy": 30,
    "ranking_proxy": 6,
    "measurement_required": 3,
    "not_identifiable": 9,
    "unresolved": 4,
}


def assign(row: pd.Series, rules: OPSTierRules) -> str:
    """The rule in the precedence the published table uses.

    Reliability is read first, so a target whose own measurement does not replicate
    is never scored on prediction quality. Everything after that follows the order
    Methods lists: quantitative, then ranking, then measurement required.
    """
    reliability = row.get("reliability")
    if pd.isna(reliability) or float(reliability) < rules.reliability_min:
        return OPSTier.NOT_IDENTIFIABLE.value
    required = ("recoverability_r", "magnitude_spearman", "variance_ratio", "top5pct_recall")
    if any(name not in row or pd.isna(row[name]) for name in required):
        return OPSTier.UNRESOLVED.value
    recoverability = float(row["recoverability_r"])
    spearman = float(row["magnitude_spearman"])
    variance = float(row["variance_ratio"])
    recall = float(row["top5pct_recall"])
    if (
        recoverability >= rules.recoverability_min
        and spearman >= rules.magnitude_spearman_min
        and rules.variance_ratio_min <= variance <= rules.variance_ratio_max
        and recall >= rules.top5pct_quantitative_min
    ):
        return OPSTier.QUANTITATIVE_PROXY.value
    if spearman >= rules.magnitude_spearman_min and recall >= rules.top5pct_ranking_min:
        return OPSTier.RANKING_PROXY.value
    if recoverability < rules.measurement_recoverability_max_exclusive:
        return OPSTier.MEASUREMENT_REQUIRED.value
    return OPSTier.UNRESOLVED.value


def perturbed(rules: OPSTierRules, gate: str | None, step: float, scheme: str) -> OPSTierRules:
    """Return a copy of `rules` with one gate, or every gate, moved by `step`.

    `step` is signed: positive tightens. `gate=None` moves all of them together,
    which is the scenario a reader is really asking about when they ask whether the
    map depends on the cut points rather than on any one cut point.
    """
    targets = GATES if gate is None else {gate: GATES[gate]}
    changes = {}
    for name, direction in targets.items():
        base = getattr(rules, name)
        delta = step * direction * (abs(base) if scheme == "relative" else 1.0)
        changes[name] = round(base + delta, 10)
    return dataclasses.replace(rules, **changes)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--group-column", default="reporter_slug")
    parser.add_argument("--absolute-step", type=float, default=0.05,
                        help="correlation units each cut is moved in the absolute scheme")
    parser.add_argument("--relative-step", type=float, default=0.10,
                        help="fraction of its own value each cut is moved in the relative scheme")
    parser.add_argument("--expect-baseline", action="store_true",
                        help="fail unless the unperturbed rule reproduces the published counts")
    args = parser.parse_args()

    evidence = pd.read_csv(args.evidence).rename(columns=RENAME)
    if args.group_column not in evidence:
        raise SystemExit(f"evidence lacks the group column {args.group_column!r}")
    if evidence[args.group_column].duplicated().any():
        raise SystemExit("evidence must hold one row per reporter")

    baseline = evidence.apply(assign, axis=1, rules=OPS_TIER_RULES)
    observed = baseline.value_counts().to_dict()
    if args.expect_baseline:
        if {k: observed.get(k, 0) for k in PUBLISHED_COUNTS} != PUBLISHED_COUNTS:
            raise SystemExit(
                "the unperturbed rule does not reproduce Figure 6c: "
                f"expected {PUBLISHED_COUNTS}, observed {observed}"
            )

    scenarios: list[dict] = []
    assignments = pd.DataFrame({args.group_column: evidence[args.group_column], "baseline": baseline})
    for scheme, step in (("absolute", args.absolute_step), ("relative", args.relative_step)):
        for gate, sign in itertools.product([None, *GATES], (-1, +1)):
            rules = perturbed(OPS_TIER_RULES, gate, sign * step, scheme)
            tiers = evidence.apply(assign, axis=1, rules=rules)
            label = f"{scheme}:{gate or 'all gates'}:{'tighter' if sign > 0 else 'looser'}"
            assignments[label] = tiers
            counts = tiers.value_counts().to_dict()
            scenarios.append({
                "scheme": scheme,
                "gate": gate or "all gates",
                "direction": "tighter" if sign > 0 else "looser",
                "step": step,
                "n_reporters_changed": int((tiers != baseline).sum()),
                **{tier: int(counts.get(tier, 0)) for tier in TIER_ORDER},
            })

    summary = pd.DataFrame(scenarios)
    scenario_columns = [c for c in assignments.columns if c not in (args.group_column, "baseline")]
    stability = (
        assignments[scenario_columns]
        .eq(assignments["baseline"], axis=0)
        .mean(axis=1)
        .rename("fraction_of_scenarios_unchanged")
    )
    per_reporter = pd.concat([assignments[[args.group_column, "baseline"]], stability], axis=1)

    # A transition matrix over every scenario at once: how often a reporter that the
    # published rule put in row i lands in column j when a cut moves.
    long = assignments.melt(id_vars=[args.group_column, "baseline"], value_vars=scenario_columns,
                            var_name="scenario", value_name="tier")
    transitions = (
        pd.crosstab(long["baseline"], long["tier"])
        .reindex(index=TIER_ORDER, columns=TIER_ORDER, fill_value=0)
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_dir / "scenario_counts.csv", index=False)
    per_reporter.to_csv(args.output_dir / "reporter_stability.csv", index=False)
    transitions.to_csv(args.output_dir / "transition_matrix.csv")
    assignments.to_csv(args.output_dir / "assignments_by_scenario.csv", index=False)

    # Moving one gate and moving all eight at once answer different questions, so
    # they are never pooled into a single stability number. The single-gate block is
    # the one a reader should read as "does this cut decide the map"; the all-gates
    # block is a deliberate worst case in which every cut moves the same way at once.
    single = summary[summary["gate"] != "all gates"]
    single_labels = [c for c in scenario_columns if ":all gates:" not in c]
    single_unchanged = assignments[single_labels].eq(assignments["baseline"], axis=0).all(axis=1)

    proxies = {OPSTier.QUANTITATIVE_PROXY.value, OPSTier.RANKING_PROXY.value}
    coarse = assignments[scenario_columns].isin(proxies)
    coarse_unchanged = coarse.eq(assignments["baseline"].isin(proxies), axis=0).all(axis=1)

    report = {
        "n_reporters": int(len(evidence)),
        "n_scenarios": int(len(scenarios)),
        "baseline_counts": {tier: int(observed.get(tier, 0)) for tier in TIER_ORDER},
        "single_gate": {
            "n_scenarios": int(len(single)),
            "reporters_unchanged_in_every_scenario": int(single_unchanged.sum()),
            "median_reporters_changed": float(single["n_reporters_changed"].median()),
            "max_reporters_changed": int(single["n_reporters_changed"].max()),
            "gate_that_moves_most": single.loc[single["n_reporters_changed"].idxmax(), "gate"],
            "proxy_total_range": [
                int((single["quantitative_proxy"] + single["ranking_proxy"]).min()),
                int((single["quantitative_proxy"] + single["ranking_proxy"]).max()),
            ],
            "not_identifiable_range": [int(single["not_identifiable"].min()),
                                       int(single["not_identifiable"].max())],
        },
        "all_gates_together": {
            "n_scenarios": int(len(summary) - len(single)),
            "max_reporters_changed": int(summary.loc[summary["gate"] == "all gates",
                                                     "n_reporters_changed"].max()),
        },
        "reporters_unchanged_in_every_scenario": int(
            (per_reporter["fraction_of_scenarios_unchanged"] == 1.0).sum()),
        "reporters_keeping_proxy_status_in_every_scenario": int(coarse_unchanged.sum()),
        "proxy_total_range_all_scenarios": [
            int((summary["quantitative_proxy"] + summary["ranking_proxy"]).min()),
            int((summary["quantitative_proxy"] + summary["ranking_proxy"]).max()),
        ],
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
