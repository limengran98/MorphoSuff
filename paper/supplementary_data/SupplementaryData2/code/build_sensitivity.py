"""Portable deterministic sensitivity from frozen summaries; never train models."""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import io
import itertools
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RULES = dict(recoverability_min=.7, magnitude_spearman_min=.7,
             top5pct_quantitative_min=.6, top5pct_ranking_min=.5,
             reliability_min=.3, measurement_recoverability_max_exclusive=.6,
             variance_ratio_min=.5, variance_ratio_max=1.5)
DIRECTIONS = {key: (-1 if key == "variance_ratio_max" else 1) for key in RULES}
METRICS = ["recoverability_r", "magnitude_spearman", "variance_ratio", "top5pct_recall", "reliability"]
TIERS = ["quantitative_proxy", "ranking_proxy", "measurement_required", "not_identifiable", "unresolved"]
SHORT = dict(zip(TIERS, ["Q", "R", "M", "N", "U"]))
PREDICATES = {
    "recoverability_min": ("recoverability_r", ">=", "quantitative_fidelity"),
    "magnitude_spearman_min": ("magnitude_spearman", ">=", "quantitative_and_ranking_fidelity"),
    "top5pct_quantitative_min": ("top5pct_recall", ">=", "quantitative_fidelity"),
    "top5pct_ranking_min": ("top5pct_recall", ">=", "ranking_fidelity"),
    "reliability_min": ("reliability", ">=", "reference_identifiability"),
    "measurement_recoverability_max_exclusive": ("recoverability_r", "<", "measurement_required_eligibility_not_good_performance"),
    "variance_ratio_min": ("variance_ratio", ">=", "quantitative_fidelity"),
    "variance_ratio_max": ("variance_ratio", "<=", "quantitative_fidelity"),
}
SOURCES = {
    "main_metrics": "paper/figure_sources/final/figure6/f_cross_context_decisions/figure_source_gate_rows.csv",
    "main_baseline_labels": "paper/figure_sources/final/figure6/f_cross_context_decisions/figure_source_data.csv",
    "batch_metrics": "paper/figure_sources/supplementary_figure1/source_data/external_context/batch_verdicts.csv",
    "ops_scenario_reference": "paper/figure_sources/final/figure6/code/rebuild_coherent_statistics.py",
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def save_text(path, content):
    path = Path(path)
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise FileExistsError(f"Refuse to replace different content: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def save_csv(path, rows):
    rows = list(rows)
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    save_text(path, stream.getvalue())


def make_scenarios():
    rows = [dict(scenario_order=0, scenario_id="baseline", scenario_group="baseline",
                 scheme="baseline", varied_rule="none", direction="baseline", step=0., **RULES)]
    for scheme, step in [("absolute", .05), ("relative", .10)]:
        for gate, sign in itertools.product([None, *RULES], [-1, 1]):
            rules = RULES.copy()
            for key in ([gate] if gate else RULES):
                rules[key] = round(RULES[key] + sign * step * DIRECTIONS[key] *
                                   (abs(RULES[key]) if scheme == "relative" else 1.), 10)
            direction = "tighter" if sign > 0 else "looser"
            rows.append(dict(scenario_order=len(rows),
                scenario_id=f"{scheme}:{gate or 'all gates'}:{direction}",
                scenario_group="single32" if gate else "joint4", scheme=scheme,
                varied_rule=gate or "all gates", direction=direction, step=step, **rules))
    assert len(rows) == 37 and sum(r["scenario_group"] == "single32" for r in rows) == 32
    assert sum(r["scenario_group"] == "joint4" for r in rows) == 4
    return rows


def freeze(repo_root):
    if (ROOT / "analysis_freeze.json").exists():
        raise FileExistsError("Inputs were already frozen; do not silently re-import")
    repo_root = Path(repo_root).resolve()
    source_rows = [dict(source_id=key, repository_relative_path=value,
                        bytes=(repo_root/value).stat().st_size, sha256=digest(repo_root/value))
                   for key, value in SOURCES.items()]
    main = read_csv(repo_root/SOURCES["main_metrics"])
    labels = read_csv(repo_root/SOURCES["main_baseline_labels"])
    batches = read_csv(repo_root/SOURCES["batch_metrics"])
    rows = []
    main_order = ["oasis:mtt_viability", "oasis:ldh_release", "periscope_cells:tomm20_mito", "periscope_guide:tomm20_mito"]
    display = ["OASIS metabolic activity", "OASIS LDH release", "PERISCOPE anti-TOMM20 (cell)", "PERISCOPE anti-TOMM20 (guide)"]
    for key, label in zip(main_order, display):
        selected = [r for r in main if r["row_key"] == key and r["band"] == "admitted"]
        assert len(selected) == 5 and {r["quantity"] for r in selected} == set(METRICS)
        baseline = {r["verdict"].replace(" ", "_") for r in labels if r["row_key"] == key}
        assert len(baseline) == 1
        rows.append(dict(display_order=len(rows)+1, context_id=key, context_group="main4",
            resource="OASIS" if key.startswith("oasis:") else "PERISCOPE", label=label,
            batch="pooled", biological_unit="compound" if key.startswith("oasis:") else "held_out_gene",
            fitting_unit="well" if key.startswith("oasis:") else ("cell" if "cells:" in key else "guide_profile"),
            source_id="main_metrics", source_key=f"band=admitted;row_key={key}",
            published_baseline_tier=next(iter(baseline)),
            **{metric:next(r["value"] for r in selected if r["quantity"] == metric) for metric in METRICS}))
    for batch, reporter in itertools.product(["prod_25", "prod_26", "prod_27", "prod_30"], ["mtt_viability", "ldh_release"]):
        selected = [r for r in batches if r["batch"] == batch and r["reporter_id"] == reporter]
        assert len(selected) == 1
        source = selected[0]
        label = f"OASIS {'metabolic activity' if reporter == 'mtt_viability' else 'LDH release'}, batch {batch[-2:]}"
        rows.append(dict(display_order=len(rows)+1, context_id=f"oasis:{batch}:{reporter}",
            context_group="OASIS_batch8", resource="OASIS", label=label, batch=batch,
            biological_unit="compound", fitting_unit="well", source_id="batch_metrics", source_key=f"batch={batch};reporter_id={reporter}",
            published_baseline_tier=source["tier"].replace(" ", "_"), **{metric:source[metric] for metric in METRICS}))
    assert len(rows) == 12 and all(math.isfinite(float(r[k])) for r in rows for k in METRICS)
    save_csv(ROOT/"inputs/context_metrics.csv", rows)
    save_csv(ROOT/"inputs/threshold_scenarios.csv", make_scenarios())
    save_csv(ROOT/"inputs/source_manifest.csv", source_rows)
    paths = ["protocol.md", "code/build_sensitivity.py", "inputs/context_metrics.csv",
             "inputs/threshold_scenarios.csv", "inputs/source_manifest.csv"]
    record = dict(frozen_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        status="fixed_before_this_bundle_evaluation_of_already_known_external_metrics",
        source_manifest="inputs/source_manifest.csv", n_contexts=12, n_main=4, n_batch=8,
        n_baseline=1, n_single_scenarios=32, n_joint_scenarios=4,
        historical_rank_only_thresholds=[.65,.60,.55], historical_contexts=main_order[:2],
        frozen_files={path:digest(ROOT/path) for path in paths})
    save_text(ROOT/"analysis_freeze.json", json.dumps(record, indent=2)+"\n")
    print(json.dumps({k:v for k,v in record.items() if k != "frozen_files"}, indent=2))


def condition(value, cutoff, operator):
    if not math.isfinite(value):
        return False
    return value >= cutoff if operator == ">=" else value <= cutoff if operator == "<=" else value < cutoff


def evidence(row, rules):
    values = {key:float(row[key]) for key in METRICS}
    atomic = {key:condition(values[metric], rules[key], op) for key,(metric,op,_) in PREDICATES.items()}
    reliable = atomic["reliability_min"]
    finite = all(math.isfinite(values[key]) for key in METRICS[:-1])
    quantitative = all(atomic[key] for key in ["recoverability_min", "magnitude_spearman_min", "variance_ratio_min", "variance_ratio_max", "top5pct_quantitative_min"])
    ranking = atomic["magnitude_spearman_min"] and atomic["top5pct_ranking_min"]
    tier = ("not_identifiable" if not reliable else "unresolved" if not finite else
            "quantitative_proxy" if quantitative else "ranking_proxy" if ranking else
            "measurement_required" if atomic["measurement_recoverability_max_exclusive"] else "unresolved")
    return tier, atomic, dict(reference_reliability_pass=reliable, fidelity_complete=finite,
        quantitative_fidelity_pass=quantitative, ranking_fidelity_pass=ranking,
        quantitative_recovery_pass=atomic["recoverability_min"],
        quantitative_rank_pass=atomic["magnitude_spearman_min"],
        quantitative_amplitude_pass=atomic["variance_ratio_min"] and atomic["variance_ratio_max"],
        quantitative_hit_recall_pass=atomic["top5pct_quantitative_min"],
        measurement_required_eligible=atomic["measurement_recoverability_max_exclusive"])


def evaluate():
    frozen = json.loads((ROOT/"analysis_freeze.json").read_text())
    for path, expected in frozen["frozen_files"].items():
        assert digest(ROOT/path) == expected, f"Frozen input changed: {path}"
    contexts = read_csv(ROOT/"inputs/context_metrics.csv")
    scenarios = read_csv(ROOT/"inputs/threshold_scenarios.csv")
    assignments, criteria = [], []
    for context in contexts:
        baseline, base_atomic, base_aggregate = evidence(context, RULES)
        assert baseline == context["published_baseline_tier"], context["context_id"]
        for scenario in scenarios:
            rules = {key:float(scenario[key]) for key in RULES}
            tier, atomic, aggregate = evidence(context, rules)
            changed = [key for key in RULES if atomic[key] != base_atomic[key]]
            fidelity_changed = [key for key in ["quantitative_recovery_pass", "quantitative_rank_pass",
                "quantitative_amplitude_pass", "quantitative_hit_recall_pass"] if aggregate[key] != base_aggregate[key]]
            assignments.append(dict(context_id=context["context_id"], context_group=context["context_group"],
                label=context["label"], scenario_id=scenario["scenario_id"], scenario_group=scenario["scenario_group"],
                baseline_tier=baseline, tier=tier, tier_short=SHORT[tier], tier_changed=tier!=baseline,
                proxy_status_changed=(tier in TIERS[:2])!=(baseline in TIERS[:2]),
                n_criterion_conditions_changed=len(changed), changed_criterion_conditions=";".join(changed),
                quantitative_fidelity_vector_changed=bool(fidelity_changed),
                n_quantitative_fidelity_conditions_changed=len(fidelity_changed),
                changed_quantitative_fidelity_conditions=";".join(fidelity_changed), **aggregate))
            for key,(metric,op,role) in PREDICATES.items():
                criteria.append(dict(context_id=context["context_id"], context_group=context["context_group"],
                    scenario_id=scenario["scenario_id"], scenario_group=scenario["scenario_group"],
                    criterion=key, criterion_role=role, metric=metric, metric_value=context[metric],
                    comparator=op, threshold=rules[key], baseline_threshold=RULES[key],
                    threshold_changed=rules[key]!=RULES[key], gate_condition_met=atomic[key],
                    baseline_gate_condition_met=base_atomic[key], gate_condition_changed=atomic[key]!=base_atomic[key],
                    tier=tier, hierarchy_reference_pass=aggregate["reference_reliability_pass"]))
    assert len(assignments) == 444 and len(criteria) == 3552
    save_csv(ROOT/"derived/assignments.csv", assignments)
    save_csv(ROOT/"derived/criterion_evidence.csv", criteria)
    summary = []
    for context in contexts:
        baseline = context["published_baseline_tier"]
        row = dict(display_order=context["display_order"], context_id=context["context_id"],
                   context_group=context["context_group"], label=context["label"], baseline_tier=baseline,
                   baseline_short=SHORT[baseline])
        for scope,n in [("single32",32),("joint4",4),("all36",36)]:
            chosen = [a for a in assignments if a["context_id"] == context["context_id"] and
                      a["scenario_group"] != "baseline" and (scope == "all36" or a["scenario_group"] == scope)]
            assert len(chosen) == n
            changes = sum(a["tier_changed"] for a in chosen)
            possibilities = set(a["tier"] for a in chosen) | {baseline}
            row.update({f"{scope}_n_scenarios":n, f"{scope}_tier_changes":changes,
                f"{scope}_tier_unchanged_fraction":(n-changes)/n,
                f"{scope}_proxy_changes":sum(a["proxy_status_changed"] for a in chosen),
                f"{scope}_proxy_unchanged_fraction":sum(not a["proxy_status_changed"] for a in chosen)/n,
                f"{scope}_scenarios_with_criterion_changes":sum(a["n_criterion_conditions_changed"]>0 for a in chosen),
                f"{scope}_scenarios_with_quantitative_fidelity_vector_changes":sum(a["quantitative_fidelity_vector_changed"] for a in chosen),
                f"{scope}_possible_tiers":";".join(t for t in TIERS if t in possibilities),
                f"{scope}_possible_short":"/".join(SHORT[t] for t in TIERS if t in possibilities)})
        summary.append(row)
    save_csv(ROOT/"derived/context_summary.csv", summary)
    scenario_summary=[]
    for scenario,group in itertools.product(scenarios,["main4","OASIS_batch8"]):
        chosen=[a for a in assignments if a["scenario_id"]==scenario["scenario_id"] and a["context_group"]==group]
        scenario_summary.append(dict(context_group=group, scenario_id=scenario["scenario_id"],
            scenario_group=scenario["scenario_group"], n_contexts=len(chosen),
            tier_changes=sum(a["tier_changed"] for a in chosen), proxy_changes=sum(a["proxy_status_changed"] for a in chosen),
            contexts_with_criterion_changes=sum(a["n_criterion_conditions_changed"]>0 for a in chosen),
            contexts_with_quantitative_fidelity_vector_changes=sum(a["quantitative_fidelity_vector_changed"] for a in chosen),
            **{tier:sum(a["tier"]==tier for a in chosen) for tier in TIERS}))
    save_csv(ROOT/"derived/scenario_summary.csv", scenario_summary)
    historical=[]
    for context in contexts:
        if context["context_id"] not in frozen["historical_contexts"]:
            continue
        for cutoff in frozen["historical_rank_only_thresholds"]:
            rules={**RULES,"magnitude_spearman_min":cutoff}
            tier,atomic,aggregate=evidence(context,rules)
            historical.append(dict(context_id=context["context_id"], rank_threshold=cutoff,
                analysis_group="historical_rank_only_not_in_36", baseline_tier=context["published_baseline_tier"],
                tier=tier, tier_short=SHORT[tier], tier_changed=tier!=context["published_baseline_tier"],
                **{k:context[k] for k in METRICS}, **aggregate))
    assert len(historical)==6
    save_csv(ROOT/"derived/historical_rank_only.csv",historical)
    latex=["% Generated Table 7 rows: context, baseline, unchanged single32, unchanged joint4, other tiers."]
    bs=chr(92); eol=bs*2
    for i,row in enumerate(summary):
        if i in [0,4]:
            if i==4:
                latex.append(bs+"midrule")
            label="Main applications" if i==0 else "OASIS batch applications"
            latex.append(f"{bs}multicolumn{{5}}{{l}}{{{bs}textit{{{label}}}}} {eol}")
        other="/".join(SHORT[t] for t in row["all36_possible_tiers"].split(";") if t != row["baseline_tier"]) or "---"
        latex.append(f"{row['label']} & {row['baseline_short']} & {32-row['single32_tier_changes']}/32 & {4-row['joint4_tier_changes']}/4 & {other} {eol}")
    save_text(ROOT/"table_rows.tex","\n".join(latex)+"\n")
    print(json.dumps(dict(status="complete",assignments=len(assignments),atomic_criteria=len(criteria),
                         contexts=len(summary),historical_rows=len(historical)),indent=2))
    for row in summary:
        print(row["context_id"],row["all36_tier_changes"],row["all36_possible_short"])


def manifest():
    rows=[]
    for path in sorted(ROOT.rglob("*")):
        if path.is_file() and path.name != "manifest_sha256.csv" and "__pycache__" not in path.parts:
            rows.append(dict(path=path.relative_to(ROOT).as_posix(),bytes=path.stat().st_size,sha256=digest(path)))
    # The manifest is derived inventory, not a frozen source or first result.
    stream=io.StringIO(newline="")
    writer=csv.DictWriter(stream,fieldnames=["path","bytes","sha256"],lineterminator="\n")
    writer.writeheader(); writer.writerows(rows)
    (ROOT/"manifest_sha256.csv").write_text(stream.getvalue(),encoding="utf-8")
    print(f"Manifest: {len(rows)} files")


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=["freeze","evaluate","manifest"])
    parser.add_argument("--repo-root",type=Path,default=ROOT.parents[2])
    args=parser.parse_args()
    if args.action=="freeze":
        freeze(args.repo_root)
    elif args.action=="evaluate":
        evaluate()
    else:
        manifest()
