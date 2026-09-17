#!/usr/bin/env python3
"""Freeze or regenerate the predictor-tier tables; never fit prediction models.

Normal use reads the deposited evidence.csv. --analysis-root is an optional
import of the completed frozen-output analysis, not a training command.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REFERENCE_STATE = "checkpoint_average"
REFERENCE = "coherent_median_ensemble"
STATES = ["checkpoint_average", "single"]
TIERS = ["quantitative_proxy", "ranking_proxy", "measurement_required", "not_identifiable", "unresolved"]
LETTERS = dict(zip(TIERS, "QRMNU"))
PROXY = {"quantitative_proxy", "ranking_proxy"}
METRICS = ["recoverability_r", "magnitude_spearman", "variance_ratio", "top5pct_recall", "reliability"]
GATES = ["reliability", "recovery", "rank", "variance_lower", "variance_upper", "quantitative_hit", "ranking_hit", "measurement_recovery"]
METHODS = ["ridge", "gbdt", "catboost_multirmse", "mlp", "tabm_specialists", "scbutterfly_ops_b", "resmlp_52head", "multitab_pilot", "midas_ops", "scpair_ops"]
DEFINITIONS = {
    "individual_method": "All metrics use the named method in run_state; four specialist methods have one frozen evaluation output shared by both state labels.",
    "coherent_median_ensemble": "Element-wise median across ten same-state method responses within fold and gene-screen-endpoint; all metrics use that predictor. Recovery: endpoint-macro Pearson then mean of five folds. Fidelity: equal-screen mean per gene, then magnitudes across 1,000 genes.",
    "state_matched_hybrid": "Recovery is the median of ten same-state method recovery scores; fidelity is computed from the same-state profile-median ensemble before screen averaging. Definition sensitivity, not a unified predictor.",
    "hybrid_median_before_screen_mean": "Historical mixed-state construction: recovery is the median of legacy single-state method scores; fidelity uses run_state profile medians before screen averaging. The checkpoint_average row replays the pre-revision historical ledger; it is not the adopted reference.",
}


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")


def write_csv(name, frame):
    path = ROOT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    compression = {"method": "gzip", "mtime": 0} if path.suffix == ".gz" else None
    frame.to_csv(path, index=False, compression=compression)


def sanitize(value):
    if isinstance(value, str) and "/MM/" in value:
        return value.split("/MM/", 1)[1]
    if isinstance(value, dict):
        return {k: sanitize(v) for k, v in value.items() if k != "elapsed_seconds"}
    if isinstance(value, list):
        return [sanitize(v) for v in value]
    return value


def tier(row):
    if pd.isna(row.reliability) or row.reliability < .3:
        return "not_identifiable"
    if any(pd.isna(row[m]) for m in METRICS):
        return "unresolved"
    if row.recoverability_r >= .7 and row.magnitude_spearman >= .7 and .5 <= row.variance_ratio <= 1.5 and row.top5pct_recall >= .6:
        return "quantitative_proxy"
    if row.magnitude_spearman >= .7 and row.top5pct_recall >= .5:
        return "ranking_proxy"
    if row.recoverability_r < .6:
        return "measurement_required"
    return "unresolved"


def gate_flags(row):
    return dict(reliability=row.reliability >= .3, recovery=row.recoverability_r >= .7,
                rank=row.magnitude_spearman >= .7, variance_lower=row.variance_ratio >= .5,
                variance_upper=row.variance_ratio <= 1.5, quantitative_hit=row.top5pct_recall >= .6,
                ranking_hit=row.top5pct_recall >= .5, measurement_recovery=row.recoverability_r < .6)


def import_analysis(path):
    frames, folds, aligned, inventories = [], [], [], []
    for state in STATES:
        block = path / f"full_{state}"
        frames.append(pd.read_csv(block / "all_evidence_candidates.csv").rename(columns={"published_tier": "historical_reference_tier"}))
        folds.append(pd.read_csv(block / "fold_recoverability.csv").assign(run_state=state))
        aligned.append(pd.read_csv(block / "alignment_audit.csv").assign(run_state=state))
        inventory = pd.read_csv(block / "source_inventory.csv")
        inventory["path"] = inventory.path.map(sanitize)
        inventory["run_state"] = state
        inventories.append(inventory)
    evidence = pd.concat(frames, ignore_index=True)
    main = evidence[evidence.method_id.isin(METHODS + [REFERENCE])].drop(columns="historical_reference_tier")
    definitions = evidence[evidence.method_id.isin(["state_matched_hybrid", "hybrid_median_before_screen_mean"])]
    write_csv("evidence.csv", main)
    write_csv("definition_sensitivity.csv", definitions)
    write_csv("provenance/per_method_fold_recoverability.csv", pd.concat(folds, ignore_index=True))
    write_csv("provenance/alignment_audit.csv", pd.concat(aligned, ignore_index=True))
    inventory = pd.concat(inventories, ignore_index=True)
    inventory = inventory.groupby(["path", "bytes", "sha256"], dropna=False, as_index=False).agg(
        run_state=("run_state", lambda x: ";".join(sorted(set(x)))))
    write_csv("provenance/frozen_input_manifest.csv.gz", inventory)
    audits = json.loads((path / "full_checkpoint_average/control_scale_audit.json").read_text())
    scales = []
    for audit in audits:
        for fold in audit["folds"]:
            for i, (name, scale, yscale) in enumerate(zip(audit["endpoint_names"], audit["endpoint_scales"], fold["y_scale"])):
                scales.append(dict(reporter_slug=audit["reporter_slug"], endpoint_index=i, endpoint_name=name,
                                   fold=fold["fold"], common_control_scale=scale, fold_target_scale=yscale,
                                   fold_to_common_factor=yscale/scale))
    write_csv("provenance/endpoint_scales.csv", pd.DataFrame(scales))
    # Complete audit retains counts/digests and relative locators, not private paths.
    clean_audits = [{k: v for k, v in a.items() if k not in {"endpoint_scales", "endpoint_names", "folds"}} for a in audits]
    write_json(ROOT / "provenance/control_scale_audit.json", sanitize(clean_audits))
    spec = json.loads((path / "full_checkpoint_average/run_spec.json").read_text())
    source_rows = []
    for state in STATES:
        p = path / f"full_{state}/all_evidence_candidates.csv"
        source_rows.append(dict(run_state=state, logical_source=f"full_{state}/all_evidence_candidates.csv",
                                sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    write_csv("provenance/analysis_evidence_digests.csv", pd.DataFrame(source_rows))
    write_json(ROOT / "analysis_spec.json", {
        "reference_state": REFERENCE_STATE, "reference_method_id": REFERENCE,
        "states": STATES, "method_ids": METHODS, "reporter_order": spec["reporters"],
        "rule": spec["rule"], "unit": "reporter", "n_reporters": 52,
        "reliability_passing_denominator": 43, "same_state_all_method_denominator": 10,
        "scale": "common within-screen-centered population SD of unique NTC cells",
        "definition_sensitivity": DEFINITIONS,
        "source_analysis_sha256": spec["analysis_script_sha256"],
        "source_scale_sha256": spec["scale_script_sha256"],
        "reference_semantics": "Every changes_vs_reference and delta field in this package uses checkpoint_average coherent_median_ensemble, including single-state sensitivity rows.",
    })


def build():
    main = pd.read_csv(ROOT / "evidence.csv")
    assert len(main) == 2 * 52 * 11
    data = pd.concat([main, pd.read_csv(ROOT / "definition_sensitivity.csv")], ignore_index=True)
    spec = json.loads((ROOT / "analysis_spec.json").read_text())
    assert len(data) == 2 * 52 * 13
    assert not data.duplicated(["run_state", "reporter_slug", "method_id"]).any()
    assert data[METRICS].notna().all().all()
    assert data.apply(tier, axis=1).eq(data.tier).all()
    for row in data.itertuples(index=False):
        assert all(bool(getattr(row, f"pass_{gate}")) == flag for gate, flag in gate_flags(row).items())
    assert data.groupby("reporter_slug").reliability.nunique().eq(1).all()
    ref = data[data.run_state.eq(REFERENCE_STATE) & data.method_id.eq(REFERENCE)].set_index("reporter_slug")
    assert len(ref) == 52 and ref.reliability.ge(.3).sum() == 43
    order = pd.DataFrame({"reporter_slug": spec["reporter_order"], "display_order": range(1, 53)})
    order = order.merge(ref[["short_name", "biological_category", "reliability", "tier"]], on="reporter_slug", validate="one_to_one")
    order = order.rename(columns={"tier": "reference_tier"})
    order["reliability_pass"] = order.reliability.ge(.3)
    write_csv("reporter_order.csv", order)
    roster = data[data.method_id.isin(METHODS)][["method_id", "method"]].drop_duplicates().set_index("method_id").loc[METHODS].reset_index()
    roster["method_order"] = range(1, 11)
    write_csv("method_roster.csv", roster)
    changes, counts, stability, ensembles = [], [], [], []
    for state in STATES:
        block = data[data.run_state.eq(state)]
        matrix = block.pivot(index="reporter_slug", columns="method_id", values="tier").loc[spec["reporter_order"], METHODS + [REFERENCE]]
        write_csv(f"derived/{state}_tier_matrix.csv", matrix.reset_index())
        for method, rows in block.groupby("method_id", sort=False):
            rows = rows.set_index("reporter_slug").loc[ref.index]
            reliability_pass = rows.reliability.ge(.3)
            tier_equal = rows.tier.eq(ref.tier)
            proxy_equal = rows.tier.isin(PROXY).eq(ref.tier.isin(PROXY))
            record = dict(run_state=state, method_id=method, reference_state=REFERENCE_STATE,
                          reference_method_id=REFERENCE, n_reporters=52, n_reliability_passing=43,
                          exact_tier_agreement_43=int((tier_equal & reliability_pass).sum()),
                          proxy_agreement_43=int((proxy_equal & reliability_pass).sum()),
                          exact_tier_agreement_52=int(tier_equal.sum()), proxy_agreement_52=int(proxy_equal.sum()))
            record.update({f"n_{LETTERS[t]}": int(rows.tier.eq(t).sum()) for t in TIERS})
            record["n_proxy"] = int(rows.tier.isin(PROXY).sum())
            if method in METHODS + [REFERENCE]:
                counts.append(record)
            if method not in METHODS:
                ensembles.append(dict(**record, definition=DEFINITIONS[method]))
            for reporter, row in rows.iterrows():
                baseline = ref.loc[reporter]
                gf, rf = gate_flags(row), gate_flags(baseline)
                gate_changes = [g for g in GATES if gf[g] != rf[g]]
                source_tier, destination_tier = baseline.tier, row.tier
                if source_tier == destination_tier:
                    transition = "unchanged"
                elif source_tier in PROXY and destination_tier in PROXY:
                    transition = f"{LETTERS[source_tier]}_to_{LETTERS[destination_tier]}"
                else:
                    transition = f"{LETTERS[source_tier]}_to_{LETTERS[destination_tier]}"
                entry = dict(run_state=state, reporter_slug=reporter, method_id=method,
                             reference_state=REFERENCE_STATE, reference_method_id=REFERENCE,
                             reference_tier=source_tier, tier=destination_tier, transition=transition,
                             exact_tier_changed=source_tier != destination_tier,
                             proxy_status_changed=(source_tier in PROXY) != (destination_tier in PROXY),
                             reliability_pass=bool(reliability_pass.loc[reporter]),
                             changed_gate_flags=";".join(gate_changes), n_changed_gate_flags=len(gate_changes))
                entry.update({f"delta_{metric}": float(row[metric]-baseline[metric]) for metric in METRICS})
                if method in METHODS + [REFERENCE]:
                    changes.append(entry)
        for reporter, rows in block[block.method_id.isin(METHODS)].groupby("reporter_slug", sort=False):
            assert len(rows) == 10
            n_proxy = int(rows.tier.isin(PROXY).sum())
            passed = bool(rows.reliability.iloc[0] >= .3)
            category = "fixed_reliability_N" if not passed else "all_proxy" if n_proxy == 10 else "all_no_proxy" if n_proxy == 0 else "proxy_boundary_crossing"
            stability.append(dict(run_state=state, reporter_slug=reporter, reliability_pass=passed,
                                  n_methods=10, n_proxy=n_proxy, n_distinct_tiers=rows.tier.nunique(),
                                  tiers=";".join(t for t in TIERS if t in set(rows.tier)),
                                  exact_tier_stable=rows.tier.nunique() == 1, stability_category=category))
    changes = pd.DataFrame(changes)
    write_csv("derived/changes_vs_reference.csv", changes)
    write_csv("derived/method_tier_counts.csv", pd.DataFrame(counts))
    write_csv("derived/ensemble_definition_comparison.csv", pd.DataFrame(ensembles))
    write_csv("derived/reporter_stability.csv", pd.DataFrame(stability))
    transitions = changes.groupby(["run_state", "method_id", "reference_state", "reference_method_id", "reference_tier", "tier", "reliability_pass"]).size().rename("n_reporters").reset_index()
    write_csv("derived/transition_counts.csv", transitions)
    state_comparison = data[data.method_id.isin(METHODS)].pivot(index=["reporter_slug", "method_id"], columns="run_state", values=METRICS + ["tier"])
    state_comparison.columns = [f"{metric}__{state}" for metric, state in state_comparison.columns]
    state_comparison["exact_tier_changed"] = state_comparison.tier__single != state_comparison.tier__checkpoint_average
    state_comparison["proxy_status_changed"] = state_comparison.tier__single.isin(PROXY) != state_comparison.tier__checkpoint_average.isin(PROXY)
    write_csv("derived/checkpoint_state_comparison.csv", state_comparison.reset_index())
    summary = {"reference_state": REFERENCE_STATE, "reference_method_id": REFERENCE, "n_reporters": 52,
               "reliability_passing_denominator": 43, "fixed_reliability_N": 9, "states": {}}
    for state in STATES:
        stable = pd.DataFrame(stability).query("run_state == @state")
        summary["states"][state] = stable.stability_category.value_counts().to_dict()
    assert [summary["states"]["checkpoint_average"][k] for k in ["all_proxy", "all_no_proxy", "proxy_boundary_crossing"]] == [20, 6, 17]
    assert [summary["states"]["single"][k] for k in ["all_proxy", "all_no_proxy", "proxy_boundary_crossing"]] == [22, 6, 15]
    write_json(ROOT / "derived/summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-root", type=Path, help="Optional directory containing completed full_checkpoint_average and full_single analyses")
    args = parser.parse_args()
    if args.analysis_root:
        import_analysis(args.analysis_root)
    build()
