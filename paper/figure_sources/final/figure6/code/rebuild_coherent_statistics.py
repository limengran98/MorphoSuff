#!/usr/bin/env python3
"""Rebuild Figure 6 derived statistics from its compact, frozen leaf inputs.

No image-to-reporter predictor is fitted. The four-feature reporter-level utility
Ridge is refitted with training-reference targets in both levels of nested LOO.
All 52 reporters are retained for that descriptive utility audit; the independent
reliability floor admits 43 reporters for replacement decisions.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "source_data"
sys.path.insert(0, str(ROOT))
from b_nested_loo_audit.build_panel_b import bootstrap_spearman, utility_spearman

RECOVERY = "ensemble_recoverability_r"
PREDICTORS = [(RECOVERY, "Recovery"), ("magnitude_spearman", "Rank"),
              ("amplitude_fidelity", "Amplitude"),
              ("within_screen_split_half_pearson_median", "Reliability")]
COMPONENTS = ["top5pct_recall", "go_bp_top10_jaccard", "go_cc_top10_jaccard",
              "ebi_complex_top10_jaccard"]
ALPHAS = [.01, .1, 1., 10., 100.]
TIERS = ["quantitative_proxy", "ranking_proxy", "measurement_required",
         "not_identifiable", "unresolved"]
RULES = dict(recoverability_min=.7, magnitude_spearman_min=.7,
             top5pct_quantitative_min=.6, top5pct_ranking_min=.5,
             reliability_min=.3, measurement_recoverability_max_exclusive=.6,
             variance_ratio_min=.5, variance_ratio_max=1.5)
DIRECTIONS = {key: (-1 if key == "variance_ratio_max" else 1) for key in RULES}
GATES = [("Recovery", RECOVERY, .7), ("Rank", "magnitude_spearman", .7),
         ("Amplitude", "magnitude_variance_ratio", .5),
         ("Top-5%", "top5pct_recall", .6), ("Reliability", "reliability", .3)]


def reference_utility(raw, train, held):
    """Training average-rank percentiles; held-out training empirical CDF.

    At a value observed in training, use its average training rank percentile;
    at an absent value, use the fraction strictly below it. This mapping is not
    linear interpolation, and never includes the held-out value in a reference.
    """
    values = raw[train]
    y_train = pd.DataFrame(values).rank(method="average", pct=True).mean(axis=1).to_numpy()
    less = (values < raw[held]).sum(axis=0)
    equal = (values == raw[held]).sum(axis=0)
    y_held = np.mean((less + np.where(equal > 0, (equal + 1) / 2, 0)) / len(train))
    return y_train, float(y_held)


def nested_loo(x, raw):
    n = len(x)
    output = []
    for held in range(n):
        train = np.delete(np.arange(n), held)
        y_train, y_held = reference_utility(raw, train, held)
        # Inner reference/scaler are reconstructed from the other 50 reporters.
        inner = []
        for valid in train:
            inner_train = train[train != valid]
            inner_y, valid_y = reference_utility(raw, inner_train, valid)
            scaler = StandardScaler().fit(x[inner_train])
            inner.append((scaler.transform(x[inner_train]), inner_y,
                          scaler.transform(x[[valid]]), valid_y))
        losses = []
        for alpha in ALPHAS:
            losses.append(np.mean([
                (Ridge(alpha=alpha).fit(xx, yy).predict(vx)[0] - vy) ** 2
                for xx, yy, vx, vy in inner
            ]))
        # np.argmin chooses the first alpha in the declared grid on an exact tie.
        alpha = ALPHAS[int(np.argmin(losses))]
        scaler = StandardScaler().fit(x[train])
        prediction = Ridge(alpha=alpha).fit(scaler.transform(x[train]), y_train).predict(
            scaler.transform(x[[held]]))[0]
        output.append(dict(scientific_utility_index=y_held,
                           outer_loo_prediction=float(prediction), selected_alpha=alpha))
    return pd.DataFrame(output)


def assign(row, rules=RULES):
    if pd.isna(row.reliability) or row.reliability < rules["reliability_min"]:
        return "not_identifiable"
    r, rank, variance, recall = [row[c] for c in
        [RECOVERY, "magnitude_spearman", "magnitude_variance_ratio", "top5pct_recall"]]
    if not np.isfinite([r, rank, variance, recall]).all():
        return "unresolved"
    if (r >= rules["recoverability_min"] and rank >= rules["magnitude_spearman_min"]
            and rules["variance_ratio_min"] <= variance <= rules["variance_ratio_max"]
            and recall >= rules["top5pct_quantitative_min"]):
        return "quantitative_proxy"
    if rank >= rules["magnitude_spearman_min"] and recall >= rules["top5pct_ranking_min"]:
        return "ranking_proxy"
    if r < rules["measurement_recoverability_max_exclusive"]:
        return "measurement_required"
    return "unresolved"


def thresholds(ledger):
    assignments = ledger[["reporter_slug"]].copy()
    assignments["baseline"] = ledger.apply(assign, axis=1)
    scenarios = []
    for scheme, step in [("absolute", .05), ("relative", .10)]:
        for gate, sign in itertools.product([None, *RULES], [-1, 1]):
            rules = RULES.copy()
            for key in ([gate] if gate else RULES):
                rules[key] = round(RULES[key] + sign * step * DIRECTIONS[key] *
                                   (abs(RULES[key]) if scheme == "relative" else 1.), 10)
            label = f"{scheme}:{gate or 'all gates'}:{'tighter' if sign > 0 else 'looser'}"
            result = ledger.apply(assign, axis=1, rules=rules)
            assignments[label] = result
            scenarios.append(dict(scenario=label, scheme=scheme, gate=gate or "all gates",
                direction="tighter" if sign > 0 else "looser", step=step,
                n_reporters_changed=int((result != assignments.baseline).sum()),
                **{tier: int(result.eq(tier).sum()) for tier in TIERS}))
    scenario_columns = list(assignments.columns[2:])
    single_columns = [c for c in scenario_columns if ":all gates:" not in c]
    equal = assignments[scenario_columns].eq(assignments.baseline, axis=0)
    same_proxy = assignments[scenario_columns].isin(TIERS[:2]).eq(
        assignments.baseline.isin(TIERS[:2]), axis=0)
    ledger["baseline"] = assignments.baseline
    ledger["fraction_of_all_scenarios_unchanged"] = equal.mean(axis=1)
    ledger["fraction_of_single_gate_scenarios_unchanged"] = equal[single_columns].mean(axis=1)
    ledger["exact_tier_unchanged_in_every_scenario"] = equal.all(axis=1)
    ledger["proxy_status_unchanged_in_every_scenario"] = same_proxy.all(axis=1)
    long = assignments.melt(id_vars=["reporter_slug", "baseline"],
                            value_vars=scenario_columns, value_name="tier")
    transition = pd.crosstab(long.baseline, long.tier).reindex(index=TIERS, columns=TIERS, fill_value=0)
    scenario = pd.DataFrame(scenarios)
    single = scenario.loc[scenario.gate.ne("all gates")]
    report = dict(n_reporters=52, n_scenarios=36,
        baseline_counts=assignments.baseline.value_counts().to_dict(),
        reliability_passing=int(ledger.reliability.ge(.3).sum()),
        single_gate=dict(n_scenarios=32,
            reporters_unchanged_in_every_scenario=int(equal[single_columns].all(axis=1).sum()),
            median_reporters_changed=float(single.n_reporters_changed.median()),
            max_reporters_changed=int(single.n_reporters_changed.max()),
            proxy_total_range=[int((single.quantitative_proxy+single.ranking_proxy).min()),
                               int((single.quantitative_proxy+single.ranking_proxy).max())]),
        all_gates_together=dict(n_scenarios=4,
            max_reporters_changed=int(scenario.loc[scenario.gate.eq("all gates"), "n_reporters_changed"].max())),
        reporters_unchanged_in_every_scenario=int(equal.all(axis=1).sum()),
        reporters_keeping_proxy_status_in_every_scenario=int(same_proxy.all(axis=1).sum()),
        proxy_total_range_all_scenarios=[int((scenario.quantitative_proxy+scenario.ranking_proxy).min()),
                                       int((scenario.quantitative_proxy+scenario.ranking_proxy).max())])
    return ledger, assignments, scenario, transition, report


def write_csv(frame, relative):
    path = ROOT / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-legacy", type=Path,
                        help="Read-only validation against an archived historical nested-LOO CSV")
    args = parser.parse_args()
    evidence = pd.read_csv(INPUT / "coherent_ensemble_evidence.csv")
    base = pd.read_csv(INPUT / "response_replacement_utility_and_annotations.csv")
    if len(base) != 52 or base.reporter_slug.nunique() != 52:
        raise RuntimeError("Expected 52 unique reporter annotations")
    ledger = base.merge(evidence, on="reporter_slug", validate="one_to_one")
    ledger["amplitude_fidelity"] = np.exp(-np.abs(np.log(ledger.magnitude_variance_ratio.clip(lower=1e-6))))
    ledger["within_screen_split_half_pearson_median"] = ledger.reliability
    data = ledger.sort_values("nested_loo_order").reset_index(drop=True)
    raw = data[COMPONENTS].to_numpy(float)
    if args.verify_legacy:
        old = pd.read_csv(args.verify_legacy).set_index("reporter_slug").loc[data.reporter_slug]
        x = old[["consensus10_gene_pearson", "magnitude_spearman", "amplitude_fidelity",
                 "within_screen_split_half_pearson_median"]].to_numpy(float)
        replay = nested_loo(x, raw)
        cols = ["scientific_utility_index", "outer_loo_prediction", "selected_alpha"]
        errors = {c: float(np.max(np.abs(replay[c].to_numpy()-old[c].to_numpy()))) for c in cols}
        np.testing.assert_allclose(replay[cols], old[cols], rtol=0, atol=1e-12)
        print(json.dumps(dict(status="PASS", legacy_replay_max_errors=errors)), flush=True)
        return
    folds = pd.read_csv(INPUT / "coherent_ensemble_fold_recoverability.csv")
    if len(folds) != 260 or folds.groupby("reporter_slug").fold.nunique().ne(5).any():
        raise RuntimeError("Expected 52 reporters × 5 coherent ensemble folds")
    recovery = folds.groupby("reporter_slug")[RECOVERY].mean().loc[ledger.reporter_slug].to_numpy()
    np.testing.assert_allclose(recovery, ledger[RECOVERY], rtol=0, atol=5e-12)
    ledger["substitutability_tier"] = ledger.apply(assign, axis=1)
    np.testing.assert_array_equal(ledger.substitutability_tier, ledger.expected_tier)
    ledger, assignments, scenario, transition, threshold_report = thresholds(ledger)
    for label, column, threshold in GATES:
        values = ledger[column]
        if label == "Amplitude":
            margin = np.minimum(values - .5, 1.5 - values) / .5
            passes = values.between(.5, 1.5)
        else:
            margin = (values - threshold) / (1 - threshold)
            passes = values.ge(threshold)
        ledger[f"gate_value__{label}"] = values
        ledger[f"gate_margin__{label}"] = margin
        ledger[f"gate_pass__{label}"] = passes
    cpath = "c_decision_ledger/source_data/"
    write_csv(ledger, cpath+"figure6c_decision_ledger_source.csv")
    write_csv(ledger, "d_prediction_utility/figure_source_decision_ledger.csv")
    write_csv(assignments, cpath+"figure6c_threshold_assignments.csv")
    write_csv(scenario, cpath+"figure6c_threshold_scenario_counts.csv")
    write_csv(transition.reset_index(), cpath+"figure6c_threshold_transition_matrix.csv")
    (ROOT / cpath / "figure6c_threshold_sensitivity_summary.json").write_text(json.dumps(threshold_report, indent=2)+"\n")
    data = ledger.sort_values("nested_loo_order").reset_index(drop=True)
    fitted = nested_loo(data[[c for c, _ in PREDICTORS]].to_numpy(float), data[COMPONENTS].to_numpy(float))
    b = data[["reporter_slug", *[c for c, _ in PREDICTORS], "short_name", "substitutability_tier"]].copy()
    b["short_name"] = data.nested_loo_short_name
    b = pd.concat([b, fitted], axis=1)
    b["mean_utility"] = (b.scientific_utility_index+b.outer_loo_prediction)/2
    b["prediction_residual"] = b.outer_loo_prediction-b.scientific_utility_index
    b["label_flag"] = False
    b.loc[b.prediction_residual.abs().sort_values(ascending=False, kind="stable").index[:4], "label_flag"] = True
    b["label_reason"] = np.where(b.label_flag, "four largest absolute outer-LOO residuals", "")
    associations = []
    for column, label in PREDICTORS:
        rho, low, high, n = bootstrap_spearman(b[column].to_numpy(float), b.scientific_utility_index.to_numpy(float))
        associations.append(dict(predictor=column, label=label, rho=rho, ci_low=low, ci_high=high,
            n_reporters=n, ci_method="reporter-level percentile bootstrap", bootstrap_reps=10000,
            bootstrap_seed=20260831))
    y, p = b.scientific_utility_index.to_numpy(), b.outer_loo_prediction.to_numpy()
    summary = pd.DataFrame([dict(n_reporters=52, outer_loo_r2=float(1-np.sum((p-y)**2)/np.sum((y-y.mean())**2)),
        outer_loo_spearman=float(utility_spearman(p,y).statistic),
        median_selected_alpha=float(b.selected_alpha.median()))])
    write_csv(b, "b_nested_loo_audit/figure_source_nested_loo.csv")
    write_csv(b, "d_prediction_utility/figure_source_reporter_utility.csv")
    write_csv(pd.DataFrame(associations), "b_nested_loo_audit/figure_source_association_bootstrap.csv")
    write_csv(summary, "b_nested_loo_audit/figure_source_nested_loo_summary.csv")
    # SI1 keeps local source tables so its rendering remains standalone.
    si1 = ROOT.parents[1] / "supplementary_figure1"
    margin_rows = []
    for idx, row in ledger.sort_values("display_order").reset_index(drop=True).iterrows():
        for metric_index, (label, column, threshold) in enumerate(GATES):
            margin_rows.append(dict(reporter_index=idx, metric_index=metric_index,
                reporter_slug=row.reporter_slug, short_name=row.short_name, tier=row.substitutability_tier,
                metric=label, source_column=column, value=row[column],
                threshold="[0.5, 1.5]" if label == "Amplitude" else str(threshold),
                scaled_margin=row[f"gate_margin__{label}"], passes=row[f"gate_pass__{label}"]))
    pd.DataFrame(margin_rows).to_csv(si1 / "source_data/ops_reporter_threshold_margins.csv", index=False)
    import importlib.util
    spec = importlib.util.spec_from_file_location("figure6_panel_f", ROOT / "code/build_panel_f.py")
    panel_f = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(panel_f)
    external = pd.read_csv(ROOT / "f_cross_context_decisions/figure_source_gate_rows.csv")
    display = panel_f.build_display_table(ledger, external)
    write_csv(display, "f_cross_context_decisions/figure_source_data.csv")
    display.to_csv(si1 / "source_data/external_context/cross_context_gate_summary.csv", index=False)
    report = dict(estimand="checkpoint-average elementwise median ensemble before equal-screen gene averaging",
        recovery_definition="endpoint macro Pearson within fold, arithmetic mean of five held-out-gene folds",
        nested_loo=summary.iloc[0].to_dict(), associations=associations,
        selected_alpha_counts={str(k): int(v) for k,v in b.selected_alpha.value_counts().items()},
        residual_labels=b.loc[b.label_flag, ["reporter_slug", "short_name", "prediction_residual"]].to_dict("records"),
        thresholds=threshold_report,
        psmb7=ledger.loc[ledger.short_name.eq("PSMB7"), [RECOVERY, "substitutability_tier"]].to_dict("records"),
        source_sha256={path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                       for path in sorted(INPUT.glob("*.csv"))})
    (ROOT / "source_data/derived_statistics_summary.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
