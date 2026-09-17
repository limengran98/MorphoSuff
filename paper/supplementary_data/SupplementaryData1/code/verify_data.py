#!/usr/bin/env python3
"""Independent table-level scientific QA and complete package file manifest."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
METRICS = ["recoverability_r", "magnitude_spearman", "variance_ratio", "top5pct_recall", "reliability"]


def main():
    evidence = pd.read_csv(ROOT / "evidence.csv")
    spec = json.loads((ROOT / "analysis_spec.json").read_text())
    assert len(evidence) == 1144
    assert not evidence.duplicated(["run_state", "method_id", "reporter_slug"]).any()
    assert evidence[METRICS].notna().all().all() and evidence.n_genes.eq(1000).all()
    # Implement precedence independently, in reverse update order.
    expected = pd.Series("unresolved", index=evidence.index)
    expected[evidence.recoverability_r.lt(.60)] = "measurement_required"
    ranking = evidence.magnitude_spearman.ge(.70) & evidence.top5pct_recall.ge(.50)
    expected[ranking] = "ranking_proxy"
    quantitative = (evidence.recoverability_r.ge(.70) & evidence.magnitude_spearman.ge(.70)
                    & evidence.variance_ratio.between(.50, 1.50) & evidence.top5pct_recall.ge(.60))
    expected[quantitative] = "quantitative_proxy"
    expected[evidence.reliability.isna() | evidence.reliability.lt(.30)] = "not_identifiable"
    assert expected.equals(evidence.tier)
    ref = evidence[evidence.run_state.eq("checkpoint_average") & evidence.method_id.eq("coherent_median_ensemble")].set_index("reporter_slug")
    assert ref.reliability.ge(.3).sum() == 43 and ref.reliability.lt(.3).sum() == 9
    assert evidence.groupby("reporter_slug").reliability.nunique().eq(1).all()
    changes = pd.read_csv(ROOT / "derived/changes_vs_reference.csv")
    assert len(changes) == len(evidence)
    assert changes.reference_state.eq("checkpoint_average").all()
    assert changes.reference_method_id.eq("coherent_median_ensemble").all()
    indexed = evidence.set_index(["run_state", "reporter_slug", "method_id"])
    max_delta_error = 0.0
    for change in changes.itertuples(index=False):
        row = indexed.loc[(change.run_state, change.reporter_slug, change.method_id)]
        baseline = ref.loc[change.reporter_slug]
        assert change.reference_tier == baseline.tier and change.tier == row.tier
        for metric in METRICS:
            error = abs(float(getattr(change, "delta_"+metric)) - (float(row[metric])-float(baseline[metric])))
            max_delta_error = max(max_delta_error, error)
            assert error < 1e-12
        flags = [name for name in evidence.columns if name.startswith("pass_")]
        expected_changed = {name.removeprefix("pass_") for name in flags if row[name] != baseline[name]}
        observed_changed = set(str(change.changed_gate_flags).split(";")) if pd.notna(change.changed_gate_flags) else set()
        assert expected_changed == observed_changed
        assert len(expected_changed) == change.n_changed_gate_flags
    expected_summary = {"checkpoint_average": [20, 6, 17], "single": [22, 6, 15]}
    for state, counts in expected_summary.items():
        frame = evidence[evidence.run_state.eq(state) & evidence.method_id.isin(spec["method_ids"]) & evidence.reliability.ge(.3)]
        n = frame.tier.isin(["quantitative_proxy", "ranking_proxy"]).groupby(frame.reporter_slug).sum()
        assert len(n) == 43 and [int(n.eq(10).sum()), int(n.eq(0).sum()), int(n.between(1,9).sum())] == counts
    scales = pd.read_csv(ROOT / "provenance/endpoint_scales.csv")
    assert len(scales) == 5*1604 and scales.common_control_scale.gt(0).all()
    assert np.allclose(scales.fold_target_scale/scales.common_control_scale, scales.fold_to_common_factor, rtol=1e-12, atol=1e-12)
    folds = pd.read_csv(ROOT / "provenance/per_method_fold_recoverability.csv")
    assert len(folds) == 2*52*10*5
    means = folds.groupby(["run_state", "reporter_slug", "method_id"]).recoverability_r.mean()
    singles = indexed[indexed.index.get_level_values("method_id").isin(spec["method_ids"])]
    assert np.allclose(means.sort_index(), singles.recoverability_r.sort_index(), atol=1e-12, rtol=0)
    scale_sensitivity = pd.read_csv(ROOT / "scale_sensitivity.csv")
    assert len(scale_sensitivity) == 104
    assert scale_sensitivity.prediction_state.eq("checkpoint_average").all()
    primary_scale = scale_sensitivity[scale_sensitivity.response_scale.eq("common_control")].set_index("reporter_slug")
    assert np.allclose(primary_scale[METRICS].sort_index(), ref[METRICS].sort_index(), atol=1e-10, rtol=0)
    scale_matrix = scale_sensitivity.pivot(index="reporter_slug", columns="response_scale", values="tier")
    assert scale_matrix.index[scale_matrix.common_control.ne(scale_matrix.fold_standardized)].tolist() == ["er_ncln"]
    report = {"status": "PASS", "scope": "Independent complete-table rule/reference/aggregation QA; no training",
              "n_evidence_rows": len(evidence), "n_reporters": 52, "n_reliability_passing": 43,
              "primary_state": "checkpoint_average", "reference": "checkpoint_average/coherent_median_ensemble",
              "max_delta_error": max_delta_error, "stability_counts_all_proxy_none_crossing": expected_summary,
              "per_method_fold_rows": len(folds), "endpoint_scale_rows": len(scales),
              "scale_sensitivity_rows": 104, "scale_only_changed_reporter": "er_ncln",
              "all_method_recoveries_reproduced_by_mean_of_five_folds": True,
              "required_prediction_archive_included": False}
    (ROOT / "qa_report.json").write_text(json.dumps(report, indent=2) + "\n")
    files = [p for p in ROOT.rglob("*") if p.is_file() and p.name != "manifest_sha256.csv" and "__pycache__" not in p.parts]
    manifest = pd.DataFrame([dict(path=p.relative_to(ROOT).as_posix(), bytes=p.stat().st_size,
                                  sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(files)])
    manifest.to_csv(ROOT / "manifest_sha256.csv", index=False)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
