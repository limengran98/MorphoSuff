#!/usr/bin/env python3
"""One-time frozen-prediction import; public rebuilding uses the compact CSVs.

This import requires the independent audit and archived prediction arrays, and
does not fit any predictor. Refuse re-import to protect the deposited snapshot.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "source_data"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-root", type=Path, required=True)
    parser.add_argument("--mm-root", type=Path, required=True,
                        help="Archive root containing results/, reports/ and the repository")
    args = parser.parse_args()
    if (OUT / "coherent_ensemble_evidence.csv").exists():
        raise RuntimeError("Frozen inputs already exist; do not silently re-import")
    OUT.mkdir(exist_ok=True)
    source = args.analysis_root / "full_checkpoint_average/all_evidence_candidates.csv"
    evidence = pd.read_csv(source).query("method_id == 'coherent_median_ensemble'").copy()
    ledger = pd.read_csv(ROOT / "c_decision_ledger/source_data/figure6c_decision_ledger_source.csv")
    old_b = pd.read_csv(ROOT / "b_nested_loo_audit/figure_source_nested_loo.csv")
    keys = ["reporter_slug", "recoverability_r", "magnitude_spearman", "variance_ratio", "top5pct_recall", "reliability", "tier"]
    frozen = evidence[keys].rename(columns={"recoverability_r":"ensemble_recoverability_r",
        "variance_ratio":"magnitude_variance_ratio", "tier":"expected_tier"})
    fields = ["reporter_slug", "short_name", "biological_category", "true_assay_stability",
        "model_screen_penalty", "domain_sensitive", "scientific_utility_index", "display_order",
        "tier_lane", "tier_lane_x", "go_bp_top10_jaccard", "go_cc_top10_jaccard", "ebi_complex_top10_jaccard"]
    base = ledger[fields].copy()
    base["nested_loo_order"] = base.reporter_slug.map(dict(zip(old_b.reporter_slug, range(52))))
    base["nested_loo_short_name"] = base.reporter_slug.map(dict(zip(old_b.reporter_slug, old_b.short_name)))
    audit = ROOT / "build/coherent_update_20260911"
    audit.mkdir(parents=True, exist_ok=True)
    old_b.to_csv(audit / "legacy_nested_loo_reference.csv", index=False)
    sys.path.insert(0, str(args.analysis_root))
    spec = importlib.util.spec_from_file_location("frozen_prediction_replay", args.analysis_root / "analyze.py")
    replay = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(replay)
    replay.MM = args.mm_root.resolve()
    replay.F2 = ROOT.parent / "figure2/source_data/frozen_figure2_package/B_reporter_ko/source_data"
    roster_path = replay.F2 / "method_roster.csv"
    methods = pd.read_csv(roster_path).sort_values("method_order").method_id.tolist()
    provenance = {str(source): digest(source), str(roster_path): digest(roster_path),
                  str(args.analysis_root / "analyze.py"): digest(args.analysis_root / "analyze.py")}
    fold_rows = []
    for number, reporter in enumerate(base.reporter_slug, 1):
        for fold in range(5):
            arrays = []
            reference_keys = None
            for method in methods:
                gp, sp, pp, yp, names_path, _ = replay.profile_paths(method, fold, reporter, "checkpoint_average")
                for path in [gp, sp, pp, yp, names_path]:
                    provenance[str(path)] = digest(path)
                gene, screen, pred, observed = [np.load(p, allow_pickle=False) for p in [gp, sp, pp, yp]]
                keys = list(zip(gene.tolist(), screen.tolist()))
                assert len(set(keys)) == len(keys)
                if reference_keys is None:
                    reference_keys = keys
                    truth = observed.astype(float)
                    names = names_path.read_text().splitlines()
                assert set(keys) == set(reference_keys)
                assert names == names_path.read_text().splitlines()
                mapping = {k: i for i,k in enumerate(keys)}
                order_index = [mapping[k] for k in reference_keys]
                np.testing.assert_allclose(observed[order_index], truth, rtol=2e-6, atol=2e-7)
                arrays.append(pred[order_index].astype(float))
            # Positive endpoint scaling commutes with median and cancels from
            # endpoint Pearson, so common control-scale factors are unnecessary.
            value = replay.macro_r(truth, np.median(np.stack(arrays), axis=0))
            fold_rows.append(dict(reporter_slug=reporter, fold=fold,
                ensemble_recoverability_r=value, n_gene_screen_pairs=len(reference_keys),
                n_endpoints=truth.shape[1], n_methods=len(methods), evaluation_state="checkpoint_average"))
        if number % 10 == 0:
            print(f"Frozen coherent folds: {number}/52 reporters", flush=True)
    folds = pd.DataFrame(fold_rows)
    mean = folds.groupby("reporter_slug").ensemble_recoverability_r.mean()
    error = float(np.max(np.abs(mean.loc[frozen.reporter_slug].to_numpy()-frozen.ensemble_recoverability_r)))
    assert error < 5e-12, error
    frozen.to_csv(OUT / "coherent_ensemble_evidence.csv", index=False)
    folds.to_csv(OUT / "coherent_ensemble_fold_recoverability.csv", index=False)
    base.to_csv(OUT / "response_replacement_utility_and_annotations.csv", index=False)
    contract = dict(estimator="elementwise median of ten aligned checkpoint-average model predictions at gene×screen level",
        predictor_state="checkpoint_average (parameter-averaged checkpoints) for six neural methods; fixed predictions for four specialist methods",
        recovery_aggregation="arithmetic fivefold mean of macro-endpoint Pearson on held-out gene×screen profiles",
        fidelity_aggregation="common control scale; median before equal-screen gene averaging; 1000 genes",
        utility_components="unchanged top-5% recall and GO BP/GO CC/complex top-10 Jaccard",
        fold_recovery_max_error_vs_independent_analysis=error, n_reporters=52, n_folds=260,
        method_order=methods, archive_root_parameter="--mm-root",
        origin_sources_sha256={Path(p).resolve().relative_to(args.mm_root.resolve()).as_posix(): value
                               for p, value in provenance.items()},
        released_inputs_sha256={p.name:digest(p) for p in sorted(OUT.glob("*.csv"))})
    (OUT / "coherent_source_contract.json").write_text(json.dumps(contract, indent=2)+"\n")
    print(json.dumps({k:v for k,v in contract.items() if "sha256" not in k}, indent=2))


if __name__ == "__main__":
    main()
