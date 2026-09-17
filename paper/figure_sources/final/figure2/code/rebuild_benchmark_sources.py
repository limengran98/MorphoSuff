#!/usr/bin/env python3
"""Rebuild Figure 2 from portable frozen fold/direction scores; never fit models.

The original deposition used a groupby median despite its arithmetic-mean
Methods contract.  The corrected reporter score is the equal arithmetic mean
over five predefined folds or eligible destination-screen directions.  Physical
screen aggregation remains the median over independently scored assay members.

Normal use needs only this figure package.  --freeze-input is a deliberate
author-approved import of an existing metric audit, not part of normal rebuilding.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import fig2_data as D

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "source_data"
FROZEN = SOURCE / "frozen_partition_scores.csv.gz"
AUDIT = SOURCE / "aggregation_correction"
KEYS = ["method_id", "split", "reporter_slug"]
METRICS = ["gene_macro_feature_pearson", "cell_macro_feature_pearson"]
FIELD, GENE, SCREEN = "field_holdout_sanity", "gene_holdout_main", "strict_whole_screen"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def table_path(panel: str) -> Path:
    folder, filename, _, _ = D.PANELS[panel]
    return D.PKG / folder / "source_data" / filename


def validate_partitions(frame: pd.DataFrame) -> None:
    methods = set(D.roster().method_id)
    assert set(frame.method_id) == methods
    assert len(frame) == 6010, "expected 5200 fold + 810 directed evaluation rows"
    assert np.isfinite(frame[METRICS].to_numpy(float)).all()
    for split in (FIELD, GENE):
        part = frame[frame.split.eq(split)]
        assert len(part) == 2600
        assert not part.duplicated(KEYS + ["fold"]).any()
        assert part.groupby(KEYS).fold.agg(lambda x: set(x) == set(range(5))).all()
        assert part.groupby("method_id").reporter_slug.nunique().eq(52).all()
    strict = frame[frame.split.eq(SCREEN)]
    assert len(strict) == 810 and not strict.duplicated(KEYS + ["destination_screen"]).any()
    assert strict.groupby("method_id").size().eq(81).all()
    assert strict.groupby("method_id").reporter_slug.nunique().eq(34).all()
    assert strict.destination_screen.nunique() == 66
    # All ten predictors must use exactly the same eligible physical directions.
    direction_sets = [set(zip(part.reporter_slug, part.destination_screen))
                      for _, part in strict.groupby("method_id")]
    assert all(keys == direction_sets[0] for keys in direction_sets)


def freeze_input(partition_audit: Path, reporter_audit: Path | None) -> None:
    columns = KEYS + ["fold", "destination_screen", "evaluation_unit"] + METRICS
    frame = pd.read_csv(partition_audit, usecols=columns, float_precision="round_trip")
    frame = frame[frame.method_id.isin(D.roster().method_id)].copy()
    frame = frame.sort_values(KEYS + ["fold", "destination_screen"], kind="stable")
    validate_partitions(frame)
    summary = frame.groupby(KEYS)[METRICS].mean()
    source_records = [{"original_filename": partition_audit.name,
                       "sha256": digest(partition_audit), "bytes": partition_audit.stat().st_size}]
    oracle_max_error = None
    if reporter_audit is not None:
        oracle = pd.read_csv(reporter_audit, float_precision="round_trip")
        oracle = oracle[oracle.method_id.isin(D.roster().method_id)].set_index(KEYS)
        error = (summary - oracle.loc[summary.index, METRICS]).abs()
        oracle_max_error = float(error.to_numpy().max())
        assert oracle_max_error < 1e-12
        source_records.append({"original_filename": reporter_audit.name,
                               "sha256": digest(reporter_audit), "bytes": reporter_audit.stat().st_size})
    frame.to_csv(FROZEN, index=False, compression={"method": "gzip", "mtime": 0})
    write_json(SOURCE / "partition_source_provenance.json", {
        "scope": "ten published predictors; frozen field/gene folds and strict destination directions",
        "n_rows": len(frame), "frozen_table": FROZEN.name,
        "frozen_table_sha256": digest(FROZEN), "upstream_sources": source_records,
        "reporter_arithmetic_mean_oracle_max_abs_error": oracle_max_error,
        "runtime_requires_original_private_paths": False,
        "model_training_or_prediction_recalculation": False,
        "root_cause": "legacy reporter_scores grouped partition metrics with median although its comment and Methods specified averaging",
        "legacy_code_reference": "ops_manuscript_benchmark_three_units_v2/code/build.py:246",
        "scope_boundary": "independent KO assay/screen sources retained; no new common-scale profile calculation",
    })


def replace_metric(frame: pd.DataFrame, score: str, values: pd.Series, keys: list[str]) -> pd.DataFrame:
    out = frame.copy()
    index = pd.MultiIndex.from_frame(out[keys])
    assert values.index.is_unique and index.isin(values.index).all()
    out[score] = values.reindex(index).to_numpy(float)
    return out


def rebuild() -> dict:
    provenance = json.loads((SOURCE / "partition_source_provenance.json").read_text())
    assert digest(FROZEN) == provenance["frozen_table_sha256"]
    frame = pd.read_csv(FROZEN, float_precision="round_trip")
    validate_partitions(frame)
    mean = frame.groupby(KEYS)[METRICS].mean()
    median = frame.groupby(KEYS)[METRICS].median()
    counts = frame.groupby(KEYS).size()
    corrections = []
    for metric in METRICS:
        part = pd.DataFrame({"metric": metric, "legacy_partition_median": median[metric],
                             "corrected_partition_mean": mean[metric], "n_partitions": counts})
        part["mean_minus_legacy_median"] = part.corrected_partition_mean - part.legacy_partition_median
        corrections.append(part.reset_index())
    corrections = pd.concat(corrections, ignore_index=True)
    AUDIT.mkdir(parents=True, exist_ok=True)
    corrections.to_csv(AUDIT / "reporter_aggregation_corrections.csv", index=False)

    # b/c: the only revised estimator is the reporter-level partition average.
    for panel, metric in zip(("b", "c"), METRICS):
        old = pd.read_csv(table_path(panel), float_precision="round_trip")
        corrected = replace_metric(old, D.PANELS[panel][2], mean[metric], KEYS)
        corrected.to_csv(table_path(panel), index=False)

    # f/d: independent KO assay estimates are not a function of reporter scores.
    ko_assays = pd.read_csv(table_path("f"), float_precision="round_trip")
    ko_screens = pd.read_csv(table_path("d"), float_precision="round_trip")
    screen_keys = ["method_id", "split", "screen_id"]
    expected_ko_screen = ko_assays.groupby(screen_keys).ko_pearson.median()
    actual_ko_screen = ko_screens.set_index(screen_keys).ko_pearson
    np.testing.assert_allclose(actual_ko_screen, expected_ko_screen.loc[actual_ko_screen.index],
                               rtol=0, atol=1e-14)
    direct = frame[frame.split.eq(SCREEN)].rename(columns={"destination_screen": "screen_id"})
    assay_keys = KEYS + ["screen_id"]
    direct = direct.set_index(assay_keys)
    strict_ko = ko_assays[ko_assays.split.eq(SCREEN)].set_index(assay_keys).ko_pearson
    np.testing.assert_allclose(strict_ko, direct.loc[strict_ko.index, METRICS[0]], rtol=0, atol=1e-14)
    preserved = {panel: {"relative_path": table_path(panel).relative_to(ROOT).as_posix(),
                         "sha256": digest(table_path(panel)), "bytes": table_path(panel).stat().st_size}
                 for panel in ("d", "f")}

    # g: field/gene repeat each corrected reporter score over physical membership.
    # Strict destination-cell evaluations remain direct and numerically unchanged.
    cells = pd.read_csv(table_path("g"), float_precision="round_trip")
    strict_cell = cells[cells.split.eq(SCREEN)].set_index(assay_keys).cell_pearson
    np.testing.assert_allclose(strict_cell, direct.loc[strict_cell.index, METRICS[1]], rtol=0, atol=1e-14)
    mask = cells.split.isin([FIELD, GENE])
    corrected_cell = replace_metric(cells.loc[mask], "cell_pearson", mean[METRICS[1]], KEYS)
    cells.loc[mask, "cell_pearson"] = corrected_cell.cell_pearson
    cells.to_csv(table_path("g"), index=False)
    # e: retain the existing within-screen median over member assays.
    cell_screens = pd.read_csv(table_path("e"), float_precision="round_trip")
    corrected_screen = replace_metric(cell_screens, "cell_pearson",
                                      cells.groupby(screen_keys).cell_pearson.median(), screen_keys)
    corrected_screen.to_csv(table_path("e"), index=False)

    inference = D.compute_inference()
    inference.to_csv(D.INFERENCE, index=False)
    descriptive = []
    for panel in ("b", "c"):
        scores = D.load(panel)
        for (test, method), part in scores.groupby(["split_label", "method"], sort=False):
            descriptive.append({"view": panel, "split": test, "method": method,
                                "n_reporters": len(part), "median": part.score.median(),
                                "q25": part.score.quantile(.25), "q75": part.score.quantile(.75),
                                "minimum": part.score.min(), "maximum": part.score.max()})
    descriptive = pd.DataFrame(descriptive)
    descriptive.to_csv(SOURCE / "statistics/benchmark_descriptive_summary.csv", index=False)
    leading = (descriptive.sort_values("median", ascending=False, kind="stable")
               .groupby(["view", "split"], sort=False).head(1))
    leading.to_csv(SOURCE / "statistics/descriptive_leading_medians.csv", index=False)
    spread = []
    for panel in ("d", "e"):
        con = D.consensus(panel)
        con.to_csv(SOURCE / f"statistics/view_{panel}_screen_consensus.csv", index=False)
        for test, part in con.groupby("split_label"):
            spread.append({"view": panel, "split": test, "n_screens": len(part),
                           "across_screen_consensus_iqr": part.consensus.quantile(.75) - part.consensus.quantile(.25),
                           "median_across_method_iqr": (part.q75 - part.q25).median()})
    pd.DataFrame(spread).to_csv(SOURCE / "statistics/screen_spread_summary.csv", index=False)
    for panel in ("f", "g"):
        D.assay_axes(panel).to_csv(SOURCE / f"statistics/view_{panel}_assay_consensus.csv", index=False)
    for panel in ("b", "c"):
        D.consensus(panel).to_csv(SOURCE / f"statistics/view_{panel}_reporter_consensus.csv", index=False)
    summary_rows = []
    for (metric, split), part in corrections.groupby(["metric", "split"]):
        summary_rows.append({"metric": metric, "split": split, "n_reporter_method_rows": len(part),
                             "n_mean_different_from_median": int(part.mean_minus_legacy_median.abs().gt(1e-12).sum()),
                             "max_abs_change": float(part.mean_minus_legacy_median.abs().max())})
    passes = inference[inference.meets_reporter_inference_gate]
    report = {
        "status": "PASS", "n_frozen_partition_rows": len(frame),
        "partition_aggregation": "arithmetic mean; five equal-weight folds or all eligible destination directions",
        "cross_method_consensus": "unweighted median of ten method-specific scores",
        "correction_summary": summary_rows,
        "independent_ko_tables_preserved_byte_exact": preserved,
        "strict_destination_assay_scores_unchanged": True,
        "n_primary_tests": len(inference), "n_passing_holm_and_effect": len(passes),
        "passing_comparisons": passes[["panel", "split", "method", "median_paired_delta", "p_holm_reporter54"]].to_dict("records"),
        "leading_marginal_medians": leading.to_dict("records"),
        "model_training_or_prediction_recalculation": False,
    }
    write_json(AUDIT / "aggregation_rebuild_audit.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-input", type=Path, help="author-approved original partition metric audit")
    parser.add_argument("--reporter-audit", type=Path, help="independent arithmetic-mean oracle during import")
    args = parser.parse_args()
    if args.freeze_input:
        freeze_input(args.freeze_input, args.reporter_audit)
    report = rebuild()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
