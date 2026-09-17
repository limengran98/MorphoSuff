#!/usr/bin/env python3
"""Replay deposited OASIS scores without fitting or modifying frozen outputs."""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd

import frozen_scoring as scorer

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def read(name):
    return pd.read_csv(DATA / name, float_precision="round_trip")


def compare(actual, expected, keys):
    a = actual.sort_values(keys).reset_index(drop=True)
    b = expected.sort_values(keys).reset_index(drop=True)
    assert a.shape == b.shape and list(a.columns) == list(b.columns)
    for field in a:
        if pd.api.types.is_numeric_dtype(a[field]) and a[field].dtype != bool:
            np.testing.assert_allclose(a[field], b[field], atol=2e-11, rtol=2e-11,
                                       equal_nan=True, err_msg=field)
        else:
            assert a[field].fillna("").equals(b[field].fillna("")), field


def main():
    provenance = json.loads((ROOT / "provenance.json").read_text())
    for name, expected in provenance["prior_independent_audit"]["result_hashes"].items():
        path = DATA / name
        if path.exists():
            assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, name
    assert hashlib.sha256((ROOT / "code/frozen_scoring.py").read_bytes()).hexdigest() == provenance["original_scoring_sha256"]
    partition = read("compound_partition.csv")
    train, test = read("training_membership.csv"), read("test_membership.csv")
    development = set(partition.loc[partition.fold.ge(0), "compound"])
    heldout = set(partition.loc[partition.fold.eq(-1), "compound"])
    assert len(development) == 868 and len(heldout) == 217 and not development & heldout
    assert set(train.Metadata_Compound) == development and set(test.Metadata_Compound) == heldout
    assert len(train) == 13157 and len(test) == 3300
    predictions = pd.read_parquet(DATA / "heldout_predictions.parquet")
    assert len(predictions) == 19800
    context = read("context_compound_scores.csv")
    compare(scorer.contexts(predictions), context, ["context", "compound", "model"])
    # The frozen CSV doubles define tie behavior. Tiny aggregation-order errors
    # do not replace them in replayed near-tied diagnostic predictions.
    with contextlib.redirect_stdout(io.StringIO()):
        scores, selections = scorer.evaluate(context)
        null, distribution = scorer.correspondence_null(predictions, context, scores)
    compare(scores, read("utility_scores.csv"), ["context", "model", "fraction"])
    compare(selections, read("selection_ledger.csv"), ["context", "model", "fraction", "compound"])
    compare(null, read("correspondence_null_summary.csv"), ["model", "fraction"])
    compare(distribution, read("correspondence_null_distribution.csv.gz"), ["model", "fraction", "iteration"])
    print(json.dumps({"status": "PASS", "training_performed": False,
                      "prediction_rows": len(predictions), "compound_context_rows": len(context),
                      "utility_rows": len(scores), "selection_rows": len(selections),
                      "null_rows": len(distribution), "bootstrap_per_model_context": scorer.REPEATS}, indent=2))


if __name__ == "__main__":
    main()
