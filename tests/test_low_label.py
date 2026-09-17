import pandas as pd
import pytest

from measurement_sufficiency.low_label import LowLabelManifestError, build_low_label_manifest


def _assignments():
    return pd.DataFrame(
        {
            "observation_id": ["a", "b", "c", "d", "e"],
            "fold": [0, 0, 0, 0, 0],
            "role": ["train", "train", "train", "train", "test"],
        }
    )


def test_low_label_manifest_has_exact_nested_budgets_and_anchor():
    first = build_low_label_manifest(_assignments(), fold=0, fractions=[0.5, 1.0], seed=12)
    second = build_low_label_manifest(_assignments().sample(frac=1, random_state=8), fold=0, fractions=[1.0, 0.5], seed=12)
    assert len(first.selected_ids(0.5)) == 2
    assert first.selected_ids(0.5).issubset(first.selected_ids(1.0))
    assert first.selected_ids(1.0) == {"a", "b", "c", "d"}
    assert first.selected_ids(0.5) == second.selected_ids(0.5)


def test_low_label_rejects_duplicate_roles():
    bad = pd.concat([_assignments(), _assignments().iloc[[0]]], ignore_index=True)
    with pytest.raises(LowLabelManifestError, match="one role"):
        build_low_label_manifest(bad, fold=0, fractions=[0.5])
