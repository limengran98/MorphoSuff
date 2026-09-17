import pandas as pd
import pytest

from measurement_sufficiency.splits import LeakageError, check_split_leakage, make_field_splits, make_gene_splits, make_whole_screen_splits


def observations():
    return pd.DataFrame({"observation_id": [f"o{i}" for i in range(8)], "field_id": ["f1", "f1", "f2", "f2", "f3", "f3", "f4", "f4"], "perturbation_id": ["g1", "g1", "g2", "g2", "g3", "g3", "g4", "g4"], "screen_id": ["s1", "s1", "s2", "s2", "s3", "s3", "s4", "s4"]})


@pytest.mark.parametrize("builder,column", [(make_field_splits, "field_id"), (make_gene_splits, "perturbation_id"), (make_whole_screen_splits, "screen_id")])
def test_group_splits_are_leak_free(builder, column):
    data = observations()
    manifest = builder(data, n_folds=2, seed=3)
    manifest.validate(data)
    for _, assignments in manifest.assignments.groupby("fold"):
        joined = assignments.merge(data, on="observation_id")
        assert not set(joined.loc[joined.role == "train", column]) & set(joined.loc[joined.role == "test", column])


def test_leakage_is_rejected():
    data = observations()
    assignments = pd.DataFrame({"observation_id": ["o0", "o1"], "fold": [0, 0], "role": ["train", "test"]})
    with pytest.raises(LeakageError, match="leakage"):
        check_split_leakage(data, assignments, "field_id")
