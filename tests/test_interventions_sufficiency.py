import numpy as np
import pandas as pd
import pytest

from measurement_sufficiency.adapters import AdapterCapabilities, EligibilityError
from measurement_sufficiency.interventions import cross_fitted_residuals, derange_targets, select_features, validate_exact_pairing
from measurement_sufficiency.schemas import SchemaError
from measurement_sufficiency.sufficiency import SufficiencyTier, TierThresholds, assign_tier


def test_pairing_capability_and_derangement():
    pairing = pd.DataFrame({"observation_id": ["o1", "o2"], "input_cell_id": ["i1", "i2"], "target_cell_id": ["t1", "t2"], "assay_id": ["a1", "a1"], "pairing_key": ["a", "b"], "pairing_confidence": [1.0, 1.0]})
    with pytest.raises(EligibilityError):
        validate_exact_pairing(pairing, AdapterCapabilities())
    validate_exact_pairing(pairing, AdapterCapabilities(exact_pairing=True))
    source = pd.DataFrame({"group": ["x", "x", "x"], "y": [1, 2, 3]})
    shuffled = derange_targets(source, target_columns=["y"], group_columns=["group"], seed=1)
    assert sorted(shuffled.y) == [1, 2, 3]
    assert not shuffled.y.eq(source.y).any()


def test_exact_pairing_allows_reused_physical_cells_across_assays_only():
    pairing = pd.DataFrame(
        {
            "observation_id": ["o1", "o2"],
            "input_cell_id": ["i1", "i1"],
            "target_cell_id": ["t1", "t1"],
            "assay_id": ["a1", "a2"],
            "pairing_key": ["cell", "cell"],
            "pairing_confidence": [1.0, 1.0],
        }
    )
    validate_exact_pairing(pairing, AdapterCapabilities(exact_pairing=True))
    pairing.loc[1, "assay_id"] = "a1"
    with pytest.raises(SchemaError, match="within each assay"):
        validate_exact_pairing(pairing, AdapterCapabilities(exact_pairing=True))


def test_feature_and_residual_interventions():
    frame = pd.DataFrame({"cell_size": [1, 2], "texture": [3, 4], "cell_shape": [5, 6]})
    assert select_features(frame, list(frame), mode="size_shape_only").columns.tolist() == ["cell_size", "cell_shape"]
    residual = cross_fitted_residuals(pd.Series([1., 2., 5., 6.]), pd.DataFrame({"x": [0., 1., 0., 1.]}), [0, 0, 1, 1])
    assert np.isfinite(residual).all()


def test_tier_rules_keep_missing_evidence_unresolved():
    full = {"recoverability_r": .9, "ranking_spearman": .9, "reliability": .8, "amplitude_ratio": .8, "environment_drop": .1, "ambiguity": .1}
    # The portable rule has no default calibration: a caller must declare one.
    thresholds = TierThresholds(
        quantitative_r=0.8,
        ranking_spearman=0.7,
        minimum_reliability=0.5,
        minimum_amplitude_ratio=0.5,
        maximum_environment_drop=0.2,
        maximum_ambiguity=0.25,
    )
    assert assign_tier(full, thresholds) is SufficiencyTier.QUANTITATIVE_PROXY
    full["recoverability_r"] = .6
    assert assign_tier(full, thresholds) is SufficiencyTier.RANKING_PROXY
    assert assign_tier({"recoverability_r": .9}, thresholds) is SufficiencyTier.UNRESOLVED
    assert assign_tier({"identifiable": False}, thresholds) is SufficiencyTier.NOT_IDENTIFIABLE
