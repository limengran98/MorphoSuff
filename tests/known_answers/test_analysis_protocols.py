import pandas as pd
import pytest

from measurement_sufficiency.analysis.conditional_ambiguity import conditional_ambiguity
from measurement_sufficiency.analysis.counterfactual import validate_assay_deletion
from measurement_sufficiency.analysis.response_fidelity import (
    response_fidelity,
    top_fraction_recall,
)
from measurement_sufficiency.analysis.tiers import OPSTier, assign_ops_tier
from measurement_sufficiency.consensus import FINAL_TEN_MODELS, consensus_median
from measurement_sufficiency.interventions import covariate_matched_derangement, derange_targets
from measurement_sufficiency.schemas import SchemaError, validate_canonical_tables


def _prediction_rows():
    rows = []
    # g1 truth response=2, prediction response=3; g2 truth response=-4,
    # prediction response=-3.  Both top-|response| and rank are exact.
    for obs, perturbation, true, pred in [("c", "control", 10.0, 5.0), ("a", "g1", 12.0, 8.0), ("b", "g2", 6.0, 2.0)]:
        rows.append({"observation_id": obs, "reporter_id": "r", "endpoint_id": "e", "split_name": "field", "fold": 0, "model_id": "ridge", "y_true": true, "y_pred": pred, "screen_id": "s", "perturbation_id": perturbation, "is_control": perturbation == "control"})
    return pd.DataFrame(rows)


def test_response_fidelity_uses_same_screen_controls_and_top_five_percent_recall():
    output = response_fidelity(_prediction_rows())
    responses = output["responses"].set_index("perturbation_id")
    assert responses.loc["g1", "response_true"] == 2.0
    assert responses.loc["g2", "response_true"] == -4.0

    # Scores are keyed by reporter, not by endpoint: the manuscript defines them on
    # the gene response vector, whose Euclidean norm equals |response| here because
    # this fixture has a single endpoint.
    magnitudes = output["magnitudes"].set_index("perturbation_id")
    assert set(magnitudes.index) == {"g1", "g2"}
    assert magnitudes.loc["g1", "magnitude_true"] == 2.0
    assert magnitudes.loc["g2", "magnitude_true"] == 4.0
    assert len(output["magnitude_spearman"]) == 1

    recall = output["strong_hit_recall"]
    assert recall.top_fraction.iloc[0] == 0.05
    # Two genes at 5% floors to a single-gene cut. The predicted magnitudes are
    # 3.0 and 3.0, an exact tie at the boundary, which the deterministic policy
    # breaks by gene name; the reported tie counts make that visible instead of
    # letting an inflated score hide it.
    assert recall.k.iloc[0] == 1
    assert recall.n_pred_tied_at_boundary.iloc[0] == 2
    assert 0.0 <= recall.top5pct_recall.iloc[0] <= 1.0


def test_top_fraction_recall_cannot_exceed_one_under_ties():
    """The old implementation divided a tie-expanded intersection by the rank count."""
    magnitudes = pd.DataFrame(
        {
            "reporter_id": "r",
            "split_name": "gene",
            "fold": 0,
            "model_id": "ridge",
            "perturbation_id": [f"g{index}" for index in range(10)],
            "magnitude_true": [1.0] * 10,
            "magnitude_pred": [1.0] * 10,
        }
    )
    for policy in ("deterministic", "inclusive"):
        scored = top_fraction_recall(
            magnitudes, top_fraction=0.5, output_column="top50pct_recall", ties=policy
        )
        assert scored.top50pct_recall.iloc[0] == 1.0


def test_top_five_percent_column_name_is_reserved_for_five_percent():
    magnitudes = pd.DataFrame(
        {
            "reporter_id": "r",
            "split_name": "gene",
            "fold": 0,
            "model_id": "ridge",
            "perturbation_id": ["g0", "g1"],
            "magnitude_true": [1.0, 2.0],
            "magnitude_pred": [1.0, 2.0],
        }
    )
    with pytest.raises(ValueError, match="reserved"):
        top_fraction_recall(magnitudes, top_fraction=0.10)


def test_continuous_covariate_derangement_and_nondefault_indices_are_safe():
    frame = pd.DataFrame({"group": ["a"] * 4, "cov": [0.1, 0.2, 0.8, 0.9], "target": [1, 2, 3, 4]}, index=["x", "y", "z", "w"])
    shuffled = covariate_matched_derangement(frame, target_columns=["target"], group_columns=["group"], covariate_column="cov")
    assert sorted(shuffled.target) == [1, 2, 3, 4]
    assert not shuffled.target.eq(frame.target).any()
    direct = derange_targets(frame, target_columns=["target"], group_columns=["group"], seed=7)
    assert not direct.target.eq(frame.target).any()


def test_consensus_rejects_cross_model_truth_disagreement():
    base = _prediction_rows().drop(columns="is_control")
    all_models = pd.concat([base.assign(model_id=model) for model in FINAL_TEN_MODELS], ignore_index=True)
    all_models.loc[all_models.model_id.eq("midas") & all_models.observation_id.eq("a"), "y_true"] = 99.0
    with pytest.raises(ValueError, match="identical y_true"):
        consensus_median(all_models)


def test_canonical_foreign_key_and_pairing_completeness_are_enforced():
    tables = {
        "observations": pd.DataFrame({"observation_id": ["o"], "input_cell_id": ["i"], "target_cell_id": ["t"], "assay_id": ["a"], "perturbation_id": ["p"], "guide_id": [None], "screen_id": ["s"], "well_id": ["w"], "field_id": ["f"], "is_control": [False]}),
        "reporters": pd.DataFrame({"reporter_id": ["r"], "reporter_name": ["r"], "biological_system": ["x"], "target_modality": ["x"], "endpoint_schema_id": ["schema"]}),
        "assays": pd.DataFrame({"assay_id": ["a"], "reporter_id": ["missing"], "screen_id": ["s"], "endpoint_ids": ["e"]}),
        "pairing": pd.DataFrame({"observation_id": ["o"], "input_cell_id": ["i"], "target_cell_id": ["t"], "assay_id": ["a"], "pairing_key": ["k"], "pairing_confidence": [1.0]}),
    }
    with pytest.raises(SchemaError, match="unknown reporter"):
        validate_canonical_tables(tables)


def test_strict_ops_tier_rule_and_proxy_leakage_guard():
    evidence = {"recoverability_r": .70, "magnitude_spearman": .70, "variance_ratio": .5, "top5pct_recall": .6, "reliability": .3}
    assert assign_ops_tier(evidence) is OPSTier.QUANTITATIVE_PROXY
    with pytest.raises(ValueError, match="re-enters"):
        validate_assay_deletion(deleted_assay_id="a", feature_provenance=pd.DataFrame({"assay_id": ["a"], "feature_id": ["f"]}))


def test_conditional_ambiguity_is_not_prediction_error():
    values = pd.DataFrame({"reporter_id": ["r"] * 4, "endpoint_id": ["e"] * 4, "condition": ["a", "a", "b", "b"], "value": [0., 2., 8., 10.]})
    out = conditional_ambiguity(values, condition_columns=["condition"])
    assert out.ambiguity.iloc[0] > 0


def test_magnitude_reports_when_one_endpoint_dominates_the_norm() -> None:
    """A Euclidean norm is only meaningful over comparable coordinates.

    Unequal vector *length* was already refused. Unequal *scale* is the other way the
    norm stops meaning what its name says, and it is silent: an endpoint with a much
    larger spread than the rest turns a 823-coordinate norm into that one number
    wearing the costume of a profile. Measured on the PERISCOPE anti-TOMM20 block,
    ``Cells_Intensity_IntegratedIntensity_Mito`` alone holds 86% of the squared spread.
    Every row therefore carries the share, so the condition cannot go unseen again.
    """
    import numpy as np
    import pandas as pd

    from measurement_sufficiency.analysis.response_fidelity import response_magnitudes

    rows = []
    for gene, size in (("g1", 1.0), ("g2", 2.0), ("g3", 3.0)):
        # e_big has a spread a hundred times the others, so it owns the norm.
        for endpoint, scale in (("e_big", 100.0), ("e_small_a", 1.0), ("e_small_b", 1.0)):
            rows.append({
                "reporter_id": "r", "split_name": "gene", "fold": 0, "model_id": "ridge",
                "perturbation_id": gene, "endpoint_id": endpoint, "screen_id": "s",
                "response_true": size * scale, "response_pred": size * scale * 0.5,
            })
    responses = pd.DataFrame(rows)

    raw = response_magnitudes(responses)
    assert "max_endpoint_scale_share" in raw.columns
    assert raw["max_endpoint_scale_share"].iloc[0] > 0.99, "the diagnostic must expose the collapse"

    # With the coordinates put on a common footing the three endpoints contribute
    # evenly, so the norm describes the profile rather than its largest coordinate.
    standardized = response_magnitudes(responses, standardize_endpoints="observed_sd")
    assert standardized["n_endpoints"].iloc[0] == 3
    assert not np.isclose(
        standardized["magnitude_true"].iloc[0], raw["magnitude_true"].iloc[0]
    ), "standardising must actually change the norm on a scale-imbalanced block"


def test_magnitude_default_is_the_unstandardised_published_unit() -> None:
    """The default must not silently restate published numbers in new units."""
    import pandas as pd

    from measurement_sufficiency.analysis.response_fidelity import response_magnitudes

    responses = pd.DataFrame([
        {"reporter_id": "r", "split_name": "gene", "fold": 0, "model_id": "ridge",
         "perturbation_id": gene, "endpoint_id": endpoint, "screen_id": "s",
         "response_true": value, "response_pred": value}
        for gene, values in (("g1", (3.0, 4.0)), ("g2", (6.0, 8.0)))
        for endpoint, value in zip(("e1", "e2"), values)
    ])
    magnitudes = response_magnitudes(responses).set_index("perturbation_id")
    # 3-4-5 and 6-8-10, in the raw units.
    assert magnitudes.loc["g1", "magnitude_true"] == 5.0
    assert magnitudes.loc["g2", "magnitude_true"] == 10.0


def test_standardising_refuses_to_divide_by_a_zero_spread_endpoint() -> None:
    """An endpoint with no observed spread says nothing about which gene responded."""
    import pandas as pd

    from measurement_sufficiency.analysis.response_fidelity import response_magnitudes

    responses = pd.DataFrame([
        {"reporter_id": "r", "split_name": "gene", "fold": 0, "model_id": "ridge",
         "perturbation_id": gene, "endpoint_id": endpoint, "screen_id": "s",
         "response_true": value, "response_pred": value}
        for gene, values in (("g1", (1.0, 7.0)), ("g2", (2.0, 7.0)))
        for endpoint, value in zip(("varies", "flat"), values)
    ])
    standardized = response_magnitudes(responses, standardize_endpoints="observed_sd")
    assert standardized["n_endpoints"].unique().tolist() == [1], "the flat endpoint must be dropped"


def test_standardize_endpoints_rejects_an_unknown_mode() -> None:
    import pandas as pd
    import pytest

    from measurement_sufficiency.analysis.response_fidelity import response_magnitudes

    with pytest.raises(ValueError, match="standardize_endpoints"):
        response_magnitudes(pd.DataFrame(), standardize_endpoints="zscore")


def test_a_barely_varying_endpoint_cannot_take_over_the_standardised_norm() -> None:
    """The mirror image of the scale-collapse guard, and it bit before it was written.

    Standardising divides by the observed spread, so an endpoint that barely varies
    becomes the loudest coordinate in the norm: the divisor is small, not the signal.
    On the single-cell PERISCOPE fit, ``RadialCV_mito_tubeness_8of20`` had an observed
    spread of 4.3e-06 across one fold's 4,078 genes and a predicted spread of 1.5e-04,
    both negligible in absolute terms. Their ratio is 35, that endpoint took 63% of the
    standardised squared norm, and the fold reported predicted response magnitudes 46
    times more variable than the observed ones. Dropping only an exactly zero spread
    does not catch it, because 4.3e-06 is not zero.
    """
    import numpy as np
    import pandas as pd

    from measurement_sufficiency.analysis.response_fidelity import response_magnitudes

    rows = []
    for index, gene in enumerate(("g1", "g2", "g3", "g4")):
        for endpoint, true, pred in (
            # Two ordinary endpoints, and one that is constant to five decimal places
            # while the model still predicts a spread a thousand times its own.
            ("e_a", 1.0 + index, 0.9 * (1.0 + index)),
            ("e_b", 2.0 - index, 0.9 * (2.0 - index)),
            ("e_degenerate", 1e-6 * index, 1e-3 * index),
        ):
            rows.append({
                "reporter_id": "r", "split_name": "gene", "fold": 0, "model_id": "ridge",
                "perturbation_id": gene, "endpoint_id": endpoint, "screen_id": "s",
                "response_true": true, "response_pred": pred,
            })
    responses = pd.DataFrame(rows)

    unguarded = response_magnitudes(
        responses, standardize_endpoints="observed_sd", min_spread_ratio=0.0
    )
    assert unguarded["n_endpoints"].iloc[0] == 3
    assert unguarded["max_standardized_share"].iloc[0] > 0.9, (
        "without the guard the degenerate endpoint must be shown to own the norm; "
        "if this ever drops, the fixture no longer reproduces the defect"
    )
    ratio_unguarded = (
        unguarded["magnitude_pred"].std() / unguarded["magnitude_true"].std()
    )

    guarded = response_magnitudes(responses, standardize_endpoints="observed_sd")
    assert guarded["n_endpoints"].iloc[0] == 2, "the degenerate endpoint must be dropped"
    assert guarded["max_standardized_share"].iloc[0] < 0.9
    assert (
        guarded["magnitude_pred"].std() / guarded["magnitude_true"].std()
    ) < ratio_unguarded / 10, (
        "dropping it must remove the inflation, not merely relabel it"
    )


def test_the_standardised_share_is_not_reported_when_nothing_was_standardised() -> None:
    """A diagnostic that reports a number for a step that did not run is worse than none."""
    import numpy as np
    import pandas as pd

    from measurement_sufficiency.analysis.response_fidelity import response_magnitudes

    rows = [
        {"reporter_id": "r", "split_name": "gene", "fold": 0, "model_id": "ridge",
         "perturbation_id": gene, "endpoint_id": endpoint, "screen_id": "s",
         "response_true": value, "response_pred": value * 0.5}
        for gene, value in (("g1", 1.0), ("g2", 2.0))
        for endpoint in ("e_a", "e_b")
    ]
    magnitudes = response_magnitudes(pd.DataFrame(rows))
    assert magnitudes["max_standardized_share"].isna().all()
