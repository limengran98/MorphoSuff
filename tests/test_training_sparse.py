import numpy as np
import pandas as pd
import pytest

from measurement_sufficiency.model_registry import (
    ModelEligibilityError,
    create_model,
    eligible_models,
    get_model_spec,
)
from measurement_sufficiency.models import SparseTargets, observed_only_mse
from measurement_sufficiency.training import fit_sparse_model, prediction_long_table


def _inputs():
    return pd.DataFrame(
        {
            "observation_id": ["o1", "o2", "o3", "o4"],
            "x": [0.0, 1.0, 2.0, 3.0],
            "screen_id": ["s", "s", "s", "s"],
            "perturbation_id": ["p1", "p1", "p2", "p2"],
            "reporter_id": ["r", "r", "r", "r"],
        }
    )


def _targets():
    return pd.DataFrame(
        {
            "observation_id": ["o1", "o1", "o2", "o2", "o3", "o4"],
            "endpoint_id": ["a", "b", "a", "b", "a", "a"],
            "y_true": [0.0, 10.0, 1.0, 11.0, 2.0, 3.0],
            "is_observed": [True, True, True, False, True, True],
        }
    )


def test_sparse_loss_ignores_unobserved_values():
    truth = np.array([[1.0, 999.0]])
    prediction = np.array([[2.0, -999.0]])
    assert observed_only_mse(truth, prediction, np.array([[True, False]])) == 1.0


def test_specialist_fit_and_long_export_excludes_unobserved_endpoint():
    fitted = fit_sparse_model(_inputs(), _targets(), feature_columns=["x"], model_id="ridge", seed=1)
    exported = prediction_long_table(fitted, _inputs(), metadata=_inputs(), split_name="field", fold=0, targets=_targets())
    assert set(exported.endpoint_id) == {"a", "b"}
    assert not ((exported.observation_id == "o2") & (exported.endpoint_id == "b")).any()
    assert exported.y_true.notna().all()


def test_shared_protocol_can_fit_heterogeneous_mask():
    fitted = fit_sparse_model(_inputs(), _targets(), feature_columns=["x"], task_mode="shared_masked")
    assert fitted.predict_matrix(_inputs()).shape == (4, 2)


def test_study_runner_model_routes_away_from_generic_factory():
    capability = get_model_spec("scbutterfly").capability
    assert capability.executable
    assert capability.study_runner_executable
    assert not capability.generic_factory_executable
    with pytest.raises(ModelEligibilityError, match="runnable through the included OPS study-runner"):
        create_model("scbutterfly", task_mode="specialist")


def test_all_frozen_ten_methods_have_a_released_execution_route():
    expected = {
        "ridge", "gbdt", "catboost", "mlp", "tabm", "scbutterfly",
        "resmlp", "multitab", "midas", "scpair",
    }
    assert {spec.model_id for spec in eligible_models(executable_only=True)} == expected


def test_generic_resmlp_is_an_executable_observed_only_shared_adapter():
    pytest.importorskip("torch")
    fitted = fit_sparse_model(
        _inputs(),
        _targets(),
        feature_columns=["x"],
        endpoint_ids=["a", "b"],
        model_id="resmlp",
        task_mode="shared_masked",
        seed=3,
        model_kwargs={
            "endpoint_groups": {"reporter": ["a", "b"]},
            "epochs": 2,
            "batch_size": 4,
            "architecture_kwargs": {
                "shared_width": 8,
                "expansion_width": 16,
                "residual_blocks": 1,
                "head_width": 4,
            },
        },
    )
    assert fitted.predict_matrix(_inputs()).shape == (4, 2)
    assert np.isfinite(fitted.predict_matrix(_inputs())).all()
