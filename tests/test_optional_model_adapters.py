import hashlib

import numpy as np
import pytest

from measurement_sufficiency.model_registry import create_model, get_model_spec
from measurement_sufficiency.models import ModelDependencyError, SparseTargets
from measurement_sufficiency.torch_models import TabMProvenance, load_official_tabm


def test_new_optional_adapters_are_registered_as_executable():
    for model_id in ("catboost", "tabm", "resmlp", "multitab"):
        assert get_model_spec(model_id).capability.executable


def test_catboost_adapter_fails_cleanly_when_dependency_is_missing(monkeypatch):
    import builtins

    original_import = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name == "catboost":
            raise ImportError("not installed")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    model = create_model("catboost", task_mode="specialist")
    with pytest.raises(ModelDependencyError, match="optional CatBoost"):
        model.estimator_factory()


def test_tabm_source_wrapper_rejects_wrong_checksum(tmp_path):
    source = tmp_path / "tabm.py"
    source.write_text("__version__ = '0.0.3'\nclass TabM: pass\n")
    wrong = "0" * 64
    assert hashlib.sha256(source.read_bytes()).hexdigest() != wrong
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        load_official_tabm(TabMProvenance(source_path=source, expected_sha256=wrong))


def test_tabm_requires_explicit_provenance():
    pytest.importorskip("torch")
    model = create_model("tabm", task_mode="specialist", epochs=1)
    targets = SparseTargets(("o1",), ("e1",), np.array([[1.0]]), np.array([[True]]))
    with pytest.raises(ValueError, match="explicit provenance"):
        model.fit(np.array([[0.0]]), targets)


def test_study_multitab_runs_with_sparse_endpoint_groups():
    pytest.importorskip("torch")
    model = create_model(
        "multitab",
        task_mode="shared_masked",
        seed=2,
        endpoint_groups={"r": ["e1", "e2"]},
        epochs=1,
        batch_size=2,
        architecture_kwargs={
            "token_dim": 4, "num_heads": 1, "num_blocks": 1,
            "feedforward_dim": 8, "head_width": 4,
        },
    )
    targets = SparseTargets(
        ("o1", "o2"), ("e1", "e2"),
        np.array([[1.0, 2.0], [3.0, np.nan]]),
        np.array([[True, True], [True, False]]),
    )
    model.fit(np.array([[0.0], [1.0]]), targets)
    assert np.isfinite(model.predict(np.array([[0.5]]))).all()
