from __future__ import annotations

import importlib
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pytest


torch = pytest.importorskip("torch")

from measurement_sufficiency.ops_models import (  # noqa: E402
    IndependentMLPSpecialists,
    MultiTab,
    SharedResMLP,
)
from measurement_sufficiency.ops_training import (  # noqa: E402
    ReporterBalancedSampler,
    ReporterBatch,
    atomic_training_checkpoint,
    capture_rng_state,
    endpoint_balanced_mse,
    fit_independent_mlp,
    load_training_checkpoint,
    reporter_balanced_mse,
)
from measurement_sufficiency.model_registry import create_ops_model  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = (
    ROOT
    / "studies"
    / "ops"
    / "runners"
    / "low_label"
    / "legacy_exact"
    / "scripts"
)


def _historical_modules():
    assert SCRIPTS.is_dir(), "repository-owned frozen study source is missing"
    value = str(SCRIPTS)
    if value not in sys.path:
        sys.path.insert(0, value)
    candidate = importlib.import_module("ops_reporter_candidate_models")
    multitask = importlib.import_module("ops_reporter_masked_multitask_v2_lib")
    return candidate, multitask


def _dimensions():
    return OrderedDict(
        (("reporter.with.dot", 2), ("r2", 3), ("r3", 1), ("r4", 2), ("r5", 1))
    )


def _assert_exact_model_parity(historical_class, migrated_class, kwargs):
    dimensions = _dimensions()
    torch.manual_seed(41)
    historical = historical_class(dimensions, input_dim=4, **kwargs)
    torch.manual_seed(41)
    migrated = migrated_class(dimensions, input_dim=4, **kwargs)
    assert tuple(historical.state_dict()) == tuple(migrated.state_dict())
    for name, value in historical.state_dict().items():
        assert torch.equal(value, migrated.state_dict()[name]), name
    historical.eval()
    migrated.eval()
    values = torch.Generator().manual_seed(9)
    x = torch.randn(6, 4, generator=values)
    for name in dimensions:
        assert torch.equal(historical(x, name), migrated(x, name))


# The source-digest anchors moved to tests/test_ops_frozen_source_anchors.py:
# they need no PyTorch, and gating them on the importorskip below meant they
# never executed in the dependency-minimal CI job.


def test_independent_mlp_is_state_and_forward_exact_to_frozen_source():
    candidate, _ = _historical_modules()
    _assert_exact_model_parity(
        candidate.IndependentMLPSpecialists,
        IndependentMLPSpecialists,
        {"hidden_dims": (7, 5), "dropout": 0.0},
    )


def test_shared_resmlp_is_state_and_forward_exact_to_frozen_source():
    candidate, _ = _historical_modules()
    _assert_exact_model_parity(
        candidate.SharedResMLP52Head,
        SharedResMLP,
        {
            "shared_width": 8,
            "expansion_width": 13,
            "residual_blocks": 2,
            "head_width": {name: 4 + index for index, name in enumerate(_dimensions())},
            "dropout": 0.0,
        },
    )


def test_multitab_is_state_and_forward_exact_to_frozen_source():
    candidate, _ = _historical_modules()
    options = {
        "reporter_panel": tuple(_dimensions()),
        "token_dim": 8,
        "num_heads": 2,
        "num_blocks": 1,
        "feedforward_dim": 11,
        "head_hidden_dims": (5,),
        "dropout": 0.0,
    }
    _assert_exact_model_parity(candidate.MultiTabColumnPilot, MultiTab, options)


def test_multitab_has_no_cross_cell_attention_gradient_path():
    model = MultiTab(
        _dimensions(),
        reporter_panel=tuple(_dimensions()),
        input_dim=4,
        token_dim=8,
        num_heads=2,
        num_blocks=1,
        feedforward_dim=11,
    ).eval()
    x = torch.randn(3, 4, requires_grad=True)
    gradient = torch.autograd.grad(model(x, "r2")[0].sum(), x)[0]
    assert torch.count_nonzero(gradient[1:]) == 0
    assert torch.count_nonzero(gradient[0]) > 0


def test_equal_endpoint_equal_reporter_loss_matches_frozen_v2_and_masks_gradients():
    _, historical = _historical_modules()
    predictions = {
        "a": torch.tensor([[1.0, 100.0], [3.0, -100.0]], requires_grad=True),
        "b": torch.tensor([[2.0, 4.0]], requires_grad=True),
    }
    targets = {
        "a": torch.tensor([[0.0, float("nan")], [1.0, 0.0]]),
        "b": torch.tensor([[0.0, 0.0]]),
    }
    masks = {
        "a": torch.tensor([[True, False], [True, False]]),
        "b": torch.tensor([[True, True]]),
    }
    migrated = reporter_balanced_mse(predictions, targets, masks)
    frozen = historical.masked_balanced_mse(predictions, targets, masks)
    assert torch.equal(migrated, frozen)
    assert migrated.item() == pytest.approx(6.25)
    migrated.backward()
    assert torch.count_nonzero(predictions["a"].grad[:, 1]) == 0
    assert torch.count_nonzero(predictions["a"].grad[:, 0]) == 2


def test_every_active_resmlp_head_receives_gradient_and_inactive_head_does_not():
    dimensions = OrderedDict((name, 2) for name in ("a", "b", "c"))
    model = SharedResMLP(
        dimensions,
        input_dim=3,
        shared_width=6,
        expansion_width=9,
        residual_blocks=1,
        head_width=4,
    )
    outputs = model.forward_grouped(torch.randn(8, 3), ("a", "c"), (4, 4))
    loss = reporter_balanced_mse(
        outputs,
        {name: torch.zeros_like(value) for name, value in outputs.items()},
    )
    loss.backward()
    assert model.backbone.heads[0].linear_out.weight.grad is not None
    assert model.backbone.heads[1].linear_out.weight.grad is None
    assert model.backbone.heads[2].linear_out.weight.grad is not None
    assert model.backbone.input_projection.weight.grad is not None


def test_sampler_batches_and_resume_are_exact_to_frozen_source():
    _, historical = _historical_modules()
    options = dict(
        task_sizes=OrderedDict((name, size) for name, size in (("a", 2), ("b", 7), ("c", 3))),
        tasks_per_batch=2,
        samples_per_task=4,
        epoch_samples_per_task=8,
        seed=73,
    )
    migrated = ReporterBalancedSampler(**options)
    frozen = historical.DeterministicCyclicTaskSampler(**options)
    for _ in range(7):
        left, right = migrated.next_batch(), frozen.next_batch()
        assert (left.epoch, left.batch_index, left.round_index, left.task_names) == (
            right.epoch,
            right.batch_index,
            right.round_index,
            right.task_names,
        )
        assert all(np.array_equal(a, b) for a, b in zip(left.task_indices, right.task_indices))
    state = migrated.state_dict()
    resumed = ReporterBalancedSampler(**options)
    resumed.load_state_dict(state)
    assert all(
        np.array_equal(a, b)
        for a, b in zip(migrated.next_batch().task_indices, resumed.next_batch().task_indices)
    )


def test_independent_mlp_restores_validation_selected_state():
    rng = np.random.default_rng(5)
    train_x = rng.normal(size=(24, 3)).astype(np.float32)
    validation_x = rng.normal(size=(12, 3)).astype(np.float32)
    weights = np.array([[1.0, -0.5], [0.3, 0.2], [-0.4, 0.7]], dtype=np.float32)
    train = ReporterBatch(train_x, train_x @ weights, np.ones((24, 2), dtype=bool))
    validation = ReporterBatch(
        validation_x, validation_x @ weights, np.ones((12, 2), dtype=bool)
    )
    fitted = fit_independent_mlp(
        train,
        validation,
        hidden_dims=(8, 4),
        batch_size=6,
        epochs=5,
        patience=5,
        learning_rate=2e-3,
        seed=19,
    )
    with torch.no_grad():
        restored = endpoint_balanced_mse(
            fitted.model(torch.from_numpy(validation_x)),
            torch.from_numpy(validation.targets),
            torch.from_numpy(validation.observed),
        ).item()
    assert restored == pytest.approx(fitted.best_validation_mse, abs=1e-7)
    assert fitted.best_epoch == min(
        fitted.history, key=lambda row: row["validation_mse"]
    )["epoch"]


def test_checkpoint_restores_sampler_and_rng_exactly(tmp_path):
    model = IndependentMLPSpecialists({"a": 1}, input_dim=2, hidden_dims=(3,))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    sampler = ReporterBalancedSampler({"a": 3}, tasks_per_batch=1, samples_per_task=2, seed=7)
    sampler.next_batch()
    path = atomic_training_checkpoint(
        tmp_path / "checkpoint.pt",
        model,
        optimizer=optimizer,
        sampler=sampler,
        training_state={"epoch": 1},
    )
    expected_numpy = np.random.random(4)
    expected_torch = torch.rand(4)
    expected_batch = sampler.next_batch().task_indices[0]
    np.random.random(5)
    torch.rand(5)
    model.apply(lambda module: module.reset_parameters() if hasattr(module, "reset_parameters") else None)
    restored_sampler = ReporterBalancedSampler(
        {"a": 3}, tasks_per_batch=1, samples_per_task=2, seed=7
    )
    payload = load_training_checkpoint(
        path, model, optimizer=optimizer, sampler=restored_sampler
    )
    assert payload["training_state"] == {"epoch": 1}
    assert np.array_equal(np.random.random(4), expected_numpy)
    assert torch.equal(torch.rand(4), expected_torch)
    assert np.array_equal(restored_sampler.next_batch().task_indices[0], expected_batch)


def test_capture_rng_state_contains_all_required_streams():
    state = capture_rng_state()
    assert set(state) == {"python", "numpy", "torch_cpu", "torch_cuda"}


@pytest.mark.parametrize(
    ("model_id", "expected"),
    (("mlp", IndependentMLPSpecialists), ("resmlp", SharedResMLP), ("multitab", MultiTab)),
)
def test_registry_exposes_seeded_ops_native_architectures(model_id, expected):
    kwargs = {"reporter_panel": tuple(_dimensions())} if model_id == "multitab" else {}
    first = create_ops_model(
        model_id,
        head_dimensions=_dimensions(),
        input_dim=4,
        seed=23,
        **kwargs,
    )
    second = create_ops_model(
        model_id,
        head_dimensions=_dimensions(),
        input_dim=4,
        seed=23,
        **kwargs,
    )
    assert isinstance(first, expected)
    assert tuple(first.state_dict()) == tuple(second.state_dict())
    assert all(torch.equal(value, second.state_dict()[name]) for name, value in first.state_dict().items())
