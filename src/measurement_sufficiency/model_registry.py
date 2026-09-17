"""Explicit model registry with fail-closed third-party adapter declarations."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any, Literal, Mapping

from .models import SharedSparseRegressor, SpecialistSparseRegressor, catboost_estimator, sklearn_estimator
from .torch_models import TorchGroupedSparseRegressor


class ModelEligibilityError(ValueError):
    """Raised when a requested method cannot honestly run under a protocol."""


@dataclass(frozen=True)
class ModelCapability:
    task_modes: tuple[str, ...]
    heterogeneous_endpoint_mask: bool
    declared_split: bool
    reproducible_seed: bool
    generic_factory_executable: bool
    study_runner_executable: bool = False

    @property
    def executable(self) -> bool:
        """Whether the method has any released execution route."""

        return self.generic_factory_executable or self.study_runner_executable

    @property
    def execution_route(self) -> str:
        if self.generic_factory_executable:
            return "generic_factory"
        if self.study_runner_executable:
            return "study_runner"
        return "none"


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    display_name: str
    family: str
    provenance: str
    license: str
    adaptation_boundary: str
    capability: ModelCapability


_BUILTIN = ModelCapability(("specialist", "shared_masked"), True, True, True, True)
_SPECIALIST = ModelCapability(("specialist",), True, True, True, True)
_SHARED = ModelCapability(("shared_masked",), True, True, True, True)
_STUDY_SPECIALIST = ModelCapability(("specialist",), True, True, True, False, True)
_STUDY_ADAPTER = ModelCapability(("declared_adapter",), True, True, True, False, True)

MODEL_REGISTRY: dict[str, ModelSpec] = {
    "ridge": ModelSpec("ridge", "Ridge", "classical_tabular", "scikit-learn", "BSD-3-Clause", "Built-in sklearn adapter; no study-specific code.", _BUILTIN),
    "gbdt": ModelSpec("gbdt", "GBDT", "classical_tabular", "scikit-learn", "BSD-3-Clause", "Built-in sklearn adapter; no study-specific code.", _BUILTIN),
    "mlp": ModelSpec("mlp", "MLP", "neural_specialist", "generic adapter: scikit-learn; OPS identity: checksummed study implementation", "BSD-3-Clause for generic adapter; MIT for the study implementation", "create_model() returns a portable sklearn reference. create_ops_model() plus ops_training exposes the source-anchored frozen identity.", _BUILTIN),
    "catboost": ModelSpec("catboost", "CatBoost", "classical_tabular", "CatBoost", "Apache-2.0", "Optional official CatBoostRegressor specialist adapter.", _SPECIALIST),
    "tabm": ModelSpec("tabm", "TabM", "neural_specialist", "yandex-research/tabm@28e47ae301c92ec37787dde1ce923a0793f405b4", "Apache-2.0", "Executable only with explicit version or source SHA-256 provenance.", _SPECIALIST),
    "scbutterfly": ModelSpec("scbutterfly", "scButterfly", "neural_specialist", "BioX-NKU/scButterfly@eb31e04bb8c4abdf85c4cbecd044fcd359105caa", "MIT", "Included OPS study runner: studies/ops/runners/run_external_fold.py --method scbutterfly.", _STUDY_SPECIALIST),
    "resmlp": ModelSpec("resmlp", "ResMLP", "generic_multitask", "generic adapter: local MIT code; OPS identity: checksummed study implementation", "MIT", "create_model() returns a generic shared residual adapter. create_ops_model() plus ops_training exposes the distinct frozen OPS identity.", _SHARED),
    "multitab": ModelSpec("multitab", "MultiTab", "generic_multitask", "generic adapter: local MIT code; OPS identity: checksummed study implementation", "MIT", "create_model() returns a generic within-cell transformer. create_ops_model() exposes the distinct frozen OPS identity with pilot/full-registry constraints.", _SHARED),
    "midas": ModelSpec("midas", "MIDAS", "biological_completion", "labomics/midas@3ef7847c88c90583c05147cafef496d862986dd3", "NOASSERTION at pinned upstream commit; MIT study-owned OPS adapter", "Included OPS study runner: studies/ops/runners/run_external_fold.py --method midas.", _STUDY_ADAPTER),
    "scpair": ModelSpec("scpair", "scPair", "biological_completion", "quon-titative-biology/scPair@c585949ca8ea1314f5e68b260e3d9c5b2dabe61c", "MIT", "Included OPS study runner: studies/ops/runners/run_external_fold.py --method scpair with a digest-verified public checkout.", _STUDY_ADAPTER),
}


def get_model_spec(model_id: str) -> ModelSpec:
    try:
        return MODEL_REGISTRY[model_id]
    except KeyError as exc:
        raise ModelEligibilityError(f"unknown model_id {model_id!r}; registered: {', '.join(MODEL_REGISTRY)}") from exc


def eligible_models(*, task_mode: str | None = None, executable_only: bool = False) -> list[ModelSpec]:
    return [spec for spec in MODEL_REGISTRY.values() if (task_mode is None or task_mode in spec.capability.task_modes) and (not executable_only or spec.capability.executable)]


def ensure_eligible(model_id: str, *, task_mode: str, requires_sparse_mask: bool = True) -> ModelSpec:
    spec = get_model_spec(model_id)
    if not spec.capability.generic_factory_executable:
        raise ModelEligibilityError(
            f"{model_id} is runnable through the included OPS study-runner route "
            f"rather than the generic create_model() factory. {spec.adaptation_boundary} "
            f"Provenance: {spec.provenance}."
        )
    if task_mode not in spec.capability.task_modes:
        raise ModelEligibilityError(f"{model_id} is not eligible for task_mode={task_mode!r}")
    if requires_sparse_mask and not spec.capability.heterogeneous_endpoint_mask:
        raise ModelEligibilityError(f"{model_id} does not declare heterogeneous endpoint-mask support")
    return spec


def create_model(model_id: str, *, task_mode: Literal["specialist", "shared_masked"] = "specialist", seed: int = 0, **kwargs: object) -> SpecialistSparseRegressor | SharedSparseRegressor | TorchGroupedSparseRegressor:
    """Create a portable sklearn/CatBoost/torch sparse-predictor adapter.

    This factory implements the generic ``SparsePredictor`` protocol. The
    OPS-frozen MLP, ResMLP and MultiTab identities require reporter-aware train
    and validation batches and are created by :func:`create_ops_model`, then
    fit with :mod:`measurement_sufficiency.ops_training`.
    """
    ensure_eligible(model_id, task_mode=task_mode, requires_sparse_mask=True)
    if model_id == "catboost":
        return SpecialistSparseRegressor(partial(catboost_estimator, seed=seed, **kwargs))
    if model_id in {"resmlp", "multitab", "tabm"}:
        return TorchGroupedSparseRegressor(architecture=model_id, seed=seed, **kwargs)
    factory = partial(sklearn_estimator, model_id, seed=seed, **kwargs)
    if task_mode == "specialist":
        return SpecialistSparseRegressor(factory)
    return SharedSparseRegressor(factory)


def create_ops_model(
    model_id: Literal["mlp", "resmlp", "multitab"],
    *,
    head_dimensions: Mapping[str, int],
    input_dim: int = 172,
    seed: int = 0,
    **kwargs: Any,
) -> Any:
    """Create an OPS-native architecture without importing torch at package import.

    Initialization is seeded before module construction. Training is
    intentionally separate because the frozen protocol requires explicit
    reporter-aware training and validation batches rather than an implicit
    split of a flat sparse matrix.
    """

    if model_id not in {"mlp", "resmlp", "multitab"}:
        raise ModelEligibilityError(
            "OPS-native architecture factory supports only mlp, resmlp and multitab"
        )
    spec = get_model_spec(model_id)
    expected_mode = "specialist" if model_id == "mlp" else "shared_masked"
    if expected_mode not in spec.capability.task_modes:
        raise ModelEligibilityError(f"{model_id} lacks its required OPS task mode")
    from .ops_models import IndependentMLPSpecialists, MultiTab, SharedResMLP
    from .ops_training import seed_all

    seed_all(seed)
    model_class = {
        "mlp": IndependentMLPSpecialists,
        "resmlp": SharedResMLP,
        "multitab": MultiTab,
    }[model_id]
    return model_class(head_dimensions, input_dim=input_dim, **kwargs)
