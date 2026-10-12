#!/usr/bin/env python3
"""Train and export one OPS fold for scButterfly, MIDAS or scPair.

The command consumes canonical public tables, applies the supplied frozen
assignment manifest, selects checkpoints only on the validation role, and
emits observation-level predictions in the common project schema.  Data,
results and third-party source remain outside this repository.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
RUNNER_ROOT = Path(__file__).resolve().parent
SCBUTTERFLY_ROOT = REPOSITORY_ROOT / "studies/ops/external_runners/scbutterfly"
MIDAS_ROOT = REPOSITORY_ROOT / "studies/ops/external_runners/midas"
for value in (SOURCE_ROOT, RUNNER_ROOT, SCBUTTERFLY_ROOT, MIDAS_ROOT):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from common import (  # noqa: E402
    FoldTables,
    ReporterTargetBlock,
    RunnerContractError,
    Standardizer,
    atomic_json,
    canonical_predictions,
    file_sha256,
    fit_input_standardizer,
    fit_target_standardizer,
    input_matrix,
    load_fold_tables,
    load_targets_for_ids,
    read_table,
    reporter_target_block,
    require,
    resolve_device,
    set_seed,
    stable_seed,
    target_transform,
    write_frame,
)
from measurement_sufficiency.schemas import validate_predictions  # noqa: E402


# PyTorch and the three biological backends are imported on demand rather than at
# module scope. Every use of them lies past the --dry-run / --contract-check-only
# exit in run(), so deferring the import keeps the contract check, the argument
# parser and --help runnable in the dependency-minimal install. The module uses
# `from __future__ import annotations`, so the type annotations that mention these
# classes are strings and need no runtime binding.
_BACKENDS_LOADED = False


def _load_backends() -> None:
    """Bind PyTorch and the scButterfly, MIDAS and scPair backends as globals."""
    global _BACKENDS_LOADED, torch, scpair_core
    global ScButterflyOPSConfig, ScButterflyOPSSpecialist
    global ScButterflyPairedBatch, ScButterflyTrainingConfig, ScButterflyTrainingController
    global MIDASMultimodalOPSConfig
    global MIDASMultimodalOPSTrainingAdapter, MIDASMultimodalReporterTrainingBatch

    if _BACKENDS_LOADED:
        return
    try:
        import torch  # noqa: F401
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "Fitting an external OPS biological model requires PyTorch; install "
            "measurement-sufficiency[torch]. The contract check "
            "(--contract-check-only) and --dry-run do not need it."
        ) from exc
    import scpair_core  # noqa: F401
    from ops_biological_baseline_scbutterfly import (
        ScButterflyOPSConfig,
        ScButterflyOPSSpecialist,
    )
    from ops_biological_baseline_scbutterfly_training import (
        ScButterflyPairedBatch,
        ScButterflyTrainingConfig,
        ScButterflyTrainingController,
    )
    from ops_biological_baseline_midas_multimodal import MIDASMultimodalOPSConfig
    from ops_biological_baseline_midas_multimodal_training import (
        MIDASMultimodalOPSTrainingAdapter,
        MIDASMultimodalReporterTrainingBatch,
    )
    _BACKENDS_LOADED = True


SCHEMA_VERSION = "measurement-sufficiency-ops-external-fold-v1"
METHOD_IDS = {"scbutterfly": "scbutterfly", "midas": "midas", "scpair": "scpair"}


DEFAULT_CONFIG: dict[str, Any] = {
    "selection": {"metric": "observed_validation_mse", "patience": 0},
    "scbutterfly": {
        "architecture": {
            "phase_encoder_widths": [256, 128],
            "phenotype_encoder_widths": [128, 128],
            "phase_decoder_widths": [256],
            "phenotype_decoder_widths": [128],
            "latent_dim": 128,
            "discriminator_hidden_widths": [],
            "discriminator_output_batch_norm": True,
            "dropout": 0.1,
            "phase_input_mask_rate": 0.5,
            "phenotype_input_mask_rate": 0.0,
        },
        "phase_pretrain_epochs": 100,
        "phenotype_pretrain_epochs": 100,
        "joint_epochs": 200,
        "steps_per_epoch": 64,
        "batch_size": 1024,
        "ema_decay": 0.999,
    },
    "midas": {
        "architecture": {
            "biological_dim": 128,
            "technical_dim": 32,
            "modality_width": 768,
            "shared_encoder_hidden_dims": [768],
            "shared_decoder_hidden_dims": [768, 768],
            "dropout": 0.1,
            "phase_reconstruction_weight": 1.0,
            "reporter_reconstruction_weight": 1.0,
            "kl_biological_weight": 1.0,
            "kl_technical_weight": 1.0,
            "technical_information_bottleneck_multiplier": 5.0,
            "modality_alignment_weight": 50.0,
            "reporter_reconstruction_reduction": "modality_mean",
            "batch_disentanglement": False,
            "batch_covariate_kind": "none",
        },
        "epochs": 100,
        "steps_per_epoch": 64,
        "batch_size": 1024,
        "reporters_per_step": 4,
        "learning_rate": 0.0001,
        "weight_decay": 0.01,
        "gradient_clip_norm": 1.0,
    },
    "scpair": {
        "epochs": 40,
        "steps_per_epoch": 64,
        "batch_size": 16384,
        "learning_rate": 0.0001,
        "weight_decay": 0.0001,
        "gradient_clip_norm": 1.0,
    },
}


def _deep_update(base: dict[str, Any], update: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in update.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _deep_update(dict(result[key]), value)
        else:
            result[key] = value
    return result


def load_config(path: Path | None, *, smoke: bool, method: str) -> dict[str, Any]:
    supplied: Mapping[str, Any] = {}
    if path is not None:
        supplied = json.loads(path.read_text(encoding="utf-8"))
    config = _deep_update(DEFAULT_CONFIG, supplied)
    if smoke:
        if method == "scbutterfly":
            config[method].update(
                {
                    "phase_pretrain_epochs": 1,
                    "phenotype_pretrain_epochs": 1,
                    "joint_epochs": 1,
                    "steps_per_epoch": 1,
                    "batch_size": 8,
                }
            )
            config[method]["architecture"].update(
                {
                    "phase_encoder_widths": [24, 16],
                    "phenotype_encoder_widths": [16, 12],
                    "phase_decoder_widths": [18, 24],
                    "phenotype_decoder_widths": [14, 12],
                    "latent_dim": 10,
                    "dropout": 0.0,
                    "phase_input_mask_rate": 0.0,
                }
            )
        elif method == "midas":
            config[method].update(
                {"epochs": 1, "steps_per_epoch": 1, "batch_size": 8, "reporters_per_step": 1}
            )
            config[method]["architecture"].update(
                {
                    "biological_dim": 8,
                    "technical_dim": 4,
                    "modality_width": 32,
                    "shared_encoder_hidden_dims": [24],
                    "shared_decoder_hidden_dims": [32],
                    "dropout": 0.0,
                    "modality_alignment_weight": 0.1,
                }
            )
        else:
            config[method].update({"epochs": 1, "steps_per_epoch": 1, "batch_size": 8})
    return config


@dataclass
class RoleBundle:
    reporter_id: str
    endpoint_ids: tuple[str, ...]
    observation_ids: tuple[str, ...]
    x: np.ndarray
    y_raw: np.ndarray
    y: np.ndarray
    observed: np.ndarray
    metadata: pd.DataFrame
    phase_row_indices: np.ndarray


@dataclass
class ReporterBundle:
    reporter_id: str
    target_scaler: Standardizer
    train: RoleBundle
    validation: RoleBundle


def _role_bundle(
    *,
    tables: FoldTables,
    role_ids: Sequence[str],
    targets: pd.DataFrame,
    reporter_id: str,
    endpoint_ids: Sequence[str],
    input_scaler: Standardizer,
    target_scaler: Standardizer,
) -> RoleBundle:
    role_inputs = tables.inputs.loc[tables.inputs.observation_id.isin(role_ids)].copy()
    block = reporter_target_block(targets, role_inputs, reporter_id, endpoint_ids=endpoint_ids)
    metadata, x = input_matrix(tables, block.observation_ids, input_scaler)
    rows = tables.inputs.reset_index(drop=True)
    index = {str(value): i for i, value in enumerate(rows.observation_id.astype(str))}
    phase_rows = np.asarray([index[value] for value in block.observation_ids], dtype=np.int64)
    return RoleBundle(
        reporter_id,
        tuple(endpoint_ids),
        block.observation_ids,
        x,
        block.values,
        target_transform(block, target_scaler),
        block.observed,
        metadata,
        phase_rows,
    )


def prepare_train_validation(
    tables: FoldTables,
    targets_path: Path,
    reporters: Sequence[str],
    input_scaler: Standardizer,
) -> dict[str, ReporterBundle]:
    role_ids = (*tables.train_ids, *tables.validation_ids)
    targets = load_targets_for_ids(targets_path, role_ids)
    result: dict[str, ReporterBundle] = {}
    for reporter in reporters:
        train_inputs = tables.inputs.loc[tables.inputs.observation_id.isin(tables.train_ids)].copy()
        train_block = reporter_target_block(targets, train_inputs, reporter)
        scaler = fit_target_standardizer(train_block)
        train = _role_bundle(
            tables=tables,
            role_ids=tables.train_ids,
            targets=targets,
            reporter_id=reporter,
            endpoint_ids=train_block.endpoint_ids,
            input_scaler=input_scaler,
            target_scaler=scaler,
        )
        validation = _role_bundle(
            tables=tables,
            role_ids=tables.validation_ids,
            targets=targets,
            reporter_id=reporter,
            endpoint_ids=train_block.endpoint_ids,
            input_scaler=input_scaler,
            target_scaler=scaler,
        )
        result[reporter] = ReporterBundle(reporter, scaler, train, validation)
    return result


def _observed_mse(prediction: np.ndarray, bundle: RoleBundle) -> float:
    error = np.square(np.asarray(prediction) - bundle.y)
    return float(error[bundle.observed].mean())


def _draw(bundle: RoleBundle, batch_size: int, rng: np.random.Generator) -> np.ndarray:
    return rng.integers(0, len(bundle.x), size=int(batch_size), endpoint=False)


def _tensor(value: np.ndarray, device: str, *, boolean: bool = False) -> torch.Tensor:
    dtype = torch.bool if boolean else torch.float32
    return torch.as_tensor(value, dtype=dtype, device=device)


def _predict_scbutterfly(
    model: ScButterflyOPSSpecialist,
    bundle: RoleBundle,
    device: str,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    values: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(bundle.x), batch_size):
            values.append(model.predict(_tensor(bundle.x[start : start + batch_size], device)).cpu().numpy())
    return np.concatenate(values)


def train_scbutterfly_reporter(
    bundle: ReporterBundle,
    config: Mapping[str, Any],
    *,
    device: str,
    seed: int,
) -> tuple[dict[str, torch.Tensor], list[dict[str, float]], dict[str, Any]]:
    cfg = dict(config)
    architecture = dict(cfg["architecture"])
    architecture.update({"phenotype_dim": len(bundle.train.endpoint_ids), "reporter_name": bundle.reporter_id})
    model = ScButterflyOPSSpecialist(ScButterflyOPSConfig(**architecture)).to(device)
    controller = ScButterflyTrainingController(
        model,
        ScButterflyTrainingConfig.source_anchor(phase_dim=172),
    )
    rng = np.random.default_rng(seed)
    batch_size = int(cfg["batch_size"])
    steps = int(cfg["steps_per_epoch"])

    def paired_batch() -> ScButterflyPairedBatch:
        take = _draw(bundle.train, batch_size, rng)
        return ScButterflyPairedBatch(
            _tensor(bundle.train.x[take], device),
            _tensor(bundle.train.y[take], device),
            _tensor(bundle.train.observed[take], device, boolean=True),
        )

    for _epoch in range(int(cfg["phase_pretrain_epochs"])):
        for _ in range(steps):
            controller.phase_pretrain_step(paired_batch())
    controller.finish_phase_pretraining()
    for _epoch in range(int(cfg["phenotype_pretrain_epochs"])):
        for _ in range(steps):
            controller.phenotype_pretrain_step(paired_batch())
    controller.finish_phenotype_pretraining()
    best = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    trajectory: list[dict[str, float]] = []
    for epoch in range(1, int(cfg["joint_epochs"]) + 1):
        losses = []
        for _ in range(steps):
            output = controller.joint_alternating_step(paired_batch())
            losses.append(float(output.generator.total.detach().cpu()))
        validation = _observed_mse(
            _predict_scbutterfly(model, bundle.validation, device, batch_size), bundle.validation
        )
        trajectory.append(
            {"epoch": float(epoch), "train_loss": float(np.mean(losses)), "validation_mse": validation}
        )
        if validation < best:
            best = validation
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    controller.finish_joint_training()
    require(best_state is not None, "scButterfly produced no validation-selected state")
    return best_state, trajectory, model.architecture_manifest()


def _complete_rows(bundle: RoleBundle) -> np.ndarray:
    rows = np.flatnonzero(bundle.observed.all(axis=1))
    require(bool(len(rows)), f"{bundle.reporter_id} has no complete endpoint rows")
    return rows


def _predict_scpair(
    modules: Mapping[str, torch.nn.Module],
    bundle: RoleBundle,
    device: str,
    batch_size: int,
) -> np.ndarray:
    for module in modules.values():
        module.eval()
    values: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(bundle.x), batch_size):
            values.append(scpair_core.predict(modules, _tensor(bundle.x[start : start + batch_size], device)).cpu().numpy())
    return np.concatenate(values)


def train_scpair_reporter(
    bundle: ReporterBundle,
    config: Mapping[str, Any],
    *,
    device: str,
    seed: int,
    upstream_root: Path,
    strict_source: bool,
) -> tuple[dict[str, dict[str, torch.Tensor]], list[dict[str, float]], dict[str, Any]]:
    cfg = dict(config)
    modules, provenance = scpair_core.build_modules(
        upstream_root, len(bundle.train.endpoint_ids), strict_source=strict_source
    )
    for module in modules.values():
        module.to(device)
    parameters = [parameter for module in modules.values() for parameter in module.parameters()]
    optimizer = torch.optim.AdamW(
        parameters, lr=float(cfg["learning_rate"]), weight_decay=float(cfg["weight_decay"])
    )
    train_rows = _complete_rows(bundle.train)
    validation_rows = _complete_rows(bundle.validation)
    rng = np.random.default_rng(seed)
    batch_size = int(cfg["batch_size"])
    best = float("inf")
    best_state = None
    trajectory: list[dict[str, float]] = []
    for epoch in range(1, int(cfg["epochs"]) + 1):
        for module in modules.values():
            module.train()
        losses = []
        for _ in range(int(cfg["steps_per_epoch"])):
            take = rng.choice(train_rows, size=batch_size, replace=True)
            optimizer.zero_grad(set_to_none=True)
            loss = scpair_core.paired_loss(
                modules, _tensor(bundle.train.x[take], device), _tensor(bundle.train.y[take], device)
            )
            require(bool(torch.isfinite(loss)), "non-finite scPair training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, float(cfg["gradient_clip_norm"]))
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        prediction = _predict_scpair(modules, bundle.validation, device, batch_size)
        validation = float(np.square(prediction[validation_rows] - bundle.validation.y[validation_rows]).mean())
        trajectory.append(
            {"epoch": float(epoch), "train_loss": float(np.mean(losses)), "validation_mse": validation}
        )
        if validation < best:
            best = validation
            best_state = scpair_core.state_dict(modules)
    require(best_state is not None, "scPair produced no validation-selected state")
    return best_state, trajectory, provenance


def _midas_local_batch(
    bundle: RoleBundle,
    take: np.ndarray,
    device: str,
) -> MIDASMultimodalReporterTrainingBatch:
    return MIDASMultimodalReporterTrainingBatch(
        reporter=bundle.reporter_id,
        phase_x=_tensor(bundle.x[take], device),
        endpoint_target=_tensor(bundle.y[take], device),
        endpoint_observed_mask=_tensor(bundle.observed[take], device, boolean=True),
        phase_row_indices=torch.as_tensor(bundle.phase_row_indices[take], dtype=torch.long, device=device),
    )


def _predict_midas(
    adapter: MIDASMultimodalOPSTrainingAdapter,
    bundle: RoleBundle,
    device: str,
    batch_size: int,
) -> np.ndarray:
    adapter.eval()
    values: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(bundle.x), batch_size):
            values.append(adapter.predict_reporter(_tensor(bundle.x[start : start + batch_size], device), bundle.reporter_id).cpu().numpy())
    return np.concatenate(values)


def train_midas(
    bundles: Mapping[str, ReporterBundle],
    config: Mapping[str, Any],
    *,
    device: str,
    seed: int,
) -> tuple[dict[str, torch.Tensor], list[dict[str, float]], dict[str, Any]]:
    cfg = dict(config)
    dimensions = {name: len(bundle.train.endpoint_ids) for name, bundle in bundles.items()}
    architecture = dict(cfg["architecture"])
    for key in ("shared_encoder_hidden_dims", "shared_decoder_hidden_dims"):
        architecture[key] = tuple(architecture[key])
    adapter = MIDASMultimodalOPSTrainingAdapter(
        dimensions, config=MIDASMultimodalOPSConfig(**architecture)
    ).to(device)
    optimizer = torch.optim.AdamW(
        adapter.parameters(), lr=float(cfg["learning_rate"]), weight_decay=float(cfg["weight_decay"])
    )
    names = tuple(bundles)
    rng = np.random.default_rng(seed)
    best = float("inf")
    best_state = None
    trajectory: list[dict[str, float]] = []
    batch_size = int(cfg["batch_size"])
    for epoch in range(1, int(cfg["epochs"]) + 1):
        adapter.train()
        losses = []
        for _ in range(int(cfg["steps_per_epoch"])):
            selected = rng.choice(
                names,
                size=min(int(cfg["reporters_per_step"]), len(names)),
                replace=False,
            )
            groups = []
            for name in selected:
                bundle = bundles[str(name)].train
                take = _draw(bundle, batch_size, rng)
                groups.append(_midas_local_batch(bundle, take, device))
            optimizer.zero_grad(set_to_none=True)
            result = adapter.forward_grouped(groups, sample=True)
            require(bool(torch.isfinite(result.total_loss)), "non-finite MIDAS training loss")
            result.total_loss.backward()
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), float(cfg["gradient_clip_norm"]))
            optimizer.step()
            losses.append(float(result.total_loss.detach().cpu()))
        validation_parts = []
        for bundle in bundles.values():
            prediction = _predict_midas(adapter, bundle.validation, device, batch_size)
            validation_parts.append(_observed_mse(prediction, bundle.validation))
        validation = float(np.mean(validation_parts))
        trajectory.append(
            {"epoch": float(epoch), "train_loss": float(np.mean(losses)), "validation_mse": validation}
        )
        if validation < best:
            best = validation
            best_state = {key: value.detach().cpu().clone() for key, value in adapter.state_dict().items()}
    require(best_state is not None, "MIDAS produced no validation-selected state")
    return best_state, trajectory, adapter.model.config_manifest()


def prepare_test_bundle(
    tables: FoldTables,
    targets_path: Path,
    reporter: ReporterBundle,
    input_scaler: Standardizer,
) -> tuple[ReporterTargetBlock, RoleBundle]:
    targets = load_targets_for_ids(targets_path, tables.test_ids)
    test_inputs = tables.inputs.loc[tables.inputs.observation_id.isin(tables.test_ids)].copy()
    block = reporter_target_block(
        targets, test_inputs, reporter.reporter_id, endpoint_ids=reporter.train.endpoint_ids
    )
    role = _role_bundle(
        tables=tables,
        role_ids=tables.test_ids,
        targets=targets,
        reporter_id=reporter.reporter_id,
        endpoint_ids=reporter.train.endpoint_ids,
        input_scaler=input_scaler,
        target_scaler=reporter.target_scaler,
    )
    return block, role


def _selected_reporters(tables: FoldTables, requested: str, smoke: bool) -> tuple[str, ...]:
    training = tables.inputs.loc[tables.inputs.observation_id.isin(tables.train_ids)]
    available = tuple(dict.fromkeys(training.reporter_id.astype(str)))
    if requested.strip().casefold() == "all":
        result = available
    else:
        result = tuple(value.strip() for value in requested.split(",") if value.strip())
        require(set(result).issubset(available), "requested reporter is absent from training role")
    if smoke:
        test_reporters = set(
            tables.inputs.loc[
                tables.inputs.observation_id.isin(tables.test_ids), "reporter_id"
            ].astype(str)
        )
        evaluable = tuple(name for name in result if name in test_reporters)
        result = (evaluable or result)[:1]
    require(bool(result), "reporter selection is empty")
    return result


def _contract(args: argparse.Namespace, tables: FoldTables, reporters: Sequence[str], config: Mapping[str, Any]) -> dict[str, Any]:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "CONTRACT_PASS",
        "method": args.method,
        "split_name": args.split_name,
        "fold": int(args.fold),
        "reporters": list(reporters),
        "n_reporters": len(reporters),
        "n_features": len(tables.feature_columns),
        "n_train": len(tables.train_ids),
        "n_validation": len(tables.validation_ids),
        "n_test": len(tables.test_ids),
        "checkpoint_selection": "validation_only",
        "test_targets_loaded": False,
        "config": config[args.method],
        "input_path": str(args.inputs),
        "target_path": str(args.targets),
        "assignment_path": str(args.assignments),
    }
    if args.method == "scpair":
        # Imported here rather than at module scope: verifying the pinned scPair
        # checkout is the one backend call the contract check makes, and only for
        # --method scpair. The other two methods stay importable without PyTorch.
        import scpair_core as _scpair_core

        payload["upstream"] = _scpair_core.verify_source(
            args.scpair_source, strict=not args.allow_unpinned_scpair
        )
    return payload


def run(args: argparse.Namespace) -> int:
    started = time.time()
    feature_columns = tuple(value.strip() for value in args.features.split(",") if value.strip()) if args.features else None
    tables = load_fold_tables(
        args.inputs, args.assignments, fold=args.fold, feature_columns=feature_columns
    )
    require(len(tables.feature_columns) == 172, f"OPS biological runners require 172 features, found {len(tables.feature_columns)}")
    requested_reporters = _selected_reporters(tables, args.reporters, args.smoke)
    test_reporters = set(
        tables.inputs.loc[tables.inputs.observation_id.isin(tables.test_ids), "reporter_id"].astype(str)
    )
    evaluation_reporters = tuple(name for name in requested_reporters if name in test_reporters)
    require(bool(evaluation_reporters), "selected fold has no test reporter")
    # MIDAS uses every source reporter as sparse-atlas supervision.  The two
    # reporter-specialist biological methods fit only evaluable heads.
    reporters = requested_reporters if args.method == "midas" else evaluation_reporters
    config = load_config(args.config, smoke=args.smoke, method=args.method)
    contract = _contract(args, tables, reporters, config)
    contract["evaluation_reporters"] = list(evaluation_reporters)
    contract["n_evaluation_reporters"] = len(evaluation_reporters)
    if args.dry_run or args.contract_check_only:
        print(json.dumps(contract, indent=2, sort_keys=True))
        return 0

    _load_backends()
    device = resolve_device(args.device)
    set_seed(args.seed)
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    atomic_json(output_root / "run_contract.json", contract)
    input_scaler = fit_input_standardizer(tables)
    bundles = prepare_train_validation(tables, args.targets, reporters, input_scaler)
    preprocessing = {
        "input": input_scaler.manifest(),
        "targets": {name: bundle.target_scaler.manifest() for name, bundle in bundles.items()},
        "feature_columns": list(tables.feature_columns),
        "fit_role": "train",
    }
    atomic_json(output_root / "preprocessing.json", preprocessing)

    prediction_tables: list[pd.DataFrame] = []
    trajectories: list[pd.DataFrame] = []
    provenance: dict[str, Any] = {}
    if args.method == "midas":
        state, trajectory, provenance = train_midas(
            bundles, config["midas"], device=device, seed=args.seed
        )
        checkpoint = output_root / "checkpoint.pt"
        torch.save({"method": "midas", "state_dict": state, "reporters": list(reporters)}, checkpoint)
        dimensions = {name: len(bundle.train.endpoint_ids) for name, bundle in bundles.items()}
        architecture = dict(config["midas"]["architecture"])
        for key in ("shared_encoder_hidden_dims", "shared_decoder_hidden_dims"):
            architecture[key] = tuple(architecture[key])
        adapter = MIDASMultimodalOPSTrainingAdapter(
            dimensions, config=MIDASMultimodalOPSConfig(**architecture)
        ).to(device)
        adapter.load_state_dict(state, strict=True)
        trajectories.append(pd.DataFrame(trajectory).assign(reporter_id="shared"))
        for reporter in bundles.values():
            if reporter.reporter_id not in test_reporters:
                continue
            block, test = prepare_test_bundle(tables, args.targets, reporter, input_scaler)
            scaled = _predict_midas(adapter, test, device, int(config["midas"]["batch_size"]))
            raw = reporter.target_scaler.inverse(scaled)
            prediction_tables.append(
                canonical_predictions(
                    metadata=test.metadata,
                    block=block,
                    prediction=raw,
                    model_id=METHOD_IDS[args.method],
                    split_name=args.split_name,
                    fold=args.fold,
                )
            )
    else:
        for reporter_name, reporter in bundles.items():
            local_seed = stable_seed(args.seed, args.method, reporter_name, args.fold)
            if args.method == "scbutterfly":
                state, trajectory, local_provenance = train_scbutterfly_reporter(
                    reporter, config["scbutterfly"], device=device, seed=local_seed
                )
                architecture = dict(config["scbutterfly"]["architecture"])
                architecture.update(
                    {"phenotype_dim": len(reporter.train.endpoint_ids), "reporter_name": reporter_name}
                )
                model = ScButterflyOPSSpecialist(ScButterflyOPSConfig(**architecture)).to(device)
                model.load_state_dict(state, strict=True)
                predictor = lambda role: _predict_scbutterfly(  # noqa: E731
                    model, role, device, int(config["scbutterfly"]["batch_size"])
                )
                checkpoint_payload = {"method": "scbutterfly", "state_dict": state}
            else:
                state, trajectory, local_provenance = train_scpair_reporter(
                    reporter,
                    config["scpair"],
                    device=device,
                    seed=local_seed,
                    upstream_root=args.scpair_source,
                    strict_source=not args.allow_unpinned_scpair,
                )
                modules, _source = scpair_core.build_modules(
                    args.scpair_source,
                    len(reporter.train.endpoint_ids),
                    strict_source=not args.allow_unpinned_scpair,
                )
                for module in modules.values():
                    module.to(device)
                scpair_core.load_state_dict(modules, state)
                predictor = lambda role: _predict_scpair(  # noqa: E731
                    modules, role, device, int(config["scpair"]["batch_size"])
                )
                checkpoint_payload = {"method": "scpair", "module_states": state}
            provenance[reporter_name] = local_provenance
            checkpoint = output_root / "reporters" / reporter_name / "checkpoint.pt"
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save(checkpoint_payload, checkpoint)
            trajectories.append(pd.DataFrame(trajectory).assign(reporter_id=reporter_name))
            block, test = prepare_test_bundle(tables, args.targets, reporter, input_scaler)
            raw = reporter.target_scaler.inverse(predictor(test))
            prediction_tables.append(
                canonical_predictions(
                    metadata=test.metadata,
                    block=block,
                    prediction=raw,
                    model_id=METHOD_IDS[args.method],
                    split_name=args.split_name,
                    fold=args.fold,
                )
            )

    predictions = pd.concat(prediction_tables, ignore_index=True)
    validate_predictions(predictions)
    prediction_path = output_root / "predictions.csv"
    trajectory_path = output_root / "validation_trajectory.csv"
    write_frame(predictions, prediction_path)
    write_frame(pd.concat(trajectories, ignore_index=True), trajectory_path)
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "method_id": METHOD_IDS[args.method],
        "split_name": args.split_name,
        "fold": int(args.fold),
        "reporters": list(reporters),
        "n_reporters": len(reporters),
        "n_prediction_rows": len(predictions),
        "outer_test_used_for_selection": False,
        "checkpoint_selection": "validation_only",
        "smoke": bool(args.smoke),
        "device": device,
        "runtime_seconds": time.time() - started,
        "predictions": str(prediction_path.resolve()),
        "predictions_sha256": file_sha256(prediction_path),
        "validation_trajectory": str(trajectory_path.resolve()),
        "provenance": provenance,
    }
    atomic_json(output_root / "training_result.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--method", required=True, choices=tuple(METHOD_IDS))
    result.add_argument("--inputs", required=True, type=Path)
    result.add_argument("--targets", required=True, type=Path)
    result.add_argument("--assignments", required=True, type=Path)
    result.add_argument("--split-name", required=True)
    result.add_argument("--fold", required=True, type=int)
    result.add_argument("--output-root", required=True, type=Path)
    result.add_argument("--reporters", default="all", help="all or comma-separated reporter IDs")
    result.add_argument("--features", help="comma-separated 172 feature columns; defaults to numeric non-metadata columns")
    result.add_argument("--config", type=Path, help="optional JSON overrides for the frozen public defaults")
    result.add_argument("--device", default="cuda")
    result.add_argument("--seed", type=int, default=20260721)
    result.add_argument("--scpair-source", type=Path, help="public scPair checkout; required by scpair")
    result.add_argument("--allow-unpinned-scpair", action="store_true", help="development only: accept a model.py digest other than the pinned revision")
    result.add_argument("--contract-check-only", action="store_true")
    result.add_argument("--dry-run", action="store_true")
    result.add_argument("--smoke", action="store_true", help="one reporter, one epoch/step, reduced architecture")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    for name in ("inputs", "targets", "assignments", "output_root", "config", "scpair_source"):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, Path(value).expanduser().resolve())
    if args.method == "scpair":
        require(args.scpair_source is not None, "--scpair-source is required for scpair")
    return run(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RunnerContractError as error:
        raise SystemExit(f"ERROR: {error}") from error
