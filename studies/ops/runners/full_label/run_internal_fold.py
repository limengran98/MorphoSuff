#!/usr/bin/env python3
"""Fit one frozen OPS full-label fold for the seven internal method families.

The command consumes the same canonical wide input, sparse long target and
immutable role-assignment tables as the biological-model runner.  All
preprocessing is fit from the training role.  Validation labels select neural
checkpoints; outer-test targets are read only after that state is frozen.
"""

from __future__ import annotations

import argparse
import copy
import json
import pickle
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[4]
SRC = REPO / "src"
RUNNERS = REPO / "studies" / "ops" / "runners"
for value in (str(SRC), str(RUNNERS)):
    if value not in sys.path:
        sys.path.insert(0, value)

from measurement_sufficiency.model_registry import create_model, create_ops_model  # noqa: E402
from measurement_sufficiency.models import SparseTargets  # noqa: E402

# measurement_sufficiency.ops_training requires PyTorch at import time, so it is
# imported inside the neural branches only. Importing it here would make the
# pure scikit-learn and CatBoost routes (ridge, gbdt, catboost) unrunnable
# without PyTorch, which is not a dependency they have.
from measurement_sufficiency.schemas import validate_predictions  # noqa: E402
from common import (  # noqa: E402
    FoldTables,
    Standardizer,
    atomic_json,
    canonical_predictions,
    file_sha256,
    fit_input_standardizer,
    fit_target_standardizer,
    input_matrix,
    load_fold_tables,
    load_targets_for_ids,
    reporter_target_block,
    require,
    set_seed,
    stable_seed,
    target_transform,
    write_frame,
)


METHODS = ("ridge", "gbdt", "catboost", "mlp", "tabm", "resmlp", "multitab")
SHARED = {"resmlp", "multitab"}
SCHEMA_VERSION = "measurement-sufficiency-ops-internal-fold-v1"
DEFAULT_CONFIG = Path(__file__).with_name("internal_methods.json")


def _deep_update(base: dict[str, Any], update: Mapping[str, Any]) -> dict[str, Any]:
    output = copy.deepcopy(base)
    for key, value in update.items():
        if isinstance(value, Mapping) and isinstance(output.get(key), Mapping):
            output[key] = _deep_update(dict(output[key]), value)
        else:
            output[key] = value
    return output


def load_config(path: Path | None, *, method: str, smoke: bool) -> dict[str, Any]:
    base = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    if path is not None:
        base = _deep_update(base, json.loads(path.read_text(encoding="utf-8")))
    config = copy.deepcopy(base[method])
    if smoke:
        if method in {"mlp", "tabm", "resmlp", "multitab"}:
            config["epochs"] = 1
            config["batch_size"] = min(16, int(config.get("batch_size", 16)))
            config["samples_per_task"] = min(16, int(config.get("samples_per_task", 16)))
        if method == "mlp":
            config.update({"hidden_dims": [16, 8], "patience": 1})
        if method == "resmlp":
            config["architecture"].update(
                {"shared_width": 24, "expansion_width": 32, "residual_blocks": 1, "head_width": 12}
            )
        if method == "multitab":
            config["architecture"].update(
                {"token_dim": 8, "num_heads": 2, "num_blocks": 1, "feedforward_dim": 16, "head_hidden_dims": [8]}
            )
        if method == "gbdt":
            config["max_iter"] = 2
        if method == "catboost":
            config["iterations"] = 2
    return config


class Role:
    def __init__(
        self,
        *,
        reporter: str,
        endpoint_ids: Sequence[str],
        observation_ids: Sequence[str],
        metadata: pd.DataFrame,
        x: np.ndarray,
        y_raw: np.ndarray,
        y: np.ndarray,
        observed: np.ndarray,
    ) -> None:
        self.reporter = reporter
        self.endpoint_ids = tuple(map(str, endpoint_ids))
        self.observation_ids = tuple(map(str, observation_ids))
        self.metadata = metadata
        self.x = np.asarray(x, dtype=np.float32)
        self.y_raw = np.asarray(y_raw, dtype=np.float32)
        self.y = np.asarray(y, dtype=np.float32)
        self.observed = np.asarray(observed, dtype=bool)


class ReporterBundle:
    def __init__(self, reporter: str, scaler: Standardizer, train: Role, validation: Role) -> None:
        self.reporter, self.scaler, self.train, self.validation = reporter, scaler, train, validation


def _role(
    tables: FoldTables,
    targets: pd.DataFrame,
    role_ids: Sequence[str],
    reporter: str,
    endpoint_ids: Sequence[str],
    input_scaler: Standardizer,
    target_scaler: Standardizer,
) -> Role:
    role_inputs = tables.inputs.loc[tables.inputs.observation_id.isin(role_ids)].copy()
    block = reporter_target_block(targets, role_inputs, reporter, endpoint_ids=endpoint_ids)
    metadata, x = input_matrix(tables, block.observation_ids, input_scaler)
    return Role(
        reporter=reporter,
        endpoint_ids=endpoint_ids,
        observation_ids=block.observation_ids,
        metadata=metadata,
        x=x,
        y_raw=block.values,
        y=target_transform(block, target_scaler),
        observed=block.observed,
    )


def prepare_train_validation(
    tables: FoldTables, targets_path: Path, reporters: Sequence[str], input_scaler: Standardizer
) -> OrderedDict[str, ReporterBundle]:
    role_ids = (*tables.train_ids, *tables.validation_ids)
    targets = load_targets_for_ids(targets_path, role_ids)
    output: OrderedDict[str, ReporterBundle] = OrderedDict()
    for reporter in reporters:
        train_inputs = tables.inputs.loc[tables.inputs.observation_id.isin(tables.train_ids)]
        block = reporter_target_block(targets, train_inputs, reporter)
        scaler = fit_target_standardizer(block)
        train = _role(tables, targets, tables.train_ids, reporter, block.endpoint_ids, input_scaler, scaler)
        validation = _role(
            tables, targets, tables.validation_ids, reporter, block.endpoint_ids, input_scaler, scaler
        )
        output[reporter] = ReporterBundle(reporter, scaler, train, validation)
    return output


def prepare_test(
    tables: FoldTables, targets_path: Path, bundle: ReporterBundle, input_scaler: Standardizer
) -> tuple[Any, Role]:
    targets = load_targets_for_ids(targets_path, tables.test_ids)
    test_inputs = tables.inputs.loc[tables.inputs.observation_id.isin(tables.test_ids)]
    block = reporter_target_block(
        targets, test_inputs, bundle.reporter, endpoint_ids=bundle.train.endpoint_ids
    )
    return block, _role(
        tables,
        targets,
        tables.test_ids,
        bundle.reporter,
        bundle.train.endpoint_ids,
        input_scaler,
        bundle.scaler,
    )


def selected_reporters(tables: FoldTables, requested: str, smoke: bool) -> tuple[str, ...]:
    training = tables.inputs.loc[tables.inputs.observation_id.isin(tables.train_ids)]
    available = tuple(dict.fromkeys(training.reporter_id.astype(str)))
    if requested.casefold() == "all":
        values = available
    else:
        values = tuple(value.strip() for value in requested.split(",") if value.strip())
        require(set(values).issubset(available), "requested reporter is absent from training role")
    if smoke:
        # Five is the smallest legal MultiTab pilot panel.  Keeping the same
        # smoke contract for every method also exercises multi-reporter data
        # routing instead of silently changing the method definition.
        test_reporters = set(
            tables.inputs.loc[
                tables.inputs.observation_id.isin(tables.test_ids), "reporter_id"
            ].astype(str)
        )
        ordered = tuple(name for name in values if name in test_reporters) + tuple(
            name for name in values if name not in test_reporters
        )
        values = ordered[:5]
    require(bool(values), "reporter selection is empty")
    return values


def _sparse(role: Role) -> SparseTargets:
    return SparseTargets(role.observation_ids, role.endpoint_ids, role.y, role.observed)


def _validation_mse(prediction: np.ndarray, role: Role) -> float:
    error = np.square(np.asarray(prediction, dtype=float) - role.y)
    require(bool(role.observed.any()), f"validation has no observed values for {role.reporter}")
    return float(error[role.observed].mean())


def fit_sklearn(bundle: ReporterBundle, method: str, config: Mapping[str, Any], seed: int) -> tuple[Any, list[dict[str, Any]]]:
    predictor = create_model(method, task_mode="specialist", seed=seed, **dict(config))
    predictor.fit(bundle.train.x, _sparse(bundle.train))
    value = _validation_mse(predictor.predict(bundle.validation.x), bundle.validation)
    return predictor, [{"epoch": 0, "validation_mse": value, "selection": "configuration_frozen_before_test"}]


def fit_mlp(bundle: ReporterBundle, config: Mapping[str, Any], seed: int, device: str) -> tuple[Any, list[dict[str, Any]]]:
    from measurement_sufficiency.ops_training import (  # local optional dependency
        ReporterBatch,
        fit_independent_mlp,
    )

    fitted = fit_independent_mlp(
        ReporterBatch(bundle.train.x, bundle.train.y, bundle.train.observed),
        ReporterBatch(bundle.validation.x, bundle.validation.y, bundle.validation.observed),
        hidden_dims=tuple(config["hidden_dims"]),
        dropout=float(config["dropout"]),
        batch_size=int(config["batch_size"]),
        epochs=int(config["epochs"]),
        patience=int(config["patience"]),
        learning_rate=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
        seed=seed,
        device=device,
    )
    return fitted.model, list(fitted.history)


def fit_tabm(bundle: ReporterBundle, config: Mapping[str, Any], seed: int, device: str, source: Path, allow_unpinned: bool) -> tuple[Any, list[dict[str, Any]], dict[str, Any]]:
    from measurement_sufficiency.torch_models import (  # local optional dependency
        FROZEN_TABM_SOURCE_SHA256,
        TabMProvenance,
        load_official_tabm,
    )
    import torch

    expected = None if allow_unpinned else FROZEN_TABM_SOURCE_SHA256
    tabm_class, provenance = load_official_tabm(
        TabMProvenance(
            mode="source",
            source_path=source,
            expected_sha256=expected,
            expected_version=None if allow_unpinned else "0.0.3",
            # Stated explicitly rather than implied by passing None expectations,
            # and recorded as provenance["unpinned"] in the run contract.
            allow_unpinned=allow_unpinned,
        )
    )
    kwargs = dict(config.get("tabm_kwargs", {}))
    model = tabm_class.make(
        n_num_features=bundle.train.x.shape[1],
        cat_cardinalities=[],
        d_out=len(bundle.train.endpoint_ids),
        **kwargs,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"])
    )
    generator = np.random.default_rng(seed)
    best, best_state, stale = float("inf"), None, 0
    history: list[dict[str, Any]] = []
    batch_size = int(config["batch_size"])

    def predict(values: np.ndarray) -> np.ndarray:
        model.eval()
        parts = []
        with torch.no_grad():
            for start in range(0, len(values), batch_size):
                raw = model(torch.as_tensor(values[start : start + batch_size], device=device))
                if raw.ndim == 3:
                    raw = raw.mean(dim=1)
                parts.append(raw.detach().cpu().numpy())
        return np.concatenate(parts)

    for epoch in range(1, int(config["epochs"]) + 1):
        model.train()
        order = generator.permutation(len(bundle.train.x))
        for start in range(0, len(order), batch_size):
            rows = order[start : start + batch_size]
            pred = model(torch.as_tensor(bundle.train.x[rows], device=device))
            truth = torch.as_tensor(bundle.train.y[rows], device=device)
            mask = torch.as_tensor(bundle.train.observed[rows], device=device)
            if pred.ndim == 3:
                square = (pred - truth[:, None, :]).square()
                loss = torch.where(mask[:, None, :], square, torch.zeros_like(square)).sum() / mask.sum().clamp_min(1) / pred.shape[1]
            else:
                square = (pred - truth).square()
                loss = torch.where(mask, square, torch.zeros_like(square)).sum() / mask.sum().clamp_min(1)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        value = _validation_mse(predict(bundle.validation.x), bundle.validation)
        history.append({"epoch": epoch, "validation_mse": value})
        if value < best:
            best, stale = value, 0
            best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
        else:
            stale += 1
            if stale >= int(config.get("patience", 10)):
                break
    require(best_state is not None, "TabM produced no validation-selected checkpoint")
    model.load_state_dict(best_state, strict=True)
    model.eval()
    return (model, history, provenance)


def model_prediction(model: Any, values: np.ndarray, *, method: str, device: str) -> np.ndarray:
    if method in {"ridge", "gbdt", "catboost"}:
        return np.asarray(model.predict(values), dtype=np.float32)
    import torch

    model.eval()
    parts = []
    with torch.no_grad():
        for start in range(0, len(values), 8192):
            pred = model(torch.as_tensor(values[start : start + 8192], device=device))
            if pred.ndim == 3:
                pred = pred.mean(dim=1)
            parts.append(pred.detach().cpu().numpy())
    return np.concatenate(parts).astype(np.float32)


def run(args: argparse.Namespace) -> int:
    started = time.time()
    features = tuple(value.strip() for value in args.features.split(",") if value.strip()) if args.features else None
    tables = load_fold_tables(args.inputs, args.assignments, fold=args.fold, feature_columns=features)
    require(len(tables.feature_columns) == 172, f"OPS full-label runners require 172 features, found {len(tables.feature_columns)}")
    requested_reporters = selected_reporters(tables, args.reporters, args.smoke)
    test_reporters = set(
        tables.inputs.loc[tables.inputs.observation_id.isin(tables.test_ids), "reporter_id"].astype(str)
    )
    evaluation_reporters = tuple(name for name in requested_reporters if name in test_reporters)
    require(bool(evaluation_reporters), "selected fold has no test reporter")
    # Shared models retain source-only reporter heads as atlas supervision.
    # Specialists fit only heads that have an outer-test assay in this fold.
    reporters = requested_reporters if args.method in SHARED else evaluation_reporters
    config = load_config(args.config, method=args.method, smoke=args.smoke)
    contract = {
        "schema_version": SCHEMA_VERSION,
        "status": "CONTRACT_PASS",
        "method": args.method,
        "split_name": args.split_name,
        "fold": int(args.fold),
        "reporters": list(reporters),
        "n_reporters": len(reporters),
        "evaluation_reporters": list(evaluation_reporters),
        "n_evaluation_reporters": len(evaluation_reporters),
        "n_features": len(tables.feature_columns),
        "n_train": len(tables.train_ids),
        "n_validation": len(tables.validation_ids),
        "n_test": len(tables.test_ids),
        "checkpoint_selection": "validation_only" if args.method in {"mlp", "tabm", "resmlp", "multitab"} else "configuration_frozen_before_test",
        "test_targets_loaded": False,
        "config": config,
    }
    if args.method == "tabm":
        require(args.tabm_source is not None, "TabM requires --tabm-source pointing to pinned tabm.py or its directory")
        contract["tabm_source"] = str(args.tabm_source)
    if args.dry_run or args.contract_check_only:
        print(json.dumps(contract, indent=2, sort_keys=True))
        return 0

    if str(args.device).startswith("cuda"):
        import torch

        require(torch.cuda.is_available(), "CUDA was requested but is unavailable")
    set_seed(args.seed)
    args.output_root.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output_root / "run_contract.json", contract)
    input_scaler = fit_input_standardizer(tables)
    bundles = prepare_train_validation(tables, args.targets, reporters, input_scaler)
    atomic_json(
        args.output_root / "preprocessing.json",
        {
            "fit_role": "train",
            "feature_columns": list(tables.feature_columns),
            "input": input_scaler.manifest(),
            "targets": {name: bundle.scaler.manifest() for name, bundle in bundles.items()},
        },
    )
    predictions: list[pd.DataFrame] = []
    trajectories: list[pd.DataFrame] = []
    provenance: dict[str, Any] = {}
    fitted_models: dict[str, Any] = {}

    if args.method in SHARED:
        from measurement_sufficiency.ops_training import (  # local optional dependency
            ReporterBatch,
            fit_reporter_balanced,
        )

        dimensions = OrderedDict((name, len(bundle.train.endpoint_ids)) for name, bundle in bundles.items())
        architecture = dict(config["architecture"])
        if args.method == "multitab":
            architecture["reporter_panel"] = tuple(dimensions)
        model = create_ops_model(
            args.method,
            head_dimensions=dimensions,
            input_dim=172,
            seed=args.seed,
            **architecture,
        )
        fitted = fit_reporter_balanced(
            model,
            OrderedDict((name, ReporterBatch(bundle.train.x, bundle.train.y, bundle.train.observed)) for name, bundle in bundles.items()),
            OrderedDict((name, ReporterBatch(bundle.validation.x, bundle.validation.y, bundle.validation.observed)) for name, bundle in bundles.items()),
            epochs=int(config["epochs"]),
            tasks_per_batch=int(config["tasks_per_batch"]),
            samples_per_task=int(config["samples_per_task"]),
            learning_rate=float(config["learning_rate"]),
            weight_decay=float(config["weight_decay"]),
            seed=args.seed,
            device=args.device,
        )
        import torch

        torch.save({"method": args.method, "state_dict": fitted.model.state_dict(), "reporters": list(reporters)}, args.output_root / "checkpoint.pt")
        trajectories.append(pd.DataFrame(fitted.history).assign(reporter_id="shared"))
        for name, bundle in bundles.items():
            if name not in test_reporters:
                continue
            block, test = prepare_test(tables, args.targets, bundle, input_scaler)
            fitted.model.eval()
            with torch.no_grad():
                scaled = fitted.model(torch.as_tensor(test.x, device=args.device), name).detach().cpu().numpy()
            raw = bundle.scaler.inverse(scaled)
            predictions.append(canonical_predictions(metadata=test.metadata, block=block, prediction=raw, model_id=args.method, split_name=args.split_name, fold=args.fold))
    else:
        for name, bundle in bundles.items():
            seed = stable_seed(args.seed, args.method, name, args.fold)
            if args.method in {"ridge", "gbdt", "catboost"}:
                model, history = fit_sklearn(bundle, args.method, config, seed)
                local_provenance: dict[str, Any] = {}
            elif args.method == "mlp":
                model, history = fit_mlp(bundle, config, seed, args.device)
                local_provenance = {}
            else:
                model, history, local_provenance = fit_tabm(
                    bundle, config, seed, args.device, args.tabm_source, args.allow_unpinned_tabm
                )
            provenance[name] = local_provenance
            trajectories.append(pd.DataFrame(history).assign(reporter_id=name))
            fitted_models[name] = model
            block, test = prepare_test(tables, args.targets, bundle, input_scaler)
            scaled = model_prediction(model, test.x, method=args.method, device=args.device)
            raw = bundle.scaler.inverse(scaled)
            predictions.append(canonical_predictions(metadata=test.metadata, block=block, prediction=raw, model_id=args.method, split_name=args.split_name, fold=args.fold))
        checkpoint = args.output_root / ("checkpoint.pkl" if args.method in {"ridge", "gbdt", "catboost"} else "checkpoint.pt")
        if args.method in {"ridge", "gbdt", "catboost"}:
            with checkpoint.open("wb") as handle:
                pickle.dump(fitted_models, handle)
        else:
            import torch

            torch.save({"method": args.method, "states": {name: model.state_dict() for name, model in fitted_models.items()}}, checkpoint)

    output = pd.concat(predictions, ignore_index=True)
    validate_predictions(output)
    prediction_path = args.output_root / "predictions.csv"
    trajectory_path = args.output_root / "validation_trajectory.csv"
    write_frame(output, prediction_path)
    write_frame(pd.concat(trajectories, ignore_index=True), trajectory_path)
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "method_id": args.method,
        "split_name": args.split_name,
        "fold": int(args.fold),
        "reporters": list(reporters),
        "n_reporters": len(reporters),
        "n_prediction_rows": len(output),
        "outer_test_used_for_selection": False,
        "checkpoint_selection": contract["checkpoint_selection"],
        "smoke": bool(args.smoke),
        "runtime_seconds": time.time() - started,
        "predictions": str(prediction_path.resolve()),
        "predictions_sha256": file_sha256(prediction_path),
        "provenance": provenance,
    }
    atomic_json(args.output_root / "training_result.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def parser() -> argparse.ArgumentParser:
    output = argparse.ArgumentParser(description=__doc__)
    output.add_argument("--method", required=True, choices=METHODS)
    output.add_argument("--inputs", required=True, type=Path)
    output.add_argument("--targets", required=True, type=Path)
    output.add_argument("--assignments", required=True, type=Path)
    output.add_argument("--split-name", required=True, choices=("field", "gene", "strict_whole_screen"))
    output.add_argument("--fold", required=True, type=int)
    output.add_argument("--output-root", required=True, type=Path)
    output.add_argument("--reporters", default="all")
    output.add_argument("--features")
    output.add_argument("--config", type=Path)
    output.add_argument("--device", default="cuda")
    output.add_argument("--seed", type=int, default=20260721)
    output.add_argument("--tabm-source", type=Path)
    output.add_argument("--allow-unpinned-tabm", action="store_true")
    output.add_argument("--contract-check-only", action="store_true")
    output.add_argument("--dry-run", action="store_true")
    output.add_argument("--smoke", action="store_true")
    return output


if __name__ == "__main__":
    raise SystemExit(run(parser().parse_args()))
