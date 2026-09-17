#!/usr/bin/env python3
"""Source-faithful paired-model OPS formal benchmark task.

One process owns one scPair/APOLLO reporter task.  It deliberately supports
the three already-frozen full-label regimes without inventing a 52-reporter
shared variant of either source method:

* field / gene: one reporter and one frozen outer fold;
* strict whole-screen: one frozen reporter-to-destination-screen direction.

The only OPS adaptation is continuous Gaussian/MSE reconstruction of the
frozen, train-standardized phenotype vector.  The released paired encoders,
decoders and cross-modal objectives are retained.  Outer-test fluorescence is
not opened until the inner-validation checkpoint is frozen.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch

import run_ops_independent_specialist_protocol_task as strict_evaluator
import run_ops_neural_specialist_screen_task as screen_task
import run_ops_paired_source_gate as source_core
import run_ops_reporter_candidate_training as common
import run_ops_reporter_masked_multitask_resmlp_v2 as frozen
from ops_reporter_specialist_lib import PhaseCache, atomic_json


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "ops-paired-source-formal-task-v1"
METHODS = ("scpair_ops", "apollo_ops")
NORMAL_SPLITS = ("field_holdout_sanity", "gene_holdout_main")
SPLIT = (*NORMAL_SPLITS, "strict_whole_screen")
DEFAULT_PYTHON = Path(sys.executable)
DEFAULT_PHASE = ROOT / "data/processed/ops_phase172_indexed"
DEFAULT_EXACT = ROOT / "data/processed/ops_full_reporter_exact/reporters"
DEFAULT_TARGETS = ROOT / "results/ops_phase0_asset_audit/reporter_targets.csv"
DEFAULT_FEATURES = ROOT / "results/ops_phase0_asset_audit/target_feature_dictionary.csv"
# These two frozen artefact trees serve different purposes.  The ResMLP V1
# tree stores the preprocessing fit; the original specialist tree stores the
# sealed per-reporter outer-test identities.  Keeping them distinct prevents a
# successful training pass from failing only when it opens the test cohort.
DEFAULT_PREPROCESSING_REFERENCE = ROOT / "results/ops_reporter_masked_multitask_resmlp_v1"
DEFAULT_SPECIALIST_REFERENCE = ROOT / "results/ops_reporter_specialists_v1"
DEFAULT_STRICT = ROOT / "configs/ops_specialist_strict_screen_plan_v1/strict_tasks.json"

# Fixed, source-shaped recipe declared before any outer-test labels are read.
# It is intentionally not parameter-matched to the other methods.
RECIPE = {
    "scpair_ops": {"epochs": 40, "steps_per_epoch": 64, "batch_size": 16384, "learning_rate": 1.0e-4, "weight_decay": 1.0e-4},
    "apollo_ops": {"epochs": 40, "steps_per_epoch": 64, "batch_size": 16384, "learning_rate": 1.0e-4, "weight_decay": 1.0e-4},
}


class TaskError(RuntimeError):
    pass


def require(value: bool, message: str) -> None:
    if not value:
        raise TaskError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_seed(*parts: object) -> int:
    value = "\0".join(map(str, parts)).encode()
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "little") % (2**31 - 1)


def normalise_method(value: str) -> str:
    value = str(value)
    require(value in METHODS, f"Unknown paired source method: {value}")
    return value


def source_name(method: str) -> str:
    return "scpair" if method == "scpair_ops" else "apollo"


def build_modules(method: str, n_targets: int) -> tuple[dict[str, torch.nn.Module], str]:
    """Instantiate the released target branch at the frozen reporter width."""
    return source_core.build_scpair(n_targets) if method == "scpair_ops" else source_core.build_apollo(n_targets)


class Predictor:
    """Evaluator-facing adapter for a released paired source model."""

    def __init__(self, method: str, modules: Mapping[str, torch.nn.Module]) -> None:
        self.method = source_name(method)
        self.modules = dict(modules)

    def eval(self) -> "Predictor":
        for module in self.modules.values():
            module.eval()
        return self

    def forward_grouped(
        self, x: torch.Tensor, task_names: list[str] | tuple[str, ...], group_sizes: list[int] | tuple[int, ...]
    ) -> dict[str, torch.Tensor]:
        require(len(task_names) == len(group_sizes) == 1, "Paired specialist accepts one reporter group")
        prediction = source_core.predict(self.method, self.modules, x)
        require(len(prediction) == int(group_sizes[0]), "Paired prediction group size drifted")
        return {str(task_names[0]): prediction}


def module_state(modules: Mapping[str, torch.nn.Module]) -> dict[str, dict[str, torch.Tensor]]:
    return {
        name: {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}
        for name, module in modules.items()
    }


def load_module_state(modules: Mapping[str, torch.nn.Module], payload: Mapping[str, Any]) -> None:
    states = payload.get("module_states")
    require(isinstance(states, Mapping) and set(states) == set(modules), "Checkpoint module set drifted")
    for name, module in modules.items():
        module.load_state_dict(states[name], strict=True)
        module.to("cuda").eval()


def average_states(states: list[Mapping[str, Mapping[str, torch.Tensor]]]) -> dict[str, dict[str, torch.Tensor]]:
    require(bool(states), "No states supplied for checkpoint averaging")
    out: dict[str, dict[str, torch.Tensor]] = {}
    for module_name in states[0]:
        out[module_name] = {}
        for parameter_name, first in states[0][module_name].items():
            if not torch.is_floating_point(first):
                out[module_name][parameter_name] = first.clone()
                continue
            total = torch.zeros_like(first, dtype=torch.float64)
            for state in states:
                total += state[module_name][parameter_name].to(torch.float64)
            out[module_name][parameter_name] = (total / len(states)).to(first.dtype)
    return out


def technical_features(args: argparse.Namespace, slug: str) -> tuple[pd.Series, tuple[str, ...]]:
    table = pd.read_csv(args.target_table)
    require(len(table) == 52 and table.reporter_slug.nunique() == 52, "Frozen 52-reporter target table changed")
    rows = table.loc[table.reporter_slug.astype(str) == slug]
    require(len(rows) == 1, f"Unknown reporter: {slug}")
    technical = frozen.load_technical_core_features(args.target_feature_dictionary, table)
    return rows.iloc[0], tuple(map(str, technical[slug]))


def prepare_normal(args: argparse.Namespace) -> tuple[Any, Any, Any, dict[str, Any]]:
    require(args.reporter is not None, "Normal full-label task requires --reporter")
    reporter_row, features = technical_features(args, args.reporter)
    phase = PhaseCache.open(args.phase_cache)
    data = common.load_sealed_training_data(
        args.exact_root / f"all_cells_fluor_{args.reporter}.exact.h5",
        args.reporter, features, phase, args.split, args.fold,
    )
    head = common.build_sealed_training_partition(args, phase, reporter_row, data)
    reference = args.preprocessing_reference_root / args.split / f"fold_{args.fold}" / "train" / "preprocessing.json"
    x_state = common.load_reference_preprocessing(reference, [head])
    seal = {
        "mode": "normal_frozen_split",
        "outer_test_label_access_during_training": False,
        "reference_preprocessing": str(reference),
        "reference_preprocessing_sha256": sha256(reference),
        "n_train": len(head.active_train_indices),
        "n_validation": len(head.complete_validation_indices),
        "n_endpoints": len(head.feature_names),
    }
    return phase, head, x_state, seal


def prepare_screen(args: argparse.Namespace) -> tuple[Any, Any, Any, dict[str, Any], Mapping[str, Any]]:
    require(args.task_id is not None, "Strict screen task requires --task-id")
    task, direction_index, plan_sha = screen_task.load_screen_task(args.strict_plan, args.task_id)
    args.direction_index = direction_index
    args.strict_plan_sha256 = plan_sha
    args.seed = int(task["seed"])
    reporter_row, features = technical_features(args, str(task["reporter_slug"]))
    phase = PhaseCache.open(args.phase_cache)
    head, x_state, _contract, prepared = screen_task.load_accessible_training_head(
        args=args, task=task, phase_cache=phase, reporter_row=reporter_row,
        feature_names=features, strict_plan_sha256=plan_sha,
    )
    seal = dict(prepared["seal"])
    seal.update({"mode": "strict_screen", "task_id": args.task_id, "strict_plan_sha256": plan_sha})
    return phase, head, x_state, seal, prepared["test_identity"]


def tensor_values(
    phase: PhaseCache, head: Any, x_state: Any, indices: np.ndarray
) -> tuple[torch.Tensor, torch.Tensor]:
    require(head.y_preprocessing is not None, "Y preprocessing is not bound")
    rows = np.asarray(head.data.phase_rows[indices], dtype=np.int64)
    x = x_state.transform(np.asarray(phase.x[rows], dtype=np.float32))
    y = head.y_preprocessing.transform(np.asarray(head.data.y[indices], dtype=np.float32))
    require(np.isfinite(x).all() and np.isfinite(y).all(), "Non-finite training values after train-only preprocessing")
    return torch.from_numpy(x).to("cuda"), torch.from_numpy(y).to("cuda")


def validation_mse(method: str, modules: Mapping[str, torch.nn.Module], x: torch.Tensor, y: torch.Tensor, batch: int) -> float:
    for module in modules.values():
        module.eval()
    losses = []
    with torch.no_grad():
        for start in range(0, len(x), batch):
            stop = min(start + batch, len(x))
            predicted = source_core.predict(source_name(method), dict(modules), x[start:stop])
            losses.append(torch.mean((predicted - y[start:stop]).square()).detach())
    return float(torch.stack(losses).mean().item())


def train(args: argparse.Namespace, phase: Any, head: Any, x_state: Any) -> tuple[dict[str, Path], dict[str, Any]]:
    recipe = dict(RECIPE[args.method])
    if args.development_smoke:
        recipe.update({"epochs": 2, "steps_per_epoch": 2, "batch_size": min(2048, recipe["batch_size"])})
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    x_train, y_train = tensor_values(phase, head, x_state, head.active_train_indices)
    x_val, y_val = tensor_values(phase, head, x_state, head.complete_validation_indices)
    modules, fidelity = build_modules(args.method, len(head.feature_names))
    for module in modules.values():
        module.to("cuda")
    parameters = [parameter for module in modules.values() for parameter in module.parameters()]
    optimizer = torch.optim.AdamW(parameters, lr=recipe["learning_rate"], weight_decay=recipe["weight_decay"])
    ema = {name: {key: value.detach().clone() for key, value in module.state_dict().items()} for name, module in modules.items()}
    best_mse = float("inf"); best_state: dict[str, dict[str, torch.Tensor]] | None = None
    best_epochs: list[tuple[float, dict[str, dict[str, torch.Tensor]]]] = []
    trajectory: list[dict[str, Any]] = []
    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    started = time.time()
    for epoch in range(1, int(recipe["epochs"]) + 1):
        for module in modules.values(): module.train()
        losses = []
        for _ in range(int(recipe["steps_per_epoch"])):
            take = torch.randint(len(x_train), (int(recipe["batch_size"]),), device="cuda", generator=generator)
            optimizer.zero_grad(set_to_none=True)
            loss = source_core.loss_step(source_name(args.method), dict(modules), x_train[take], y_train[take])
            require(bool(torch.isfinite(loss)), f"Non-finite {args.method} loss")
            loss.backward(); torch.nn.utils.clip_grad_norm_(parameters, 1.0); optimizer.step()
            with torch.no_grad():
                for name, module in modules.items():
                    for key, value in module.state_dict().items():
                        if torch.is_floating_point(value): ema[name][key].mul_(0.999).add_(value.detach(), alpha=0.001)
                        else: ema[name][key].copy_(value.detach())
            losses.append(float(loss.detach()))
        score = validation_mse(args.method, modules, x_val, y_val, int(recipe["batch_size"]))
        current = module_state(modules)
        best_epochs.append((score, current)); best_epochs.sort(key=lambda item: item[0]); best_epochs = best_epochs[:3]
        if score < best_mse:
            best_mse, best_state = score, current
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), "validation_macro_mse": score, "best_validation_macro_mse": best_mse}
        trajectory.append(row)
        print(f"{args.method} reporter={head.slug} split={args.split} epoch={epoch:02d}/{recipe['epochs']} train_loss={row['train_loss']:.6f} validation_macro_mse={score:.6f} best={best_mse:.6f}", flush=True)
    require(best_state is not None, "No validation-selected source-model state")
    states = {"single": best_state, "ema": {name: {key: value.detach().cpu() for key, value in values.items()} for name, values in ema.items()}, "checkpoint_average": average_states([state for _, state in best_epochs])}
    paths: dict[str, Path] = {}
    for state_name, state in states.items():
        destination = args.output_root / "evaluation_state_candidates" / f"{state_name}.pt"
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"schema_version": SCHEMA, "method_id": args.method, "reporter": head.slug, "module_states": state, "source_fidelity": fidelity}, destination)
        paths[state_name] = destination
    pd.DataFrame(trajectory).to_csv(args.output_root / "validation_trajectory.csv", index=False)
    torch.cuda.synchronize()
    complete = {"schema_version": SCHEMA, "status": "complete", "method_id": args.method, "reporter_slug": head.slug, "split": args.split, "fold": int(args.fold), "recipe": recipe, "source_fidelity": fidelity, "checkpoint_selection": {"selection_partition": "validation", "test_used_for_selection": False, "validation_macro_mse": best_mse, "selected_epoch": int(np.argmin([x['validation_macro_mse'] for x in trajectory]) + 1)}, "state_paths": {name: str(path.resolve()) for name, path in paths.items()}, "runtime_seconds": time.time() - started, "outer_test_label_accessed": False}
    atomic_json(args.output_root / "training_complete.json", complete)
    del x_train, y_train, x_val, y_val, modules, optimizer
    gc.collect(); torch.cuda.empty_cache()
    return paths, complete


def predict(method: str, modules: Mapping[str, torch.nn.Module], phase: Any, rows: np.ndarray, x_state: Any, batch: int) -> np.ndarray:
    output = []
    for start in range(0, len(rows), batch):
        values = x_state.transform(np.asarray(phase.x[rows[start:start + batch]], dtype=np.float32))
        tensor = torch.from_numpy(values).to("cuda")
        with torch.no_grad(): output.append(source_core.predict(source_name(method), dict(modules), tensor).float().cpu().numpy())
    return np.concatenate(output, axis=0)


def load_normal_outer_test(args: argparse.Namespace, phase: Any, head: Any) -> dict[str, Any]:
    """Open the frozen outer fold only after checkpoint selection.

    Earlier comparators place their duplicated test identities inside each
    method-specific evaluation tree rather than a central specialist-reference
    directory.  Reconstructing the fold directly from the immutable phase
    cache is therefore both method-independent and exactly the split contract
    used during sealed training.
    """
    _row, features = technical_features(args, head.slug)
    cache = args.exact_root / f"all_cells_fluor_{head.slug}.exact.h5"
    with __import__("h5py").File(cache, "r") as source:
        phase_rows = np.asarray(source["phase_row_index"][:], dtype=np.int64)
        controls = np.asarray(source["metadata/is_control"][:], dtype=bool)
        columns, names = common.selected_feature_columns(source, features, head.slug)
        folds = np.asarray(phase.folds[args.split][phase_rows], dtype=np.uint8)
        h5_rows = np.flatnonzero(folds == args.fold).astype(np.int64)
        truth = common.read_h5_rows_columns(source["fluorescence"], h5_rows, columns)
    complete = np.isfinite(truth).all(axis=1)
    require(bool(complete.any()), f"No complete outer-test rows for {head.slug}")
    return {
        "truth_raw": np.asarray(truth[complete], dtype=np.float32),
        "phase_rows": phase_rows[h5_rows][complete],
        "is_control": controls[h5_rows][complete],
        "feature_names": np.asarray(names, dtype=str),
        "n_outer_identity_rows": int(len(h5_rows)),
        "n_outer_complete_rows": int(complete.sum()),
    }


def evaluate_normal(args: argparse.Namespace, phase: Any, head: Any, x_state: Any, states: Mapping[str, Path]) -> list[dict[str, Any]]:
    assert head.y_preprocessing is not None
    test = load_normal_outer_test(args, phase, head)
    rows = []
    for state_name, path in states.items():
        modules, _ = build_modules(args.method, len(head.feature_names))
        checkpoint = torch.load(path, map_location="cpu", weights_only=False); load_module_state(modules, checkpoint)
        prediction_scaled = predict(args.method, modules, phase, test["phase_rows"], x_state, args.prediction_batch_size)
        prediction_raw = head.y_preprocessing.inverse(prediction_scaled)
        result = strict_evaluator.evaluate_prediction(
            output_dir=args.output_root / "evaluation_states" / state_name / "reporters" / head.slug,
            truth_raw=test["truth_raw"], prediction_raw=prediction_raw, truth_scaled=head.y_preprocessing.transform(test["truth_raw"]), prediction_scaled=prediction_scaled,
            is_control=test["is_control"], screen_codes=np.asarray(phase.metadata["screen_code"][test["phase_rows"]], dtype=np.int32), gene_codes=np.asarray(phase.metadata["gene_code"][test["phase_rows"]], dtype=np.int32), feature_names=test["feature_names"], baseline_mean_raw=head.y_preprocessing.mean, seed=args.seed,
            binding={"method_id": args.method, "reporter_slug": head.slug, "split": args.split, "fold": args.fold, "evaluation_state": state_name}, save_predictions=False,
        )
        rows.append({"evaluation_state": state_name, **{f"cell_{k}": v for k, v in result["cell_metrics"].items()}, **{f"gene_{k}": v for k, v in result["gene_metrics"].items()}})
        del modules; gc.collect(); torch.cuda.empty_cache()
    return rows


def evaluate_screen(args: argparse.Namespace, phase: Any, head: Any, x_state: Any, identity: Mapping[str, Any], states: Mapping[str, Path]) -> list[dict[str, Any]]:
    assert head.y_preprocessing is not None
    test = screen_task.load_test_labels(identity, head.y_preprocessing)
    rows = []
    for state_name, path in states.items():
        modules, _ = build_modules(args.method, len(head.feature_names))
        checkpoint = torch.load(path, map_location="cpu", weights_only=False); load_module_state(modules, checkpoint)
        prediction_scaled = predict(args.method, modules, phase, test["phase_rows"], x_state, args.prediction_batch_size)
        result = strict_evaluator.evaluate_prediction(
            output_dir=args.output_root / "evaluation_states" / state_name / "directions" / args.task_id,
            truth_raw=test["truth_raw"], prediction_raw=head.y_preprocessing.inverse(prediction_scaled), truth_scaled=test["truth_scaled"], prediction_scaled=prediction_scaled,
            is_control=test["is_control"], screen_codes=np.asarray(phase.metadata["screen_code"][test["phase_rows"]], dtype=np.int32), gene_codes=np.asarray(phase.metadata["gene_code"][test["phase_rows"]], dtype=np.int32), feature_names=test["feature_names"], baseline_mean_raw=head.y_preprocessing.mean, seed=args.seed,
            binding={"method_id": args.method, "task_id": args.task_id, "reporter_slug": head.slug, "split": args.split, "fold": args.fold, "evaluation_state": state_name}, save_predictions=False,
        )
        rows.append({"evaluation_state": state_name, "task_id": args.task_id, "destination_screen": identity.get("destination_screen"), **{f"cell_{k}": v for k, v in result["cell_metrics"].items()}, **{f"gene_{k}": v for k, v in result["gene_metrics"].items()}})
        del modules; gc.collect(); torch.cuda.empty_cache()
    return rows


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--method", required=True, choices=METHODS)
    p.add_argument("--split", required=True, choices=SPLIT)
    p.add_argument("--fold", required=True, type=int, choices=range(5))
    p.add_argument("--reporter")
    p.add_argument("--task-id")
    p.add_argument("--output-root", required=True, type=Path)
    p.add_argument("--phase-cache", type=Path, default=DEFAULT_PHASE); p.add_argument("--exact-root", type=Path, default=DEFAULT_EXACT)
    p.add_argument("--target-table", type=Path, default=DEFAULT_TARGETS); p.add_argument("--target-feature-dictionary", type=Path, default=DEFAULT_FEATURES)
    p.add_argument("--preprocessing-reference-root", type=Path, default=DEFAULT_PREPROCESSING_REFERENCE); p.add_argument("--specialist-reference-root", type=Path, default=DEFAULT_SPECIALIST_REFERENCE)
    p.add_argument("--strict-plan", type=Path, default=DEFAULT_STRICT); p.add_argument("--source-validation-field-fold", type=int, default=0, choices=range(5))
    p.add_argument("--seed", type=int, default=20260721); p.add_argument("--prediction-batch-size", type=int, default=65536)
    # Kept identical to the established strict-screen loader.  It is not a
    # tunable model choice: it simply asserts that the frozen phase172 input
    # columns are finite in the source-training cohort before preprocessing.
    p.add_argument("--min-x-finite-fraction", type=float, default=0.80)
    p.add_argument("--workers", type=int, default=4); p.add_argument("--contract-check-only", action="store_true"); p.add_argument("--development-smoke", action="store_true")
    return p


def main() -> None:
    args = parser().parse_args(); args.method = normalise_method(args.method)
    for name in ("output_root", "phase_cache", "exact_root", "target_table", "target_feature_dictionary", "preprocessing_reference_root", "specialist_reference_root", "strict_plan"):
        setattr(args, name, Path(getattr(args, name)).expanduser().resolve())
    if args.split in NORMAL_SPLITS: require(args.reporter is not None and args.task_id is None, "Normal split requires --reporter only")
    else: require(args.task_id is not None and args.reporter is None, "Strict screen split requires --task-id only")
    if args.contract_check_only:
        print(json.dumps({"status": "PASS", "method": args.method, "split": args.split, "fold": args.fold, "cuda_required": True, "cpu_fallback_allowed": False, "outer_test_sealed_until_checkpoint": True, "source_core": "released scPair/APOLLO"}, indent=2)); return
    require(torch.cuda.is_available(), "CUDA is required; CPU fallback is forbidden")
    if args.output_root.exists() and (args.output_root / "training_result.json").is_file() and not args.development_smoke:
        previous = json.loads((args.output_root / "training_result.json").read_text())
        if previous.get("status") == "complete" and previous.get("method_id") == args.method and previous.get("split") == args.split and int(previous.get("fold", -1)) == args.fold:
            print(json.dumps({"status": "ALREADY_COMPLETE", "output_root": str(args.output_root)}, indent=2)); return
        raise TaskError(f"Incompatible final marker: {args.output_root / 'training_result.json'}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    if args.split in NORMAL_SPLITS: phase, head, x_state, seal = prepare_normal(args); test_identity = None
    else: phase, head, x_state, seal, test_identity = prepare_screen(args)
    atomic_json(args.output_root / "formal_training_data_seal.json", seal)
    gpu = {"cuda_required": True, "cpu_fallback_allowed": False, "device": torch.cuda.get_device_name(), "torch": torch.__version__}; atomic_json(args.output_root / "gpu_preflight.json", gpu)
    states, complete = train(args, phase, head, x_state)
    if args.development_smoke:
        print(json.dumps({"status": "SMOKE_COMPLETE", "output_root": str(args.output_root), "training_complete": complete}, indent=2)); return
    results = evaluate_normal(args, phase, head, x_state, states) if test_identity is None else evaluate_screen(args, phase, head, x_state, test_identity, states)
    pd.DataFrame(results).to_csv(args.output_root / "evaluation_states_summary.csv", index=False)
    result = {"schema_version": SCHEMA, "status": "complete", "completed": True, "mode": "formal", "method_id": args.method, "reporters_argument": args.reporter or "strict_direction", "reporter_slug": head.slug, "split": args.split, "fold": args.fold, "task_id": args.task_id, "n_reporters": 1, "n_evaluation_states": len(results), "outer_test_loaded_after_checkpoint_selection": True, "outer_test_used_for_selection": False, "training_complete_sha256": sha256(args.output_root / "training_complete.json"), "evaluation_states_summary_sha256": sha256(args.output_root / "evaluation_states_summary.csv")}
    if args.split == "strict_whole_screen": result["global_screen_removed_from_every_reporter"] = True
    atomic_json(args.output_root / "training_result.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    try: main()
    except TaskError as error: raise SystemExit(f"ERROR: {error}")
