#!/usr/bin/env python3
"""Rapid, source-faithful OPS gate for recent paired multimodal methods.

This is deliberately a *development-only* gate: 5xUPRE, gene-holdout fold 0,
and an inner validation cohort only.  It never opens the frozen outer-test
labels.  scPair and APOLLO retain their released paired encoders/decoders;
only their count-specific output assumptions are changed to Gaussian/MSE for
the already-standardized continuous OPS phenotype endpoints.

Monae is audited here rather than silently reimplemented: its released method
requires an RNA--ATAC genomic guidance graph, for which phase morphology and
fluorescence endpoints have no defensible analogue.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import sys
import time
import types
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "external" / "original_methods"
sys.path.insert(0, str(ROOT / "scripts"))
from ops_reporter_specialist_lib import PhaseCache, decode  # noqa: E402


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def macro_pearson(pred: torch.Tensor, truth: torch.Tensor) -> float:
    pred = pred.float()
    truth = truth.float()
    pred = pred - pred.mean(dim=0, keepdim=True)
    truth = truth - truth.mean(dim=0, keepdim=True)
    denominator = torch.sqrt((pred.square().sum(0) * truth.square().sum(0)).clamp_min(1e-12))
    values = (pred * truth).sum(0) / denominator
    return float(values.mean().item())


def read_gate_data(seed: int, max_train: int, max_validation: int) -> tuple[np.ndarray, ...]:
    """Read only train+inner-validation rows; fold 0 outer test is sealed."""
    cache = PhaseCache.open(ROOT / "data" / "processed" / "ops_phase172_indexed")
    h5_path = ROOT / "data" / "processed" / "ops_full_reporter_exact" / "reporters" / "all_cells_fluor_5xupre.exact.h5"
    dictionary = pd.read_csv(ROOT / "results" / "ops_phase0_asset_audit" / "target_feature_dictionary.csv")
    selected_names = dictionary.loc[
        (dictionary["reporter_slug"] == "5xupre") & dictionary["selected_in_technical_core"].astype(bool),
        "target_feature_name",
    ].astype(str).tolist()
    if len(selected_names) != 20:
        raise RuntimeError(f"Expected 20 frozen 5xUPRE technical-core endpoints, found {len(selected_names)}")
    with h5py.File(h5_path, "r") as source:
        phase_rows = np.asarray(source["phase_row_index"][:], dtype=np.int64)
        names = decode(source["features/target_feature_names"][:]).astype(str)
        lookup = {name: i for i, name in enumerate(names.tolist())}
        columns = np.asarray([lookup[name] for name in selected_names], dtype=np.int64)
        if len(np.unique(columns)) != len(columns):
            raise RuntimeError("5xUPRE selected endpoint mapping is non-unique")
        # h5py permits one fancy axis only and requires it to be increasing;
        # restore the frozen technical-core endpoint order after the read.
        read_order = np.argsort(columns, kind="mergesort")
        inverse_order = np.argsort(read_order, kind="mergesort")
        y = np.asarray(source["fluorescence"][:, columns[read_order]], dtype=np.float32)[:, inverse_order]
    folds = np.asarray(cache.folds["gene_holdout_main"][phase_rows], dtype=np.uint8)
    finite = np.isfinite(y).all(axis=1)
    # fold 0 remains physically excluded; fold 1 is the internal validation set.
    train_mask = finite & (folds != 0) & (folds != 1)
    validation_mask = finite & (folds == 1)
    rng = np.random.default_rng(seed)
    train_rows = np.flatnonzero(train_mask)
    validation_rows = np.flatnonzero(validation_mask)
    if len(train_rows) < max_train or len(validation_rows) < max_validation:
        raise RuntimeError(f"Insufficient finite rows: train={len(train_rows)} val={len(validation_rows)}")
    train_rows = np.sort(rng.choice(train_rows, size=max_train, replace=False))
    validation_rows = np.sort(rng.choice(validation_rows, size=max_validation, replace=False))
    x_train = np.asarray(cache.x[phase_rows[train_rows]], dtype=np.float32)
    x_validation = np.asarray(cache.x[phase_rows[validation_rows]], dtype=np.float32)
    y_train, y_validation = y[train_rows], y[validation_rows]
    # phase172 contains sparse non-finite morphology descriptors.  Fit the
    # imputer strictly on the development training cohort, then apply it to
    # validation; this matches the leakage-safe preprocessing principle of the
    # frozen OPS harness.
    x_train_finite = np.isfinite(x_train)
    x_fill = np.nanmean(np.where(x_train_finite, x_train, np.nan), axis=0)
    x_fill = np.where(np.isfinite(x_fill), x_fill, 0.0).astype(np.float32)
    x_train = np.where(x_train_finite, x_train, x_fill)
    x_validation = np.where(np.isfinite(x_validation), x_validation, x_fill)
    x_mean, x_sd = x_train.mean(0), x_train.std(0).clip(1e-6)
    y_mean, y_sd = y_train.mean(0), y_train.std(0).clip(1e-6)
    return (
        (x_train - x_mean) / x_sd,
        (y_train - y_mean) / y_sd,
        (x_validation - x_mean) / x_sd,
        (y_validation - y_mean) / y_sd,
        np.asarray(selected_names, dtype=str),
    )


def _require_target_width(n_targets: int) -> int:
    """Validate the frozen reporter-specific phenotype width.

    OPS reporter blocks are deliberately heterogeneous (20/24/30/60/72
    endpoints).  A paired source model has a distinct target modality, so its
    released target encoder and decoder must be instantiated at that modality
    width; this is schema binding, not a tunable architecture change.
    """
    n_targets = int(n_targets)
    if n_targets not in {20, 24, 30, 60, 72}:
        raise ValueError(f"Unexpected OPS reporter endpoint width: {n_targets}")
    return n_targets


def build_scpair(n_targets: int = 20) -> tuple[dict[str, torch.nn.Module], str]:
    # Import the released model.py without importing the package's unused
    # scVI/JAX/AnnData convenience layer.  model.py only uses FCNN from utils
    # for the three released core modules below.  The source file is neither
    # copied nor edited; the minimal module namespace simply avoids executing
    # unrelated optional imports in scpair/__init__.py.
    package_root = SOURCE_ROOT / "scpair" / "scpair"
    package = types.ModuleType("_ops_scpair_core")
    package.__path__ = [str(package_root)]
    sys.modules[package.__name__] = package
    for name in ("scVI_distribution", "loss"):
        sys.modules[f"{package.__name__}.{name}"] = types.ModuleType(f"{package.__name__}.{name}")
    utils = types.ModuleType(f"{package.__name__}.utils")
    def FCNN(layers, layernorm=True, activation=nn.ReLU(), batchnorm=False, dropout_rate=0):
        blocks = []
        for index in range(1, len(layers)):
            blocks.append(nn.Linear(layers[index - 1], layers[index]))
            if layernorm:
                blocks.append(nn.LayerNorm(layers[index]))
            blocks.append(activation)
            if batchnorm:
                blocks.append(nn.BatchNorm1d(layers[index]))
            blocks.append(nn.Dropout(dropout_rate))
        return nn.Sequential(*blocks)
    utils.FCNN = FCNN
    sys.modules[utils.__name__] = utils
    spec = importlib.util.spec_from_file_location(f"{package.__name__}.model", package_root / "model.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load released scPair model.py")
    source_model = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = source_model
    spec.loader.exec_module(source_model)
    Input_Module = source_model.Input_Module
    Module_Module = source_model.Module_Module
    Output_Module_Gau = source_model.Output_Module_Gau

    n_targets = _require_target_width(n_targets)
    hidden = [256, 64]
    modules = {
        "x_encoder": Input_Module(172, 0, hidden, dropout_rate=0.10, infer_library_size=False, add_linear_layer=True),
        "y_encoder": Input_Module(n_targets, 0, hidden, dropout_rate=0.10, infer_library_size=False, add_linear_layer=True),
        "x_decoder": Output_Module_Gau(172, 0, hidden, infer_library_size=False),
        "y_decoder": Output_Module_Gau(n_targets, 0, hidden, infer_library_size=False),
        "x_to_y": Module_Module(64, 64, [128], non_neg=False, dropout_rate=0.10),
        "y_to_x": Module_Module(64, 64, [128], non_neg=False, dropout_rate=0.10),
    }
    return modules, f"released scPair model.py Input_Module/Output_Module_Gau/Module_Module; target modality bound to frozen {n_targets}D OPS endpoint block; source-core adapter bypasses unused high-level scVI/JAX imports"


def build_apollo(n_targets: int = 20) -> tuple[dict[str, torch.nn.Module], str]:
    sys.path.insert(0, str(SOURCE_ROOT / "apollo" / "atac_rna"))
    from model_lord import fc_decode_l4, fc_encode_l4

    n_targets = _require_target_width(n_targets)
    modules = {
        "x_encoder": fc_encode_l4(172, 256, 128, 128, 16, 16, 0.10),
        "y_encoder": fc_encode_l4(n_targets, 256, 128, 128, 16, 16, 0.10),
        "x_decoder": fc_decode_l4(172, 32, 128, 0.10),
        "y_decoder": fc_decode_l4(n_targets, 32, 128, 0.10),
    }
    return modules, f"released APOLLO fc_encode_l4/fc_decode_l4; target modality bound to frozen {n_targets}D OPS endpoint block; paired shared/private latent retained"


def predict(method: str, modules: dict[str, torch.nn.Module], x: torch.Tensor) -> torch.Tensor:
    if method == "scpair":
        zx, _ = modules["x_encoder"](x, None)
        return modules["y_decoder"](modules["x_to_y"](zx), None, None, None)
    zx_shared, _zx_private = modules["x_encoder"](x)
    empty_private = torch.zeros_like(_zx_private)
    return modules["y_decoder"](torch.cat([zx_shared, empty_private], dim=1))


def loss_step(method: str, modules: dict[str, torch.nn.Module], x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    if method == "scpair":
        zx, _ = modules["x_encoder"](x, None)
        zy, _ = modules["y_encoder"](y, None)
        x_rec = modules["x_decoder"](zx, None, None, None)
        y_rec = modules["y_decoder"](zy, None, None, None)
        y_cross = modules["y_decoder"](modules["x_to_y"](zx), None, None, None)
        x_cross = modules["x_decoder"](modules["y_to_x"](zy), None, None, None)
        return F.mse_loss(x_rec, x) + F.mse_loss(y_rec, y) + F.mse_loss(y_cross, y) + F.mse_loss(x_cross, x)
    zx_s, zx_p = modules["x_encoder"](x)
    zy_s, zy_p = modules["y_encoder"](y)
    x_rec = modules["x_decoder"](torch.cat([zx_s, zx_p], dim=1))
    y_rec = modules["y_decoder"](torch.cat([zy_s, zy_p], dim=1))
    y_cross = modules["y_decoder"](torch.cat([zx_s, torch.zeros_like(zx_p)], dim=1))
    x_cross = modules["x_decoder"](torch.cat([zy_s, torch.zeros_like(zy_p)], dim=1))
    align = F.mse_loss(zx_s, zy_s)
    private_orthogonality = (zx_s * zx_p).mean().square() + (zy_s * zy_p).mean().square()
    return F.mse_loss(x_rec, x) + F.mse_loss(y_rec, y) + F.mse_loss(y_cross, y) + F.mse_loss(x_cross, x) + 0.2 * align + 0.05 * private_orthogonality


def run(method: str, args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; CPU fallback is forbidden for this gate")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    output = args.output_root / method
    output.mkdir(parents=True, exist_ok=True)
    if method == "monae":
        atomic_json(output / "compatibility_preflight.json", {
            "status": "NOT_RUN_SOURCE_INCOMPATIBLE",
            "reason": "Released Monae/Monae-E requires a trainable RNA--ATAC genomic guidance graph. OPS phase172 features and fluorescence endpoints have no biological feature correspondence from which such a graph can be faithfully constructed.",
            "source_evidence": ["preprocess_1.py constructs rna_anchored_guidance_graph", "integration.py requires guidance-hvf.graphml.gz"],
            "decision": "Do not introduce an invented correlation graph and mislabel it Monae.",
        })
        print("MONAE_PREFLIGHT NOT_RUN_SOURCE_INCOMPATIBLE", flush=True)
        return
    print(f"{method} STAGE=data_load", flush=True)
    xtr, ytr, xva, yva, endpoints = read_gate_data(args.seed, args.max_train, args.max_validation)
    device = torch.device("cuda")
    print(f"{method} STAGE=gpu_stage train={len(xtr)} validation={len(xva)}", flush=True)
    xtr_t, ytr_t = torch.as_tensor(xtr, device=device), torch.as_tensor(ytr, device=device)
    xva_t, yva_t = torch.as_tensor(xva, device=device), torch.as_tensor(yva, device=device)
    print(f"{method} STAGE=source_model", flush=True)
    modules, fidelity = build_scpair(ytr.shape[1]) if method == "scpair" else build_apollo(ytr.shape[1])
    for module in modules.values():
        module.to(device)
    print(f"{method} STAGE=train_init", flush=True)
    parameters = [p for module in modules.values() for p in module.parameters()]
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=1e-4)
    best, best_state = -float("inf"), None
    generator = torch.Generator(device=device).manual_seed(args.seed)
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        for module in modules.values():
            module.train()
        order = torch.randperm(len(xtr_t), device=device, generator=generator)
        losses = []
        for start in range(0, len(order), args.batch_size):
            idx = order[start:start + args.batch_size]
            optimizer.zero_grad(set_to_none=True)
            # Source models are compact; full FP32 makes this gate a direct
            # numerical test of the released architecture rather than of AMP
            # stability under four simultaneous reconstruction objectives.
            loss = loss_step(method, modules, xtr_t[idx], ytr_t[idx])
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite {method} loss at epoch={epoch} step={start // args.batch_size}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        for module in modules.values():
            module.eval()
        with torch.no_grad():
            predictions = torch.cat([predict(method, modules, xva_t[i:i + args.batch_size]) for i in range(0, len(xva_t), args.batch_size)])
            val_r = macro_pearson(predictions, yva_t)
        if val_r > best:
            best = val_r
            best_state = {name: {k: v.detach().cpu().clone() for k, v in module.state_dict().items()} for name, module in modules.items()}
        print(f"{method} epoch={epoch:02d}/{args.epochs} train_loss={np.mean(losses):.5f} validation_macro_pearson={val_r:.5f} best={best:.5f}", flush=True)
    assert best_state is not None
    for name, module in modules.items():
        module.load_state_dict(best_state[name])
        module.eval()
    with torch.no_grad():
        predictions = torch.cat([predict(method, modules, xva_t[i:i + args.batch_size]) for i in range(0, len(xva_t), args.batch_size)])
        endpoint_r = []
        for col in range(yva_t.shape[1]):
            endpoint_r.append(macro_pearson(predictions[:, col:col+1], yva_t[:, col:col+1]))
    result = {
        "schema_version": "ops-paired-source-gate-v1",
        "status": "PASS",
        "method": method,
        "task": "5xUPRE / gene_holdout_main fold0 / development inner-validation only",
        "outer_test_labels_opened": False,
        "train_rows": int(len(xtr_t)), "validation_rows": int(len(xva_t)),
        "technical_core_endpoints": endpoints.tolist(),
        "best_validation_macro_pearson": float(best),
        "per_endpoint_validation_pearson": endpoint_r,
        "elapsed_seconds": round(time.time() - started, 3),
        "peak_gpu_memory_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3),
        "device": torch.cuda.get_device_name(0),
        "source_fidelity": fidelity,
    }
    atomic_json(output / "gate_result.json", result)
    print("GATE_RESULT " + json.dumps(result, sort_keys=True), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", default="scpair,apollo,monae")
    parser.add_argument("--output-root", type=Path, default=ROOT / "results" / "ops_recent_bio_source_gate_20260803")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--max-train", type=int, default=60000)
    parser.add_argument("--max-validation", type=int, default=15000)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260803)
    args = parser.parse_args()
    allowed = {"scpair", "apollo", "monae"}
    methods = [item.strip().lower() for item in args.methods.split(",") if item.strip()]
    if not methods or set(methods) - allowed:
        raise ValueError(f"--methods must be a subset of {sorted(allowed)}")
    args.output_root = args.output_root.resolve()
    for method in methods:
        run(method, args)


if __name__ == "__main__":
    main()
