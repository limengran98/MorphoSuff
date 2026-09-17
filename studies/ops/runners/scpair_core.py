"""Thin OPS bridge to the pinned public scPair source model.

No scPair source is copied into this repository.  The caller supplies a clone
of the public repository; this module verifies the pinned ``model.py`` digest
and loads only the released encoder/decoder building blocks needed by OPS.
"""

from __future__ import annotations

import hashlib
import importlib.util
import sys
import types
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn
from torch.nn import functional as F


PINNED_COMMIT = "c585949ca8ea1314f5e68b260e3d9c5b2dabe61c"
PINNED_MODEL_SHA256 = "2aa5e9afdc9c634c0f55aaba3994b4254e219e3db749e336cef3c1e86c055409"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_model_file(upstream_root: str | Path) -> Path:
    root = Path(upstream_root).expanduser().resolve()
    candidates = (
        root / "scpair" / "model.py",
        root / "model.py",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"Could not find scpair/model.py below {root}; pass the public scPair checkout root"
    )


def verify_source(upstream_root: str | Path, *, strict: bool = True) -> dict[str, str]:
    model_file = resolve_model_file(upstream_root)
    observed = _sha256(model_file)
    if strict and observed != PINNED_MODEL_SHA256:
        raise RuntimeError(
            "scPair model.py does not match the pinned public revision: "
            f"observed={observed}, expected={PINNED_MODEL_SHA256}"
        )
    return {
        "repository": "https://github.com/quon-titative-biology/scPair.git",
        "revision": PINNED_COMMIT,
        "model_file": str(model_file),
        "model_sha256": observed,
        "expected_model_sha256": PINNED_MODEL_SHA256,
    }


def _fcnn(
    layers: list[int],
    layernorm: bool = True,
    activation: nn.Module = nn.ReLU(),
    batchnorm: bool = False,
    dropout_rate: float = 0,
) -> nn.Sequential:
    blocks: list[nn.Module] = []
    for index in range(1, len(layers)):
        blocks.append(nn.Linear(layers[index - 1], layers[index]))
        if layernorm:
            blocks.append(nn.LayerNorm(layers[index]))
        blocks.append(activation)
        if batchnorm:
            blocks.append(nn.BatchNorm1d(layers[index]))
        blocks.append(nn.Dropout(dropout_rate))
    return nn.Sequential(*blocks)


def _load_public_model(upstream_root: str | Path, *, strict_source: bool) -> Any:
    source = verify_source(upstream_root, strict=strict_source)
    model_file = Path(source["model_file"])
    package_root = model_file.parent
    namespace = f"_measurement_sufficiency_scpair_{source['model_sha256'][:12]}"
    existing = sys.modules.get(f"{namespace}.model")
    if existing is not None:
        return existing
    package = types.ModuleType(namespace)
    package.__path__ = [str(package_root)]
    sys.modules[namespace] = package
    for name in ("scVI_distribution", "loss"):
        sys.modules[f"{namespace}.{name}"] = types.ModuleType(f"{namespace}.{name}")
    utils = types.ModuleType(f"{namespace}.utils")
    utils.FCNN = _fcnn
    sys.modules[utils.__name__] = utils
    spec = importlib.util.spec_from_file_location(f"{namespace}.model", model_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load public scPair source: {model_file}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def build_modules(
    upstream_root: str | Path,
    n_targets: int,
    *,
    strict_source: bool = True,
) -> tuple[dict[str, nn.Module], dict[str, str]]:
    n_targets = int(n_targets)
    if n_targets <= 0:
        raise ValueError("n_targets must be positive")
    source_model = _load_public_model(upstream_root, strict_source=strict_source)
    hidden = [256, 64]
    modules = {
        "x_encoder": source_model.Input_Module(
            172, 0, hidden, dropout_rate=0.10, infer_library_size=False, add_linear_layer=True
        ),
        "y_encoder": source_model.Input_Module(
            n_targets, 0, hidden, dropout_rate=0.10, infer_library_size=False, add_linear_layer=True
        ),
        "x_decoder": source_model.Output_Module_Gau(172, 0, hidden, infer_library_size=False),
        "y_decoder": source_model.Output_Module_Gau(n_targets, 0, hidden, infer_library_size=False),
        "x_to_y": source_model.Module_Module(64, 64, [128], non_neg=False, dropout_rate=0.10),
        "y_to_x": source_model.Module_Module(64, 64, [128], non_neg=False, dropout_rate=0.10),
    }
    return modules, verify_source(upstream_root, strict=strict_source)


def predict(modules: Mapping[str, nn.Module], phase_x: torch.Tensor) -> torch.Tensor:
    latent, _ = modules["x_encoder"](phase_x, None)
    return modules["y_decoder"](modules["x_to_y"](latent), None, None, None)


def paired_loss(
    modules: Mapping[str, nn.Module],
    phase_x: torch.Tensor,
    phenotype_y: torch.Tensor,
) -> torch.Tensor:
    phase_latent, _ = modules["x_encoder"](phase_x, None)
    phenotype_latent, _ = modules["y_encoder"](phenotype_y, None)
    phase_reconstruction = modules["x_decoder"](phase_latent, None, None, None)
    phenotype_reconstruction = modules["y_decoder"](phenotype_latent, None, None, None)
    phenotype_cross = modules["y_decoder"](
        modules["x_to_y"](phase_latent), None, None, None
    )
    phase_cross = modules["x_decoder"](
        modules["y_to_x"](phenotype_latent), None, None, None
    )
    return (
        F.mse_loss(phase_reconstruction, phase_x)
        + F.mse_loss(phenotype_reconstruction, phenotype_y)
        + F.mse_loss(phenotype_cross, phenotype_y)
        + F.mse_loss(phase_cross, phase_x)
    )


def state_dict(modules: Mapping[str, nn.Module]) -> dict[str, dict[str, torch.Tensor]]:
    return {
        name: {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}
        for name, module in modules.items()
    }


def load_state_dict(
    modules: Mapping[str, nn.Module],
    state: Mapping[str, Mapping[str, torch.Tensor]],
) -> None:
    if set(modules) != set(state):
        raise ValueError("scPair checkpoint module set differs")
    for name, module in modules.items():
        module.load_state_dict(state[name], strict=True)

