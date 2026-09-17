"""Public DINOv2 and Cytoland encoder loaders used by the OPS image pilot."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np


def _strip_prefix(state: dict[str, Any], prefix: str) -> dict[str, Any]:
    if state and all(key.startswith(prefix) for key in state):
        return {key[len(prefix) :]: value for key, value in state.items()}
    return state


def load_dinov2(
    repository: Path,
    checkpoint: Path,
    entrypoint: str,
    device: "torch.device",
) -> "torch.nn.Module":
    import torch

    if not (repository / "hubconf.py").is_file():
        raise FileNotFoundError(f"DINOv2 repository lacks hubconf.py: {repository}")
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    model = torch.hub.load(str(repository), entrypoint, source="local", pretrained=False)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if isinstance(payload, dict):
        for key in ("teacher", "model", "state_dict", "backbone"):
            if key in payload and isinstance(payload[key], dict):
                payload = payload[key]
                break
    if not isinstance(payload, dict):
        raise RuntimeError("DINOv2 checkpoint is not a state dictionary")
    for prefix in ("module.", "backbone.", "teacher."):
        payload = _strip_prefix(payload, prefix)
    missing, unexpected = model.load_state_dict(payload, strict=False)
    loaded = len(model.state_dict()) - len(missing)
    if loaded < 0.8 * len(model.state_dict()):
        raise RuntimeError(
            f"DINOv2 checkpoint mismatch: loaded={loaded}/{len(model.state_dict())}; "
            f"missing={missing[:8]}; unexpected={unexpected[:8]}"
        )
    return model.eval().requires_grad_(False).to(device)


def normalize_dinov2(
    images: np.ndarray,
    windows: np.ndarray,
    *,
    low: float,
    high: float,
    output_size: int,
    device: "torch.device",
) -> "torch.Tensor":
    import torch
    import torch.nn.functional as functional

    if not high > low:
        raise ValueError("DINOv2 clip_high must exceed clip_low")
    values = np.asarray(images, dtype=np.float32).copy()
    for index, (y0, y1, x0, x1) in enumerate(np.asarray(windows, dtype=int)):
        valid = values[index, y0:y1, x0:x1]
        if valid.size == 0:
            raise RuntimeError(f"Empty valid window for image {index}")
        fill = float(np.median(valid))
        values[index, :y0], values[index, y1:] = fill, fill
        values[index, :, :x0], values[index, :, x1:] = fill, fill
    values = np.clip(values, low, high)
    values = (values - low) / (high - low)
    tensor = torch.from_numpy(values[:, None]).to(device, non_blocking=True)
    tensor = functional.interpolate(
        tensor, size=(output_size, output_size), mode="bilinear", align_corners=False
    ).repeat(1, 3, 1, 1)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device)[None, :, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], device=device)[None, :, None, None]
    return (tensor - mean) / std


def dinov2_embedding(model: "torch.nn.Module", batch: "torch.Tensor") -> "torch.Tensor":
    import torch

    result = model(batch)
    if isinstance(result, dict):
        for key in ("x_norm_clstoken", "cls_token", "embedding"):
            if key in result:
                result = result[key]
                break
    if isinstance(result, (tuple, list)):
        result = result[0]
    if not isinstance(result, torch.Tensor) or result.ndim != 2:
        raise RuntimeError(f"Unexpected DINOv2 output: {type(result)}, {getattr(result, 'shape', None)}")
    return result


def load_cytoland(
    repository: Path,
    checkpoint: Path,
    model_config: dict[str, Any],
    device: "torch.device",
) -> "torch.nn.Module":
    import torch

    source_candidates = (
        repository / "packages" / "viscy-models" / "src",
        repository / "src",
    )
    source = next((candidate for candidate in source_candidates if candidate.exists()), None)
    if source is None:
        raise FileNotFoundError(f"Cannot find VisCy model source below {repository}")
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    sys.path.insert(0, str(source))
    try:
        from viscy_models.unet.fcmae import MaskedMultiscaleEncoder
    except ImportError as error:
        raise ImportError(
            "Pinned VisCy source cannot expose viscy_models.unet.fcmae; install the "
            "repository revision documented in README.md"
        ) from error
    model = MaskedMultiscaleEncoder(
        in_channels=int(model_config["in_channels"]),
        stage_blocks=tuple(model_config["encoder_blocks"]),
        dims=tuple(model_config["dims"]),
        stem_kernel_size=tuple(model_config["stem_kernel_size"]),
        in_stack_depth=int(model_config["in_stack_depth"]),
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
    if not isinstance(state, dict):
        raise RuntimeError("Cytoland checkpoint is not a state dictionary")
    prefix = "model.encoder."
    encoder_state = {
        key.removeprefix(prefix): value for key, value in state.items() if key.startswith(prefix)
    }
    if not encoder_state:
        raise RuntimeError("Cytoland checkpoint has no model.encoder.* weights")
    missing, unexpected = model.load_state_dict(encoder_state, strict=False)
    loaded = len(model.state_dict()) - len(missing)
    if loaded < 0.98 * len(model.state_dict()) or unexpected:
        raise RuntimeError(
            f"Cytoland checkpoint mismatch: loaded={loaded}/{len(model.state_dict())}; "
            f"missing={missing[:8]}; unexpected={unexpected[:8]}"
        )
    return model.eval().requires_grad_(False).to(device)


def normalize_cytoland(
    images: np.ndarray, windows: np.ndarray, *, device: "torch.device"
) -> "torch.Tensor":
    import torch

    values = np.asarray(images, dtype=np.float32).copy()
    for index, (y0, y1, x0, x1) in enumerate(np.asarray(windows, dtype=int)):
        valid = values[index, y0:y1, x0:x1]
        if valid.size == 0:
            raise RuntimeError(f"Empty valid window for image {index}")
        median = float(np.median(valid))
        q25, q75 = np.percentile(valid, [25.0, 75.0])
        scale = max(float(q75 - q25), 1e-6)
        values[index, :y0], values[index, y1:] = median, median
        values[index, :, :x0], values[index, :, x1:] = median, median
        values[index] = (values[index] - median) / scale
    return torch.from_numpy(values[:, None, None]).to(device, non_blocking=True)


def cytoland_embedding(model: "torch.nn.Module", batch: "torch.Tensor") -> "torch.Tensor":
    import torch

    feature_maps, mask = model(batch, mask_ratio=0.0)
    if mask is not None or len(feature_maps) != 4:
        raise RuntimeError("Cytoland encoder did not return four unmasked feature maps")
    return torch.cat([feature.float().mean(dim=(-2, -1)) for feature in feature_maps], dim=1)
