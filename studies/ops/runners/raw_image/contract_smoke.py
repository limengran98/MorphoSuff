#!/usr/bin/env python3
"""Exercise every raw-image entry point plus a fine-tuning backward pass."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from common import sha256_file
from finetune_cytoland import endpoint_model_from_encoder
from smoke import build_fixture


HERE = Path(__file__).resolve().parent
ENTRY_POINTS = (
    "build_crop_plan.py", "download_crops.py", "preflight.py",
    "extract_embeddings.py", "train_frozen_head.py", "finetune_cytoland.py",
    "build_plan.py", "run_plan.py",
)


def run(*arguments: str) -> str:
    command = [sys.executable, *map(str, arguments)]
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(
            "Raw-image contract command failed\n"
            f"command: {' '.join(command)}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    return completed.stdout


def tiny_backward(config: dict) -> None:
    import torch
    from torch import nn

    class FakeEncoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.stem = nn.Conv2d(1, 1, 1)
            self.stages = nn.ModuleList(
                [nn.Conv2d(1, width, 1) for width in (96, 192, 384, 768)]
            )

        def forward(self, image, mask_ratio: float = 0.0):
            del mask_ratio
            value = image.mean(dim=(-3, -2, -1)).reshape(len(image), 1, 1, 1)
            return [stage(value) for stage in self.stages], None

    model, counts = endpoint_model_from_encoder(
        config, 3, FakeEncoder(), torch.device("cpu")
    )
    prediction = model(torch.randn(2, 1, 1, 8, 8))
    if prediction.shape != (2, 3) or counts["active_parameters"] <= 0:
        raise RuntimeError("Fine-tuning model seam returned an invalid contract")
    prediction.square().mean().backward()
    if any(parameter.grad is not None for parameter in model.encoder.stages[0].parameters()):
        raise RuntimeError("Frozen Cytoland stage 0 received gradients")
    if not any(parameter.grad is not None for parameter in model.encoder.stages[2].parameters()):
        raise RuntimeError("Trainable Cytoland stage 2 did not receive gradients")
    if not any(parameter.grad is not None for parameter in model.head.parameters()):
        raise RuntimeError("Endpoint head did not receive gradients")


def main() -> None:
    help_checked = 0
    for entry in ENTRY_POINTS:
        output = run(str(HERE / entry), "--help")
        if "usage:" not in output.casefold():
            raise RuntimeError(f"--help failed for {entry}")
        help_checked += 1
    with tempfile.TemporaryDirectory(prefix="raw-image-contract-") as temporary:
        root = Path(temporary)
        fixture = build_fixture(root)
        checkpoint = root / "checkpoint.bin"
        checkpoint.write_bytes(b"public-checkpoint-smoke")
        digest = sha256_file(checkpoint)
        dinov2_repo = root / "dinov2"; dinov2_repo.mkdir()
        (dinov2_repo / "hubconf.py").write_text("# smoke\n", encoding="utf-8")
        viscy_repo = root / "VisCy"; (viscy_repo / "src").mkdir(parents=True)
        stage_b = root / "stage_b"; stage_b.mkdir()
        (stage_b / "manifest.json").write_text("{}", encoding="utf-8")
        locations = root / "locations.csv"
        pd.DataFrame({
            "phase_row_index": np.arange(150),
            "reporter_slugs": ["smoke_reporter"] * 150,
            "remote_zarr_https": ["https://ops-explorer-public.s3.amazonaws.com/smoke"] * 150,
            "level0_array_path": ["0"] * 150,
            "x_pheno": np.arange(150) + 256,
            "y_pheno": np.arange(150) + 256,
        }).to_csv(locations, index=False)
        config = {
            "reporters": ["smoke_reporter"],
            "encoder": {
                "entrypoint": "smoke", "expected_embedding_dimension": 1440,
                "checkpoint_sha256": digest,
                "model_config": {"in_channels": 1, "encoder_blocks": [3, 3, 9, 3],
                                 "dims": [96, 192, 384, 768],
                                 "stem_kernel_size": [1, 2, 2], "in_stack_depth": 1},
            },
            "image_transform": {"clip_low": -1, "clip_high": 1, "resize": 16},
            "evaluation": {"folds": [0, 1, 2, 3, 4]},
            "training": {
                "trainable_stage_indices": [2, 3], "head_hidden": [16, 8],
                "head_learning_rate": 0.0003, "backbone_learning_rate": 0.00001,
                "head_weight_decay": 0.0001, "backbone_weight_decay": 0.01,
                "warmup_epochs": 1, "epochs": 2, "patience": 1, "batch_size": 4,
            },
        }
        config_path = root / "config.json"
        config_path.write_text(json.dumps(config), encoding="utf-8")
        crop_plan = root / "crop_plan"
        run(str(HERE / "build_crop_plan.py"), "--locations", locations,
            "--config", config_path, "--output-root", crop_plan, "--format", "csv")
        run(str(HERE / "download_crops.py"), "--plan-root", crop_plan,
            "--output-root", root / "crops", "--dry-run")
        cells = crop_plan / "cells.csv"
        assets = {
            "phase_cache": str(fixture["phase"]), "exact_cache_root": str(fixture["exact"]),
            "target_feature_dictionary": str(fixture["dictionary"]),
            "stage_b_root": str(stage_b), "stage_b_cells": str(cells),
            "dinov2_repository": str(dinov2_repo), "dinov2_checkpoint": str(checkpoint),
            "dinov2_embedding_root": str(fixture["embedding"]),
            "viscy_repository": str(viscy_repo), "cytoland_checkpoint": str(checkpoint),
            "cytoland_embedding_root": str(fixture["embedding"]),
        }
        assets_path = root / "assets.json"
        assets_path.write_text(json.dumps(assets), encoding="utf-8")
        run(str(HERE / "preflight.py"), "--mode", "dinov2",
            "--config", config_path, "--assets", assets_path)
        run(str(HERE / "extract_embeddings.py"), "--encoder", "dinov2",
            "--config", config_path, "--data-root", stage_b, "--cells", cells,
            "--repository", dinov2_repo, "--checkpoint", checkpoint,
            "--output-root", root / "extract", "--dry-run")
        run(str(HERE / "finetune_cytoland.py"), "--reporter", "smoke_reporter",
            "--fold", "0", "--output-dir", root / "finetune", "--config", config_path,
            "--stage-b-root", stage_b, "--stage-b-plan", cells,
            "--viscy-repo", viscy_repo, "--checkpoint", checkpoint,
            "--phase-cache", fixture["phase"], "--exact-cache-root", fixture["exact"],
            "--target-feature-dictionary", fixture["dictionary"], "--dry-run")
        plan = root / "plan.json"
        run(str(HERE / "build_plan.py"), "--mode", "cytoland-finetune",
            "--config", config_path, "--assets", assets_path,
            "--result-root", root / "results", "--output-plan", plan)
        output = run(str(HERE / "run_plan.py"), "--plan", plan,
                     "--run-root", root / "run", "--max-jobs", "1", "--dry-run")
        if "DRY_RUN_PASS" not in output:
            raise RuntimeError("Plan-runner dry-run contract failed")
        tiny_backward(config)
    print(json.dumps({
        "status": "CONTRACT_SMOKE_PASS", "entry_point_help_checks": help_checked,
        "dry_run_contracts": 7, "finetune_forward_backward": 1,
    }, indent=2))


if __name__ == "__main__":
    main()
