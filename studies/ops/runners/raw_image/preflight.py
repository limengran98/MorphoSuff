#!/usr/bin/env python3
"""Fail-closed validation of public model and OPS raw-image assets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import sha256_file


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mode", choices=("dinov2", "cytoland-frozen", "cytoland-finetune"), required=True)
    return parser.parse_args()


def main() -> None:
    args = arguments()
    assets = json.loads(args.assets.read_text(encoding="utf-8"))
    config = json.loads(args.config.read_text(encoding="utf-8"))
    asset_base = args.assets.resolve().parent

    def path_for(key: str) -> Path:
        value = Path(str(assets.get(key, ""))).expanduser()
        return value if value.is_absolute() else asset_base / value

    file_keys = ["target_feature_dictionary"]
    directory_keys = ["phase_cache", "exact_cache_root"]
    if args.mode == "dinov2":
        file_keys += ["dinov2_checkpoint", "stage_b_cells"]
        directory_keys += ["dinov2_repository", "stage_b_root"]
        expected = config["encoder"].get("checkpoint_sha256")
        checkpoint_key = "dinov2_checkpoint"
    else:
        file_keys += ["cytoland_checkpoint", "stage_b_cells"]
        directory_keys += ["viscy_repository", "stage_b_root"]
        expected = config["encoder"].get("checkpoint_sha256")
        checkpoint_key = "cytoland_checkpoint"
    missing = []
    for key in file_keys:
        if not path_for(key).is_file():
            missing.append(f"{key}={assets.get(key)}")
    for key in directory_keys:
        if not path_for(key).is_dir():
            missing.append(f"{key}={assets.get(key)}")
    if missing:
        raise FileNotFoundError(f"Missing raw-image assets: {missing}")
    if args.mode == "dinov2" and not (path_for("dinov2_repository") / "hubconf.py").is_file():
        raise FileNotFoundError("DINOv2 repository does not contain hubconf.py")
    if args.mode != "dinov2":
        repository = path_for("viscy_repository")
        candidates = (repository / "packages" / "viscy-models" / "src", repository / "src")
        if not any(path.is_dir() for path in candidates):
            raise FileNotFoundError("VisCy repository does not expose its model source tree")
    if not (path_for("stage_b_root") / "manifest.json").is_file():
        raise FileNotFoundError("Stage-B crop root lacks manifest.json")
    observed = sha256_file(path_for(checkpoint_key))
    if expected and observed != expected:
        raise RuntimeError(f"Checkpoint SHA256 mismatch: {observed} != {expected}")
    for reporter in config["reporters"]:
        path = path_for("exact_cache_root") / f"all_cells_fluor_{reporter}.exact.h5"
        if not path.is_file():
            raise FileNotFoundError(path)
    print(json.dumps({
        "status": "PREFLIGHT_PASS", "mode": args.mode,
        "n_reporters": len(config["reporters"]),
        "n_folds": len(config["evaluation"]["folds"]),
        "checkpoint_sha256": observed,
    }, indent=2))


if __name__ == "__main__":
    main()
