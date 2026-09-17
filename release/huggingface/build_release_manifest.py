#!/usr/bin/env python3
"""Build a no-copy Hugging Face upload manifest for MorphoSuff.

The private manifest records absolute local source paths and is git-ignored.
The public TSV contains only destination paths, hashes, sizes and provenance.
No data or checkpoint is copied into a staging directory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

import h5py
import numpy as np


PRIVATE_PATH = re.compile(
    r"(?<![A-Za-z0-9_.-])/(?:" + "|".join(("data", "home", "tmp", "mnt", "Users", "root")) + r")/"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--canonical-root", type=Path, required=True)
    value.add_argument("--scpair-root", type=Path)
    value.add_argument("--cytoland-root", type=Path)
    value.add_argument("--phase-cache", type=Path)
    value.add_argument("--exact-cache-root", type=Path)
    value.add_argument("--asset-audit-root", type=Path)
    value.add_argument("--spec", type=Path, default=Path(__file__).with_name("release_spec.json"))
    value.add_argument("--private-manifest", type=Path, default=Path(__file__).with_name("local_upload_manifest.json"))
    value.add_argument("--public-manifest", type=Path, default=Path(__file__).with_name("MANIFEST.tsv"))
    value.add_argument("--without-checkpoints", action="store_true")
    value.add_argument("--without-training-data", action="store_true")
    return value


def text_is_public(path: Path, rejected: list[str]) -> bool:
    if path.suffix.lower() == ".parquet":
        return True
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return False
    return not PRIVATE_PATH.search(text) and not any(pattern in text for pattern in rejected)


def entry(path: Path, destination: str, kind: str, license_name: str, source_identity: str) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "local_path": str(path.resolve()),
        "repo_path": destination,
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "kind": kind,
        "license": license_name,
        "source_identity": source_identity,
    }


def add_unique(rows: list[dict], seen: set[str], row: dict) -> None:
    destination = str(row["repo_path"])
    if destination in seen:
        raise RuntimeError(f"Duplicate destination: {destination}")
    seen.add(destination)
    rows.append(row)


def add_processed_training_data(
    args: argparse.Namespace, rows: list[dict], seen: set[str], training_spec: dict
) -> dict:
    roots = {
        "phase_cache": args.phase_cache,
        "exact_cache_root": args.exact_cache_root,
        "asset_audit_root": args.asset_audit_root,
    }
    missing_arguments = [name for name, value in roots.items() if value is None]
    if missing_arguments:
        flags = ", ".join(f"--{name.replace('_', '-')}" for name in missing_arguments)
        raise ValueError(f"Processed OPS training data requires {flags}, or use --without-training-data")
    phase_root = args.phase_cache.expanduser().resolve()
    exact_root = args.exact_cache_root.expanduser().resolve()
    audit_root = args.asset_audit_root.expanduser().resolve()
    phase_files = tuple(training_spec["phase_files"])
    audit_files = tuple(training_spec["metadata_files"])
    for name in phase_files:
        path = phase_root / name
        add_unique(
            rows,
            seen,
            entry(
                path,
                f"processed_ops/phase172/{name}",
                "processed_training_input",
                "CC-BY-4.0",
                "OPS DOI:10.5281/zenodo.20495192; MorphoSuff deterministic phase172 preparation",
            ),
        )
    # Keep the public feature declaration beside the shared phase matrix.  The
    # local phase-cache manifest contains workstation provenance paths and is
    # deliberately not redistributed.
    feature_path = audit_root / "phase_features_172d.txt"
    add_unique(
        rows,
        seen,
        entry(
            feature_path,
            "processed_ops/phase172/phase_features_172d.txt",
            "processed_training_schema",
            "CC-BY-4.0",
            "OPS DOI:10.5281/zenodo.20495192; frozen 172D feature declaration",
        ),
    )
    for name in audit_files[1:]:
        path = audit_root / name
        if not text_is_public(path, ["app-"]):
            raise RuntimeError(f"Processed metadata contains a private path or host token: {path}")
        add_unique(
            rows,
            seen,
            entry(
                path,
                f"processed_ops/metadata/{name}",
                "processed_training_metadata",
                "CC-BY-4.0",
                "OPS DOI:10.5281/zenodo.20495192; MorphoSuff exact-link audit",
            ),
        )

    phase = np.load(phase_root / "phase172.npy", mmap_mode="r")
    if phase.shape != (8_410_291, 172) or phase.dtype != np.dtype("float32"):
        raise RuntimeError(f"Unexpected phase172 identity: shape={phase.shape}, dtype={phase.dtype}")
    exact_paths = sorted(exact_root.glob("all_cells_fluor_*.exact.h5"))
    if len(exact_paths) != 52:
        raise RuntimeError(f"Expected 52 exact reporter caches, found {len(exact_paths)}")
    required_h5 = {
        "phase_row_index",
        "fluorescence_row_index",
        "fluorescence",
        "features/target_feature_names",
        "metadata/screen_categories",
    }
    n_observations = 0
    n_assays = 0
    endpoint_distribution: dict[int, int] = {}
    screen_ids: set[str] = set()
    for path in exact_paths:
        with h5py.File(path, "r") as handle:
            absent = [key for key in required_h5 if key not in handle]
            if absent:
                raise RuntimeError(f"{path.name} lacks required datasets: {absent}")
            n_rows = int(handle["phase_row_index"].shape[0])
            if int(handle["fluorescence"].shape[0]) != n_rows:
                raise RuntimeError(f"Pairing/target row mismatch in {path.name}")
            n_endpoints = int(handle["fluorescence"].shape[1])
            endpoint_distribution[n_endpoints] = endpoint_distribution.get(n_endpoints, 0) + 1
            raw_screens = handle["metadata/screen_categories"][:]
            decoded = {value.decode() if isinstance(value, bytes) else str(value) for value in raw_screens}
            n_observations += n_rows
            n_assays += len(decoded)
            screen_ids.update(decoded)
        add_unique(
            rows,
            seen,
            entry(
                path,
                f"processed_ops/exact_reporters/{path.name}",
                "processed_training_target",
                "CC-BY-4.0",
                "OPS DOI:10.5281/zenodo.20495192; MorphoSuff exact same-cell reporter cache",
            ),
        )
    census = {
        "n_phase_cells": int(phase.shape[0]),
        "n_phase_features": int(phase.shape[1]),
        "n_reporters": len(exact_paths),
        "n_reporter_screen_assays": n_assays,
        "n_screens": len(screen_ids),
        "n_exact_same_cell_observations": n_observations,
        "endpoint_block_distribution": {str(key): value for key, value in sorted(endpoint_distribution.items())},
    }
    expected = training_spec["expected_census"]
    if census != expected:
        raise RuntimeError(f"Processed training census changed:\nobserved={census}\nexpected={expected}")
    return census


def main() -> int:
    args = parser().parse_args()
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    root = args.canonical_root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    canonical = spec["canonical"]
    allowed = set(canonical["include_extensions"])
    excluded = tuple(token.lower() for token in canonical["exclude_path_tokens"])
    rows: list[dict] = []
    seen: set[str] = set()

    for relative_root in canonical["include_roots"]:
        source_root = root / relative_root
        if not source_root.is_dir():
            raise FileNotFoundError(source_root)
        for path in sorted(source_root.rglob("*")):
            relative = path.relative_to(root).as_posix()
            lowered = relative.lower()
            if not path.is_file() or path.suffix.lower() not in allowed:
                continue
            if any(token in lowered for token in excluded):
                continue
            if not text_is_public(path, canonical["reject_text_patterns"]):
                continue
            parts = Path(relative).parts
            dataset_relative = "/".join((parts[0], *parts[2:])) if len(parts) > 2 and parts[1] == "source_data" else relative
            destination = f"source_data/{dataset_relative}"
            add_unique(rows, seen, entry(path, destination, "canonical_source_data", "CC-BY-4.0", "MorphoSuff canonical final"))

    for figure in spec["figures"]:
        path = root / figure["source"]
        destination = figure["destination"]
        add_unique(rows, seen, entry(path, destination, "rendered_figure", "CC-BY-4.0", "MorphoSuff canonical final"))

    training_census = None
    if not args.without_training_data:
        training_census = add_processed_training_data(args, rows, seen, spec["processed_training"])

    if not args.without_checkpoints:
        roots = {"scpair_root": args.scpair_root, "cytoland_root": args.cytoland_root}
        for checkpoint in spec["checkpoints"]:
            selected_root = roots[checkpoint["root_argument"]]
            if selected_root is None:
                raise ValueError(f"--{checkpoint['root_argument'].replace('_', '-')} is required unless --without-checkpoints is used")
            path = selected_root.resolve() / checkpoint["source"]
            destination = checkpoint["destination"]
            add_unique(rows, seen, entry(path, destination, checkpoint["kind"], checkpoint["license"], checkpoint["source_identity"]))

    rows.sort(key=lambda item: item["repo_path"])
    args.private_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.private_manifest.write_text(json.dumps({"schema_version": "morphosuff-local-upload-v1", "files": rows}, indent=2) + "\n", encoding="utf-8")
    with args.public_manifest.open("w", newline="", encoding="utf-8") as handle:
        fields = ["repo_path", "size_bytes", "sha256", "kind", "license", "source_identity"]
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            delimiter="\t",
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    total = sum(row["size_bytes"] for row in rows)
    print(json.dumps({"files": len(rows), "bytes": total, "mib": round(total / 1048576, 2), "training_census": training_census, "private_manifest": str(args.private_manifest), "public_manifest": str(args.public_manifest)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
