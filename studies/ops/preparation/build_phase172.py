#!/usr/bin/env python3
"""Build the row-addressable public OPS 172D phase cache and frozen folds."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np

try:  # direct script and package import are both supported
    from .features import derive_feature_names
    from .h5ad import categories, decode, file_sha256, require_h5ad_columns
except ImportError:  # pragma: no cover - exercised by CLI smoke
    from features import derive_feature_names
    from h5ad import categories, decode, file_sha256, require_h5ad_columns


SCHEMA_VERSION = "ops-phase172-index-v1"


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_npy(path: Path, values: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, values, allow_pickle=False)
    os.replace(temporary, path)


def splitmix64(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.uint64).copy()
    with np.errstate(over="ignore"):
        x += np.uint64(0x9E3779B97F4A7C15)
        x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        x ^= x >> np.uint64(31)
    return x


def combine_group_codes(*codes_: np.ndarray, seed: int) -> np.ndarray:
    result = splitmix64(np.zeros(len(codes_[0]), dtype=np.uint64) + np.uint64(seed))
    for offset, code in enumerate(codes_, start=1):
        encoded = (np.asarray(code, dtype=np.int64) + 2).astype(np.uint64, copy=False)
        result ^= splitmix64(encoded + np.uint64(seed + offset * 0x9E3779B1))
    return splitmix64(result)


def target_genes_from_caches(exact_root: Path, expected: int = 1000) -> set[str]:
    paths = sorted(exact_root.glob("all_cells_fluor_*.exact.h5"))
    if not paths:
        raise FileNotFoundError(f"no exact reporter caches under {exact_root}")
    sets: list[set[str]] = []
    for path in (paths[0], paths[-1]):
        with h5py.File(path, "r") as source:
            gene_categories = decode(source["metadata/gene_categories"][:])
            gene_codes = np.asarray(source["metadata/gene_codes"][:], dtype=np.int64)
            is_control = np.asarray(source["metadata/is_control"][:], dtype=bool)
        target_codes = np.unique(gene_codes[(gene_codes >= 0) & ~is_control])
        sets.append(set(gene_categories[target_codes]))
    if sets[0] != sets[1]:
        raise ValueError("reporter caches disagree on the target-gene library")
    if len(sets[0]) != expected:
        raise ValueError(f"expected {expected:,} target genes, found {len(sets[0])}")
    return sets[0]


def copy_phase_matrix(
    phase: h5py.File,
    output_dir: Path,
    feature_indices: np.ndarray,
    feature_names: list[str],
    chunk_rows: int,
) -> Path:
    source = phase["X"]
    if not isinstance(source, h5py.Dataset) or len(source.shape) != 2:
        raise ValueError("phase H5AD X must be a dense two-dimensional dataset")
    n_rows = int(source.shape[0])
    final = output_dir / "phase172.npy"
    temporary = output_dir / "phase172.building.npy"
    state_path = output_dir / "phase172.build_state.json"
    signature = {
        "schema_version": SCHEMA_VERSION,
        "n_rows": n_rows,
        "n_features": len(feature_indices),
        "feature_names": feature_names,
        "chunk_rows": int(chunk_rows),
    }
    if final.exists():
        matrix = np.load(final, mmap_mode="r")
        if matrix.shape != (n_rows, 172) or matrix.dtype != np.float32:
            raise ValueError(f"existing phase cache is incompatible: {final}")
        return final
    start = 0
    if temporary.exists() and state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if any(state.get(key) != value for key, value in signature.items()):
            raise ValueError("interrupted phase-cache state has a different contract")
        destination = np.lib.format.open_memmap(temporary, mode="r+")
        start = int(state["next_row"])
        print(f"resuming phase172 at row {start:,}/{n_rows:,}", flush=True)
    else:
        temporary.unlink(missing_ok=True)
        state_path.unlink(missing_ok=True)
        destination = np.lib.format.open_memmap(
            temporary, mode="w+", dtype=np.float32, shape=(n_rows, len(feature_indices))
        )
        atomic_json(state_path, {**signature, "next_row": 0})
    order = np.argsort(feature_indices)
    sorted_indices, restore = feature_indices[order], np.argsort(order)
    started = time.time()
    for block_number, row_start in enumerate(range(start, n_rows, chunk_rows), start=1):
        row_stop = min(row_start + chunk_rows, n_rows)
        block = np.asarray(source[row_start:row_stop, sorted_indices], dtype=np.float32)
        destination[row_start:row_stop] = block[:, restore]
        destination.flush()
        atomic_json(state_path, {**signature, "next_row": row_stop})
        if block_number % 16 == 0 or row_stop == n_rows:
            rate = (row_stop - start) / max(time.time() - started, 1e-6)
            eta = (n_rows - row_stop) / max(rate, 1e-6)
            print(f"phase172 {row_stop:,}/{n_rows:,}; {rate:,.0f} rows/s; ETA {eta / 60:.1f} min", flush=True)
    del destination
    os.replace(temporary, final)
    state_path.unlink(missing_ok=True)
    return final


def validate_matrix(
    phase: h5py.File, cache_path: Path, feature_indices: np.ndarray, seed: int
) -> None:
    cached = np.load(cache_path, mmap_mode="r")
    rng = np.random.default_rng(seed)
    rows = np.sort(rng.choice(cached.shape[0], size=min(1024, cached.shape[0]), replace=False))
    order = np.argsort(feature_indices)
    sorted_indices, restore = feature_indices[order], np.argsort(order)
    dataset = phase["X"]
    source = np.empty((len(rows), len(sorted_indices)), dtype=np.float32)
    row_chunk = int(dataset.chunks[0]) if dataset.chunks else 8192
    for chunk_id in np.unique(rows // row_chunk):
        positions = np.flatnonzero(rows // row_chunk == chunk_id)
        start = int(chunk_id * row_chunk)
        stop = min(start + row_chunk, dataset.shape[0])
        block = np.asarray(dataset[start:stop, sorted_indices], dtype=np.float32)
        source[positions] = block[rows[positions] - start]
    if not np.allclose(source[:, restore], np.asarray(cached[rows]), equal_nan=True, rtol=0, atol=0):
        raise RuntimeError("phase172 random-row source validation failed")


def build_metadata_and_folds(
    phase: h5py.File,
    output_dir: Path,
    target_genes: set[str],
    n_folds: int,
    seed: int,
) -> dict[str, Any]:
    column_map = {
        "screen_code": "portal_screen_name",
        "well_code": "well_canonical",
        "tile_code": "tile_pheno",
        "gene_code": "gene",
        "sgrna_code": "sgRNA",
    }
    arrays: dict[str, np.ndarray] = {}
    category_payload: dict[str, list[str]] = {}
    for output_name, source_name in column_map.items():
        arrays[output_name] = np.asarray(phase[f"obs/{source_name}/codes"][:], dtype=np.int32)
        category_payload[output_name] = categories(phase, "obs", source_name).tolist()
        atomic_npy(output_dir / f"{output_name}.npy", arrays[output_name])
    gene_categories = np.asarray(category_payload["gene_code"], dtype=str)
    category_is_target = np.asarray([name in target_genes for name in gene_categories], dtype=bool)
    gene_codes = arrays["gene_code"]
    valid = gene_codes >= 0
    is_target = np.zeros(len(gene_codes), dtype=bool)
    is_target[valid] = category_is_target[gene_codes[valid]]
    atomic_npy(output_dir / "is_target.npy", is_target)
    field_hash = combine_group_codes(
        arrays["screen_code"], arrays["well_code"], arrays["tile_code"], seed=seed
    )
    field_fold = np.asarray(field_hash % np.uint64(n_folds), dtype=np.uint8)
    gene_hash = splitmix64(gene_codes.astype(np.int64).astype(np.uint64) + np.uint64(seed))
    gene_fold = field_fold.copy()
    gene_fold[is_target] = np.asarray(gene_hash[is_target] % np.uint64(n_folds), dtype=np.uint8)
    atomic_npy(output_dir / "field_holdout_sanity.fold.npy", field_fold)
    atomic_npy(output_dir / "gene_holdout_main.fold.npy", gene_fold)
    atomic_json(output_dir / "categories.json", category_payload)
    return {
        "n_phase_rows": int(len(gene_codes)),
        "n_target_rows": int(is_target.sum()),
        "field_fold_counts": np.bincount(field_fold, minlength=n_folds).astype(int).tolist(),
        "gene_holdout_target_fold_counts": np.bincount(gene_fold[is_target], minlength=n_folds).astype(int).tolist(),
        "control_assignment": "screen + well + tile",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", type=Path, required=True)
    parser.add_argument("--exact-cache-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--features", type=Path, help="optional frozen 172-line manifest; otherwise derived from H5AD metadata")
    parser.add_argument("--chunk-rows", type=int, default=8192)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--expected-target-genes", type=int, default=1000)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    phase_path = args.phase.expanduser().resolve()
    exact_root = args.exact_cache_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not phase_path.is_file():
        raise FileNotFoundError(phase_path)
    if args.features:
        feature_names = [line.strip() for line in args.features.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        feature_names = derive_feature_names(phase_path)
    if len(feature_names) != 172 or len(set(feature_names)) != 172:
        raise ValueError("feature manifest must contain 172 unique names")
    target_genes = target_genes_from_caches(exact_root, args.expected_target_genes)
    plan = {
        "phase": str(phase_path),
        "exact_cache_root": str(exact_root),
        "output_dir": str(output_dir),
        "shape": ["phase_rows", 172],
        "n_target_genes": len(target_genes),
        "n_folds": args.n_folds,
        "seed": args.seed,
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return 0
    if args.n_folds < 3:
        raise ValueError("at least three folds are needed for train/validation/test")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists() and not args.overwrite:
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        required = [
            "phase172.npy",
            "screen_code.npy",
            "well_code.npy",
            "tile_code.npy",
            "gene_code.npy",
            "sgrna_code.npy",
            "is_target.npy",
            "field_holdout_sanity.fold.npy",
            "gene_holdout_main.fold.npy",
            "categories.json",
        ]
        if existing.get("schema_version") == SCHEMA_VERSION and all((output_dir / name).exists() for name in required):
            print(f"reusing verified phase172 cache: {output_dir}")
            return 0
        raise RuntimeError(f"existing phase172 output is incomplete: {output_dir}")
    if args.overwrite:
        for path in output_dir.glob("*.npy"):
            path.unlink()
        for name in ("manifest.json", "categories.json", "phase172.build_state.json"):
            (output_dir / name).unlink(missing_ok=True)
    started = time.time()
    with h5py.File(phase_path, "r") as phase:
        require_h5ad_columns(
            phase,
            obs=("portal_screen_name", "well_canonical", "tile_pheno", "gene", "sgRNA"),
            var=("_index",),
        )
        all_names = decode(phase["var/_index"][:])
        lookup = {name: index for index, name in enumerate(all_names)}
        missing = [name for name in feature_names if name not in lookup]
        if missing:
            raise ValueError(f"phase H5AD lacks locked features: {missing[:5]}")
        indices = np.asarray([lookup[name] for name in feature_names], dtype=np.int64)
        matrix_path = copy_phase_matrix(phase, output_dir, indices, feature_names, args.chunk_rows)
        validate_matrix(phase, matrix_path, indices, args.seed)
        split_summary = build_metadata_and_folds(phase, output_dir, target_genes, args.n_folds, args.seed)
    (output_dir / "phase_features_172d.txt").write_text("\n".join(feature_names) + "\n", encoding="utf-8")
    matrix = np.load(matrix_path, mmap_mode="r")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "source_phase_file": phase_path.name,
        "source_phase_sha256": file_sha256(phase_path),
        "feature_names": feature_names,
        "feature_rule": "frozen ordered OPS 172D morphology panel",
        "shape": list(matrix.shape),
        "dtype": str(matrix.dtype),
        "n_folds": args.n_folds,
        "seed": args.seed,
        "split_summary": split_summary,
        "validation": "1,024 source rows compared exactly",
        "build_seconds": time.time() - started,
    }
    atomic_json(manifest_path, manifest)
    print(f"wrote verified phase172 cache: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
