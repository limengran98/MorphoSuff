#!/usr/bin/env python3
"""Export exact OPS caches as runnable per-reporter canonical bundles.

Each bundle contains wide 172D inputs, observed sparse targets, exact pairing
tables, and field/gene split assignments. A compact atlas directory contains
the reporter, endpoint and reporter-screen assay registries plus the 81
strict whole-screen destination tasks. CSV and Parquet are both supported;
Parquet is recommended for the full public atlas.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np
import pandas as pd

try:  # direct script and package import are both supported
    from .h5ad import decode, file_sha256
except ImportError:  # pragma: no cover - exercised by CLI smoke
    from h5ad import decode, file_sha256


def reporter_slug(path: Path) -> str:
    prefix, suffix = "all_cells_fluor_", ".exact.h5"
    if not path.name.startswith(prefix) or not path.name.endswith(suffix):
        raise ValueError(f"unexpected exact-cache name: {path.name}")
    return path.name[len(prefix) : -len(suffix)]


def _sha256_payload(payload: Any) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class StreamTable:
    def __init__(self, path: Path, format_: str):
        self.path, self.format = path, format_
        self.first = True
        self.writer: Any = None

    def write(self, frame: pd.DataFrame) -> None:
        if frame.empty:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.format == "csv":
            frame.to_csv(self.path, mode="w" if self.first else "a", header=self.first, index=False)
        else:
            try:
                import pyarrow as pa
                import pyarrow.parquet as pq
            except ImportError as error:
                raise RuntimeError("Parquet export requires pyarrow; install .[parquet] or use --format csv") from error
            table = pa.Table.from_pandas(frame, preserve_index=False)
            if self.writer is None:
                self.writer = pq.ParquetWriter(self.path, table.schema, compression="zstd")
            self.writer.write_table(table)
        self.first = False

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()
        if self.first:
            raise RuntimeError(f"no rows were written to {self.path}")


def output_path(root: Path, stem: str, format_: str) -> Path:
    return root / f"{stem}.{format_}"


def _values(categories_: np.ndarray, codes_: np.ndarray) -> np.ndarray:
    codes_ = np.asarray(codes_, dtype=np.int64)
    output = np.full(len(codes_), "", dtype=object)
    valid = codes_ >= 0
    output[valid] = categories_[codes_[valid]]
    return output.astype(str)


def load_registry(path: Path) -> pd.DataFrame:
    registry = pd.read_csv(path)
    required = {"reporter_slug", "reporter_name", "short_name", "biological_system"}
    missing = required.difference(registry.columns)
    if missing:
        raise ValueError(f"reporter registry lacks {sorted(missing)}")
    if len(registry) != 52 or registry.reporter_slug.duplicated().any():
        raise ValueError("the frozen OPS registry must contain 52 unique reporters")
    return registry


def load_phase_feature_names(phase_root: Path) -> list[str]:
    text_path = phase_root / "phase_features_172d.txt"
    if text_path.is_file():
        names = [line.strip() for line in text_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        manifest = json.loads((phase_root / "manifest.json").read_text(encoding="utf-8"))
        names = [str(value) for value in manifest.get("feature_names", [])]
    if len(names) != 172 or len(set(names)) != 172:
        raise ValueError("phase cache must declare 172 unique feature names")
    return names


def build_atlas_tables(
    cache_paths: list[Path], registry: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    registry_by_slug = registry.set_index("reporter_slug")
    reporters: list[dict[str, Any]] = []
    assays: list[dict[str, Any]] = []
    endpoints: list[dict[str, Any]] = []
    strict_tasks: list[dict[str, Any]] = []
    for cache_path in cache_paths:
        slug = reporter_slug(cache_path)
        if slug not in registry_by_slug.index:
            raise ValueError(f"reporter is absent from frozen registry: {slug}")
        metadata = registry_by_slug.loc[slug]
        with h5py.File(cache_path, "r") as cache:
            endpoint_ids = decode(cache["features/target_feature_names"][:]).tolist()
            screens = decode(cache["metadata/screen_categories"][:]).tolist()
        schema_id = f"ops-target-{slug}-v1"
        reporters.append(
            {
                "reporter_id": slug,
                "reporter_name": metadata.reporter_name,
                "short_name": metadata.short_name,
                "biological_system": metadata.biological_system,
                "target_modality": "targeted_fluorescence",
                "endpoint_schema_id": schema_id,
            }
        )
        endpoint_text = ",".join(endpoint_ids)
        assays.extend(
            {
                "assay_id": f"{slug}::{screen}",
                "reporter_id": slug,
                "screen_id": screen,
                "endpoint_ids": endpoint_text,
            }
            for screen in screens
        )
        endpoints.extend(
            {
                "endpoint_schema_id": schema_id,
                "reporter_id": slug,
                "endpoint_id": endpoint,
                "target_modality": "targeted_fluorescence",
            }
            for endpoint in endpoint_ids
        )
        if len(screens) > 1:
            strict_tasks.extend(
                {
                    "task_id": f"strict-screen::{slug}::to::{destination}",
                    "reporter_id": slug,
                    "destination_screen": destination,
                    "source_screens": ";".join(screen for screen in screens if screen != destination),
                    "source_validation_field_fold": 0,
                }
                for destination in screens
            )
    return (
        pd.DataFrame(reporters),
        pd.DataFrame(assays),
        pd.DataFrame(endpoints),
        pd.DataFrame(strict_tasks),
    )


def write_small_tables(
    output_root: Path,
    tables: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> None:
    atlas = output_root / "atlas"
    atlas.mkdir(parents=True, exist_ok=True)
    for name, table in zip(("reporters", "assays", "endpoints", "strict_screen_tasks"), tables):
        table.to_csv(atlas / f"{name}.csv", index=False)


def assignment_frame(
    observation_ids: np.ndarray,
    fold_values: np.ndarray,
    split_name: str,
    is_control: np.ndarray | None = None,
) -> Iterable[pd.DataFrame]:
    """Emit per-fold role assignments, flagging rows that are a baseline not a score.

    ``is_control`` adds two columns. ``is_control`` lets the leakage check exempt
    the control population, which is deliberately present in every fold of the
    gene split so a control-relative response has a same-screen baseline.
    ``scored`` records that those rows are excluded from accuracy metrics: in a
    gene-holdout design there is no held-out control gene to predict, and a
    control's expected response is zero by construction. The flags travel with
    the split rather than being re-derived, so a consumer cannot silently score
    the reference population.
    """
    n_folds = int(np.max(fold_values)) + 1
    for fold in range(n_folds):
        validation = (fold + 1) % n_folds
        roles = np.where(fold_values == fold, "test", np.where(fold_values == validation, "validation", "train"))
        frame = pd.DataFrame(
            {
                "observation_id": observation_ids,
                "split_name": split_name,
                "fold": fold,
                "role": roles,
            }
        )
        if is_control is not None:
            control = np.asarray(is_control, dtype=bool)
            frame["is_control"] = control
            # Only the gene split holds out a perturbation identity, so only there
            # is a control row a non-evaluable reference rather than a held-out unit.
            frame["scored"] = ~control if split_name == "gene" else True
        yield frame


def export_reporter_bundle(
    *,
    cache_path: Path,
    phase_root: Path,
    output_root: Path,
    reporter_record: pd.Series,
    format_: str,
    chunk_rows: int,
    raw_images_available: bool = False,
) -> dict[str, Any]:
    slug = reporter_slug(cache_path)
    bundle = output_root / "reporters" / slug
    bundle.mkdir(parents=True, exist_ok=True)
    extension = format_
    writers = {
        name: StreamTable(output_path(bundle, name, extension), format_)
        for name in ("inputs", "targets", "observations", "pairing")
    }
    matrix = np.load(phase_root / "phase172.npy", mmap_mode="r")
    field_fold_all = np.load(phase_root / "field_holdout_sanity.fold.npy", mmap_mode="r")
    gene_fold_all = np.load(phase_root / "gene_holdout_main.fold.npy", mmap_mode="r")
    well_codes_all = np.load(phase_root / "well_code.npy", mmap_mode="r")
    tile_codes_all = np.load(phase_root / "tile_code.npy", mmap_mode="r")
    categories_payload = json.loads((phase_root / "categories.json").read_text(encoding="utf-8"))
    feature_names = load_phase_feature_names(phase_root)
    with h5py.File(cache_path, "r") as cache:
        phase_rows = np.asarray(cache["phase_row_index"][:], dtype=np.int64)
        target_rows = np.asarray(cache["fluorescence_row_index"][:], dtype=np.int64)
        screen_codes = np.asarray(cache["metadata/screen_codes"][:], dtype=np.int64)
        screen_names = decode(cache["metadata/screen_categories"][:])
        genes = _values(decode(cache["metadata/gene_categories"][:]), cache["metadata/gene_codes"][:])
        guides = _values(decode(cache["metadata/sgRNA_categories"][:]), cache["metadata/sgRNA_codes"][:])
        wells = _values(decode(cache["metadata/well_categories"][:]), cache["metadata/well_codes"][:])
        is_control = np.asarray(cache["metadata/is_control"][:], dtype=bool)
        endpoint_ids = decode(cache["features/target_feature_names"][:])
        fluorescence = cache["fluorescence"]
        observation_ids = np.asarray(
            [f"{slug}:{int(row)}" for row in target_rows], dtype=object
        )
        input_cell_ids = np.asarray([f"phase:{int(row)}" for row in phase_rows], dtype=object)
        target_cell_ids = np.asarray([f"{slug}:{int(row)}" for row in target_rows], dtype=object)
        screens = screen_names[screen_codes]
        assays = np.asarray([f"{slug}::{screen}" for screen in screens], dtype=object)
        tile_categories = np.asarray(categories_payload["tile_code"], dtype=str)
        phase_tile_codes = np.asarray(tile_codes_all[phase_rows], dtype=np.int64)
        tiles = np.where(phase_tile_codes >= 0, tile_categories[np.maximum(phase_tile_codes, 0)], "")
        field_ids = np.asarray([f"{screen}|{well}|{tile}" for screen, well, tile in zip(screens, wells, tiles)], dtype=object)
        for start in range(0, len(phase_rows), chunk_rows):
            stop = min(start + chunk_rows, len(phase_rows))
            sl = slice(start, stop)
            common = pd.DataFrame(
                {
                    "observation_id": observation_ids[sl],
                    "reporter_id": slug,
                    "screen_id": screens[sl],
                    "perturbation_id": genes[sl],
                    "assay_id": assays[sl],
                    "field_id": field_ids[sl],
                }
            )
            # is_control travels with the inputs so the fold runners can carry it
            # into the canonical prediction table; control-relative response
            # fidelity cannot run without it. It is not a feature: the runners'
            # METADATA_COLUMNS denylist already excludes it from feature inference.
            inputs = pd.concat(
                [
                    common.assign(is_control=is_control[sl]),
                    pd.DataFrame(np.asarray(matrix[phase_rows[sl]], dtype=np.float32), columns=feature_names),
                ],
                axis=1,
            )
            writers["inputs"].write(inputs)
            observations = common.assign(
                input_cell_id=input_cell_ids[sl],
                target_cell_id=target_cell_ids[sl],
                guide_id=guides[sl],
                well_id=wells[sl],
                is_control=is_control[sl],
            )[
                [
                    "observation_id",
                    "input_cell_id",
                    "target_cell_id",
                    "assay_id",
                    "perturbation_id",
                    "guide_id",
                    "screen_id",
                    "well_id",
                    "field_id",
                    "is_control",
                ]
            ]
            writers["observations"].write(observations)
            pairing = observations[["observation_id", "input_cell_id", "target_cell_id", "assay_id"]].copy()
            pairing["pairing_key"] = [f"{p}:{t}" for p, t in zip(phase_rows[sl], target_rows[sl])]
            pairing["pairing_confidence"] = 1.0
            writers["pairing"].write(pairing)
            target = pd.DataFrame(np.asarray(fluorescence[sl], dtype=np.float32), columns=endpoint_ids)
            target.insert(0, "observation_id", observation_ids[sl])
            target = target.melt(id_vars="observation_id", var_name="endpoint_id", value_name="y_true")
            target = target.loc[np.isfinite(target.y_true)].copy()
            target["reporter_id"] = slug
            target["is_observed"] = True
            writers["targets"].write(target)
    for writer in writers.values():
        writer.close()

    split_paths: dict[str, str] = {}
    for split_name, fold_array in (
        ("field", np.asarray(field_fold_all[phase_rows], dtype=np.uint8)),
        ("gene", np.asarray(gene_fold_all[phase_rows], dtype=np.uint8)),
    ):
        writer = StreamTable(output_path(bundle, f"split_{split_name}", extension), format_)
        for frame in assignment_frame(observation_ids, fold_array, split_name, is_control=is_control):
            writer.write(frame)
        writer.close()
        split_paths[split_name] = writer.path.name

    strict_paths: dict[str, str] = {}
    if len(screen_names) > 1:
        field_fold = np.asarray(field_fold_all[phase_rows], dtype=np.uint8)
        for direction, destination in enumerate(screen_names):
            role = np.where(
                screens == destination,
                "test",
                np.where(field_fold == 0, "validation", "train"),
            )
            frame = pd.DataFrame(
                {
                    "observation_id": observation_ids,
                    "split_name": "strict_whole_screen",
                    "fold": direction,
                    "role": role,
                    "destination_screen": destination,
                    "is_control": is_control,
                    # The strict split holds out a physical screen, not a
                    # perturbation identity, so a control row there is a genuine
                    # held-out observation and is scored.
                    "scored": True,
                }
            )
            writer = StreamTable(output_path(bundle, f"split_screen_{direction:02d}", extension), format_)
            writer.write(frame)
            writer.close()
            strict_paths[str(destination)] = writer.path.name

    reporter_table = pd.DataFrame(
        [
            {
                "reporter_id": slug,
                "reporter_name": reporter_record.reporter_name,
                "biological_system": reporter_record.biological_system,
                "target_modality": "targeted_fluorescence",
                "endpoint_schema_id": f"ops-target-{slug}-v1",
            }
        ]
    )
    assays_table = pd.DataFrame(
        [
            {
                "assay_id": f"{slug}::{screen}",
                "reporter_id": slug,
                "screen_id": screen,
                "endpoint_ids": ",".join(endpoint_ids),
            }
            for screen in screen_names
        ]
    )
    endpoints_table = pd.DataFrame(
        {
            "endpoint_schema_id": f"ops-target-{slug}-v1",
            "endpoint_id": endpoint_ids,
            "reporter_id": slug,
        }
    )
    reporter_table.to_csv(bundle / "reporters.csv", index=False)
    assays_table.to_csv(bundle / "assays.csv", index=False)
    endpoints_table.to_csv(bundle / "endpoints.csv", index=False)
    manifest = {
        "schema_version": 1,
        "dataset_id": "ops-public",
        "reporter_id": slug,
        "capabilities": {
            "exact_pairing": True,
            "raw_images": raw_images_available,
            "guides": True,
            "screen_matched_controls": True,
            "repeated_screens": len(screen_names) > 1,
            "covariates": True,
        },
        "tables": {
            "observations": writers["observations"].path.name,
            "reporters": "reporters.csv",
            "assays": "assays.csv",
            "pairing": writers["pairing"].path.name,
            "endpoints": "endpoints.csv",
            "inputs": writers["inputs"].path.name,
            "targets": writers["targets"].path.name,
        },
        "splits": {**split_paths, "strict_whole_screen": strict_paths},
        "input_schema": {"schema_id": "ops-phase172-v1", "feature_names": feature_names},
        "source_cache": cache_path.name,
        "source_cache_sha256": file_sha256(cache_path),
        "n_observations": int(len(phase_rows)),
        "n_endpoints": int(len(endpoint_ids)),
        "n_screens": int(len(screen_names)),
    }
    (bundle / "canonical_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exact-cache-root", type=Path, required=True)
    parser.add_argument("--phase-cache", type=Path, required=True)
    parser.add_argument("--reporter-registry", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reporter", action="append", default=[])
    parser.add_argument("--format", choices=("parquet", "csv"), default="parquet")
    parser.add_argument("--chunk-rows", type=int, default=4096)
    parser.add_argument(
        "--raw-images-available",
        action="store_true",
        help="Declare raw-image capability only when the exported dataset root also resolves those images.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    exact_root = args.exact_cache_root.expanduser().resolve()
    phase_root = args.phase_cache.expanduser().resolve()
    output_root = args.output_dir.expanduser().resolve()
    registry = load_registry(args.reporter_registry.expanduser().resolve())
    selected = {str(value) for value in args.reporter}
    cache_paths = sorted(exact_root.glob("all_cells_fluor_*.exact.h5"))
    if selected:
        cache_paths = [path for path in cache_paths if reporter_slug(path) in selected]
    if not cache_paths:
        raise FileNotFoundError("no exact reporter caches matched the request")
    required_phase = (
        "phase172.npy",
        "field_holdout_sanity.fold.npy",
        "gene_holdout_main.fold.npy",
        "well_code.npy",
        "tile_code.npy",
        "categories.json",
    )
    missing = [name for name in required_phase if not (phase_root / name).is_file()]
    if not (phase_root / "phase_features_172d.txt").is_file() and not (phase_root / "manifest.json").is_file():
        missing.append("phase_features_172d.txt (or private preparation manifest.json)")
    if missing:
        raise FileNotFoundError(f"phase cache is incomplete: {missing}")
    plan = {
        "n_reporters": len(cache_paths),
        "reporters": [reporter_slug(path) for path in cache_paths],
        "output_dir": str(output_root),
        "format": args.format,
        "canonical_tables": ["observations", "reporters", "assays", "pairing", "endpoints", "inputs", "targets"],
        "splits": ["field", "gene", "strict_whole_screen"],
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return 0
    output_root.mkdir(parents=True, exist_ok=True)
    atlas_tables = tuple(build_atlas_tables(cache_paths, registry))
    write_small_tables(output_root, atlas_tables)  # type: ignore[arg-type]
    registry_by_slug = registry.set_index("reporter_slug")
    manifests = []
    for number, cache_path in enumerate(cache_paths, start=1):
        slug = reporter_slug(cache_path)
        print(f"canonical bundle {number}/{len(cache_paths)}: {slug}", flush=True)
        manifests.append(
            export_reporter_bundle(
                cache_path=cache_path,
                phase_root=phase_root,
                output_root=output_root,
                reporter_record=registry_by_slug.loc[slug],
                format_=args.format,
                chunk_rows=args.chunk_rows,
                raw_images_available=args.raw_images_available,
            )
        )
    release = {
        "schema_version": 1,
        "n_reporters": len(manifests),
        "n_assays": int(len(atlas_tables[1])),
        "n_strict_screen_directions": int(len(atlas_tables[3])),
        "format": args.format,
        "reporter_manifests": [f"reporters/{row['reporter_id']}/canonical_manifest.json" for row in manifests],
        "contract_sha256": _sha256_payload(
            {"reporters": atlas_tables[0].to_dict("records"), "assays": atlas_tables[1].to_dict("records")}
        ),
    }
    (output_root / "canonical_release.json").write_text(json.dumps(release, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(manifests)} runnable canonical bundles to {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
