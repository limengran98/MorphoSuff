#!/usr/bin/env python3
"""Build exact same-cell phase/target caches from public OPS processed H5ADs.

The join is performed independently within each screen from the public
well/tile/segmentation/x/y identifiers, requires ``op_match`` on both sides,
and verifies gene and guide concordance. Ambiguous coordinate keys are
excluded rather than resolved heuristically. The output schema is consumed
directly by the released full-label, low-label and raw-image runners.
"""

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
    from .h5ad import (
        categories,
        categorical_rows,
        categorical_values,
        decode,
        file_sha256,
        read_rows,
        require_h5ad_columns,
    )
except ImportError:  # pragma: no cover - exercised by CLI smoke
    from h5ad import (
        categories,
        categorical_rows,
        categorical_values,
        decode,
        file_sha256,
        read_rows,
        require_h5ad_columns,
    )


PHASE_INDEX_SCHEMA = "ops-phase-pairing-index-v1"
CACHE_SCHEMA = "ops-full-reporter-exact-v2"
FOURI_ALIASES = {
    "b-catenin": ("b-catenin",),
    "c-myc": ("c-myc",),
    "p21": ("p21",),
    "p53": ("p53",),
    "prb": ("rb", "prb"),
    "ps6": ("rsp6", "ps6"),
}


def _category_map(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    lookup = {value: index for index, value in enumerate(target)}
    return np.asarray([lookup.get(value, -1) for value in source], dtype=np.int32)


def _mapped_codes(values: np.ndarray, mapping: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.int64)
    output = np.full(len(values), -1, dtype=np.int32)
    valid = values >= 0
    output[valid] = mapping[values[valid]]
    return output


def coordinate_keys(
    well_codes: np.ndarray,
    tile_codes: np.ndarray,
    segmentation_id: np.ndarray,
    x_pheno: np.ndarray,
    y_pheno: np.ndarray,
) -> np.ndarray:
    """Create stable 64-bit keys for the public exact-cell linkage fields."""

    components = [
        np.asarray(well_codes, dtype=np.int64),
        np.asarray(tile_codes, dtype=np.int64),
        np.rint(np.asarray(segmentation_id, dtype=np.float64)).astype(np.int64),
    ]
    for values in (x_pheno, y_pheno):
        values = np.asarray(values, dtype=np.float64)
        finite = np.isfinite(values)
        encoded = np.full(len(values), np.iinfo(np.int64).min, dtype=np.int64)
        encoded[finite] = np.rint(values[finite] * 1_000_000).astype(np.int64)
        components.append(encoded)
    result = np.full(len(well_codes), np.uint64(0x9E3779B97F4A7C15), dtype=np.uint64)
    with np.errstate(over="ignore"):
        for component in components:
            value = component.view(np.uint64) + np.uint64(0x9E3779B97F4A7C15)
            value = (value ^ (value >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
            value = (value ^ (value >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
            value ^= value >> np.uint64(31)
            result ^= value + (result << np.uint64(6)) + (result >> np.uint64(2))
    return result


def unique_key_rows(keys: np.ndarray, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    unique, first, counts = np.unique(keys, return_index=True, return_counts=True)
    keep = counts == 1
    return unique[keep], rows[first[keep]], int(np.sum(counts[counts > 1]))


def build_phase_index(phase_path: Path, output_path: Path, *, overwrite: bool = False) -> None:
    if output_path.exists() and not overwrite:
        with h5py.File(output_path, "r") as index:
            valid = (
                index.attrs.get("schema_version", "") == PHASE_INDEX_SCHEMA
                and int(index.attrs.get("phase_size_bytes", -1)) == phase_path.stat().st_size
            )
        if valid:
            print(f"reusing verified phase linkage index: {output_path}", flush=True)
            return
        raise RuntimeError(f"existing phase index is incompatible: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    with h5py.File(phase_path, "r") as phase, h5py.File(temporary, "w") as output:
        require_h5ad_columns(
            phase,
            obs=(
                "portal_screen_name",
                "well_canonical",
                "tile_pheno",
                "segmentation_id",
                "x_pheno",
                "y_pheno",
            ),
        )
        screen_codes = np.asarray(phase["obs/portal_screen_name/codes"][:], dtype=np.int32)
        well_codes = np.asarray(phase["obs/well_canonical/codes"][:], dtype=np.int32)
        tile_codes = np.asarray(phase["obs/tile_pheno/codes"][:], dtype=np.int32)
        segmentation = np.asarray(phase["obs/segmentation_id"][:])
        x_pheno = np.asarray(phase["obs/x_pheno"][:])
        y_pheno = np.asarray(phase["obs/y_pheno"][:])
        screens = categories(phase, "obs", "portal_screen_name")
        output.attrs["schema_version"] = PHASE_INDEX_SCHEMA
        output.attrs["phase_size_bytes"] = phase_path.stat().st_size
        output.attrs["phase_sha256"] = file_sha256(phase_path)
        output.attrs["phase_rows"] = len(screen_codes)
        groups = output.create_group("screens")
        for code, screen in enumerate(screens):
            rows = np.flatnonzero(screen_codes == code).astype(np.int64)
            keys = coordinate_keys(
                well_codes[rows], tile_codes[rows], segmentation[rows], x_pheno[rows], y_pheno[rows]
            )
            unique_keys, unique_rows, duplicate_rows = unique_key_rows(keys, rows)
            group = groups.create_group(f"{code:03d}")
            group.attrs["screen"] = str(screen)
            group.attrs["all_rows"] = len(rows)
            group.attrs["unique_rows"] = len(unique_rows)
            group.attrs["duplicate_coordinate_rows"] = duplicate_rows
            group.create_dataset("key", data=unique_keys, compression="lzf", shuffle=True)
            group.create_dataset("phase_row_index", data=unique_rows, compression="lzf", shuffle=True)
            print(f"phase index {code + 1}/{len(screens)} {screen}: {len(unique_rows):,} unique", flush=True)
    os.replace(temporary, output_path)


def select_target_columns(reporter: h5py.File, reporter_path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    require_h5ad_columns(reporter, var=("_index", "source", "category", "organelle"))
    names = decode(reporter["var/_index"][:])
    source = categorical_values(reporter, "var", "source")
    category = categorical_values(reporter, "var", "category")
    organelle = categorical_values(reporter, "var", "organelle")
    slug = reporter_path.stem.removeprefix("all_cells_fluor_").casefold()
    lower_names = np.char.lower(names)
    if slug in FOURI_ALIASES:
        marker_match = np.zeros(len(names), dtype=bool)
        for alias in FOURI_ALIASES[slug]:
            marker_match |= np.char.find(lower_names, alias) >= 0
        selected = (
            (source == "cellprofiler")
            & np.char.startswith(lower_names, "cp_single_object_")
            & (np.char.find(lower_names, "_intensity_") >= 0)
            & marker_match
        )
        rule = "marker-specific CellProfiler single-object intensity features"
        selected_organelles = ["marker-specific-4i"]
    else:
        intensity = (source == "organelle_profiler") & (category == "intensity")
        organelles = sorted(set(organelle[intensity]) - {""})
        selected_organelles = [x for x in organelles if x.casefold() not in {"nuclei", "nucleus"}]
        selected_organelles = selected_organelles or organelles
        selected = intensity & np.isin(organelle, selected_organelles)
        rule = "OrganelleProfiler intensity; non-nuclear reporter objects when available"
    columns = np.flatnonzero(selected).astype(np.int64)
    if len(columns) == 0:
        raise ValueError(f"target selection produced zero columns for {reporter_path.name}")
    return columns, {
        "selection_rule": rule,
        "reporter_slug": slug,
        "selected_organelles": selected_organelles,
        "n_target_columns": int(len(columns)),
        "target_feature_names": names[columns].tolist(),
    }


def match_reporter(
    phase: h5py.File, phase_index: h5py.File, reporter: h5py.File
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    required = (
        "portal_screen_name",
        "well_canonical",
        "tile_pheno",
        "segmentation_id",
        "x_pheno",
        "y_pheno",
        "op_match",
        "gene",
        "sgRNA",
    )
    require_h5ad_columns(phase, obs=required)
    require_h5ad_columns(reporter, obs=required)
    phase_screens = categories(phase, "obs", "portal_screen_name")
    screen_lookup = {name: code for code, name in enumerate(phase_screens)}
    reporter_screens = categories(reporter, "obs", "portal_screen_name")
    reporter_screen_codes = np.asarray(reporter["obs/portal_screen_name/codes"][:], dtype=np.int32)
    reporter_well_codes = np.asarray(reporter["obs/well_canonical/codes"][:], dtype=np.int32)
    reporter_tile_codes = np.asarray(reporter["obs/tile_pheno/codes"][:], dtype=np.int32)
    well_map = _category_map(categories(reporter, "obs", "well_canonical"), categories(phase, "obs", "well_canonical"))
    tile_map = _category_map(categories(reporter, "obs", "tile_pheno"), categories(phase, "obs", "tile_pheno"))
    segmentation = np.asarray(reporter["obs/segmentation_id"][:])
    x_pheno = np.asarray(reporter["obs/x_pheno"][:])
    y_pheno = np.asarray(reporter["obs/y_pheno"][:])
    reporter_op = np.asarray(reporter["obs/op_match"][:], dtype=bool)
    phase_parts: list[np.ndarray] = []
    reporter_parts: list[np.ndarray] = []
    screen_parts: list[np.ndarray] = []
    audits: list[dict[str, Any]] = []
    for reporter_screen_code, screen in enumerate(reporter_screens):
        if screen not in screen_lookup:
            raise ValueError(f"reporter screen is absent from phase H5AD: {screen}")
        phase_code = screen_lookup[screen]
        index = phase_index[f"screens/{phase_code:03d}"]
        phase_keys = np.asarray(index["key"][:], dtype=np.uint64)
        phase_rows = np.asarray(index["phase_row_index"][:], dtype=np.int64)
        rows = np.flatnonzero(reporter_screen_codes == reporter_screen_code).astype(np.int64)
        wells = _mapped_codes(reporter_well_codes[rows], well_map)
        tiles = _mapped_codes(reporter_tile_codes[rows], tile_map)
        valid_categories = (wells >= 0) & (tiles >= 0)
        eligible = rows[valid_categories]
        keys = coordinate_keys(
            wells[valid_categories], tiles[valid_categories], segmentation[eligible], x_pheno[eligible], y_pheno[eligible]
        )
        unique_keys, unique_reporter_rows, duplicate_rows = unique_key_rows(keys, eligible)
        positions = np.searchsorted(phase_keys, unique_keys)
        found = positions < len(phase_keys)
        found[found] &= phase_keys[positions[found]] == unique_keys[found]
        matched_phase = phase_rows[positions[found]]
        matched_reporter = unique_reporter_rows[found]
        op = np.asarray(read_rows(phase["obs/op_match"], matched_phase), dtype=bool) & reporter_op[matched_reporter]
        matched_phase, matched_reporter = matched_phase[op], matched_reporter[op]
        concordant = (
            categorical_rows(phase, "obs", "gene", matched_phase)
            == categorical_rows(reporter, "obs", "gene", matched_reporter)
        ) & (
            categorical_rows(phase, "obs", "sgRNA", matched_phase)
            == categorical_rows(reporter, "obs", "sgRNA", matched_reporter)
        )
        matched_phase, matched_reporter = matched_phase[concordant], matched_reporter[concordant]
        phase_parts.append(matched_phase)
        reporter_parts.append(matched_reporter)
        screen_parts.append(np.repeat(screen, len(matched_phase)))
        audits.append(
            {
                "screen": str(screen),
                "reporter_rows": int(len(rows)),
                "rows_with_mappable_well_tile": int(len(eligible)),
                "duplicate_coordinate_rows_excluded": duplicate_rows,
                "coordinate_matches": int(found.sum()),
                "op_matches": int(op.sum()),
                "gene_guide_concordant_pairs": int(concordant.sum()),
            }
        )
    if not phase_parts:
        raise ValueError("reporter has no screens")
    phase_rows = np.concatenate(phase_parts)
    reporter_rows = np.concatenate(reporter_parts)
    screens = np.concatenate(screen_parts)
    order = np.argsort(reporter_rows, kind="stable")
    return {
        "phase_rows": phase_rows[order],
        "reporter_rows": reporter_rows[order],
        "screens": screens[order],
    }, audits


def _encode(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    categories_, codes_ = np.unique(np.asarray(values, dtype=str), return_inverse=True)
    return categories_, codes_.astype(np.int32)


def _write_strings(group: h5py.Group, name: str, values: np.ndarray | list[str]) -> None:
    group.create_dataset(name, data=np.asarray(values, dtype=object), dtype=h5py.string_dtype("utf-8"))


def _copy_target_rows(
    source: h5py.Dataset,
    rows: np.ndarray,
    columns: np.ndarray,
    output: h5py.Dataset,
    finite_output: h5py.Dataset,
) -> float:
    row_chunk = int(source.chunks[0]) if source.chunks else 8192
    chunk_ids = rows // row_chunk
    finite_count = 0
    for chunk_id in np.unique(chunk_ids):
        positions = np.flatnonzero(chunk_ids == chunk_id)
        start = int(chunk_id * row_chunk)
        stop = min(source.shape[0], start + row_chunk)
        block = np.asarray(source[start:stop, columns], dtype=np.float32)
        selected = block[rows[positions] - start]
        finite = np.all(np.isfinite(selected), axis=1)
        output[positions], finite_output[positions] = selected, finite
        finite_count += int(finite.sum())
    return finite_count / max(len(rows), 1)


def build_reporter_cache(
    phase_path: Path,
    index_path: Path,
    reporter_path: Path,
    output_path: Path,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    audit_path = output_path.with_suffix(".audit.json")
    source_sha256 = file_sha256(reporter_path)
    if output_path.exists() and audit_path.exists() and not overwrite:
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if audit.get("status") == "verified" and audit.get("source_sha256") == source_sha256:
            print(f"reusing verified reporter cache: {output_path}", flush=True)
            return audit
        raise RuntimeError(f"existing reporter cache is incompatible: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    started = time.time()
    with h5py.File(phase_path, "r") as phase, h5py.File(index_path, "r") as phase_index, h5py.File(reporter_path, "r") as reporter:
        phase_sha256 = str(phase_index.attrs["phase_sha256"])
        target_columns, feature_audit = select_target_columns(reporter, reporter_path)
        target_names = decode(reporter["var/_index"][:])[target_columns]
        matched, screen_audit = match_reporter(phase, phase_index, reporter)
        phase_rows, reporter_rows = matched["phase_rows"], matched["reporter_rows"]
        if len(phase_rows) == 0:
            raise ValueError(f"no exact pairs found for {reporter_path.name}")
        genes = categorical_rows(phase, "obs", "gene", phase_rows)
        guides = categorical_rows(phase, "obs", "sgRNA", phase_rows)
        wells = categorical_rows(phase, "obs", "well_canonical", phase_rows)
        upper = np.char.upper(genes)
        is_control = np.char.startswith(upper, "NTC") | np.char.startswith(upper, "NONTARGET")
        roles = np.where(is_control, "control", "targeting")
        metadata_values = {
            "gene": _encode(genes),
            "sgRNA": _encode(guides),
            "well": _encode(wells),
            "screen": _encode(matched["screens"]),
            "role": _encode(roles),
        }
        with h5py.File(temporary, "w") as output:
            output.attrs["schema_version"] = CACHE_SCHEMA
            output.attrs["source_phase_sha256"] = phase_sha256
            output.attrs["reporter_source_sha256"] = source_sha256
            output.attrs["reporter_source_size_bytes"] = reporter_path.stat().st_size
            output.attrs["n_exact_pairs"] = len(phase_rows)
            output.attrs["n_target_features"] = len(target_columns)
            output.create_dataset("phase_row_index", data=phase_rows, compression="lzf", shuffle=True)
            output.create_dataset("fluorescence_row_index", data=reporter_rows, compression="lzf", shuffle=True)
            metadata = output.create_group("metadata")
            for name, (cats, codes_) in metadata_values.items():
                metadata.create_dataset(f"{name}_codes", data=codes_, compression="lzf", shuffle=True)
                _write_strings(metadata, f"{name}_categories", cats)
            metadata.create_dataset("is_control", data=is_control, compression="lzf", shuffle=True)
            features = output.create_group("features")
            _write_strings(features, "target_feature_names", target_names)
            _write_strings(features, "target_organelles", feature_audit["selected_organelles"])
            target = output.create_dataset(
                "fluorescence",
                shape=(len(reporter_rows), len(target_columns)),
                dtype=np.float32,
                chunks=(min(8192, len(reporter_rows)), len(target_columns)),
                compression="lzf",
                shuffle=True,
            )
            finite_rows = output.create_dataset(
                "target_row_all_finite",
                shape=(len(reporter_rows),),
                dtype=np.bool_,
                chunks=(min(65536, len(reporter_rows)),),
                compression="lzf",
                shuffle=True,
            )
            finite_fraction = _copy_target_rows(reporter["X"], reporter_rows, target_columns, target, finite_rows)
            output.attrs["target_row_all_finite_fraction"] = finite_fraction
    os.replace(temporary, output_path)
    with h5py.File(output_path, "r") as check:
        if check["fluorescence"].shape != (len(phase_rows), len(target_columns)):
            raise RuntimeError(f"cache shape verification failed: {output_path}")
        if len(np.unique(check["phase_row_index"][:])) != len(phase_rows):
            raise RuntimeError(f"cache duplicates a phase row: {output_path}")
    audit: dict[str, Any] = {
        "status": "verified",
        "schema_version": CACHE_SCHEMA,
        "source_file": reporter_path.name,
        "source_sha256": source_sha256,
        "cache_file": output_path.name,
        "cache_sha256": file_sha256(output_path),
        "n_exact_pairs": int(len(phase_rows)),
        "n_targeting_pairs": int((~is_control).sum()),
        "n_control_pairs": int(is_control.sum()),
        "n_screens": int(len(metadata_values["screen"][0])),
        "n_target_features": int(len(target_columns)),
        "target_all_finite_fraction": float(finite_fraction),
        "feature_selection": feature_audit,
        "screens": screen_audit,
        "build_seconds": float(time.time() - started),
    }
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(f"verified {output_path.name}: {len(phase_rows):,} exact pairs", flush=True)
    return audit


def discover_reporters(reporter_dir: Path, reporters: list[Path]) -> list[Path]:
    found = list(reporters)
    if reporter_dir:
        found.extend(sorted(reporter_dir.glob("all_cells_fluor_*.h5ad")))
    unique = sorted({path.expanduser().resolve() for path in found})
    if not unique:
        raise ValueError("provide --reporter and/or --reporter-dir")
    missing = [path for path in unique if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing[0])
    return unique


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", type=Path, required=True)
    parser.add_argument("--reporter", type=Path, action="append", default=[])
    parser.add_argument("--reporter-dir", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--phase-index", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    phase = args.phase.expanduser().resolve()
    if not phase.is_file():
        raise FileNotFoundError(phase)
    reporters = discover_reporters(args.reporter_dir, args.reporter)
    output_root = args.output_root.expanduser().resolve()
    phase_index = (args.phase_index or output_root.parent / "phase_pairing_index.h5").expanduser().resolve()
    plan = {
        "phase": str(phase),
        "phase_index": str(phase_index),
        "output_root": str(output_root),
        "n_reporters": len(reporters),
        "reporters": [path.name for path in reporters],
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return 0
    build_phase_index(phase, phase_index, overwrite=args.overwrite)
    audits = []
    for number, reporter in enumerate(reporters, start=1):
        output = output_root / f"{reporter.stem}.exact.h5"
        print(f"reporter {number}/{len(reporters)}: {reporter.name}", flush=True)
        audits.append(build_reporter_cache(phase, phase_index, reporter, output, overwrite=args.overwrite))
    with h5py.File(phase_index, "r") as index:
        phase_sha256 = str(index.attrs["phase_sha256"])
    manifest = {
        "schema_version": CACHE_SCHEMA,
        "phase_file": phase.name,
        "phase_sha256": phase_sha256,
        "n_reporters": len(audits),
        "n_exact_pairs": int(sum(row["n_exact_pairs"] for row in audits)),
        "reporters": audits,
    }
    manifest_path = output_root.parent / "exact_cache_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote exact-cache manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
