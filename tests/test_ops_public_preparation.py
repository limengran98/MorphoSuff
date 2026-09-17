from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from measurement_sufficiency.schemas import validate_canonical_tables
from studies.ops.preparation import build_exact_caches, build_phase172, export_canonical
from studies.ops.preparation.features import (
    INTENSITY_AGGREGATIONS,
    INTENSITY_METRICS,
    INTENSITY_ORGANELLE_ORDER,
    LOCALIZATION_AGGREGATIONS,
    LOCALIZATION_METRICS,
    ORGANELLE_ORDER,
    SHAPE_METRICS,
    derive_feature_names,
)


def _strings(group: h5py.Group, name: str, values: list[str]) -> None:
    group.create_dataset(name, data=np.asarray(values, dtype=object), dtype=h5py.string_dtype("utf-8"))


def _categorical(group: h5py.Group, name: str, values: list[str]) -> None:
    categories, codes = np.unique(np.asarray(values, dtype=str), return_inverse=True)
    node = group.create_group(name)
    _strings(node, "categories", categories.tolist())
    node.create_dataset("codes", data=codes.astype(np.int32))


def _feature_rows() -> list[tuple[str, str, str, str, str]]:
    rows: list[tuple[str, str, str, str, str]] = []
    for organelle in ORGANELLE_ORDER:
        for metric in LOCALIZATION_METRICS:
            for aggregation in LOCALIZATION_AGGREGATIONS:
                rows.append((f"op_{organelle}_{metric}_{aggregation}", organelle, "localization", metric, aggregation))
    for organelle in INTENSITY_ORGANELLE_ORDER:
        for metric in INTENSITY_METRICS:
            for aggregation in INTENSITY_AGGREGATIONS:
                rows.append((f"op_{organelle}_{metric}_{aggregation}", organelle, "intensity", metric, aggregation))
    for metric in SHAPE_METRICS:
        rows.append((f"op_cell_{metric}", "cell", "cell_morphology", metric, ""))
    assert len(rows) == 172
    return rows


def _write_phase(path: Path, n_rows: int = 15) -> None:
    feature_rows = _feature_rows()
    rng = np.random.default_rng(12)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("X", data=rng.normal(size=(n_rows, 172)).astype(np.float32))
        obs = handle.create_group("obs")
        screens = ["S1" if i < n_rows // 2 else "S2" for i in range(n_rows)]
        genes = ["NTC" if i % 5 == 0 else ("G1" if i % 2 else "G2") for i in range(n_rows)]
        for name, values in (
            ("portal_screen_name", screens),
            ("well_canonical", [f"W{i:02d}" for i in range(n_rows)]),
            ("tile_pheno", [f"T{i % 4}" for i in range(n_rows)]),
            ("gene", genes),
            ("sgRNA", ["NTC" if gene == "NTC" else f"{gene}_guide" for gene in genes]),
        ):
            _categorical(obs, name, values)
        obs.create_dataset("segmentation_id", data=np.arange(n_rows, dtype=np.int64))
        obs.create_dataset("x_pheno", data=np.arange(n_rows, dtype=float) + 0.25)
        obs.create_dataset("y_pheno", data=np.arange(n_rows, dtype=float) + 0.75)
        obs.create_dataset("op_match", data=np.ones(n_rows, dtype=bool))
        var = handle.create_group("var")
        _strings(var, "_index", [row[0] for row in feature_rows])
        _categorical(var, "source", ["organelle_profiler"] * 172)
        _categorical(var, "organelle", [row[1] for row in feature_rows])
        _categorical(var, "category", [row[2] for row in feature_rows])
        _categorical(var, "metric", [row[3] for row in feature_rows])
        _categorical(var, "aggregation", [row[4] for row in feature_rows])


def _write_reporter(path: Path, phase_path: Path) -> None:
    with h5py.File(phase_path, "r") as phase, h5py.File(path, "w") as handle:
        n_rows = phase["X"].shape[0]
        values = np.arange(n_rows * 2, dtype=np.float32).reshape(n_rows, 2)
        values[1, 1] = np.nan
        handle.create_dataset("X", data=values, chunks=(5, 2))
        obs = handle.create_group("obs")
        for name in ("portal_screen_name", "well_canonical", "tile_pheno", "gene", "sgRNA"):
            source = phase[f"obs/{name}"]
            node = obs.create_group(name)
            node.create_dataset("codes", data=source["codes"][:])
            _strings(node, "categories", [x.decode() if isinstance(x, bytes) else str(x) for x in source["categories"][:]])
        for name in ("segmentation_id", "x_pheno", "y_pheno", "op_match"):
            obs.create_dataset(name, data=phase[f"obs/{name}"][:])
        var = handle.create_group("var")
        _strings(var, "_index", ["op_lysosome_intensity_mean", "op_lysosome_intensity_max"])
        _categorical(var, "source", ["organelle_profiler"] * 2)
        _categorical(var, "category", ["intensity"] * 2)
        _categorical(var, "organelle", ["lysosome"] * 2)


def test_public_h5ad_to_exact_phase172_and_canonical(tmp_path: Path) -> None:
    phase = tmp_path / "all_cells_phase.h5ad"
    reporter = tmp_path / "all_cells_fluor_lysosome_lamp1.h5ad"
    _write_phase(phase)
    _write_reporter(reporter, phase)
    assert len(derive_feature_names(phase)) == 172

    exact_root = tmp_path / "prepared" / "exact" / "reporters"
    phase_index = tmp_path / "prepared" / "exact" / "phase_pairing_index.h5"
    build_exact_caches.build_phase_index(phase, phase_index)
    cache_path = exact_root / "all_cells_fluor_lysosome_lamp1.exact.h5"
    audit = build_exact_caches.build_reporter_cache(phase, phase_index, reporter, cache_path)
    assert audit["n_exact_pairs"] == 15
    assert audit["n_screens"] == 2

    phase_root = tmp_path / "prepared" / "phase172"
    phase_root.mkdir(parents=True)
    features = derive_feature_names(phase)
    with h5py.File(phase, "r") as source:
        names = [x.decode() if isinstance(x, bytes) else str(x) for x in source["var/_index"][:]]
        indices = np.asarray([names.index(name) for name in features])
        matrix_path = build_phase172.copy_phase_matrix(source, phase_root, indices, features, 4)
        build_phase172.validate_matrix(source, matrix_path, indices, 17)
        summary = build_phase172.build_metadata_and_folds(
            source, phase_root, {"G1", "G2"}, 5, 20260716
        )
    (phase_root / "phase_features_172d.txt").write_text("\n".join(features) + "\n")
    (phase_root / "manifest.json").write_text(
        json.dumps({"schema_version": build_phase172.SCHEMA_VERSION, "shape": [15, 172], "split_summary": summary})
    )

    registry_source = Path(__file__).parents[1] / "configs" / "ops" / "data" / "reporter_registry.csv"
    registry = export_canonical.load_registry(registry_source)
    canonical = tmp_path / "prepared" / "canonical"
    atlas_tables = export_canonical.build_atlas_tables([cache_path], registry)
    export_canonical.write_small_tables(canonical, atlas_tables)
    manifest = export_canonical.export_reporter_bundle(
        cache_path=cache_path,
        phase_root=phase_root,
        output_root=canonical,
        reporter_record=registry.set_index("reporter_slug").loc["lysosome_lamp1"],
        format_="csv",
        chunk_rows=4,
    )
    bundle = canonical / "reporters" / "lysosome_lamp1"
    tables = {
        name: pd.read_csv(bundle / manifest["tables"][name])
        for name in ("observations", "reporters", "assays", "pairing", "endpoints")
    }
    validate_canonical_tables(tables)
    inputs = pd.read_csv(bundle / manifest["tables"]["inputs"])
    targets = pd.read_csv(bundle / manifest["tables"]["targets"])
    assert len(inputs) == 15 and len([c for c in inputs if c.startswith("op_")]) == 172
    assert len(targets) == 29
    assert set(manifest["splits"]) == {"field", "gene", "strict_whole_screen"}


def test_cli_help_and_dry_run_contract(tmp_path: Path) -> None:
    phase = tmp_path / "all_cells_phase.h5ad"
    reporter = tmp_path / "all_cells_fluor_lysosome_lamp1.h5ad"
    _write_phase(phase, 6)
    _write_reporter(reporter, phase)
    args = type(
        "Args",
        (),
        {
            "phase": phase,
            "reporter": [reporter],
            "reporter_dir": None,
            "output_root": tmp_path / "cache" / "reporters",
            "phase_index": None,
            "overwrite": False,
            "dry_run": True,
        },
    )
    assert len(build_exact_caches.discover_reporters(args.reporter_dir, args.reporter)) == 1
