"""Minimal, real Figure 1–6 renderer driven exclusively by source contracts."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

import pandas as pd


CONTRACTS: dict[int, dict[str, list[str]]] = {
    1: {"reporters": ["reporter_id"], "assays": ["assay_id", "reporter_id", "screen_id"], "observations": ["observation_id"], "perturbation_response_atlas": ["reporter_id"], "microscopy_selection_manifest": ["reporter_id"]},
    2: {"predictions_long": ["observation_id", "reporter_id", "endpoint_id", "model_id", "y_true", "y_pred"], "split_manifest": ["observation_id"], "methods_registry": ["model_id"]},
    3: {"prediction_metrics": ["reporter_id"], "reporter_metadata": ["reporter_id"], "endpoint_family_map": ["reporter_id"], "microscopy_selection_manifest": ["reporter_id"]},
    4: {"same_cell_arm_metrics": ["reporter_id"], "residual_metrics": ["reporter_id"], "response_profiles": ["reporter_id"], "strict_screen_metrics": ["reporter_id"]},
    5: {"stability_metrics": ["reporter_id"], "coverage_metrics": ["reporter_id"], "amplitude_metrics": ["reporter_id"], "transfer_shift_metrics": ["reporter_id"], "ambiguity_metrics": ["reporter_id"]},
    6: {"scientific_utility": ["reporter_id"], "fidelity_metrics": ["reporter_id"], "measurement_tiers": ["reporter_id", "tier"], "low_label_metrics": ["reporter_id"], "image_pilot_metrics": ["reporter_id"]},
}


CONFIG_ROOT = Path(__file__).resolve().parents[3] / "configs" / "ops" / "figures"


def _read(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.suffix.lower() in {".parquet", ".pq"} else pd.read_csv(path)


def declared_contract(number: int) -> dict[str, object]:
    """Read the frozen figure contract that configs/ops/figures declares.

    The in-module CONTRACTS table is the fallback minimum, not the contract: the
    YAML declares columns, split names and method policy that the renderer never
    read, so a manifest could omit a declared column and still be accepted.
    """
    import yaml  # optional: installed with the figures extra

    path = CONFIG_ROOT / f"figure{number}.yaml"
    if not path.is_file():
        raise ValueError(f"no frozen contract for Figure {number} at {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    declared_tables = set(payload.get("canonical_inputs", []))
    fallback_tables = set(CONTRACTS[number])
    if declared_tables != fallback_tables:
        raise ValueError(
            f"Figure {number} table names drifted: configs declare {sorted(declared_tables)}, "
            f"the renderer expects {sorted(fallback_tables)}"
        )
    required_columns = payload.get("required_columns", {}) or {}
    # Union, never replacement: the YAML declares required_columns only for some
    # figures, so taking it alone would weaken the others.
    columns = {
        name: sorted(set(required_columns.get(name, [])) | set(CONTRACTS[number].get(name, [])))
        for name in fallback_tables
    }
    return {
        "columns": columns,
        "splits": payload.get("splits"),
        "methods": payload.get("methods"),
        "consensus": payload.get("consensus"),
    }


def load_contract(number: int, manifest_path: Path) -> dict[str, pd.DataFrame]:
    payload = json.loads(manifest_path.read_text())
    specs = payload.get("tables", payload)
    contract = declared_contract(number)
    required: dict[str, list[str]] = contract["columns"]  # type: ignore[assignment]
    tables: dict[str, pd.DataFrame] = {}
    for name, columns in required.items():
        if name not in specs:
            raise ValueError(f"Figure {number} source contract lacks required table {name!r}")
        spec = specs[name]
        value = spec.get("local_path") if isinstance(spec, dict) else spec
        path = Path(value)
        path = path if path.is_absolute() else manifest_path.parent / path
        table = _read(path)
        missing = set(columns).difference(table)
        if missing:
            raise ValueError(f"Figure {number} table {name!r} lacks columns: {', '.join(sorted(missing))}")
        if table.empty:
            raise ValueError(f"Figure {number} table {name!r} is empty")
        all_null = [column for column in columns if table[column].isna().all()]
        if all_null:
            raise ValueError(
                f"Figure {number} table {name!r} has entirely null required columns: "
                f"{', '.join(sorted(all_null))}"
            )
        tables[name] = table
    _validate_declared_policy(number, contract, tables)
    return tables


def _validate_declared_policy(number: int, contract: dict[str, object], tables: dict[str, pd.DataFrame]) -> None:
    """Enforce the split and method policy the frozen contract declares."""
    from measurement_sufficiency.consensus import FINAL_TEN_MODELS
    from measurement_sufficiency.splits import canonical_split_name

    allowed_models = set(FINAL_TEN_MODELS) | {"ten_model_median"}
    for name, table in tables.items():
        if contract.get("methods") == "final_ten_registry_only" and "model_id" in table:
            extra = sorted(set(map(str, table.model_id)).difference(allowed_models))
            if extra:
                raise ValueError(
                    f"Figure {number} table {name!r} contains non-registry methods: {extra}"
                )
        if contract.get("consensus") == "ten_method_unweighted_median" and "model_id" in table:
            models = set(map(str, table.model_id))
            if models and not models.issubset(allowed_models):
                raise ValueError(
                    f"Figure {number} table {name!r} declares a ten-method consensus but "
                    f"carries unregistered methods: {sorted(models - allowed_models)}"
                )
        declared_splits = contract.get("splits")
        if declared_splits and "split_name" in table:
            allowed = {canonical_split_name(value) for value in declared_splits}
            present = {canonical_split_name(value) for value in set(map(str, table.split_name))}
            if not present.issubset(allowed):
                raise ValueError(
                    f"Figure {number} table {name!r} uses splits outside the frozen "
                    f"contract: {sorted(present - allowed)}"
                )


def render(number: int, tables_manifest: Path, output_dir: Path, *, archival_renderer: str | None = None) -> list[Path]:
    """Validate inputs then render a compact auditable diagnostic figure.

    An archival renderer can be supplied as ``module:function``; it receives
    ``(number, tables, output_dir)`` after the same contract validation.
    """
    tables = load_contract(number, tables_manifest)
    output_dir.mkdir(parents=True, exist_ok=True)
    if archival_renderer:
        module, function = archival_renderer.split(":", 1)
        return list(getattr(importlib.import_module(module), function)(number, tables, output_dir))
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(tables), figsize=(4 * len(tables), 3.2), squeeze=False)
    for axis, (name, table) in zip(axes[0], tables.items()):
        numeric = table.select_dtypes(include="number")
        if not numeric.empty:
            column = numeric.columns[0]
            axis.hist(numeric[column].dropna(), bins=min(24, max(5, numeric[column].nunique())), color="#3f7f93")
            axis.set_xlabel(column)
        else:
            axis.bar([0], [len(table)], color="#3f7f93")
            axis.set_xticks([])
            axis.set_ylabel("rows")
        axis.set_title(name.replace("_", " "))
    fig.suptitle(f"OPS Figure {number}: source-contract diagnostic", y=1.02)
    fig.tight_layout()
    paths = []
    for suffix in ("pdf", "svg", "png"):
        path = output_dir / f"figure{number}.{suffix}"
        fig.savefig(path, dpi=180 if suffix == "png" else None, bbox_inches="tight")
        paths.append(path)
    plt.close(fig)
    return paths


def cli(number: int) -> int:
    parser = argparse.ArgumentParser(description=f"Render OPS Figure {number} from its canonical source contract.")
    parser.add_argument("--tables-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--archival-renderer", help="optional module:function renderer called after validation")
    args = parser.parse_args()
    render(number, args.tables_manifest, args.output_dir, archival_renderer=args.archival_renderer)
    return 0
