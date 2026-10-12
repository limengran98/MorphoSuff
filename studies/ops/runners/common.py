"""Shared table, split, preprocessing and export utilities for OPS runners.

The runner layer consumes public, canonical tables.  It deliberately keeps
data locations outside the source tree and learns every imputation/scaling
quantity from the training role of the requested fold only.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from measurement_sufficiency.schemas import coerce_boolean_column
from measurement_sufficiency.training import infer_feature_columns


METADATA_COLUMNS = {
    "observation_id",
    "reporter_id",
    "screen_id",
    "perturbation_id",
    "field_id",
    "assay_id",
    "is_control",
    "fold",
    "role",
}


class RunnerContractError(ValueError):
    """Raised when a public table or split violates the runner contract."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RunnerContractError(message)


def read_table(path: str | Path, *, columns: Sequence[str] | None = None) -> pd.DataFrame:
    path = Path(path).expanduser().resolve()
    require(path.is_file(), f"table does not exist: {path}")
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path, columns=None if columns is None else list(columns))
    if path.suffix.lower() in {".csv", ".txt", ".tsv"}:
        separator = "\t" if path.suffix.lower() == ".tsv" else ","
        return pd.read_csv(path, usecols=None if columns is None else list(columns), sep=separator)
    raise RunnerContractError(f"unsupported table extension: {path.suffix}")


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_seed(*parts: object) -> int:
    payload = "\0".join(map(str, parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**31 - 1)


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    try:
        import torch

        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
    except ImportError:
        pass


def atomic_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    temporary.write_text(json.dumps(dict(payload), indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def write_frame(frame: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() in {".parquet", ".pq"}:
        frame.to_parquet(path, index=False)
    else:
        frame.to_csv(path, index=False)


@dataclass(frozen=True)
class FoldTables:
    inputs: pd.DataFrame
    assignments: pd.DataFrame
    feature_columns: tuple[str, ...]
    train_ids: tuple[str, ...]
    validation_ids: tuple[str, ...]
    test_ids: tuple[str, ...]


def _normalise_role(value: object) -> str:
    text = str(value).strip().casefold()
    aliases = {"val": "validation", "valid": "validation", "outer_test": "test"}
    return aliases.get(text, text)


def load_fold_tables(
    inputs_path: str | Path,
    assignments_path: str | Path,
    *,
    fold: int,
    feature_columns: Sequence[str] | None = None,
) -> FoldTables:
    inputs = read_table(inputs_path)
    assignments = read_table(assignments_path)
    required_inputs = {"observation_id", "reporter_id", "screen_id", "perturbation_id"}
    required_assignments = {"observation_id", "fold", "role"}
    require(required_inputs.issubset(inputs), f"inputs lack {sorted(required_inputs - set(inputs))}")
    require(
        required_assignments.issubset(assignments),
        f"assignments lack {sorted(required_assignments - set(assignments))}",
    )
    require(not inputs.observation_id.isna().any(), "inputs contain null observation_id")
    require(not inputs.observation_id.duplicated().any(), "inputs duplicate observation_id")
    inputs = inputs.copy()
    inputs["observation_id"] = inputs.observation_id.astype(str)
    inputs["reporter_id"] = inputs.reporter_id.astype(str)
    assignments = assignments.loc[assignments.fold.astype(int).eq(int(fold))].copy()
    require(not assignments.empty, f"fold {fold} has no assignments")
    assignments["observation_id"] = assignments.observation_id.astype(str)
    assignments["role"] = assignments.role.map(_normalise_role)
    require(
        set(assignments.role).issubset({"train", "validation", "test"}),
        f"fold {fold} contains unsupported roles: {sorted(set(assignments.role))}",
    )
    require(
        not assignments.observation_id.duplicated().any(),
        "one observation has multiple roles in the same fold",
    )
    missing = set(assignments.observation_id).difference(inputs.observation_id)
    require(not missing, f"assignments reference {len(missing)} absent input observations")
    ids = {
        role: tuple(assignments.loc[assignments.role.eq(role), "observation_id"])
        for role in ("train", "validation", "test")
    }
    for role, values in ids.items():
        require(bool(values), f"fold {fold} has no {role} observations")
    if feature_columns is None:
        # Shared with the portable toolkit so the two layers cannot disagree about
        # what counts as a feature: identifier columns are excluded by name, and a
        # column that is mostly numeric with stray tokens raises instead of being
        # dropped without notice.
        feature_columns = infer_feature_columns(inputs, exclude=sorted(METADATA_COLUMNS))
    else:
        feature_columns = tuple(map(str, feature_columns))
    require(bool(feature_columns), "no feature columns selected")
    require(len(set(feature_columns)) == len(feature_columns), "feature columns are duplicated")
    require(set(feature_columns).issubset(inputs), "one or more feature columns are absent")
    return FoldTables(
        inputs=inputs,
        assignments=assignments,
        feature_columns=tuple(feature_columns),
        train_ids=ids["train"],
        validation_ids=ids["validation"],
        test_ids=ids["test"],
    )


@dataclass(frozen=True)
class Standardizer:
    fill: np.ndarray
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray) -> "Standardizer":
        values = np.asarray(values, dtype=np.float32)
        require(values.ndim == 2 and len(values), "standardizer needs a nonempty matrix")
        finite = np.isfinite(values)
        with np.errstate(all="ignore"):
            fill = np.nanmedian(np.where(finite, values, np.nan), axis=0)
        fill = np.where(np.isfinite(fill), fill, 0.0).astype(np.float32)
        clean = np.where(finite, values, fill)
        mean = clean.mean(axis=0, dtype=np.float64).astype(np.float32)
        scale = clean.std(axis=0, dtype=np.float64).astype(np.float32)
        scale = np.where(np.isfinite(scale) & (scale > 1e-6), scale, 1.0).astype(np.float32)
        return cls(fill, mean, scale)

    def transform(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32)
        clean = np.where(np.isfinite(values), values, self.fill)
        transformed = (clean - self.mean) / self.scale
        require(np.isfinite(transformed).all(), "non-finite values after preprocessing")
        return transformed.astype(np.float32, copy=False)

    def inverse(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values, dtype=np.float32) * self.scale + self.mean

    def manifest(self) -> dict[str, list[float]]:
        return {
            "fill": self.fill.astype(float).tolist(),
            "mean": self.mean.astype(float).tolist(),
            "scale": self.scale.astype(float).tolist(),
        }


def fit_input_standardizer(tables: FoldTables) -> Standardizer:
    train = tables.inputs.loc[
        tables.inputs.observation_id.isin(tables.train_ids), tables.feature_columns
    ].to_numpy(dtype=np.float32)
    return Standardizer.fit(train)


def input_matrix(
    tables: FoldTables,
    observation_ids: Sequence[str],
    standardizer: Standardizer,
) -> tuple[pd.DataFrame, np.ndarray]:
    order = pd.DataFrame({"observation_id": list(map(str, observation_ids))})
    rows = order.merge(tables.inputs, on="observation_id", how="left", validate="one_to_one")
    require(not rows[list(tables.feature_columns)].isna().all(axis=1).any(), "input rows are absent")
    return rows, standardizer.transform(rows.loc[:, tables.feature_columns].to_numpy(np.float32))


def _target_columns(frame: pd.DataFrame) -> None:
    required = {"observation_id", "endpoint_id", "y_true"}
    require(required.issubset(frame), f"targets lack {sorted(required - set(frame))}")


def load_targets_for_ids(path: str | Path, ids: Iterable[str]) -> pd.DataFrame:
    """Read and retain targets only for the explicitly requested role IDs."""

    ids = set(map(str, ids))
    require(bool(ids), "target ID selection is empty")
    path = Path(path).expanduser().resolve()
    if path.suffix.lower() in {".csv", ".txt", ".tsv"}:
        separator = "\t" if path.suffix.lower() == ".tsv" else ","
        parts: list[pd.DataFrame] = []
        for chunk in pd.read_csv(path, sep=separator, chunksize=250_000):
            _target_columns(chunk)
            chunk["observation_id"] = chunk.observation_id.astype(str)
            selected = chunk.loc[chunk.observation_id.isin(ids)]
            if not selected.empty:
                parts.append(selected.copy())
        frame = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    else:
        frame = read_table(path)
        _target_columns(frame)
        frame["observation_id"] = frame.observation_id.astype(str)
        frame = frame.loc[frame.observation_id.isin(ids)].copy()
    require(not frame.empty, "selected role has no target rows")
    if "is_observed" in frame:
        # Strict parsing: a CSV round trip turns False into the string "False",
        # and .astype(bool) would keep every such row as an observed target.
        frame = frame.loc[
            coerce_boolean_column(
                frame["is_observed"], column="is_observed", table="target table"
            )
        ].copy()
    frame = frame.loc[np.isfinite(pd.to_numeric(frame.y_true, errors="coerce"))].copy()
    require(not frame.empty, "selected role has no finite observed targets")
    require(
        not frame.duplicated(["observation_id", "endpoint_id"]).any(),
        "targets duplicate observation_id × endpoint_id",
    )
    return frame


@dataclass(frozen=True)
class ReporterTargetBlock:
    reporter_id: str
    endpoint_ids: tuple[str, ...]
    observation_ids: tuple[str, ...]
    values: np.ndarray
    observed: np.ndarray


def reporter_target_block(
    targets: pd.DataFrame,
    inputs: pd.DataFrame,
    reporter_id: str,
    *,
    endpoint_ids: Sequence[str] | None = None,
) -> ReporterTargetBlock:
    reporter_id = str(reporter_id)
    ids = tuple(inputs.loc[inputs.reporter_id.astype(str).eq(reporter_id), "observation_id"].astype(str))
    require(bool(ids), f"no inputs for reporter {reporter_id}")
    local = targets.loc[targets.observation_id.astype(str).isin(ids)].copy()
    if "reporter_id" in local:
        require(
            local.reporter_id.astype(str).eq(reporter_id).all(),
            f"target reporter binding differs for {reporter_id}",
        )
    if endpoint_ids is None:
        endpoint_ids = tuple(dict.fromkeys(local.endpoint_id.astype(str)))
    else:
        endpoint_ids = tuple(map(str, endpoint_ids))
    require(bool(endpoint_ids), f"no endpoint schema for reporter {reporter_id}")
    lookup_row = {value: index for index, value in enumerate(ids)}
    lookup_endpoint = {value: index for index, value in enumerate(endpoint_ids)}
    values = np.zeros((len(ids), len(endpoint_ids)), dtype=np.float32)
    observed = np.zeros_like(values, dtype=bool)
    for row in local.itertuples(index=False):
        endpoint = str(row.endpoint_id)
        if endpoint not in lookup_endpoint:
            continue
        i, j = lookup_row[str(row.observation_id)], lookup_endpoint[endpoint]
        values[i, j] = float(row.y_true)
        observed[i, j] = True
    keep = observed.any(axis=1)
    require(bool(keep.any()), f"no observed target rows for reporter {reporter_id}")
    return ReporterTargetBlock(
        reporter_id,
        tuple(endpoint_ids),
        tuple(np.asarray(ids, dtype=object)[keep].astype(str)),
        values[keep],
        observed[keep],
    )


def fit_target_standardizer(block: ReporterTargetBlock) -> Standardizer:
    values = np.where(block.observed, block.values, np.nan)
    return Standardizer.fit(values)


def target_transform(block: ReporterTargetBlock, scaler: Standardizer) -> np.ndarray:
    transformed = scaler.transform(np.where(block.observed, block.values, np.nan))
    return np.where(block.observed, transformed, 0.0).astype(np.float32)


def canonical_predictions(
    *,
    metadata: pd.DataFrame,
    block: ReporterTargetBlock,
    prediction: np.ndarray,
    model_id: str,
    split_name: str,
    fold: int,
    require_is_control: bool = True,
) -> pd.DataFrame:
    prediction = np.asarray(prediction, dtype=np.float32)
    require(prediction.shape == block.values.shape, "prediction shape differs from target block")
    carries_control = "is_control" in metadata
    if require_is_control and not carries_control:
        raise RunnerContractError(
            "prediction metadata lacks is_control, so the exported predictions cannot "
            "feed control-relative response fidelity. Re-export the canonical bundle "
            "with studies/ops/preparation/export_canonical.py, which writes is_control "
            "into the inputs table, or join it from the observations table on "
            "observation_id. Pass require_is_control=False only for a dataset that "
            "genuinely has no screen-matched controls."
        )
    if carries_control:
        metadata = metadata.assign(
            is_control=coerce_boolean_column(
                metadata["is_control"], column="is_control", table="prediction metadata"
            )
        )
    meta = metadata.set_index(metadata.observation_id.astype(str), drop=False)
    rows: list[dict[str, object]] = []
    for i, observation_id in enumerate(block.observation_ids):
        require(observation_id in meta.index, f"metadata missing {observation_id}")
        source = meta.loc[observation_id]
        if isinstance(source, pd.DataFrame):
            raise RunnerContractError(f"metadata duplicates {observation_id}")
        for j, endpoint_id in enumerate(block.endpoint_ids):
            if not block.observed[i, j]:
                continue
            rows.append(
                {
                    "observation_id": observation_id,
                    "reporter_id": block.reporter_id,
                    "endpoint_id": endpoint_id,
                    "split_name": split_name,
                    "fold": int(fold),
                    "model_id": model_id,
                    "y_true": float(block.values[i, j]),
                    "y_pred": float(prediction[i, j]),
                    "screen_id": source.screen_id,
                    "perturbation_id": source.perturbation_id,
                    **({"is_control": bool(source.is_control)} if carries_control else {}),
                }
            )
    result = pd.DataFrame(rows)
    require(not result.empty, "canonical prediction export is empty")
    return result


def resolve_device(requested: str) -> str:
    import torch

    requested = str(requested)
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RunnerContractError("CUDA was requested but is unavailable; use --device cpu for a smoke run")
    return str(torch.device(requested))

