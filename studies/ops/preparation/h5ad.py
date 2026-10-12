"""Small H5AD/HDF5 helpers shared by the OPS preparation commands."""

from __future__ import annotations

import hashlib
from pathlib import Path

import h5py
import numpy as np


def decode(values: np.ndarray) -> np.ndarray:
    """Decode an H5AD string vector without depending on anndata."""

    return np.asarray(
        [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in values],
        dtype=str,
    )


def categorical_values(handle: h5py.File, table: str, column: str) -> np.ndarray:
    """Materialize one categorical or dense H5AD column as strings."""

    node = handle[f"{table}/{column}"]
    if isinstance(node, h5py.Dataset):
        return decode(node[:])
    categories = decode(node["categories"][:])
    codes = np.asarray(node["codes"][:], dtype=np.int64)
    output = np.full(len(codes), "", dtype=object)
    valid = codes >= 0
    output[valid] = categories[codes[valid]]
    return output.astype(str)


def categorical_rows(
    handle: h5py.File, table: str, column: str, indices: np.ndarray
) -> np.ndarray:
    """Read selected rows from one H5AD categorical column."""

    node = handle[f"{table}/{column}"]
    if isinstance(node, h5py.Dataset):
        return decode(read_rows(node, indices))
    codes = np.asarray(read_rows(node["codes"], indices), dtype=np.int64)
    categories = decode(node["categories"][:])
    output = np.full(len(indices), "", dtype=object)
    valid = codes >= 0
    output[valid] = categories[codes[valid]]
    return output.astype(str)


def categories(handle: h5py.File, table: str, column: str) -> np.ndarray:
    return decode(handle[f"{table}/{column}/categories"][:])


def codes(handle: h5py.File, table: str, column: str) -> np.ndarray:
    return np.asarray(handle[f"{table}/{column}/codes"][:], dtype=np.int32)


def read_rows(dataset: h5py.Dataset, indices: np.ndarray) -> np.ndarray:
    """h5py-safe fancy indexing that restores duplicates and caller order."""

    indices = np.asarray(indices, dtype=np.int64)
    unique, inverse = np.unique(indices, return_inverse=True)
    return np.asarray(dataset[unique])[inverse]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_h5ad_columns(
    handle: h5py.File,
    *,
    obs: tuple[str, ...] = (),
    var: tuple[str, ...] = (),
) -> None:
    missing = [f"obs/{name}" for name in obs if f"obs/{name}" not in handle]
    missing += [f"var/{name}" for name in var if f"var/{name}" not in handle]
    if "X" not in handle:
        missing.append("X")
    if missing:
        raise ValueError("H5AD is missing required nodes: " + ", ".join(missing))

