"""Study-owned utilities for the OPS raw-phase representation experiments.

The functions in this module are deliberately independent of the historical
cluster harness.  They implement the frozen reporter cache, split, training,
and evaluation contracts used by the DINOv2/Cytoland pilot while leaving all
data, embeddings, and weights outside the source checkout.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np
import pandas as pd
from scipy.stats import rankdata


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_seed(*tokens: object, base: int = 20260730) -> int:
    payload = "|".join(str(token) for token in tokens).encode("utf-8")
    return (base + int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")) % (
        2**31 - 1
    )


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def atomic_npy(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    with temporary.open("wb") as handle:
        np.save(handle, np.asarray(values), allow_pickle=False)
    os.replace(temporary, path)


def decode(values: np.ndarray) -> np.ndarray:
    return np.asarray(
        [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in values],
        dtype=str,
    )


@dataclass(frozen=True)
class PhaseCache:
    root: Path
    manifest: dict[str, Any]
    metadata: dict[str, np.ndarray]
    folds: np.ndarray

    @classmethod
    def open(cls, root: Path, split: str = "gene_holdout_main") -> "PhaseCache":
        manifest_path = root / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Missing phase-cache manifest: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        metadata = {
            name: np.load(root / f"{name}.npy", mmap_mode="r")
            for name in ("screen_code", "well_code", "tile_code", "gene_code")
        }
        folds = np.load(root / f"{split}.fold.npy", mmap_mode="r")
        lengths = {name: len(values) for name, values in {**metadata, "fold": folds}.items()}
        if len(set(lengths.values())) != 1:
            raise RuntimeError(f"Phase-cache row counts are inconsistent: {lengths}")
        return cls(root=root, manifest=manifest, metadata=metadata, folds=folds)


@dataclass(frozen=True)
class ReporterData:
    slug: str
    phase_rows: np.ndarray
    y: np.ndarray
    is_control: np.ndarray
    target_feature_names: np.ndarray


def load_reporter_data(
    cache_path: Path,
    slug: str,
    selected_target_feature_names: Iterable[str] | None = None,
) -> ReporterData:
    if not cache_path.is_file():
        raise FileNotFoundError(cache_path)
    with h5py.File(cache_path, "r") as source:
        phase_rows = np.asarray(source["phase_row_index"][:], dtype=np.int64)
        y = np.asarray(source["fluorescence"][:], dtype=np.float32)
        is_control = np.asarray(source["metadata/is_control"][:], dtype=bool)
        names = decode(source["features/target_feature_names"][:])
    if selected_target_feature_names is not None:
        selected = tuple(map(str, selected_target_feature_names))
        if not selected or len(set(selected)) != len(selected):
            raise RuntimeError(f"Invalid target-feature selection for {slug}")
        lookup = {name: index for index, name in enumerate(names.astype(str))}
        missing = [name for name in selected if name not in lookup]
        if missing:
            raise RuntimeError(f"Reporter {slug} lacks selected endpoints: {missing[:10]}")
        order = np.asarray([lookup[name] for name in selected], dtype=np.int64)
        y = y[:, order]
        names = names[order]
    finite = np.all(np.isfinite(y), axis=1)
    phase_rows, y, is_control = phase_rows[finite], y[finite], is_control[finite]
    if len(np.unique(phase_rows)) != len(phase_rows):
        raise RuntimeError(f"Duplicate phase rows within reporter cache: {cache_path}")
    return ReporterData(slug, phase_rows, y, is_control, names)


def technical_core(path: Path, reporter: str) -> list[str]:
    table = pd.read_csv(path)
    required = {"reporter_slug", "target_feature_name", "selected_in_technical_core"}
    missing = required - set(table.columns)
    if missing:
        raise RuntimeError(f"Target dictionary is missing columns: {sorted(missing)}")
    selected = table["selected_in_technical_core"].astype(str).str.casefold().isin(
        {"true", "1", "yes"}
    )
    frame = table.loc[selected & table["reporter_slug"].astype(str).eq(reporter)]
    names = frame["target_feature_name"].astype(str).tolist()
    if not names or len(names) != len(set(names)):
        raise RuntimeError(f"No unique technical-core endpoint set for {reporter}")
    return names


@dataclass(frozen=True)
class PreprocessingState:
    x_keep: np.ndarray
    x_median: np.ndarray
    x_mean: np.ndarray
    x_scale: np.ndarray
    y_keep: np.ndarray
    y_mean: np.ndarray
    y_scale: np.ndarray

    def transform_x(self, values: np.ndarray) -> np.ndarray:
        result = np.asarray(values[:, self.x_keep], dtype=np.float32).copy()
        bad = ~np.isfinite(result)
        if np.any(bad):
            rows, columns = np.nonzero(bad)
            result[rows, columns] = self.x_median[columns]
        return (result - self.x_mean) / self.x_scale

    def transform_y(self, values: np.ndarray) -> np.ndarray:
        return (np.asarray(values[:, self.y_keep], dtype=np.float32) - self.y_mean) / self.y_scale

    def inverse_y(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values, dtype=np.float32) * self.y_scale + self.y_mean


def fit_preprocessing(
    x_train: np.ndarray,
    y_train: np.ndarray,
    min_x_finite_fraction: float = 0.95,
    variance_epsilon: float = 1e-8,
) -> PreprocessingState:
    finite_fraction = np.mean(np.isfinite(x_train), axis=0)
    candidate = np.flatnonzero(finite_fraction >= min_x_finite_fraction)
    candidate_values = np.asarray(x_train[:, candidate], dtype=np.float32)
    with np.errstate(all="ignore"):
        medians = np.nanmedian(
            np.where(np.isfinite(candidate_values), candidate_values, np.nan), axis=0
        )
    valid = np.isfinite(medians)
    candidate, candidate_values, medians = candidate[valid], candidate_values[:, valid], medians[valid]
    bad = ~np.isfinite(candidate_values)
    if np.any(bad):
        rows, columns = np.nonzero(bad)
        candidate_values[rows, columns] = medians[columns]
    x_mean = np.mean(candidate_values, axis=0, dtype=np.float64).astype(np.float32)
    x_scale = np.std(candidate_values, axis=0, dtype=np.float64).astype(np.float32)
    x_variable = np.isfinite(x_scale) & (x_scale > variance_epsilon)
    y_mean = np.mean(y_train, axis=0, dtype=np.float64).astype(np.float32)
    y_scale = np.std(y_train, axis=0, dtype=np.float64).astype(np.float32)
    y_variable = np.isfinite(y_scale) & (y_scale > variance_epsilon)
    if not np.any(x_variable) or not np.any(y_variable):
        raise RuntimeError("No non-constant input or target endpoints in training")
    return PreprocessingState(
        x_keep=candidate[x_variable],
        x_median=np.asarray(medians[x_variable], dtype=np.float32),
        x_mean=x_mean[x_variable],
        x_scale=x_scale[x_variable],
        y_keep=np.flatnonzero(y_variable),
        y_mean=y_mean[y_variable],
        y_scale=y_scale[y_variable],
    )


def partition_indices(
    fold_values: np.ndarray, outer_fold: int, n_folds: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    validation_fold = (outer_fold + 1) % n_folds
    test = np.flatnonzero(fold_values == outer_fold)
    validation = np.flatnonzero(fold_values == validation_fold)
    train = np.flatnonzero((fold_values != outer_fold) & (fold_values != validation_fold))
    if min(map(len, (train, validation, test))) == 0:
        raise RuntimeError("Train, validation, and test partitions must all be non-empty")
    return train, validation, test, validation_fold


def assert_gene_split_integrity(
    train: np.ndarray,
    validation: np.ndarray,
    test: np.ndarray,
    is_control: np.ndarray,
    gene_codes: np.ndarray,
    phase_rows: np.ndarray,
) -> None:
    if len(np.unique(phase_rows)) != len(phase_rows):
        raise RuntimeError("A phase cell occurs more than once within one reporter")
    target = ~is_control
    groups = [set(gene_codes[index][target[index]].tolist()) for index in (train, validation, test)]
    if groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2]:
        raise RuntimeError("Target genes cross train/validation/test boundaries")


def embedding_lookup(root: Path, phase_rows: np.ndarray) -> np.ndarray:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "COMPLETE":
        raise RuntimeError(f"Embedding extraction is not complete: {manifest.get('status')}")
    stored_rows = np.load(root / "phase_row_index.npy", mmap_mode="r")
    order = np.load(root / "phase_row_sort_order.npy", mmap_mode="r")
    sorted_rows = np.asarray(stored_rows[order], dtype=np.int64)
    query = np.asarray(phase_rows, dtype=np.int64)
    locations = np.searchsorted(sorted_rows, query)
    safe = np.minimum(locations, max(len(sorted_rows) - 1, 0))
    valid = (locations < len(sorted_rows)) & (sorted_rows[safe] == query)
    if not np.all(valid):
        raise RuntimeError(f"Embeddings lack {int((~valid).sum())}/{len(query)} exact cells")
    values = np.load(root / "embedding.float16.npy", mmap_mode="r")
    return np.asarray(values[np.asarray(order[locations], dtype=np.int64)], dtype=np.float32)


def fit_mlp(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    x_test: np.ndarray,
    *,
    hidden: list[int],
    batch_size: int,
    epochs: int,
    patience: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
    device_name: str = "auto",
) -> tuple[np.ndarray, dict[str, Any]]:
    import torch
    from torch import nn

    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device(device_name)
    x_train_t = torch.as_tensor(x_train, dtype=torch.float32, device=device)
    y_train_t = torch.as_tensor(y_train, dtype=torch.float32, device=device)
    x_validation_t = torch.as_tensor(x_validation, dtype=torch.float32, device=device)
    y_validation_t = torch.as_tensor(y_validation, dtype=torch.float32, device=device)
    x_test_t = torch.as_tensor(x_test, dtype=torch.float32, device=device)
    layers: list[nn.Module] = []
    width = x_train.shape[1]
    for next_width in hidden:
        layers.extend((nn.Linear(width, next_width), nn.GELU(), nn.LayerNorm(next_width)))
        width = next_width
    layers.append(nn.Linear(width, y_train.shape[1]))
    model = nn.Sequential(*layers).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    best_loss, best_state, stale = math.inf, None, 0
    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(x_train_t), device=device)
        total = 0.0
        for start in range(0, len(order), batch_size):
            rows = order[start : start + batch_size]
            prediction = model(x_train_t[rows])
            loss = torch.mean((prediction - y_train_t[rows]) ** 2)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.detach().cpu()) * len(rows)
        model.eval()
        with torch.inference_mode():
            validation_loss = float(
                torch.mean((model(x_validation_t) - y_validation_t) ** 2).cpu()
            )
        history.append(
            {"epoch": epoch, "train_mse": total / len(order), "validation_mse": validation_loss}
        )
        if validation_loss < best_loss - 1e-6:
            best_loss = validation_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    if best_state is None:
        raise RuntimeError("MLP did not produce a valid validation checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    with torch.inference_mode():
        prediction = model(x_test_t).float().cpu().numpy()
    return prediction.astype(np.float32, copy=False), {
        "backend": "pytorch",
        "device": device_name,
        "hidden": hidden,
        "batch_size": batch_size,
        "epochs_requested": epochs,
        "epochs_completed": len(history),
        "patience": patience,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "best_validation_mse": best_loss,
        "history": history,
        "torch_version": torch.__version__,
    }


def _pearson_columns(truth: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    a = truth - np.mean(truth, axis=0, keepdims=True)
    b = prediction - np.mean(prediction, axis=0, keepdims=True)
    denominator = np.sqrt(np.sum(a**2, axis=0) * np.sum(b**2, axis=0))
    result = np.full(truth.shape[1], np.nan, dtype=float)
    valid = denominator > 0
    result[valid] = np.sum(a[:, valid] * b[:, valid], axis=0) / denominator[valid]
    return result


def _spearman_columns(truth: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    result = np.full(truth.shape[1], np.nan, dtype=float)
    for column in range(truth.shape[1]):
        result[column] = _pearson_columns(
            rankdata(truth[:, column]).reshape(-1, 1),
            rankdata(prediction[:, column]).reshape(-1, 1),
        )[0]
    return result


def matrix_metrics(
    truth: np.ndarray, prediction: np.ndarray, baseline: np.ndarray, feature_names: np.ndarray
) -> tuple[dict[str, float], pd.DataFrame]:
    truth, prediction, baseline = map(
        lambda value: np.asarray(value, dtype=np.float32), (truth, prediction, baseline)
    )
    mse = np.mean((truth - prediction) ** 2, axis=0)
    baseline_mse = np.mean((truth - baseline) ** 2, axis=0)
    gain = np.where(baseline_mse > 0, 1.0 - mse / baseline_mse, np.nan)
    pearson = _pearson_columns(truth, prediction)
    spearman = _spearman_columns(truth, prediction)
    table = pd.DataFrame(
        {"feature": feature_names, "mse": mse, "baseline_mse": baseline_mse,
         "gain_vs_train_mean": gain, "pearson": pearson, "spearman": spearman}
    )
    return {
        "mse": float(np.mean(mse)),
        "baseline_mse": float(np.mean(baseline_mse)),
        "gain_vs_train_mean": float(1.0 - np.mean(mse) / np.mean(baseline_mse)),
        "macro_feature_pearson": float(np.nanmean(pearson)),
        "macro_feature_spearman": float(np.nanmean(spearman)),
    }, table


@dataclass(frozen=True)
class GeneProfiles:
    truth: np.ndarray
    prediction: np.ndarray
    screen_code: np.ndarray
    gene_code: np.ndarray


def _group_means(
    values: np.ndarray, primary: np.ndarray, secondary: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    keys = np.asarray(primary, dtype=np.int64) if secondary is None else (
        (np.asarray(primary, dtype=np.int64) << 32)
        ^ (np.asarray(secondary, dtype=np.int64) & 0xFFFFFFFF)
    )
    order = np.argsort(keys, kind="stable")
    unique, starts, counts = np.unique(keys[order], return_index=True, return_counts=True)
    sums = np.add.reduceat(np.asarray(values[order], dtype=np.float64), starts, axis=0)
    return unique, np.asarray(sums / counts[:, None], dtype=np.float32)


def aggregate_gene_profiles(
    truth: np.ndarray,
    prediction: np.ndarray,
    screen_codes: np.ndarray,
    gene_codes: np.ndarray,
    is_control: np.ndarray,
) -> GeneProfiles:
    controls, targets = np.flatnonzero(is_control), np.flatnonzero(~is_control)
    if not len(controls) or not len(targets):
        raise RuntimeError("Matched-control aggregation requires controls and target cells")
    control_screens, control_truth = _group_means(truth[controls], screen_codes[controls])
    predicted_screens, control_prediction = _group_means(prediction[controls], screen_codes[controls])
    group_keys, group_truth = _group_means(truth[targets], screen_codes[targets], gene_codes[targets])
    predicted_keys, group_prediction = _group_means(
        prediction[targets], screen_codes[targets], gene_codes[targets]
    )
    if not np.array_equal(control_screens, predicted_screens) or not np.array_equal(group_keys, predicted_keys):
        raise RuntimeError("True and predicted aggregation groups are misaligned")
    group_screen = (group_keys >> 32).astype(np.int64)
    group_gene = (group_keys & 0xFFFFFFFF).astype(np.int64)
    where = np.searchsorted(control_screens, group_screen)
    valid = (where < len(control_screens)) & (
        control_screens[np.minimum(where, len(control_screens) - 1)] == group_screen
    )
    if not np.all(valid):
        raise RuntimeError("At least one target group lacks a matched control in its screen")
    return GeneProfiles(
        group_truth - control_truth[where],
        group_prediction - control_prediction[where],
        group_screen,
        group_gene,
    )


def profile_metrics(
    profiles: GeneProfiles, feature_names: np.ndarray
) -> tuple[dict[str, float], pd.DataFrame]:
    summary, table = matrix_metrics(
        profiles.truth, profiles.prediction, np.zeros_like(profiles.truth), feature_names
    )
    truth_norm = np.linalg.norm(profiles.truth, axis=1)
    prediction_norm = np.linalg.norm(profiles.prediction, axis=1)
    denominator = truth_norm * prediction_norm
    cosine = np.full(len(truth_norm), np.nan)
    valid = denominator > 0
    cosine[valid] = np.sum(profiles.truth[valid] * profiles.prediction[valid], axis=1) / denominator[valid]
    summary.update(
        {
            "mean_profile_cosine": float(np.nanmean(cosine)),
            "response_magnitude_spearman": float(
                _spearman_columns(truth_norm[:, None], prediction_norm[:, None])[0]
            ),
            "n_gene_screen_groups": int(len(profiles.truth)),
            "n_genes": int(len(np.unique(profiles.gene_code))),
            "n_screens": int(len(np.unique(profiles.screen_code))),
        }
    )
    return summary, table


def bootstrap_gene_metrics(profiles: GeneProfiles, draws: int, seed: int) -> dict[str, float]:
    genes = np.unique(profiles.gene_code)
    groups = {gene: np.flatnonzero(profiles.gene_code == gene) for gene in genes}
    rng = np.random.default_rng(seed)
    gains, magnitudes = np.empty(draws), np.empty(draws)
    for draw in range(draws):
        rows = np.concatenate([groups[gene] for gene in rng.choice(genes, len(genes), replace=True)])
        truth, prediction = profiles.truth[rows], profiles.prediction[rows]
        denominator = float(np.mean(truth**2))
        gains[draw] = 1.0 - float(np.mean((truth - prediction) ** 2)) / denominator
        magnitudes[draw] = _spearman_columns(
            np.linalg.norm(truth, axis=1)[:, None], np.linalg.norm(prediction, axis=1)[:, None]
        )[0]
    return {
        "gene_bootstrap_draws": draws,
        "gain_ci95_low": float(np.nanquantile(gains, 0.025)),
        "gain_ci95_high": float(np.nanquantile(gains, 0.975)),
        "magnitude_spearman_ci95_low": float(np.nanquantile(magnitudes, 0.025)),
        "magnitude_spearman_ci95_high": float(np.nanquantile(magnitudes, 0.975)),
    }
def read_table(path: Path, *, columns: list[str] | None = None) -> pd.DataFrame:
    """Read a CSV or Parquet table through one explicit file contract."""

    if path.suffix.casefold() == ".csv":
        return pd.read_csv(path, usecols=columns)
    return pd.read_parquet(path, columns=columns)

