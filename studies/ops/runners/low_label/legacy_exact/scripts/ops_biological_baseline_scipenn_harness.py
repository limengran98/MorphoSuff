#!/usr/bin/env python3
"""Common-harness collection for the OPS sciPENN baseline.

This module is sciPENN-specific and deliberately does not modify the frozen
three-method biological-baseline factory.  It exposes the small collection
interface consumed by the already-audited shared training loop while retaining
sciPENN's defining single dense endpoint union and heterogeneous-panel mask.

There is no data loading, optimizer, device policy, checkpoint selection, or
experiment launch in this module.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from typing import Any

import torch
from torch import nn

from ops_biological_baseline_harness_models import (
    OPS_REPORTER_DIMENSIONS,
    OPS_REPORTER_REGISTRY_SHA256,
    PHASE_DIMENSION,
    full_reporter_registry,
    validate_frozen_registry,
)
from ops_biological_baseline_scipenn import OPSSciPENNAdapter
from ops_biological_baseline_scipenn_training import (
    OPSSciPENNTrainingAdapter,
    SciPENNGroupedTrainingResult,
    SciPENNReporterTrainingBatch,
)


FACTORY_SCHEMA_VERSION = "ops-scipenn-harness-factory-v1"
COLLECTION_SCHEMA_VERSION = "ops-scipenn-harness-collection-v1"
METHOD_ID = "scipenn_ops"


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        result = {str(key): _json_safe(item) for key, item in value.items()}
    elif isinstance(value, (list, tuple)):
        result = [_json_safe(item) for item in value]
    elif value is None or isinstance(value, (str, int, float, bool)):
        result = value
    else:
        raise TypeError(f"sciPENN factory option is not JSON-safe: {type(value).__name__}")
    json.dumps(result, sort_keys=True, allow_nan=False)
    return result


def _normalise_active_reporters(values: Sequence[str]) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError("active_reporters must be a sequence")
    result = tuple(str(value) for value in values)
    if not result or len(result) != len(set(result)):
        raise ValueError("active_reporters must be non-empty and unique")
    registry = full_reporter_registry()
    unknown = [name for name in result if name not in registry]
    if unknown:
        raise KeyError(f"Unknown active sciPENN reporters: {unknown}")
    return result


@dataclass(frozen=True)
class SciPENNHarnessConfig:
    """Serializable construction identity for one sciPENN collection."""

    active_reporters: tuple[str, ...]
    options: Mapping[str, Any]
    method: str = METHOD_ID
    schema_version: str = FACTORY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != FACTORY_SCHEMA_VERSION:
            raise ValueError("Unsupported sciPENN harness schema")
        if self.method != METHOD_ID:
            raise ValueError(f"sciPENN method must be {METHOD_ID!r}")
        object.__setattr__(
            self, "active_reporters", _normalise_active_reporters(self.active_reporters)
        )
        options = _json_safe(self.options)
        expected = {
            "hidden_dim",
            "dropout",
            "quantile_levels",
            "loss_reduction",
        }
        if set(options) != expected:
            raise ValueError(
                "sciPENN model option keys changed: "
                f"expected={sorted(expected)}, observed={sorted(options)}"
            )
        object.__setattr__(self, "options", options)

    def to_manifest(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "method": self.method,
            "active_reporters": list(self.active_reporters),
            "options": _json_safe(self.options),
        }


class SciPENNHarnessCollection(nn.Module):
    """One native sciPENN union over the reporter panels in this run."""

    family = "scipenn_ops_dense_union_harness_collection"

    def __init__(self, factory_config: SciPENNHarnessConfig) -> None:
        super().__init__()
        validate_frozen_registry()
        self.factory_config = factory_config
        self.reporter_names = tuple(factory_config.active_reporters)
        full_registry = full_reporter_registry()
        # Native sciPENN forms the protein union over the datasets actually
        # participating in the run.  Panel12 development must therefore have
        # a 12-panel union; injecting 40 never-observed blocks would dilute
        # source_dense_mean and invalidate the source anchor.
        registry = OrderedDict(
            (name, full_registry[name]) for name in self.reporter_names
        )
        options = dict(factory_config.options)
        self.training_adapter = OPSSciPENNTrainingAdapter(
            registry,
            model_options={
                "input_dim": PHASE_DIMENSION,
                "hidden_dim": int(options["hidden_dim"]),
                "dropout": float(options["dropout"]),
                "quantile_levels": tuple(float(v) for v in options["quantile_levels"]),
                "loss_reduction": str(options["loss_reduction"]),
            },
        )
        if tuple(self.training_adapter.model.reporter_names) != self.reporter_names:
            raise RuntimeError("sciPENN model union differs from active campaign panels")

    @property
    def model(self) -> OPSSciPENNAdapter:
        return self.training_adapter.model

    def forward_grouped(
        self, batches: Sequence[SciPENNReporterTrainingBatch]
    ) -> SciPENNGroupedTrainingResult:
        names = tuple(str(batch.reporter) for batch in batches)
        if any(name not in self.reporter_names for name in names):
            raise KeyError("sciPENN grouped batch contains an inactive reporter")
        return self.training_adapter.forward_grouped(batches)

    def predict_reporter(self, phase_x: torch.Tensor, reporter: str) -> torch.Tensor:
        name = str(reporter)
        if name not in self.reporter_names:
            raise KeyError(f"Inactive sciPENN reporter {name!r}")
        return self.training_adapter.predict_reporter(phase_x, name)

    def factory_manifest(self) -> dict[str, Any]:
        payload = {
            "schema_version": COLLECTION_SCHEMA_VERSION,
            "family": self.family,
            "factory_config": self.factory_config.to_manifest(),
            "reporter_names": list(self.reporter_names),
            "full_reporter_registry": [list(value) for value in OPS_REPORTER_DIMENSIONS],
            "full_reporter_registry_sha256": OPS_REPORTER_REGISTRY_SHA256,
            "training_union_reporters": list(self.model.reporter_names),
            "training_union_endpoints": self.model.total_endpoints,
            "training_union_rule": "union_of_reporter_panels_participating_in_this_run",
            "model": self.training_adapter.config_manifest(),
            "method_distinction": {
                "single_dense_output_union": True,
                "heterogeneous_panel_mask": True,
                "reporter_queries": False,
                "semantic_queries": False,
                "reporter_specific_heads": False,
                "reporter_specific_experts": False,
            },
        }
        canonical = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        payload["manifest_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
        return payload


def create_scipenn_harness_model(
    *, active_reporters: Sequence[str], options: Mapping[str, Any]
) -> SciPENNHarnessCollection:
    return SciPENNHarnessCollection(
        SciPENNHarnessConfig(
            active_reporters=tuple(active_reporters),
            options=options,
        )
    )


# Compatibility alias for annotations in the imported common runner only.
HarnessModelCollection = SciPENNHarnessCollection


__all__ = [
    "COLLECTION_SCHEMA_VERSION",
    "FACTORY_SCHEMA_VERSION",
    "HarnessModelCollection",
    "METHOD_ID",
    "SciPENNHarnessCollection",
    "SciPENNHarnessConfig",
    "create_scipenn_harness_model",
    "full_reporter_registry",
]
