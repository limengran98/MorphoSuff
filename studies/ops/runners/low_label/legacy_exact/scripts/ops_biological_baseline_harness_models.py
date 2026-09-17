#!/usr/bin/env python3
"""Model collections for the common OPS biological-baseline harness.

This module is a construction and routing layer, not a training runner.  It
binds three deliberately different method families to the same frozen
52-reporter / 1,604-endpoint scientific registry without turning them into one
architecture:

* OPS-CAPTAIN keeps one shared endpoint-token model and its reporter-grouped
  training bridge;
* MIDAS-OPS keeps phase plus 52 independent sparse reporter modalities with
  modality-local fronts/heads and shared latent encoder/decoder trunks;
* scButterfly-OPS-B remains a ``ModuleDict`` of independent specialists, whose
  multi-stage optimization is still owned by the existing controller.

There is no data loading, split construction, optimizer, device selection,
checkpoint policy, or experiment launch here.  Loading accepts an already
deserialised state dict so the common harness remains responsible for safe
checkpoint I/O and ``map_location``.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any, Mapping, Sequence

import torch
from torch import nn

from ops_biological_baseline_captain import OPSCaptainAdapter
from ops_biological_baseline_captain_training import (
    CaptainReporterPredictionBatch,
    CaptainReporterTrainingBatch,
    CaptainReporterTrainingResult,
    OPSCaptainTrainingAdapter,
)
from ops_biological_baseline_midas_multimodal import (
    MIDASMultimodalOPS,
    MIDASMultimodalOPSConfig,
    MIDASMultimodalOPSOutput,
)
from ops_biological_baseline_midas_multimodal_training import (
    MIDASMultimodalGroupedTrainingResult,
    MIDASMultimodalOPSTrainingAdapter,
    MIDASMultimodalReporterTrainingBatch,
)
from ops_biological_baseline_scbutterfly import (
    ScButterflyOPSConfig,
    ScButterflyOPSSpecialist,
)


FACTORY_SCHEMA_VERSION = "ops-biological-baseline-harness-factory-v1"
COLLECTION_MANIFEST_SCHEMA = "ops-biological-baseline-harness-model-v1"
PHASE_DIMENSION = 172
PHENOTYPE_UNION_DIMENSION = 1604

# Frozen order and dimensions from
# results/ops_phase0_asset_audit/reporter_targets.csv::
# n_technical_core_features.  Keeping this literal makes factory manifests and
# checkpoint reconstruction independent of mutable directory enumeration.
OPS_REPORTER_DIMENSIONS: tuple[tuple[str, int], ...] = (
    ("fe2+_ferhonox_live-cell_dye", 24),
    ("b-catenin", 30),
    ("c-myc", 30),
    ("endocytic_vesicle_ph_phrodo-dextran_live_cell_dye", 24),
    ("oxidative_stress_cellrox_live-cell_dye", 24),
    ("p21", 30),
    ("p53", 30),
    ("prb", 30),
    ("ps6", 30),
    ("cell_proliferation_marker_mki67", 24),
    ("chromatin_h2bc21", 24),
    ("nuclear_speckles_srrm2", 24),
    ("nuclei_hoechst", 60),
    ("nucleoli_npm1", 72),
    ("nucleolus-dfc_fbl", 24),
    ("nucleolus-gc_npm3", 24),
    ("nuclei_nucleolive_live_cell_dye", 24),
    ("er_cona", 72),
    ("er_ncln", 24),
    ("er_sec61b", 24),
    ("er_golgi_cop-ii_sec23a", 24),
    ("er_golgi_cope", 24),
    ("er_golgi_bridge_vapa", 24),
    ("trans-golgi_vamp3", 24),
    ("chromalive_488_excitation", 24),
    ("mitochondria_chromalive_561_excitation", 24),
    ("mitochondria_tomm20", 72),
    ("mitochondria_tomm70a", 24),
    ("actin_filament_fastact_spy555_live_cell_dye", 24),
    ("f-actin_phalloidin", 72),
    ("microtubules_map4", 24),
    ("microtubules_tubulin", 72),
    ("autophagosome_atg101", 24),
    ("autophagosome_map1lc3b", 24),
    ("clathrin_vesicles_clta", 24),
    ("lipid_droplet_bodipy_live_cell_dye", 24),
    ("lipid_droplet_plin2", 24),
    ("lysosome_lamp1", 24),
    ("lysosome_lysotracker_live-cell_dye", 24),
    ("early_endosome_eea1", 24),
    ("endosome_vps35", 24),
    ("late_endosome_rab7a", 24),
    ("recycling_endosome_tfrc", 24),
    ("plasma_membrane_atp1b3", 24),
    ("plasma_membrane_slc3a2", 24),
    ("plasma_membrane_wga", 72),
    ("peroxisome_peroxi_spy650_live_cell_dye", 24),
    ("5xupre", 20),
    ("caspase_activity_cellevent-caspase_live-cell_dye", 24),
    ("chaperones_hspa1b", 24),
    ("proteasome_psmb7", 24),
    ("stress_granule_g3bp1", 24),
)
OPS_REPORTER_REGISTRY_SHA256 = (
    "f1de8234358def71c7b434c382cb89ae689a6663e014e4ad16727bd515463f9d"
)


def full_reporter_registry() -> OrderedDict[str, int]:
    """Return a fresh ordered copy of the frozen 52-reporter registry."""

    return OrderedDict(OPS_REPORTER_DIMENSIONS)


def _registry_json() -> str:
    return json.dumps(
        [list(value) for value in OPS_REPORTER_DIMENSIONS],
        ensure_ascii=True,
        separators=(",", ":"),
    )


def validate_frozen_registry() -> None:
    dimensions = full_reporter_registry()
    if len(dimensions) != 52 or len(set(dimensions)) != 52:
        raise RuntimeError("Frozen OPS registry must contain 52 unique reporters")
    if sum(dimensions.values()) != PHENOTYPE_UNION_DIMENSION:
        raise RuntimeError("Frozen OPS registry must contain 1,604 endpoints")
    distribution = {
        value: tuple(dimensions.values()).count(value)
        for value in sorted(set(dimensions.values()))
    }
    if distribution != {20: 1, 24: 38, 30: 6, 60: 1, 72: 6}:
        raise RuntimeError(f"Frozen OPS dimension distribution drifted: {distribution}")
    observed_hash = hashlib.sha256(_registry_json().encode("utf-8")).hexdigest()
    if observed_hash != OPS_REPORTER_REGISTRY_SHA256:
        raise RuntimeError("Frozen OPS reporter registry fingerprint drifted")


validate_frozen_registry()


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        result = {str(key): _json_safe(item) for key, item in value.items()}
    elif isinstance(value, (list, tuple)):
        result = [_json_safe(item) for item in value]
    elif value is None or isinstance(value, (str, int, float, bool)):
        result = value
    else:
        raise TypeError(f"Factory option is not JSON-safe: {type(value).__name__}")
    json.dumps(result, sort_keys=True, allow_nan=False)
    return result


_METHOD_ALIASES = {
    "captain": "ops_captain",
    "ops_captain": "ops_captain",
    "ops-captain": "ops_captain",
    "midas": "midas_ops",
    "midas_ops": "midas_ops",
    "midas-ops": "midas_ops",
    "scbutterfly": "scbutterfly_ops_b",
    "scbutterfly_ops_b": "scbutterfly_ops_b",
    "scbutterfly-ops-b": "scbutterfly_ops_b",
}


def _canonical_method(value: str) -> str:
    try:
        return _METHOD_ALIASES[str(value).strip().casefold()]
    except KeyError as error:
        raise KeyError(
            f"Unknown biological baseline method {value!r}; "
            f"available={sorted(set(_METHOD_ALIASES.values()))}"
        ) from error


def _normalise_active_reporters(
    reporters: Sequence[str] | None,
) -> tuple[str, ...]:
    registry = full_reporter_registry()
    if reporters is None:
        return tuple(registry)
    if isinstance(reporters, (str, bytes)) or not isinstance(reporters, Sequence):
        raise TypeError("active_reporters must be a sequence of reporter names")
    result = tuple(str(value) for value in reporters)
    if not result:
        raise ValueError("active_reporters must not be empty")
    if len(set(result)) != len(result):
        raise ValueError("active_reporters contains duplicates")
    unknown = [name for name in result if name not in registry]
    if unknown:
        raise KeyError(f"Unknown active reporters: {unknown}")
    return result


@dataclass(frozen=True)
class HarnessModelConfig:
    """JSON-safe factory configuration for one method collection."""

    method: str
    active_reporters: tuple[str, ...] | None = None
    options: Mapping[str, Any] | None = None
    schema_version: str = FACTORY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != FACTORY_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported factory schema version {self.schema_version!r}"
            )
        object.__setattr__(self, "method", _canonical_method(self.method))
        object.__setattr__(
            self,
            "active_reporters",
            _normalise_active_reporters(self.active_reporters),
        )
        object.__setattr__(self, "options", _json_safe(self.options or {}))

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "HarnessModelConfig":
        payload = dict(value)
        return cls(
            method=payload["method"],
            active_reporters=(
                tuple(payload["active_reporters"])
                if payload.get("active_reporters") is not None
                else None
            ),
            options=payload.get("options", {}),
            schema_version=payload.get("schema_version", FACTORY_SCHEMA_VERSION),
        )

    def to_manifest(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "method": self.method,
            "active_reporters": list(self.active_reporters or ()),
            "options": _json_safe(self.options or {}),
        }


def _registry_offsets() -> OrderedDict[str, tuple[int, int]]:
    offsets: OrderedDict[str, tuple[int, int]] = OrderedDict()
    cursor = 0
    for reporter, dimension in OPS_REPORTER_DIMENSIONS:
        offsets[reporter] = (cursor, cursor + dimension)
        cursor += dimension
    if cursor != PHENOTYPE_UNION_DIMENSION:
        raise AssertionError("Registry offset cursor drifted")
    return offsets


class _RegistryBoundCollection(nn.Module):
    """Registry utilities only; subclasses keep distinct scientific models."""

    def __init__(self, factory_config: HarnessModelConfig) -> None:
        super().__init__()
        self.factory_config = factory_config
        self.reporter_names = tuple(factory_config.active_reporters or ())
        self._active_reporter_set = frozenset(self.reporter_names)
        self._dimensions = full_reporter_registry()
        self._offsets = _registry_offsets()

    @property
    def full_head_dimensions(self) -> OrderedDict[str, int]:
        return full_reporter_registry()

    @property
    def active_head_dimensions(self) -> OrderedDict[str, int]:
        return OrderedDict(
            (name, self._dimensions[name]) for name in self.reporter_names
        )

    def _require_active(self, reporter: str) -> str:
        name = str(reporter)
        if name not in self._active_reporter_set:
            raise KeyError(
                f"Reporter {name!r} is not active; active={self.reporter_names}"
            )
        return name

    def endpoint_slice(self, reporter: str) -> slice:
        name = self._require_active(reporter)
        start, stop = self._offsets[name]
        return slice(start, stop)

    def _base_manifest(self) -> dict[str, Any]:
        return {
            "schema_version": COLLECTION_MANIFEST_SCHEMA,
            "factory_config": self.factory_config.to_manifest(),
            "registry": {
                "source": (
                    "results/ops_phase0_asset_audit/reporter_targets.csv::"
                    "n_technical_core_features"
                ),
                "sha256": OPS_REPORTER_REGISTRY_SHA256,
                "n_reporters": 52,
                "n_endpoints": PHENOTYPE_UNION_DIMENSION,
                "ordered_dimensions": [
                    {"reporter": name, "dimension": dimension}
                    for name, dimension in OPS_REPORTER_DIMENSIONS
                ],
                "active_reporters": list(self.reporter_names),
            },
            "parameter_policy": {
                "matching_required": False,
                "explicit_limit": None,
                "role": "reporting_only",
            },
            "ownership": {
                "data_loading": False,
                "split_construction": False,
                "optimizer": False,
                "scheduler": False,
                "checkpoint_io": False,
                "device_selection": False,
                "experiment_launch": False,
            },
        }

    def factory_manifest(self) -> dict[str, Any]:
        raise NotImplementedError


def _captain_constructor_options(options: Mapping[str, Any]) -> dict[str, Any]:
    values = dict(options)
    if "dropout" in values:
        raise ValueError(
            "CAPTAIN requires separate embedding_dropout, attention_dropout, "
            "and feedforward_dropout fields"
        )
    loss_weights = values.pop("loss_weights", None)
    if loss_weights is not None:
        weights = dict(loss_weights)
        aliases = {
            "mean": "mean_loss_weight",
            "quantile": "quantile_loss_weight",
            "phase_reconstruction": "phase_reconstruction_loss_weight",
        }
        unknown = set(weights) - set(aliases)
        if unknown:
            raise ValueError(f"Unknown CAPTAIN loss weights: {sorted(unknown)}")
        for key, constructor_key in aliases.items():
            if key in weights:
                values[constructor_key] = weights[key]
    forbidden = [
        key
        for key in values
        if "semantic" in key.casefold() or "factor" in key.casefold()
    ]
    if forbidden:
        raise ValueError(
            "OPS-CAPTAIN forbids semantic/factorized query options: "
            f"{forbidden}"
        )
    return values


class CaptainHarnessCollection(_RegistryBoundCollection):
    """Full-registry CAPTAIN plus its existing reporter training bridge."""

    family = "ops_captain_harness_collection"

    def __init__(self, factory_config: HarnessModelConfig) -> None:
        super().__init__(factory_config)
        options = _captain_constructor_options(factory_config.options or {})
        core = OPSCaptainAdapter(self.full_head_dimensions, **options)
        # Register the core exactly once, through the bridge.
        self.training_bridge = OPSCaptainTrainingAdapter(core)

    @property
    def model(self) -> OPSCaptainAdapter:
        return self.training_bridge.model

    def forward_grouped(
        self, batches: Sequence[CaptainReporterTrainingBatch]
    ) -> OrderedDict[str, CaptainReporterTrainingResult]:
        for batch in batches:
            self._require_active(str(batch.reporter))
        return self.training_bridge.forward_grouped(batches)

    def forward(
        self, batches: Sequence[CaptainReporterTrainingBatch]
    ) -> OrderedDict[str, CaptainReporterTrainingResult]:
        return self.forward_grouped(batches)

    def predict_reporter_result(
        self,
        phase_x: torch.Tensor,
        reporter: str,
        *,
        phase_observed_mask: torch.Tensor | None = None,
    ):
        name = self._require_active(reporter)
        return self.training_bridge.predict_reporter(
            phase_x, name, phase_observed_mask=phase_observed_mask
        )

    def predict_reporter(
        self,
        phase_x: torch.Tensor,
        reporter: str,
        *,
        phase_observed_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.predict_reporter_result(
            phase_x,
            reporter,
            phase_observed_mask=phase_observed_mask,
        ).prediction.mean

    def predict_grouped(
        self, phase_by_reporter: Mapping[str, torch.Tensor]
    ) -> OrderedDict[str, torch.Tensor]:
        batches = [
            CaptainReporterPredictionBatch(reporter, phase)
            for reporter, phase in phase_by_reporter.items()
        ]
        for batch in batches:
            self._require_active(str(batch.reporter))
        rich = self.training_bridge.predict_grouped(batches)
        return OrderedDict(
            (name, result.prediction.mean) for name, result in rich.items()
        )

    def factory_manifest(self) -> dict[str, Any]:
        payload = self._base_manifest()
        payload.update(
            {
                "family": self.family,
                "method_difference": (
                    "one shared global-endpoint-token cross-attention model"
                ),
                "training_api": "OPSCaptainTrainingAdapter.forward_grouped",
                "model_manifest": self.model.config_manifest(),
                "parameter_count": sum(p.numel() for p in self.parameters()),
            }
        )
        json.dumps(payload, sort_keys=True, allow_nan=False)
        return payload


MIDASReporterTrainingBatch = MIDASMultimodalReporterTrainingBatch
MIDASGroupedTrainingResult = MIDASMultimodalGroupedTrainingResult


def _midas_config_options(options: Mapping[str, Any]) -> dict[str, Any]:
    values = dict(options)
    # Accepted only to load pre-correction development manifests.  With no
    # trustworthy technical-batch covariate this legacy width has no module to
    # size and is intentionally ignored by the formal multimodal adapter.
    values.pop("batch_hidden_dim", None)
    loss_weights = values.pop("loss_weights", None)
    if loss_weights is not None:
        weights = dict(loss_weights)
        aliases = {
            "phase_reconstruction": "phase_reconstruction_weight",
            "phenotype_reconstruction": "reporter_reconstruction_weight",
            "reporter_reconstruction": "reporter_reconstruction_weight",
            "kl_biological": "kl_biological_weight",
            "kl_technical": "kl_technical_weight",
            "modality_alignment": "modality_alignment_weight",
        }
        unknown = set(weights) - set(aliases)
        if unknown:
            raise ValueError(f"Unknown MIDAS loss weights: {sorted(unknown)}")
        for key, config_key in aliases.items():
            if key in weights:
                values[config_key] = weights[key]
    for key in (
        "shared_encoder_hidden_dims",
        "shared_decoder_hidden_dims",
    ):
        if key in values:
            values[key] = tuple(values[key])
    return values


class MIDASHarnessCollection(_RegistryBoundCollection):
    """OPS-native phase plus 52 sparse reporter-modality MIDAS model."""

    family = "midas_ops_harness_collection"

    def __init__(self, factory_config: HarnessModelConfig) -> None:
        super().__init__(factory_config)
        config = MIDASMultimodalOPSConfig(
            **_midas_config_options(factory_config.options or {})
        )
        core = MIDASMultimodalOPS(self.full_head_dimensions, config)
        # The formal factory always instantiates all frozen 52 reporter
        # modalities.  active_reporters only controls campaign routing; panel12
        # leaves the remaining modalities absent rather than deleting them.
        self.training_adapter = MIDASMultimodalOPSTrainingAdapter(
            self.full_head_dimensions,
            model=core,
        )

    @property
    def model(self) -> MIDASMultimodalOPS:
        return self.training_adapter.model

    def endpoint_slice(self, reporter: str) -> slice:
        """Legacy registry view only; the model itself has no union decoder."""

        return super().endpoint_slice(reporter)

    def predict_reporter(
        self,
        phase_x: torch.Tensor,
        reporter: str,
        *,
        technical_batch_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        name = self._require_active(reporter)
        if technical_batch_ids is not None:
            raise ValueError(
                "technical_batch_ids are disabled for OPS-native MIDAS"
            )
        return self.training_adapter.predict_reporter(phase_x, name)

    def predict_grouped(
        self, phase_by_reporter: Mapping[str, torch.Tensor]
    ) -> OrderedDict[str, torch.Tensor]:
        if not isinstance(phase_by_reporter, Mapping) or not phase_by_reporter:
            raise ValueError("phase_by_reporter must be a non-empty mapping")
        names = [self._require_active(name) for name in phase_by_reporter]
        ordered = OrderedDict((name, phase_by_reporter[name]) for name in names)
        return self.training_adapter.predict_grouped(ordered)

    def forward_grouped(
        self,
        batches: Sequence[MIDASReporterTrainingBatch],
        *,
        sample: bool | None = None,
    ) -> MIDASGroupedTrainingResult:
        """Run one shared PoE objective over independent target modalities."""

        for batch in batches:
            self._require_active(str(batch.reporter))
        return self.training_adapter.forward_grouped(batches, sample=sample)

    def forward(
        self,
        batches: Sequence[MIDASReporterTrainingBatch],
        *,
        sample: bool | None = None,
    ) -> MIDASGroupedTrainingResult:
        return self.forward_grouped(batches, sample=sample)

    def factory_manifest(self) -> dict[str, Any]:
        payload = self._base_manifest()
        payload.update(
            {
                "family": self.family,
                "method_difference": (
                    "phase plus 52 independent sparse reporter modalities with "
                    "shared PoE latent trunks and reporter-local heads"
                ),
                "training_loss_scope": "single_native_multimodal_midas_loss",
                "target_representation": (
                    "independent_sparse_reporter_modalities"
                ),
                "independent_reporter_losses_fabricated": False,
                "training_adapter_manifest": (
                    self.training_adapter.config_manifest()
                ),
                "modality_binding": {
                    "implementation": (
                        "ops_biological_baseline_midas_multimodal_training."
                        "MIDASMultimodalOPSTrainingAdapter"
                    ),
                    "n_source_modalities": 1,
                    "n_target_modalities": 52,
                    "factory_registry_sha256": OPS_REPORTER_REGISTRY_SHA256,
                },
                "model_manifest": self.model.config_manifest(),
                "parameter_count": sum(p.numel() for p in self.parameters()),
            }
        )
        json.dumps(payload, sort_keys=True, allow_nan=False)
        return payload


def _scbutterfly_options(
    options: Mapping[str, Any],
) -> tuple[str, dict[str, Any]]:
    values = dict(options)
    architecture_factory = str(
        values.pop("architecture_factory", "ops_explicit")
    )
    if architecture_factory not in {"source_anchor", "ops_explicit"}:
        raise ValueError(
            "scButterfly architecture_factory must be source_anchor or "
            "ops_explicit"
        )
    for key in (
        "phase_encoder_widths",
        "phenotype_encoder_widths",
        "phase_decoder_widths",
        "phenotype_decoder_widths",
        "discriminator_hidden_widths",
    ):
        if key in values:
            values[key] = tuple(values[key])
    forbidden = {"phenotype_dim", "reporter_name", "phase_dim"} & set(values)
    if forbidden:
        raise ValueError(
            "scButterfly collection owns specialist identity/dimensions: "
            f"{sorted(forbidden)}"
        )
    return architecture_factory, values


class ScButterflySpecialistCollection(_RegistryBoundCollection):
    """Independent specialist modules; training remains controller-owned."""

    family = "scbutterfly_ops_b_specialist_collection"

    def __init__(self, factory_config: HarnessModelConfig) -> None:
        super().__init__(factory_config)
        architecture_factory, options = _scbutterfly_options(
            factory_config.options or {}
        )
        specialists: OrderedDict[str, ScButterflyOPSSpecialist] = OrderedDict()
        for reporter in self.reporter_names:
            identity = {
                "phenotype_dim": self._dimensions[reporter],
                "reporter_name": reporter,
            }
            if architecture_factory == "source_anchor":
                config = ScButterflyOPSConfig.source_anchor(
                    **identity,
                    **options,
                )
            else:
                config = ScButterflyOPSConfig(
                    **identity,
                    phase_dim=PHASE_DIMENSION,
                    **options,
                )
            specialists[reporter] = ScButterflyOPSSpecialist(config)
        self.specialists = nn.ModuleDict(specialists)
        self.architecture_factory = architecture_factory

    def specialist(self, reporter: str) -> ScButterflyOPSSpecialist:
        return self.specialists[self._require_active(reporter)]

    def predict_reporter(
        self, phase_x: torch.Tensor, reporter: str
    ) -> torch.Tensor:
        return self.specialist(reporter).predict(phase_x)

    def predict_grouped(
        self, phase_by_reporter: Mapping[str, torch.Tensor]
    ) -> OrderedDict[str, torch.Tensor]:
        if not isinstance(phase_by_reporter, Mapping) or not phase_by_reporter:
            raise ValueError("phase_by_reporter must be a non-empty mapping")
        return OrderedDict(
            (reporter, self.predict_reporter(phase, reporter))
            for reporter, phase in phase_by_reporter.items()
        )

    def forward_grouped(
        self, phase_by_reporter: Mapping[str, torch.Tensor]
    ) -> OrderedDict[str, torch.Tensor]:
        """Grouped inference only; never substitutes for staged training."""

        return self.predict_grouped(phase_by_reporter)

    def forward(
        self, phase_by_reporter: Mapping[str, torch.Tensor]
    ) -> OrderedDict[str, torch.Tensor]:
        return self.forward_grouped(phase_by_reporter)

    def factory_manifest(self) -> dict[str, Any]:
        payload = self._base_manifest()
        payload.update(
            {
                "family": self.family,
                "method_difference": (
                    "independent dual-modality specialist per active reporter"
                ),
                "container": "torch.nn.ModuleDict",
                "training_controller_owned": True,
                "architecture_factory": self.architecture_factory,
                "collection_forward_grouped_scope": "inference_only",
                "specialists": OrderedDict(
                    (name, self.specialists[name].architecture_manifest())
                    for name in self.reporter_names
                ),
                "parameter_count": sum(p.numel() for p in self.parameters()),
            }
        )
        json.dumps(payload, sort_keys=True, allow_nan=False)
        return payload


HarnessModelCollection = (
    CaptainHarnessCollection
    | MIDASHarnessCollection
    | ScButterflySpecialistCollection
)


def create_harness_model(
    config: HarnessModelConfig | Mapping[str, Any] | str,
    *,
    active_reporters: Sequence[str] | None = None,
    **options: Any,
) -> HarnessModelCollection:
    """Create one method-native collection without a runner or device policy."""

    if isinstance(config, str):
        factory_config = HarnessModelConfig(
            method=config,
            active_reporters=(
                tuple(active_reporters) if active_reporters is not None else None
            ),
            options=options,
        )
    else:
        if active_reporters is not None or options:
            raise ValueError(
                "Do not pass active_reporters/options alongside a config object"
            )
        factory_config = (
            config
            if isinstance(config, HarnessModelConfig)
            else HarnessModelConfig.from_mapping(config)
        )
    if factory_config.method == "ops_captain":
        return CaptainHarnessCollection(factory_config)
    if factory_config.method == "midas_ops":
        return MIDASHarnessCollection(factory_config)
    if factory_config.method == "scbutterfly_ops_b":
        return ScButterflySpecialistCollection(factory_config)
    raise AssertionError(f"Unreachable method {factory_config.method!r}")


def load_harness_model(
    factory_manifest: Mapping[str, Any],
    state_dict: Mapping[str, torch.Tensor] | None = None,
    *,
    strict: bool = True,
) -> HarnessModelCollection:
    """Recreate a collection and optionally load an in-memory state dict.

    File deserialisation is intentionally outside this API: the common harness
    owns checkpoint trust, ``weights_only`` and ``map_location`` policy.
    """

    if not isinstance(factory_manifest, Mapping):
        raise TypeError("factory_manifest must be a mapping")
    if factory_manifest.get("schema_version") != COLLECTION_MANIFEST_SCHEMA:
        raise ValueError("Unsupported collection manifest schema")
    registry = factory_manifest.get("registry")
    if not isinstance(registry, Mapping):
        raise ValueError("Collection manifest lacks frozen registry metadata")
    if registry.get("sha256") != OPS_REPORTER_REGISTRY_SHA256:
        raise ValueError("Collection manifest reporter registry fingerprint drifted")
    expected_registry = [
        {"reporter": name, "dimension": dimension}
        for name, dimension in OPS_REPORTER_DIMENSIONS
    ]
    if registry.get("ordered_dimensions") != expected_registry:
        raise ValueError("Collection manifest reporter registry contents drifted")
    factory_config = factory_manifest.get("factory_config")
    if not isinstance(factory_config, Mapping):
        raise ValueError("Collection manifest lacks factory_config")
    parsed_config = HarnessModelConfig.from_mapping(factory_config)
    if registry.get("active_reporters") != list(parsed_config.active_reporters or ()):
        raise ValueError(
            "Collection manifest active reporter registry differs from its "
            "factory configuration"
        )
    expected_family = {
        "ops_captain": CaptainHarnessCollection.family,
        "midas_ops": MIDASHarnessCollection.family,
        "scbutterfly_ops_b": ScButterflySpecialistCollection.family,
    }[parsed_config.method]
    if factory_manifest.get("family") != expected_family:
        raise ValueError("Collection manifest family differs from factory method")
    model = create_harness_model(parsed_config)
    if state_dict is not None:
        if not isinstance(state_dict, Mapping):
            raise TypeError("state_dict must be an already deserialised mapping")
        model.load_state_dict(state_dict, strict=strict)
    return model


__all__ = [
    "COLLECTION_MANIFEST_SCHEMA",
    "CaptainHarnessCollection",
    "FACTORY_SCHEMA_VERSION",
    "HarnessModelCollection",
    "HarnessModelConfig",
    "MIDASGroupedTrainingResult",
    "MIDASHarnessCollection",
    "MIDASReporterTrainingBatch",
    "OPS_REPORTER_DIMENSIONS",
    "OPS_REPORTER_REGISTRY_SHA256",
    "PHASE_DIMENSION",
    "PHENOTYPE_UNION_DIMENSION",
    "ScButterflySpecialistCollection",
    "create_harness_model",
    "full_reporter_registry",
    "load_harness_model",
    "validate_frozen_registry",
]
