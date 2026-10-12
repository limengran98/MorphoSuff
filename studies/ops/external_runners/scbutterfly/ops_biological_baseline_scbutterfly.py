#!/usr/bin/env python3
"""OPS-native scButterfly-B specialist adapter with audited train variants.

This module adapts the paired two-modality core of scButterfly-B to one OPS
reporter at a time::

    phase morphology (P, 172D) <-> reporter phenotype (Y, reporter-specific D)

It deliberately does *not* implement a shared 52-reporter encoder, a reporter
query, pseudo-pairing, or a dense union of all reporter endpoints.  Each model
instance is one independent P<->Y specialist and retains the defining
scButterfly mechanisms:

* two modality-specific encoders and two modality-specific decoders;
* source-specific mean/log-variance parameterisations and reparameterisation;
* the upstream BatchNorm + activation after every translator mean,
  log-variance, and projection linear;
* P->P, P->Y, Y->P, and Y->Y paths;
* one discriminator for each target-modality latent space;
* modality-wise VAE pretraining APIs followed by paired joint-training APIs;
* explicit generator/discriminator gradient ownership for alternating updates.

Reporter endpoint missingness is represented only by an explicit boolean mask.
Observed numerical zero is therefore a valid label.  Missing target storage is
cleaned before arithmetic and never becomes a zero-valued supervision target.

The implementation is intentionally architecture/training-loop only.  It does
not load OPS data, construct splits, write artifacts, or start an experiment.

Two adversarial protocols are intentionally kept distinct.  ``source_anchor``
reproduces the public RNA--ADT training mechanics (random soft labels, a
per-cell real/fake mixture, and the 1.35 generator gate).  The earlier smooth
non-saturating objective remains available only under the explicit name
``ops_stabilized``; it must not be reported as source-faithful.

Primary upstream sources frozen in this repository:

* ``external/original_methods/scbutterfly/scButterfly/model_component.py``
* ``external/original_methods/scbutterfly/scButterfly/train_model_cite.py``
* ``external/original_methods/scbutterfly_source``
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator, Literal, Sequence

import torch
from torch import nn
from torch.nn import functional as F


SCHEMA_VERSION = "ops-scbutterfly-specialist-v2"
PAPER_DOI = "10.1038/s41467-024-47418-x"
UPSTREAM_METHOD = "scButterfly-B"
UPSTREAM_LICENSE = "MIT"
DEFAULT_PHASE_DIM = 172
SOURCE_ANCHOR_PROTOCOL = "source_anchor"
OPS_STABILIZED_PROTOCOL = "ops_stabilized"
ADVERSARIAL_PROTOCOLS = frozenset(
    (SOURCE_ANCHOR_PROTOCOL, OPS_STABILIZED_PROTOCOL)
)
SOURCE_GENERATOR_GATE_THRESHOLD = 1.35
SOURCE_PHASE_RECONSTRUCTION_WEIGHT = 1.0
SOURCE_PHENOTYPE_RECONSTRUCTION_WEIGHT = 2.0
SOURCE_DISCRIMINATOR_WEIGHT = 1.0
SOURCE_PRETRAIN_KL_WEIGHT_PHASE172 = 20.0 / DEFAULT_PHASE_DIM
SOURCE_JOINT_KL_WEIGHT_PHASE172 = 40.0 / DEFAULT_PHASE_DIM


def validate_adversarial_protocol(value: str) -> str:
    """Return a canonical, explicitly named adversarial protocol."""

    value = str(value)
    if value not in ADVERSARIAL_PROTOCOLS:
        raise ValueError(
            "adversarial_protocol must be one of "
            f"{sorted(ADVERSARIAL_PROTOCOLS)}, found {value!r}"
        )
    return value


def _positive_int(value: int, name: str) -> int:
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive, found {value}")
    return value


def _probability(value: float, name: str) -> float:
    value = float(value)
    if not 0.0 <= value < 1.0:
        raise ValueError(f"{name} must be in [0, 1), found {value}")
    return value


def _positive_float(value: float, name: str, *, allow_zero: bool = False) -> float:
    value = float(value)
    valid = value >= 0.0 if allow_zero else value > 0.0
    if not valid:
        relation = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be {relation}, found {value}")
    return value


def _widths(values: Sequence[int], name: str) -> tuple[int, ...]:
    result = tuple(_positive_int(value, name) for value in values)
    if not result:
        raise ValueError(f"{name} must contain at least one width")
    return result


def _optional_widths(values: Sequence[int], name: str) -> tuple[int, ...]:
    return tuple(_positive_int(value, name) for value in values)


def parameter_count(module: nn.Module, *, trainable_only: bool = False) -> int:
    """Return the scalar parameter count without imposing an admission cap."""

    parameters = module.parameters()
    if trainable_only:
        parameters = (value for value in parameters if value.requires_grad)
    return int(sum(value.numel() for value in parameters))


def _unique_parameters(modules: Sequence[nn.Module]) -> tuple[nn.Parameter, ...]:
    """Collect parameters once even when a module graph contains shared nodes."""

    result: list[nn.Parameter] = []
    seen: set[int] = set()
    for module in modules:
        for parameter in module.parameters():
            identity = id(parameter)
            if identity not in seen:
                result.append(parameter)
                seen.add(identity)
    return tuple(result)


def _validate_float_matrix(
    value: torch.Tensor,
    *,
    name: str,
    width: int,
    require_finite: bool,
) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.ndim != 2 or value.shape[1] != width:
        raise ValueError(
            f"Expected {name} [batch, {width}], found {tuple(value.shape)}"
        )
    if not value.is_floating_point():
        raise TypeError(f"{name} must be floating point")
    if require_finite and not torch.isfinite(value).all():
        raise ValueError(f"{name} must contain only finite values")


def prepare_observed_phenotype(
    phenotype_y: torch.Tensor,
    endpoint_mask: torch.Tensor,
    *,
    expected_dim: int | None = None,
    require_observed_per_row: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Validate an explicit endpoint mask and clean missing target storage.

    False-mask entries may contain NaN, infinity, or arbitrary sentinels.  They
    are replaced before any subtraction or encoder input is formed.  True-mask
    entries, including numerical zero, are preserved exactly.
    """

    if not isinstance(phenotype_y, torch.Tensor):
        raise TypeError("phenotype_y must be a torch.Tensor")
    if phenotype_y.ndim != 2:
        raise ValueError(
            "Expected phenotype_y [batch, endpoints], found "
            f"{tuple(phenotype_y.shape)}"
        )
    if expected_dim is not None and phenotype_y.shape[1] != int(expected_dim):
        raise ValueError(
            f"Expected phenotype_y [batch, {int(expected_dim)}], "
            f"found {tuple(phenotype_y.shape)}"
        )
    if not phenotype_y.is_floating_point():
        raise TypeError("phenotype_y must be floating point")
    if not isinstance(endpoint_mask, torch.Tensor):
        raise TypeError("endpoint_mask must be a torch.Tensor")
    if endpoint_mask.shape != phenotype_y.shape:
        raise ValueError(
            f"endpoint_mask shape {tuple(endpoint_mask.shape)} does not match "
            f"phenotype_y {tuple(phenotype_y.shape)}"
        )
    if endpoint_mask.dtype is not torch.bool:
        raise TypeError("endpoint_mask must have boolean dtype")
    mask = endpoint_mask.to(device=phenotype_y.device)
    observed_per_row = mask.sum(dim=1)
    if require_observed_per_row and bool((observed_per_row == 0).any().item()):
        raise ValueError("Every paired cell must contain at least one observed endpoint")
    if int(mask.sum().item()) == 0:
        raise ValueError("endpoint_mask contains no observed endpoints")
    if not torch.isfinite(phenotype_y[mask]).all():
        raise ValueError("Observed phenotype endpoints must be finite")
    clean = torch.where(mask, phenotype_y, torch.zeros_like(phenotype_y))
    return clean, mask


def observed_endpoint_mse(
    prediction: torch.Tensor,
    phenotype_y: torch.Tensor,
    endpoint_mask: torch.Tensor,
) -> torch.Tensor:
    """MSE over observed endpoints only; observed zeros remain supervised."""

    if prediction.shape != phenotype_y.shape:
        raise ValueError(
            "prediction and phenotype_y must have equal shape, found "
            f"{tuple(prediction.shape)} and {tuple(phenotype_y.shape)}"
        )
    clean, mask = prepare_observed_phenotype(
        phenotype_y,
        endpoint_mask,
        expected_dim=prediction.shape[1],
    )
    squared_error = (prediction - clean).square()
    squared_error = torch.where(mask, squared_error, torch.zeros_like(squared_error))
    return squared_error.sum() / mask.sum().to(dtype=squared_error.dtype)


def gaussian_kl_mean(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Mean KL divergence from a diagonal Gaussian to N(0, I)."""

    if mu.shape != logvar.shape:
        raise ValueError("mu and logvar must have identical shape")
    if mu.ndim != 2:
        raise ValueError("mu and logvar must be [batch, latent]")
    return -0.5 * torch.mean(1.0 + logvar - mu.square() - logvar.exp())


@dataclass(frozen=True)
class ScButterflyOPSConfig:
    """Architecture for one independent reporter specialist.

    Width, depth, and latent dimension are intentionally unconstrained by the
    4.31M historical ResMLP parameter count.  Only positivity and numerical
    validity are enforced.
    """

    phenotype_dim: int
    reporter_name: str = "unspecified_reporter"
    phase_dim: int = DEFAULT_PHASE_DIM
    phase_encoder_widths: tuple[int, ...] = (512, 512)
    phenotype_encoder_widths: tuple[int, ...] = (256, 256)
    phase_decoder_widths: tuple[int, ...] = (512, 512)
    phenotype_decoder_widths: tuple[int, ...] = (256, 256)
    latent_dim: int = 256
    discriminator_hidden_widths: tuple[int, ...] = ()
    discriminator_output_batch_norm: bool = True
    dropout: float = 0.1
    phase_input_mask_rate: float = 0.25
    phenotype_input_mask_rate: float = 0.0
    negative_slope: float = 0.01
    logvar_min: float = -12.0
    logvar_max: float = 8.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "phenotype_dim", _positive_int(self.phenotype_dim, "phenotype_dim"))
        object.__setattr__(self, "phase_dim", _positive_int(self.phase_dim, "phase_dim"))
        object.__setattr__(self, "latent_dim", _positive_int(self.latent_dim, "latent_dim"))
        object.__setattr__(
            self,
            "discriminator_hidden_widths",
            _optional_widths(
                self.discriminator_hidden_widths, "discriminator_hidden_widths"
            ),
        )
        if not isinstance(self.discriminator_output_batch_norm, bool):
            raise TypeError("discriminator_output_batch_norm must be boolean")
        object.__setattr__(
            self,
            "phase_encoder_widths",
            _widths(self.phase_encoder_widths, "phase_encoder_widths"),
        )
        object.__setattr__(
            self,
            "phenotype_encoder_widths",
            _widths(self.phenotype_encoder_widths, "phenotype_encoder_widths"),
        )
        object.__setattr__(
            self,
            "phase_decoder_widths",
            _widths(self.phase_decoder_widths, "phase_decoder_widths"),
        )
        object.__setattr__(
            self,
            "phenotype_decoder_widths",
            _widths(self.phenotype_decoder_widths, "phenotype_decoder_widths"),
        )
        object.__setattr__(self, "dropout", _probability(self.dropout, "dropout"))
        object.__setattr__(
            self,
            "phase_input_mask_rate",
            _probability(self.phase_input_mask_rate, "phase_input_mask_rate"),
        )
        object.__setattr__(
            self,
            "phenotype_input_mask_rate",
            _probability(
                self.phenotype_input_mask_rate, "phenotype_input_mask_rate"
            ),
        )
        object.__setattr__(
            self,
            "negative_slope",
            _positive_float(self.negative_slope, "negative_slope", allow_zero=True),
        )
        if not float(self.logvar_min) < float(self.logvar_max):
            raise ValueError("logvar_min must be smaller than logvar_max")
        if not str(self.reporter_name):
            raise ValueError("reporter_name must be non-empty")

    @classmethod
    def source_anchor(
        cls,
        *,
        phenotype_dim: int,
        reporter_name: str,
        **overrides: object,
    ) -> "ScButterflyOPSConfig":
        """Construct the public RNA--ADT architecture anchor for OPS.

        ``phase`` occupies the upstream RNA/source role and reporter phenotype
        occupies the ADT/target role.  Output layers remain continuous because
        OPS endpoints are standardised continuous values; the source's ADT
        sigmoid is therefore an explicitly necessary data-domain adaptation.
        """

        anchor: dict[str, object] = {
            "phenotype_dim": phenotype_dim,
            "reporter_name": reporter_name,
            "phase_encoder_widths": (256, 128),
            "phenotype_encoder_widths": (128, 128),
            "phase_decoder_widths": (256,),
            "phenotype_decoder_widths": (128,),
            "latent_dim": 128,
            "discriminator_hidden_widths": (),
            "discriminator_output_batch_norm": True,
            "dropout": 0.1,
            "phase_input_mask_rate": 0.5,
            "phenotype_input_mask_rate": 0.0,
        }
        anchor.update(overrides)
        return cls(**anchor)


@dataclass(frozen=True)
class JointLossWeights:
    """Positive weights for the complete four-path joint objective."""

    phase_reconstruction: float = 1.0
    phenotype_reconstruction: float = 1.0
    phase_kl: float = 1.0 / 150.0
    phenotype_kl: float = 1.0 / 150.0
    adversarial: float = 1.0

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            _positive_float(value, name)


class DenseBlock(nn.Module):
    """Fully connected block following the upstream Linear/BN/LeakyReLU form."""

    def __init__(
        self,
        input_dim: int,
        widths: Sequence[int],
        *,
        dropout: float,
        input_mask_rate: float,
        negative_slope: float,
    ) -> None:
        super().__init__()
        dimensions = (int(input_dim),) + tuple(int(value) for value in widths)
        self.input_mask_rate = float(input_mask_rate)
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropout = nn.Dropout(float(dropout))
        self.activation = nn.LeakyReLU(float(negative_slope))
        for input_width, output_width in zip(dimensions[:-1], dimensions[1:]):
            layer = nn.Linear(input_width, output_width)
            nn.init.xavier_uniform_(layer.weight)
            self.layers.append(layer)
            self.norms.append(nn.BatchNorm1d(output_width))

    @property
    def output_dim(self) -> int:
        return int(self.layers[-1].out_features)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = F.dropout(
            value,
            p=self.input_mask_rate,
            training=self.training and self.input_mask_rate > 0.0,
        )
        last = len(self.layers) - 1
        for index, (layer, norm) in enumerate(zip(self.layers, self.norms)):
            value = self.activation(norm(layer(value)))
            if index != last:
                value = self.dropout(value)
        return value


class ContinuousDecoder(nn.Module):
    """Dense decoder with a linear output suitable for standardised values."""

    def __init__(
        self,
        input_dim: int,
        hidden_widths: Sequence[int],
        output_dim: int,
        *,
        dropout: float,
        negative_slope: float,
    ) -> None:
        super().__init__()
        widths = tuple(int(value) for value in hidden_widths)
        dimensions = (int(input_dim),) + widths
        self.hidden = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropout = nn.Dropout(float(dropout))
        self.activation = nn.LeakyReLU(float(negative_slope))
        for input_width, output_width in zip(dimensions[:-1], dimensions[1:]):
            layer = nn.Linear(input_width, output_width)
            nn.init.xavier_uniform_(layer.weight)
            self.hidden.append(layer)
            self.norms.append(nn.BatchNorm1d(output_width))
        self.output = nn.Linear(widths[-1], int(output_dim))
        nn.init.xavier_uniform_(self.output.weight)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        for layer, norm in zip(self.hidden, self.norms):
            value = self.dropout(self.activation(norm(layer(value))))
        return self.output(value)


@dataclass(frozen=True)
class VariationalState:
    projected_phase: torch.Tensor
    projected_phenotype: torch.Tensor
    mu: torch.Tensor
    logvar: torch.Tensor


class DualVariationalTranslator(nn.Module):
    """Source Gaussian encoders plus the official BN/activation projections."""

    def __init__(
        self,
        phase_embedding_dim: int,
        phenotype_embedding_dim: int,
        latent_dim: int,
        *,
        negative_slope: float,
        logvar_min: float,
        logvar_max: float,
    ) -> None:
        super().__init__()
        self.phase_mu = nn.Linear(phase_embedding_dim, latent_dim)
        self.phase_mu_norm = nn.BatchNorm1d(latent_dim)
        self.phase_logvar = nn.Linear(phase_embedding_dim, latent_dim)
        self.phase_logvar_norm = nn.BatchNorm1d(latent_dim)
        self.phenotype_mu = nn.Linear(phenotype_embedding_dim, latent_dim)
        self.phenotype_mu_norm = nn.BatchNorm1d(latent_dim)
        self.phenotype_logvar = nn.Linear(phenotype_embedding_dim, latent_dim)
        self.phenotype_logvar_norm = nn.BatchNorm1d(latent_dim)
        self.to_phase = nn.Linear(latent_dim, phase_embedding_dim)
        self.to_phase_norm = nn.BatchNorm1d(phase_embedding_dim)
        self.to_phenotype = nn.Linear(latent_dim, phenotype_embedding_dim)
        self.to_phenotype_norm = nn.BatchNorm1d(phenotype_embedding_dim)
        self.activation = nn.LeakyReLU(float(negative_slope))
        self.logvar_min = float(logvar_min)
        self.logvar_max = float(logvar_max)
        for module in (
            self.phase_mu,
            self.phase_logvar,
            self.phenotype_mu,
            self.phenotype_logvar,
            self.to_phase,
            self.to_phenotype,
        ):
            nn.init.xavier_uniform_(module.weight)

    def _state(
        self,
        embedding: torch.Tensor,
        *,
        source: Literal["phase", "phenotype"],
        sample: bool,
    ) -> VariationalState:
        if source == "phase":
            mu = self.activation(self.phase_mu_norm(self.phase_mu(embedding)))
            logvar = self.activation(
                self.phase_logvar_norm(self.phase_logvar(embedding))
            )
        elif source == "phenotype":
            mu = self.activation(
                self.phenotype_mu_norm(self.phenotype_mu(embedding))
            )
            logvar = self.activation(
                self.phenotype_logvar_norm(self.phenotype_logvar(embedding))
            )
        else:  # pragma: no cover - protected by the Literal-facing methods.
            raise ValueError(f"Unknown source modality {source!r}")
        logvar = logvar.clamp(self.logvar_min, self.logvar_max)
        latent = self.reparameterize(mu, logvar) if sample else mu
        return VariationalState(
            projected_phase=self.activation(
                self.to_phase_norm(self.to_phase(latent))
            ),
            projected_phenotype=self.activation(
                self.to_phenotype_norm(self.to_phenotype(latent))
            ),
            mu=mu,
            logvar=logvar,
        )

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def from_phase(self, embedding: torch.Tensor, *, sample: bool) -> VariationalState:
        return self._state(embedding, source="phase", sample=sample)

    def from_phenotype(
        self, embedding: torch.Tensor, *, sample: bool
    ) -> VariationalState:
        return self._state(embedding, source="phenotype", sample=sample)


@dataclass(frozen=True)
class SingleModalityState:
    projected: torch.Tensor
    mu: torch.Tensor
    logvar: torch.Tensor


class SingleModalityVariationalBridge(nn.Module):
    """The modality-specific variational bridge used during pretraining."""

    def __init__(
        self,
        embedding_dim: int,
        latent_dim: int,
        *,
        negative_slope: float,
        logvar_min: float,
        logvar_max: float,
    ) -> None:
        super().__init__()
        self.mu = nn.Linear(embedding_dim, latent_dim)
        self.mu_norm = nn.BatchNorm1d(latent_dim)
        self.logvar = nn.Linear(embedding_dim, latent_dim)
        self.logvar_norm = nn.BatchNorm1d(latent_dim)
        self.project = nn.Linear(latent_dim, embedding_dim)
        self.project_norm = nn.BatchNorm1d(embedding_dim)
        self.activation = nn.LeakyReLU(float(negative_slope))
        self.logvar_min = float(logvar_min)
        self.logvar_max = float(logvar_max)
        for module in (self.mu, self.logvar, self.project):
            nn.init.xavier_uniform_(module.weight)

    def forward(self, embedding: torch.Tensor, *, sample: bool) -> SingleModalityState:
        mu = self.activation(self.mu_norm(self.mu(embedding)))
        logvar = self.activation(
            self.logvar_norm(self.logvar(embedding))
        ).clamp(self.logvar_min, self.logvar_max)
        latent = (
            DualVariationalTranslator.reparameterize(mu, logvar) if sample else mu
        )
        return SingleModalityState(
            projected=self.activation(self.project_norm(self.project(latent))),
            mu=mu,
            logvar=logvar,
        )


class LatentDiscriminator(nn.Module):
    """Modality-specific discriminator returning pre-sigmoid logits.

    The source RNA--ADT anchor uses no hidden layer and applies BatchNorm before
    its terminal sigmoid.  Returning logits lets the source protocol apply the
    mathematically identical sigmoid + BCE path while the stabilized protocol
    uses BCE-with-logits.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_widths: Sequence[int],
        negative_slope: float,
        *,
        output_batch_norm: bool,
    ) -> None:
        super().__init__()
        modules: list[nn.Module] = []
        current = int(input_dim)
        for width in hidden_widths:
            modules.extend(
                (
                    nn.Linear(current, int(width)),
                    nn.BatchNorm1d(int(width)),
                    nn.LeakyReLU(float(negative_slope)),
                )
            )
            current = int(width)
        modules.append(nn.Linear(current, 1))
        if output_batch_norm:
            modules.append(nn.BatchNorm1d(1))
        self.network = nn.Sequential(*modules)
        for module in self.network:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.network(value).squeeze(-1)


@dataclass(frozen=True)
class ScButterflyLatentPaths:
    """Encoder/translator state that never evaluates either decoder.

    The adversarial discriminator and the source-anchor gate only consume
    latent representations.  Keeping that path explicit prevents their probe
    forwards from mutating decoder BatchNorm running statistics.
    """

    native_phase_latent: torch.Tensor
    native_phenotype_latent: torch.Tensor
    phase_reconstruction_latent: torch.Tensor
    phenotype_reconstruction_latent: torch.Tensor
    translated_phase_latent: torch.Tensor
    translated_phenotype_latent: torch.Tensor
    phase_mu: torch.Tensor
    phase_logvar: torch.Tensor
    phenotype_mu: torch.Tensor
    phenotype_logvar: torch.Tensor


@dataclass(frozen=True)
class ScButterflyPaths(ScButterflyLatentPaths):
    phase_to_phase: torch.Tensor
    phase_to_phenotype: torch.Tensor
    phenotype_to_phase: torch.Tensor
    phenotype_to_phenotype: torch.Tensor


@dataclass(frozen=True)
class PretrainLoss:
    total: torch.Tensor
    reconstruction: torch.Tensor
    kl: torch.Tensor


@dataclass(frozen=True)
class DiscriminatorLoss:
    total: torch.Tensor
    phase_real: torch.Tensor
    phase_fake: torch.Tensor
    phenotype_real: torch.Tensor
    phenotype_fake: torch.Tensor
    adversarial_protocol: str
    soft_labels: torch.Tensor | None
    source_uniform_draws: torch.Tensor | None


@dataclass(frozen=True)
class JointGeneratorLoss:
    total: torch.Tensor
    phase_to_phase: torch.Tensor
    phase_to_phenotype: torch.Tensor
    phenotype_to_phase: torch.Tensor
    phenotype_to_phenotype: torch.Tensor
    phase_kl: torch.Tensor
    phenotype_kl: torch.Tensor
    adversarial: torch.Tensor
    discriminator_probe: torch.Tensor
    adversarial_gate_active: bool
    adversarial_protocol: str
    paths: ScButterflyPaths


def source_soft_labels(uniform_draws: torch.Tensor) -> torch.Tensor:
    """Map U(0, 1) draws exactly as the public scButterfly discriminator.

    Upstream uses the draw itself outside ``(0.2, 0.8]``, 0.2 in
    ``(0.2, 0.5]``, and 0.8 in ``(0.5, 0.8]``.  Values above 0.5 select a
    native/real row; the remainder select its translated/fake counterpart.
    """

    if not isinstance(uniform_draws, torch.Tensor):
        raise TypeError("uniform_draws must be a torch.Tensor")
    if uniform_draws.ndim != 1 or not uniform_draws.is_floating_point():
        raise ValueError("uniform_draws must be a floating [batch] tensor")
    if not torch.isfinite(uniform_draws).all():
        raise ValueError("uniform_draws must be finite")
    if bool(((uniform_draws < 0.0) | (uniform_draws > 1.0)).any().item()):
        raise ValueError("uniform_draws must lie in [0, 1]")
    return torch.where(
        uniform_draws > 0.8,
        uniform_draws,
        torch.where(
            uniform_draws > 0.5,
            torch.full_like(uniform_draws, 0.8),
            torch.where(
                uniform_draws > 0.2,
                torch.full_like(uniform_draws, 0.2),
                uniform_draws,
            ),
        ),
    )


def source_mixed_latents(
    native: torch.Tensor,
    translated: torch.Tensor,
    soft_labels: torch.Tensor,
) -> torch.Tensor:
    """Select one real or fake latent per cell using the source soft label."""

    if native.shape != translated.shape or native.ndim != 2:
        raise ValueError("native and translated must share [batch, latent] shape")
    if soft_labels.ndim != 1 or soft_labels.shape[0] != native.shape[0]:
        raise ValueError("soft_labels must have one value per latent row")
    if soft_labels.device != native.device:
        raise ValueError("soft_labels and latents must be on the same device")
    return torch.where(
        (soft_labels > 0.5).unsqueeze(1),
        native,
        translated,
    )


@contextmanager
def _temporarily_frozen(modules: Sequence[nn.Module]) -> Iterator[None]:
    states: list[tuple[nn.Parameter, bool]] = []
    for module in modules:
        for parameter in module.parameters():
            states.append((parameter, parameter.requires_grad))
            parameter.requires_grad_(False)
    try:
        yield
    finally:
        for parameter, requires_grad in states:
            parameter.requires_grad_(requires_grad)


class ScButterflyOPSSpecialist(nn.Module):
    """One paired phase<->phenotype scButterfly-B specialist."""

    family = "scbutterfly_ops_b_specialist"

    def __init__(self, config: ScButterflyOPSConfig) -> None:
        super().__init__()
        self.config = config
        phenotype_encoder_input = config.phenotype_dim * 2
        self.phase_encoder = DenseBlock(
            config.phase_dim,
            config.phase_encoder_widths,
            dropout=config.dropout,
            input_mask_rate=config.phase_input_mask_rate,
            negative_slope=config.negative_slope,
        )
        self.phenotype_encoder = DenseBlock(
            phenotype_encoder_input,
            config.phenotype_encoder_widths,
            dropout=config.dropout,
            input_mask_rate=config.phenotype_input_mask_rate,
            negative_slope=config.negative_slope,
        )
        phase_embedding_dim = self.phase_encoder.output_dim
        phenotype_embedding_dim = self.phenotype_encoder.output_dim
        self.phase_decoder = ContinuousDecoder(
            phase_embedding_dim,
            config.phase_decoder_widths,
            config.phase_dim,
            dropout=config.dropout,
            negative_slope=config.negative_slope,
        )
        self.phenotype_decoder = ContinuousDecoder(
            phenotype_embedding_dim,
            config.phenotype_decoder_widths,
            config.phenotype_dim,
            dropout=config.dropout,
            negative_slope=config.negative_slope,
        )
        self.phase_pretrain_bridge = SingleModalityVariationalBridge(
            phase_embedding_dim,
            config.latent_dim,
            negative_slope=config.negative_slope,
            logvar_min=config.logvar_min,
            logvar_max=config.logvar_max,
        )
        self.phenotype_pretrain_bridge = SingleModalityVariationalBridge(
            phenotype_embedding_dim,
            config.latent_dim,
            negative_slope=config.negative_slope,
            logvar_min=config.logvar_min,
            logvar_max=config.logvar_max,
        )
        self.translator = DualVariationalTranslator(
            phase_embedding_dim,
            phenotype_embedding_dim,
            config.latent_dim,
            negative_slope=config.negative_slope,
            logvar_min=config.logvar_min,
            logvar_max=config.logvar_max,
        )
        self.phase_discriminator = LatentDiscriminator(
            phase_embedding_dim,
            config.discriminator_hidden_widths,
            config.negative_slope,
            output_batch_norm=config.discriminator_output_batch_norm,
        )
        self.phenotype_discriminator = LatentDiscriminator(
            phenotype_embedding_dim,
            config.discriminator_hidden_widths,
            config.negative_slope,
            output_batch_norm=config.discriminator_output_batch_norm,
        )

    @property
    def phase_dim(self) -> int:
        return self.config.phase_dim

    @property
    def phenotype_dim(self) -> int:
        return self.config.phenotype_dim

    def _validate_pair(
        self,
        phase_x: torch.Tensor,
        phenotype_y: torch.Tensor,
        endpoint_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _validate_float_matrix(
            phase_x,
            name="phase_x",
            width=self.phase_dim,
            require_finite=True,
        )
        clean_y, mask = prepare_observed_phenotype(
            phenotype_y,
            endpoint_mask,
            expected_dim=self.phenotype_dim,
        )
        if phase_x.shape[0] != phenotype_y.shape[0]:
            raise ValueError("phase_x and phenotype_y batch sizes must match")
        if phase_x.device != phenotype_y.device:
            raise ValueError("phase_x and phenotype_y must be on the same device")
        return clean_y, mask

    @staticmethod
    def _phenotype_encoder_input(
        clean_y: torch.Tensor, endpoint_mask: torch.Tensor
    ) -> torch.Tensor:
        return torch.cat(
            [clean_y, endpoint_mask.to(dtype=clean_y.dtype)], dim=1
        )

    def forward_paths(
        self,
        phase_x: torch.Tensor,
        phenotype_y: torch.Tensor,
        endpoint_mask: torch.Tensor,
        *,
        sample: bool | None = None,
    ) -> ScButterflyPaths:
        """Materialise all four scButterfly paths for one exact-paired batch."""

        latent = self.latent_paths(
            phase_x,
            phenotype_y,
            endpoint_mask,
            sample=sample,
        )
        return ScButterflyPaths(
            native_phase_latent=latent.native_phase_latent,
            native_phenotype_latent=latent.native_phenotype_latent,
            phase_reconstruction_latent=latent.phase_reconstruction_latent,
            phenotype_reconstruction_latent=(
                latent.phenotype_reconstruction_latent
            ),
            translated_phase_latent=latent.translated_phase_latent,
            translated_phenotype_latent=latent.translated_phenotype_latent,
            phase_mu=latent.phase_mu,
            phase_logvar=latent.phase_logvar,
            phenotype_mu=latent.phenotype_mu,
            phenotype_logvar=latent.phenotype_logvar,
            phase_to_phase=self.phase_decoder(
                latent.phase_reconstruction_latent
            ),
            phase_to_phenotype=self.phenotype_decoder(
                latent.translated_phenotype_latent
            ),
            phenotype_to_phase=self.phase_decoder(
                latent.translated_phase_latent
            ),
            phenotype_to_phenotype=self.phenotype_decoder(
                latent.phenotype_reconstruction_latent
            ),
        )

    def latent_paths(
        self,
        phase_x: torch.Tensor,
        phenotype_y: torch.Tensor,
        endpoint_mask: torch.Tensor,
        *,
        sample: bool | None = None,
    ) -> ScButterflyLatentPaths:
        """Materialise only encoder/translator states for adversarial work."""

        clean_y, mask = self._validate_pair(phase_x, phenotype_y, endpoint_mask)
        if sample is None:
            sample = bool(self.training)
        native_phase = self.phase_encoder(phase_x)
        native_phenotype = self.phenotype_encoder(
            self._phenotype_encoder_input(clean_y, mask)
        )
        from_phase = self.translator.from_phase(native_phase, sample=bool(sample))
        from_phenotype = self.translator.from_phenotype(
            native_phenotype, sample=bool(sample)
        )
        return ScButterflyLatentPaths(
            native_phase_latent=native_phase,
            native_phenotype_latent=native_phenotype,
            phase_reconstruction_latent=from_phase.projected_phase,
            phenotype_reconstruction_latent=(
                from_phenotype.projected_phenotype
            ),
            translated_phase_latent=from_phenotype.projected_phase,
            translated_phenotype_latent=from_phase.projected_phenotype,
            phase_mu=from_phase.mu,
            phase_logvar=from_phase.logvar,
            phenotype_mu=from_phenotype.mu,
            phenotype_logvar=from_phenotype.logvar,
        )

    def predict(self, phase_x: torch.Tensor) -> torch.Tensor:
        """Deterministic P->Y inference using the phase latent mean."""

        _validate_float_matrix(
            phase_x,
            name="phase_x",
            width=self.phase_dim,
            require_finite=True,
        )
        native_phase = self.phase_encoder(phase_x)
        from_phase = self.translator.from_phase(native_phase, sample=False)
        return self.phenotype_decoder(from_phase.projected_phenotype)

    def forward(self, phase_x: torch.Tensor) -> torch.Tensor:
        return self.predict(phase_x)

    def phase_pretrain_loss(
        self,
        phase_x: torch.Tensor,
        *,
        kl_weight: float = 1.0 / 150.0,
        sample: bool | None = None,
    ) -> PretrainLoss:
        _validate_float_matrix(
            phase_x,
            name="phase_x",
            width=self.phase_dim,
            require_finite=True,
        )
        if sample is None:
            sample = bool(self.training)
        embedding = self.phase_encoder(phase_x)
        state = self.phase_pretrain_bridge(embedding, sample=bool(sample))
        prediction = self.phase_decoder(state.projected)
        reconstruction = F.mse_loss(prediction, phase_x)
        kl = gaussian_kl_mean(state.mu, state.logvar)
        total = reconstruction + float(kl_weight) * kl
        return PretrainLoss(total=total, reconstruction=reconstruction, kl=kl)

    def phenotype_pretrain_loss(
        self,
        phenotype_y: torch.Tensor,
        endpoint_mask: torch.Tensor,
        *,
        kl_weight: float = 1.0 / 150.0,
        sample: bool | None = None,
    ) -> PretrainLoss:
        clean_y, mask = prepare_observed_phenotype(
            phenotype_y,
            endpoint_mask,
            expected_dim=self.phenotype_dim,
        )
        if sample is None:
            sample = bool(self.training)
        embedding = self.phenotype_encoder(
            self._phenotype_encoder_input(clean_y, mask)
        )
        state = self.phenotype_pretrain_bridge(embedding, sample=bool(sample))
        prediction = self.phenotype_decoder(state.projected)
        reconstruction = observed_endpoint_mse(prediction, phenotype_y, mask)
        kl = gaussian_kl_mean(state.mu, state.logvar)
        total = reconstruction + float(kl_weight) * kl
        return PretrainLoss(total=total, reconstruction=reconstruction, kl=kl)

    @staticmethod
    def _detach_discriminator_inputs(
        paths: ScButterflyLatentPaths,
        *,
        detach_generator: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        values = (
            paths.native_phase_latent,
            paths.native_phenotype_latent,
            paths.translated_phase_latent,
            paths.translated_phenotype_latent,
        )
        if detach_generator:
            values = tuple(value.detach() for value in values)
        return values  # type: ignore[return-value]

    def _source_anchor_discriminator_loss(
        self,
        paths: ScButterflyLatentPaths,
        *,
        detach_generator: bool,
        uniform_draws: torch.Tensor | None,
    ) -> DiscriminatorLoss:
        """Official random soft-label and mixed-real/fake objective."""

        (
            native_phase,
            native_phenotype,
            translated_phase,
            translated_phenotype,
        ) = self._detach_discriminator_inputs(
            paths, detach_generator=detach_generator
        )
        if uniform_draws is None:
            uniform_draws = torch.rand(
                native_phase.shape[0],
                device=native_phase.device,
                dtype=native_phase.dtype,
            )
        elif (
            uniform_draws.device != native_phase.device
            or uniform_draws.dtype != native_phase.dtype
            or uniform_draws.shape != (native_phase.shape[0],)
        ):
            raise ValueError(
                "source uniform_draws must match the latent batch device and dtype"
            )
        labels = source_soft_labels(uniform_draws)
        real_rows = (labels > 0.5).to(dtype=labels.dtype)
        fake_rows = 1.0 - real_rows
        mixed_phase = source_mixed_latents(
            native_phase, translated_phase, labels
        )
        mixed_phenotype = source_mixed_latents(
            native_phenotype, translated_phenotype, labels
        )
        phase_per_row = F.binary_cross_entropy(
            torch.sigmoid(self.phase_discriminator(mixed_phase)),
            labels,
            reduction="none",
        )
        phenotype_per_row = F.binary_cross_entropy(
            torch.sigmoid(self.phenotype_discriminator(mixed_phenotype)),
            labels,
            reduction="none",
        )
        denominator = labels.new_tensor(float(labels.numel()))
        phase_real = (phase_per_row * real_rows).sum() / denominator
        phase_fake = (phase_per_row * fake_rows).sum() / denominator
        phenotype_real = (phenotype_per_row * real_rows).sum() / denominator
        phenotype_fake = (phenotype_per_row * fake_rows).sum() / denominator
        total = phase_real + phase_fake + phenotype_real + phenotype_fake
        return DiscriminatorLoss(
            total=total,
            phase_real=phase_real,
            phase_fake=phase_fake,
            phenotype_real=phenotype_real,
            phenotype_fake=phenotype_fake,
            adversarial_protocol=SOURCE_ANCHOR_PROTOCOL,
            soft_labels=labels.detach(),
            source_uniform_draws=uniform_draws.detach(),
        )

    def _ops_stabilized_discriminator_loss(
        self,
        paths: ScButterflyLatentPaths,
        *,
        detach_generator: bool,
    ) -> DiscriminatorLoss:
        (
            native_phase,
            native_phenotype,
            translated_phase,
            translated_phenotype,
        ) = self._detach_discriminator_inputs(
            paths, detach_generator=detach_generator
        )
        phase_real_logits = self.phase_discriminator(native_phase)
        phase_fake_logits = self.phase_discriminator(translated_phase)
        phenotype_real_logits = self.phenotype_discriminator(native_phenotype)
        phenotype_fake_logits = self.phenotype_discriminator(translated_phenotype)
        phase_real = F.binary_cross_entropy_with_logits(
            phase_real_logits, torch.ones_like(phase_real_logits)
        )
        phase_fake = F.binary_cross_entropy_with_logits(
            phase_fake_logits, torch.zeros_like(phase_fake_logits)
        )
        phenotype_real = F.binary_cross_entropy_with_logits(
            phenotype_real_logits, torch.ones_like(phenotype_real_logits)
        )
        phenotype_fake = F.binary_cross_entropy_with_logits(
            phenotype_fake_logits, torch.zeros_like(phenotype_fake_logits)
        )
        total = phase_real + phase_fake + phenotype_real + phenotype_fake
        return DiscriminatorLoss(
            total=total,
            phase_real=phase_real,
            phase_fake=phase_fake,
            phenotype_real=phenotype_real,
            phenotype_fake=phenotype_fake,
            adversarial_protocol=OPS_STABILIZED_PROTOCOL,
            soft_labels=None,
            source_uniform_draws=None,
        )

    def discriminator_loss(
        self,
        paths: ScButterflyLatentPaths,
        *,
        adversarial_protocol: str = OPS_STABILIZED_PROTOCOL,
        detach_generator: bool = True,
        source_uniform_draws: torch.Tensor | None = None,
    ) -> DiscriminatorLoss:
        """Discriminator objective with an explicit source/stabilized label."""

        protocol = validate_adversarial_protocol(adversarial_protocol)
        if protocol == SOURCE_ANCHOR_PROTOCOL:
            return self._source_anchor_discriminator_loss(
                paths,
                detach_generator=bool(detach_generator),
                uniform_draws=source_uniform_draws,
            )
        if source_uniform_draws is not None:
            raise ValueError(
                "source_uniform_draws are only valid for source_anchor"
            )
        return self._ops_stabilized_discriminator_loss(
            paths, detach_generator=bool(detach_generator)
        )

    def discriminator_step_loss(
        self,
        phase_x: torch.Tensor,
        phenotype_y: torch.Tensor,
        endpoint_mask: torch.Tensor,
        *,
        adversarial_protocol: str = OPS_STABILIZED_PROTOCOL,
        source_uniform_draws: torch.Tensor | None = None,
    ) -> DiscriminatorLoss:
        paths = self.latent_paths(
            phase_x, phenotype_y, endpoint_mask, sample=True
        )
        return self.discriminator_loss(
            paths,
            adversarial_protocol=adversarial_protocol,
            detach_generator=True,
            source_uniform_draws=source_uniform_draws,
        )

    def _ops_stabilized_generator_adversarial_loss(
        self, paths: ScButterflyPaths
    ) -> torch.Tensor:
        """Smooth non-saturating OPS-stabilized translator objective.

        The discriminators remain differentiable with respect to translated
        latents, but their parameters do not acquire gradients during the
        generator step.  This is the explicit alternating-update boundary that
        the upstream implementation relies on implicitly through optimizers.
        """

        with _temporarily_frozen(
            (self.phase_discriminator, self.phenotype_discriminator)
        ):
            phase_fake_logits = self.phase_discriminator(
                paths.translated_phase_latent
            )
            phenotype_fake_logits = self.phenotype_discriminator(
                paths.translated_phenotype_latent
            )
            phase_fooling = F.binary_cross_entropy_with_logits(
                phase_fake_logits, torch.ones_like(phase_fake_logits)
            )
            phenotype_fooling = F.binary_cross_entropy_with_logits(
                phenotype_fake_logits, torch.ones_like(phenotype_fake_logits)
            )
        return phase_fooling + phenotype_fooling

    def _source_anchor_generator_adversarial_loss(
        self,
        paths: ScButterflyLatentPaths,
        *,
        threshold: float,
        source_uniform_draws: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, bool]:
        """Official ``loss_g -= d_weight * loss_d`` threshold/gate.

        Discriminator parameters are frozen solely to make optimizer ownership
        explicit.  Autograd through the mixed latent inputs is preserved, so
        the generator sees the same signed discriminator signal as upstream.
        """

        threshold = _positive_float(threshold, "source_adversarial_threshold")
        with _temporarily_frozen(
            (self.phase_discriminator, self.phenotype_discriminator)
        ):
            probe = self.discriminator_loss(
                paths,
                adversarial_protocol=SOURCE_ANCHOR_PROTOCOL,
                detach_generator=False,
                source_uniform_draws=source_uniform_draws,
            ).total
        gate_active = bool(probe.detach().item() < threshold)
        signed_objective = -probe if gate_active else probe * 0.0
        return signed_objective, probe, gate_active

    def joint_generator_loss(
        self,
        phase_x: torch.Tensor,
        phenotype_y: torch.Tensor,
        endpoint_mask: torch.Tensor,
        *,
        weights: JointLossWeights | None = None,
        sample: bool | None = None,
        adversarial_protocol: str = OPS_STABILIZED_PROTOCOL,
        source_adversarial_threshold: float = SOURCE_GENERATOR_GATE_THRESHOLD,
        source_uniform_draws: torch.Tensor | None = None,
    ) -> JointGeneratorLoss:
        """Complete four-path joint objective for a generator-only step."""

        weights = weights or JointLossWeights()
        protocol = validate_adversarial_protocol(adversarial_protocol)
        # Upstream evaluates a fresh discriminator probe before a separate
        # translator/reconstruction forward during the generator update.  Keep
        # those stochastic VAE draws separate in the source anchor.
        source_probe_paths: ScButterflyLatentPaths | None = None
        if protocol == SOURCE_ANCHOR_PROTOCOL:
            source_probe_paths = self.latent_paths(
                phase_x,
                phenotype_y,
                endpoint_mask,
                sample=sample,
            )
        paths = self.forward_paths(
            phase_x,
            phenotype_y,
            endpoint_mask,
            sample=sample,
        )
        phase_to_phase = F.mse_loss(paths.phase_to_phase, phase_x)
        phenotype_to_phase = F.mse_loss(paths.phenotype_to_phase, phase_x)
        phase_to_phenotype = observed_endpoint_mse(
            paths.phase_to_phenotype, phenotype_y, endpoint_mask
        )
        phenotype_to_phenotype = observed_endpoint_mse(
            paths.phenotype_to_phenotype, phenotype_y, endpoint_mask
        )
        phase_kl = gaussian_kl_mean(paths.phase_mu, paths.phase_logvar)
        phenotype_kl = gaussian_kl_mean(
            paths.phenotype_mu, paths.phenotype_logvar
        )
        if protocol == SOURCE_ANCHOR_PROTOCOL:
            if source_probe_paths is None:  # pragma: no cover - protocol guard.
                raise AssertionError("source probe paths were not materialised")
            adversarial, discriminator_probe, gate_active = (
                self._source_anchor_generator_adversarial_loss(
                    source_probe_paths,
                    threshold=source_adversarial_threshold,
                    source_uniform_draws=source_uniform_draws,
                )
            )
        else:
            if source_uniform_draws is not None:
                raise ValueError(
                    "source_uniform_draws are only valid for source_anchor"
                )
            adversarial = self._ops_stabilized_generator_adversarial_loss(paths)
            discriminator_probe = adversarial.new_zeros(())
            gate_active = True
        total = (
            weights.phase_reconstruction
            * (phase_to_phase + phenotype_to_phase)
            + weights.phenotype_reconstruction
            * (phase_to_phenotype + phenotype_to_phenotype)
            + weights.phase_kl * phase_kl
            + weights.phenotype_kl * phenotype_kl
            + weights.adversarial * adversarial
        )
        return JointGeneratorLoss(
            total=total,
            phase_to_phase=phase_to_phase,
            phase_to_phenotype=phase_to_phenotype,
            phenotype_to_phase=phenotype_to_phase,
            phenotype_to_phenotype=phenotype_to_phenotype,
            phase_kl=phase_kl,
            phenotype_kl=phenotype_kl,
            adversarial=adversarial,
            discriminator_probe=discriminator_probe,
            adversarial_gate_active=gate_active,
            adversarial_protocol=protocol,
            paths=paths,
        )

    def phase_pretrain_parameters(self) -> tuple[nn.Parameter, ...]:
        return _unique_parameters(
            (self.phase_encoder, self.phase_pretrain_bridge, self.phase_decoder)
        )

    def phenotype_pretrain_parameters(self) -> tuple[nn.Parameter, ...]:
        return _unique_parameters(
            (
                self.phenotype_encoder,
                self.phenotype_pretrain_bridge,
                self.phenotype_decoder,
            )
        )

    def joint_generator_parameters(self) -> tuple[nn.Parameter, ...]:
        return _unique_parameters(
            (
                self.phase_encoder,
                self.phenotype_encoder,
                self.translator,
                self.phase_decoder,
                self.phenotype_decoder,
            )
        )

    def discriminator_parameters(self) -> tuple[nn.Parameter, ...]:
        return _unique_parameters(
            (self.phase_discriminator, self.phenotype_discriminator)
        )

    def inference_parameters(self) -> tuple[nn.Parameter, ...]:
        """Parameters active for the deployed P->Y direction only."""

        return _unique_parameters(
            (self.phase_encoder, self.translator, self.phenotype_decoder)
        )

    def parameter_counts(self) -> dict[str, int]:
        return {
            "total": parameter_count(self),
            "joint_active": int(
                sum(value.numel() for value in self.joint_generator_parameters())
            ),
            "discriminator_active": int(
                sum(value.numel() for value in self.discriminator_parameters())
            ),
            "inference_active": int(
                sum(value.numel() for value in self.inference_parameters())
            ),
        }

    def architecture_manifest(self) -> dict[str, object]:
        return {
            "schema": SCHEMA_VERSION,
            "family": self.family,
            "scope": "single_reporter_specialist",
            "reporter_name": self.config.reporter_name,
            "upstream_method": UPSTREAM_METHOD,
            "paper_doi": PAPER_DOI,
            "upstream_license": UPSTREAM_LICENSE,
            "config": asdict(self.config),
            "paths": ["phase_to_phase", "phase_to_phenotype", "phenotype_to_phase", "phenotype_to_phenotype"],
            "endpoint_missingness": "explicit_observed_mask_only",
            "observed_zero_is_valid": True,
            "test_latent": "mean",
            "translator_linear_postprocessing": "batchnorm_then_leaky_relu",
            "supported_adversarial_protocols": sorted(ADVERSARIAL_PROTOCOLS),
            "source_anchor_mechanics": {
                "random_soft_labels": True,
                "per_cell_mixed_real_fake": True,
                "generator_threshold_gate": SOURCE_GENERATOR_GATE_THRESHOLD,
                "source_phase_noise_rate": 0.5,
                "source_target_noise_rate": 0.0,
            },
            "ops_required_adaptations": [
                "continuous_standardized_phase_and_phenotype_outputs",
                "explicit_observed_endpoint_mask",
                "discriminator_parameters_frozen_during_generator_backward",
            ],
            "ops_stabilized_is_source_faithful": False,
            "parameter_matching_required": False,
            "parameter_count_is_admission_or_selection_input": False,
            "parameter_counts": self.parameter_counts(),
            "upstream_local_paths": [
                str(
                    Path("external/original_methods/scbutterfly/scButterfly/model_component.py")
                ),
                str(
                    Path("external/original_methods/scbutterfly/scButterfly/train_model_cite.py")
                ),
                str(Path("external/original_methods/scbutterfly_source")),
            ],
        }


__all__ = [
    "ADVERSARIAL_PROTOCOLS",
    "DEFAULT_PHASE_DIM",
    "DiscriminatorLoss",
    "JointGeneratorLoss",
    "JointLossWeights",
    "OPS_STABILIZED_PROTOCOL",
    "PAPER_DOI",
    "PretrainLoss",
    "SCHEMA_VERSION",
    "SOURCE_ANCHOR_PROTOCOL",
    "SOURCE_DISCRIMINATOR_WEIGHT",
    "SOURCE_GENERATOR_GATE_THRESHOLD",
    "SOURCE_JOINT_KL_WEIGHT_PHASE172",
    "SOURCE_PHASE_RECONSTRUCTION_WEIGHT",
    "SOURCE_PHENOTYPE_RECONSTRUCTION_WEIGHT",
    "SOURCE_PRETRAIN_KL_WEIGHT_PHASE172",
    "ScButterflyOPSConfig",
    "ScButterflyOPSSpecialist",
    "ScButterflyLatentPaths",
    "ScButterflyPaths",
    "UPSTREAM_METHOD",
    "gaussian_kl_mean",
    "observed_endpoint_mse",
    "parameter_count",
    "prepare_observed_phenotype",
    "source_mixed_latents",
    "source_soft_labels",
    "validate_adversarial_protocol",
]
