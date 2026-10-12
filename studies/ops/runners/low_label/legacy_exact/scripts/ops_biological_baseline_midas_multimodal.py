#!/usr/bin/env python3
r"""OPS-native multimodal adaptation of MIDAS.

The official MIDAS implementation models every measured assay as a distinct
modality.  Each modality has a thin input-specific encoder in front of a
shared posterior network; available same-cell modality experts are combined
with a unit-normal Gaussian product of experts (PoE), and a shared latent
decoder reconstructs all modalities.  See the frozen source at::

    external/original_methods/midas_reproducibility/modules/models.py

commit ``3ef7847c88c90583c05147cafef496d862986dd3``.  In particular, the
source constructs one ``x_encs[m]`` per modality, applies ``poe`` to the
available experts, splits ``z`` into biological ``c`` and technical ``b``,
and splits the shared decoder output back into modality-specific blocks.

OPS-native MIDAS therefore uses **53 modalities**, not a synthetic 1,604-D
target modality:

* ``phase`` is the always-observed 172-D source modality;
* each of the frozen 52 reporter blocks is an independent sparse target
  modality with its own thin input front and Gaussian output head;
* a panel12 run still instantiates all 52 modalities; the other 40 are simply
  absent from that development batch/campaign;
* endpoint missingness is an explicit boolean mask.  A numeric zero under a
  true mask remains an observed value.

OPS endpoints are continuous, so masked diagonal-Gaussian reconstruction
replaces MIDAS' count-specific Poisson/Bernoulli likelihoods.  The modality
fronts and heads are local, while posterior and decoder trunks are shared.

No trustworthy technical-batch variable has been frozen for this OPS task.
Consequently batch encoding/decoding and adversarial batch removal are
deliberately disabled.  The biological/technical latent split and asymmetric
technical information bottleneck are retained as architectural regularisers,
but are **not identifiable disentanglement** and no batch-correction claim is
made.  Reporter identity is a modality key and is never a batch covariate.

This module is runner-agnostic and performs no I/O, data download, split
construction, optimization, checkpoint selection, or experiment launch.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import math
from typing import Any

import torch
from torch import nn


PHASE_DIM = 172
MIDAS_REPRODUCIBILITY_COMMIT = "3ef7847c88c90583c05147cafef496d862986dd3"
MIDAS_SOURCE_MODELS_SHA256 = (
    "74c72abca48c273861b86981a400031eebbf480b99ffba781b278279afc440c9"
)
MIDAS_SOURCE_MODELS_PATH = (
    "external/original_methods/midas_reproducibility/modules/models.py"
)
MODEL_SCHEMA = "ops-midas-53-modality-model-v1"

_FORBIDDEN_BATCH_KINDS = {
    "reporter",
    "reporter_id",
    "reporter-id",
    "reporter_target",
    "target_reporter",
}


def _positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _nonnegative(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


def _widths(values: Sequence[int], name: str) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{name} must be a sequence")
    result = tuple(_positive_int(value, name) for value in values)
    if not result:
        raise ValueError(f"{name} must not be empty")
    return result


def _validate_reporter_dimensions(
    reporter_dimensions: Mapping[str, int],
) -> OrderedDict[str, int]:
    if not isinstance(reporter_dimensions, Mapping) or not reporter_dimensions:
        raise ValueError("reporter_dimensions must be a non-empty mapping")
    result: OrderedDict[str, int] = OrderedDict()
    for raw_name, raw_dimension in reporter_dimensions.items():
        name = str(raw_name).strip()
        if not name:
            raise ValueError("Reporter modality names must be non-empty")
        if name == "phase":
            raise ValueError("'phase' is reserved for the source modality")
        if name in result:
            raise ValueError(f"Duplicate reporter modality {name!r}")
        result[name] = _positive_int(raw_dimension, f"dimension for {name}")
    return result


@dataclass(frozen=True)
class MIDASMultimodalOPSConfig:
    """Architecture and objective for the OPS-native 53-modality adapter.

    ``technical_information_bottleneck_multiplier`` mirrors the asymmetric
    KL pressure on MIDAS' technical latent.  With no technical-batch labels,
    it is only a regularizer.  It must not be interpreted as an identified
    technical factor or as evidence of batch correction.
    """

    phase_dim: int = PHASE_DIM
    biological_dim: int = 128
    technical_dim: int = 32
    modality_width: int = 768
    shared_encoder_hidden_dims: tuple[int, ...] = (768,)
    shared_decoder_hidden_dims: tuple[int, ...] = (768, 768)
    dropout: float = 0.10
    min_logvar: float = -10.0
    max_logvar: float = 8.0

    phase_reconstruction_weight: float = 1.0
    reporter_reconstruction_weight: float = 1.0
    kl_biological_weight: float = 1.0
    kl_technical_weight: float = 1.0
    technical_information_bottleneck_multiplier: float = 5.0
    modality_alignment_weight: float = 1.0
    reporter_reconstruction_reduction: str = "modality_mean"

    batch_disentanglement: bool = False
    batch_covariate_kind: str = "none"

    def __post_init__(self) -> None:
        if self.phase_dim != PHASE_DIM:
            raise ValueError(f"OPS phase_dim is frozen at {PHASE_DIM}")
        _positive_int(self.biological_dim, "biological_dim")
        _positive_int(self.technical_dim, "technical_dim")
        _positive_int(self.modality_width, "modality_width")
        _widths(self.shared_encoder_hidden_dims, "shared_encoder_hidden_dims")
        _widths(self.shared_decoder_hidden_dims, "shared_decoder_hidden_dims")
        dropout = float(self.dropout)
        if not math.isfinite(dropout) or not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if not float(self.min_logvar) < float(self.max_logvar):
            raise ValueError("min_logvar must be smaller than max_logvar")
        for name in (
            "phase_reconstruction_weight",
            "reporter_reconstruction_weight",
            "kl_biological_weight",
            "kl_technical_weight",
            "technical_information_bottleneck_multiplier",
            "modality_alignment_weight",
        ):
            _nonnegative(getattr(self, name), name)
        if self.reporter_reconstruction_reduction not in {
            "modality_mean",
            "observed_scalar_mean",
        }:
            raise ValueError(
                "reporter_reconstruction_reduction must be modality_mean or "
                "observed_scalar_mean"
            )
        kind = str(self.batch_covariate_kind).strip().casefold()
        if kind in _FORBIDDEN_BATCH_KINDS or "reporter" in kind:
            raise ValueError("Reporter identity is not a technical batch covariate")
        if self.batch_disentanglement:
            raise ValueError(
                "OPS-native MIDAS has no frozen trustworthy technical-batch "
                "variable; batch_disentanglement must remain disabled"
            )
        if kind != "none":
            raise ValueError(
                "Disabled batch disentanglement requires batch_covariate_kind='none'"
            )

    @property
    def latent_dim(self) -> int:
        return int(self.biological_dim + self.technical_dim)


class _MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        output_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        dims = (int(input_dim), *tuple(int(v) for v in hidden_dims), int(output_dim))
        layers: list[nn.Module] = []
        for index, (left, right) in enumerate(zip(dims[:-1], dims[1:])):
            layers.append(nn.Linear(left, right))
            if index < len(dims) - 2:
                layers.extend((nn.LayerNorm(right), nn.Mish()))
                if dropout:
                    layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value)


class _ThinModalityFront(nn.Module):
    """One modality-specific affine/Mish front before the shared encoder."""

    def __init__(self, input_dim: int, width: int, dropout: float) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Linear(input_dim, width),
            nn.LayerNorm(width),
            nn.Mish(),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value)


@dataclass(frozen=True)
class ReporterObservation:
    values: torch.Tensor
    observed_mask: torch.Tensor


@dataclass
class GaussianReconstruction:
    mean: torch.Tensor
    logvar: torch.Tensor


@dataclass
class CompactModalityExpert:
    """One reporter expert evaluated only for rows where it is observed."""

    reporter: str
    row_indices: torch.Tensor
    mean: torch.Tensor
    logvar: torch.Tensor
    unimodal_latent: torch.Tensor


@dataclass
class MIDASMultimodalOPSOutput:
    phase: GaussianReconstruction
    reporters: OrderedDict[str, GaussianReconstruction]
    joint_mean: torch.Tensor
    joint_logvar: torch.Tensor
    latent: torch.Tensor
    biological_latent: torch.Tensor
    technical_latent: torch.Tensor
    phase_expert_mean: torch.Tensor
    phase_expert_logvar: torch.Tensor
    phase_unimodal_latent: torch.Tensor
    reporter_experts: OrderedDict[str, CompactModalityExpert]


def gaussian_product_of_experts(
    means: Sequence[torch.Tensor],
    logvars: Sequence[torch.Tensor],
    *,
    include_unit_prior: bool = True,
    min_logvar: float = -30.0,
    max_logvar: float = 20.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dense diagonal-Gaussian PoE used for single-modality posteriors."""

    if not means or len(means) != len(logvars):
        raise ValueError("means/logvars must contain equal non-zero expert lists")
    shape = means[0].shape
    if len(shape) != 2:
        raise ValueError("Experts must have shape [batch, latent]")
    for mean, logvar in zip(means, logvars):
        if mean.shape != shape or logvar.shape != shape:
            raise ValueError("All expert shapes must match")
    stacked_mean = torch.stack(tuple(means), dim=0)
    stacked_logvar = torch.stack(tuple(logvars), dim=0).clamp(
        float(min_logvar), float(max_logvar)
    )
    precision = torch.exp(-stacked_logvar)
    precision_sum = precision.sum(dim=0)
    if include_unit_prior:
        precision_sum = precision_sum + 1.0
    variance = precision_sum.reciprocal()
    mean = variance * (precision * stacked_mean).sum(dim=0)
    return mean, torch.log(variance)


def rowwise_gaussian_product_of_experts(
    phase_mean: torch.Tensor,
    phase_logvar: torch.Tensor,
    compact_reporter_experts: Sequence[
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ],
    *,
    min_logvar: float = -30.0,
    max_logvar: float = 20.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """PoE where each reporter expert is available on a subset of rows.

    Each compact expert is ``(row_indices, mean, logvar)``.  Rows absent from a
    reporter expert receive exactly no precision contribution.  Phase and the
    unit prior are present for every row.
    """

    if phase_mean.ndim != 2 or phase_logvar.shape != phase_mean.shape:
        raise ValueError("phase expert must have equal [batch, latent] tensors")
    phase_lv = phase_logvar.clamp(float(min_logvar), float(max_logvar))
    phase_precision = torch.exp(-phase_lv)
    precision_sum = 1.0 + phase_precision
    weighted_sum = phase_precision * phase_mean
    batch_size, latent_dim = phase_mean.shape
    for row_indices, mean, logvar in compact_reporter_experts:
        if row_indices.dtype != torch.long or row_indices.ndim != 1:
            raise TypeError("row_indices must be a 1D torch.long tensor")
        if mean.shape != (len(row_indices), latent_dim) or logvar.shape != mean.shape:
            raise ValueError("Compact reporter expert shape mismatch")
        if bool((row_indices < 0).any()) or bool((row_indices >= batch_size).any()):
            raise IndexError("Compact reporter row index outside batch")
        precision = torch.exp(
            -logvar.clamp(float(min_logvar), float(max_logvar))
        )
        precision_sum = precision_sum.index_add(0, row_indices, precision)
        weighted_sum = weighted_sum.index_add(0, row_indices, precision * mean)
    variance = precision_sum.reciprocal()
    return variance * weighted_sum, torch.log(variance)


def masked_gaussian_nll(
    mean: torch.Tensor,
    logvar: torch.Tensor,
    target: torch.Tensor,
    observed_mask: torch.Tensor,
    *,
    min_logvar: float,
    max_logvar: float,
) -> tuple[torch.Tensor, int]:
    """Mean NLL over observed scalars; missing storage has no autograd path."""

    if mean.shape != logvar.shape or mean.shape != target.shape:
        raise ValueError("Gaussian prediction and target shapes must match")
    if observed_mask.dtype is not torch.bool or observed_mask.shape != target.shape:
        raise TypeError("observed_mask must be boolean with the target shape")
    mask = observed_mask.to(device=mean.device)
    count = int(mask.sum().item())
    if count == 0:
        return mean.new_zeros(()), 0
    observed_target = target.to(device=mean.device).masked_select(mask)
    if not bool(torch.isfinite(observed_target).all()):
        raise ValueError("Observed targets must be finite")
    observed_mean = mean.masked_select(mask)
    observed_logvar = logvar.masked_select(mask).clamp(
        float(min_logvar), float(max_logvar)
    )
    loss = 0.5 * (
        math.log(2.0 * math.pi)
        + observed_logvar
        + (observed_target - observed_mean).square() * torch.exp(-observed_logvar)
    )
    return loss.mean(), count


def standard_normal_kl(mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    if mean.shape != logvar.shape or mean.ndim != 2:
        raise ValueError("KL tensors must have equal [batch, latent] shapes")
    return (-0.5 * (1.0 + logvar - mean.square() - logvar.exp())).sum(1).mean()


def available_modality_alignment(
    phase_unimodal_latent: torch.Tensor,
    reporter_experts: Sequence[CompactModalityExpert],
) -> torch.Tensor:
    r"""Upstream MIDAS latent variance generalized to sparse modalities.

    For each cell, let :math:`A_i` be its available unimodal latent samples,
    including phase.  This returns

    .. math::

       \frac{1}{N}\sum_i\sum_{m\in A_i}
       \lVert z_i^m - |A_i|^{-1}\sum_{j\in A_i}z_i^j\rVert_2^2.

    When every modality is available for every row this is algebraically the
    public MIDAS ``calc_mod_align_loss``.  Missing reporter modalities add no
    zero-filled pseudo-expert and contribute neither count nor variance.
    """

    if phase_unimodal_latent.ndim != 2 or len(phase_unimodal_latent) == 0:
        raise ValueError("phase_unimodal_latent must be nonempty [batch, latent]")
    batch_size, latent_dim = phase_unimodal_latent.shape
    latent_sum = phase_unimodal_latent
    available_count = phase_unimodal_latent.new_ones((batch_size, 1))
    for expert in reporter_experts:
        rows = expert.row_indices
        latent = expert.unimodal_latent
        if rows.dtype is not torch.long or rows.ndim != 1:
            raise TypeError("Reporter expert row_indices must be 1D torch.long")
        if latent.shape != (len(rows), latent_dim):
            raise ValueError("Reporter expert latent shape mismatch")
        latent_sum = latent_sum.index_add(0, rows, latent)
        available_count = available_count.index_add(
            0,
            rows,
            available_count.new_ones((len(rows), 1)),
        )
    per_row_mean = latent_sum / available_count
    squared_deviation = (
        phase_unimodal_latent - per_row_mean
    ).square().sum()
    for expert in reporter_experts:
        local_mean = per_row_mean.index_select(0, expert.row_indices)
        squared_deviation = squared_deviation + (
            expert.unimodal_latent - local_mean
        ).square().sum()
    return squared_deviation / batch_size


class MIDASMultimodalOPS(nn.Module):
    """Phase plus independent reporter-modality MIDAS adaptation."""

    family = "midas_ops_53_modality_mosaic"

    def __init__(
        self,
        reporter_dimensions: Mapping[str, int],
        config: MIDASMultimodalOPSConfig | None = None,
    ) -> None:
        super().__init__()
        self.reporter_dimensions = _validate_reporter_dimensions(
            reporter_dimensions
        )
        self.config = config or MIDASMultimodalOPSConfig()
        cfg = self.config
        latent_dim = cfg.latent_dim

        self.phase_front = _ThinModalityFront(
            cfg.phase_dim, cfg.modality_width, cfg.dropout
        )
        self.reporter_fronts = nn.ModuleDict(
            (
                name,
                _ThinModalityFront(2 * dimension, cfg.modality_width, cfg.dropout),
            )
            for name, dimension in self.reporter_dimensions.items()
        )
        self.shared_encoder = _MLP(
            cfg.modality_width,
            cfg.shared_encoder_hidden_dims,
            2 * latent_dim,
            cfg.dropout,
        )

        decoder_widths = _widths(
            cfg.shared_decoder_hidden_dims, "shared_decoder_hidden_dims"
        )
        self.shared_decoder = _MLP(
            latent_dim,
            decoder_widths[:-1],
            decoder_widths[-1],
            cfg.dropout,
        )
        decoder_width = decoder_widths[-1]
        self.phase_decoder = nn.Linear(decoder_width, 2 * cfg.phase_dim)
        self.reporter_decoders = nn.ModuleDict(
            (name, nn.Linear(decoder_width, 2 * dimension))
            for name, dimension in self.reporter_dimensions.items()
        )

    @property
    def modality_names(self) -> tuple[str, ...]:
        return ("phase", *tuple(self.reporter_dimensions))

    def _validate_phase(self, phase: torch.Tensor) -> None:
        if not isinstance(phase, torch.Tensor) or not phase.is_floating_point():
            raise TypeError("phase must be a floating-point torch.Tensor")
        if phase.ndim != 2 or phase.shape[1] != self.config.phase_dim:
            raise ValueError(f"phase must have shape [batch, {self.config.phase_dim}]")
        if not bool(torch.isfinite(phase).all()):
            raise ValueError("phase is always observed and must be finite")

    def _normalise_reporter_observations(
        self,
        observations: Mapping[str, ReporterObservation | tuple[torch.Tensor, torch.Tensor]] | None,
        *,
        batch_size: int,
        device: torch.device,
    ) -> OrderedDict[str, ReporterObservation]:
        if observations is None:
            return OrderedDict()
        if not isinstance(observations, Mapping):
            raise TypeError("reporter_observations must be a mapping")
        result: OrderedDict[str, ReporterObservation] = OrderedDict()
        for raw_name, raw_observation in observations.items():
            name = str(raw_name)
            if name not in self.reporter_dimensions:
                raise KeyError(f"Unknown reporter modality {name!r}")
            if isinstance(raw_observation, ReporterObservation):
                observation = raw_observation
            else:
                if not isinstance(raw_observation, tuple) or len(raw_observation) != 2:
                    raise TypeError(
                        "Reporter observation must be ReporterObservation or "
                        "(values, observed_mask)"
                    )
                observation = ReporterObservation(*raw_observation)
            values, mask = observation.values, observation.observed_mask
            dimension = self.reporter_dimensions[name]
            if not isinstance(values, torch.Tensor) or not values.is_floating_point():
                raise TypeError(f"{name} values must be a floating-point tensor")
            if values.shape != (batch_size, dimension):
                raise ValueError(
                    f"{name} values must have shape [{batch_size}, {dimension}]"
                )
            if values.device != device:
                raise ValueError(f"{name} values and phase must share a device")
            if not isinstance(mask, torch.Tensor) or mask.dtype is not torch.bool:
                raise TypeError(f"{name} observed_mask must have dtype torch.bool")
            if mask.shape != values.shape or mask.device != device:
                raise ValueError(f"{name} mask must match values on the phase device")
            if not bool(torch.isfinite(values.masked_select(mask)).all()):
                raise ValueError(f"{name} observed targets must be finite")
            result[name] = ReporterObservation(values, mask)
        return result

    def _encode_front(self, front: nn.Module, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.shared_encoder(front(value))
        return encoded.chunk(2, dim=1)

    @staticmethod
    def _sample(mean: torch.Tensor, logvar: torch.Tensor, sample: bool) -> torch.Tensor:
        if not sample:
            return mean
        return mean + torch.randn_like(mean) * torch.exp(0.5 * logvar)

    def _decode_reporters(
        self,
        shared: torch.Tensor,
        reporter_names: Sequence[str],
    ) -> OrderedDict[str, GaussianReconstruction]:
        result: OrderedDict[str, GaussianReconstruction] = OrderedDict()
        for raw_name in reporter_names:
            name = str(raw_name)
            if name not in self.reporter_dimensions:
                raise KeyError(f"Unknown reporter modality {name!r}")
            if name in result:
                raise ValueError(f"Duplicate reporter decoder request {name!r}")
            mean, logvar = self.reporter_decoders[name](shared).chunk(2, dim=1)
            result[name] = GaussianReconstruction(mean, logvar)
        return result

    def forward(
        self,
        phase: torch.Tensor,
        reporter_observations: Mapping[
            str, ReporterObservation | tuple[torch.Tensor, torch.Tensor]
        ] | None = None,
        *,
        decode_reporters: Sequence[str] | None = None,
        sample: bool | None = None,
    ) -> MIDASMultimodalOPSOutput:
        self._validate_phase(phase)
        observations = self._normalise_reporter_observations(
            reporter_observations,
            batch_size=len(phase),
            device=phase.device,
        )
        if decode_reporters is None:
            decode_names = tuple(observations)
        else:
            decode_names = tuple(str(value) for value in decode_reporters)
        do_sample = self.training if sample is None else bool(sample)
        cfg = self.config

        phase_mean, phase_logvar = self._encode_front(self.phase_front, phase)
        phase_uni_mean, phase_uni_logvar = gaussian_product_of_experts(
            (phase_mean,),
            (phase_logvar,),
            min_logvar=cfg.min_logvar,
            max_logvar=cfg.max_logvar,
        )
        phase_unimodal = self._sample(phase_uni_mean, phase_uni_logvar, do_sample)

        compact: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        reporter_experts: OrderedDict[str, CompactModalityExpert] = OrderedDict()
        for name, observation in observations.items():
            present = observation.observed_mask.any(dim=1)
            rows = present.nonzero(as_tuple=False).flatten()
            if len(rows) == 0:
                # A completely absent modality is semantically identical to it
                # not being supplied, and none of its parameters enter the graph.
                continue
            local_values = observation.values.index_select(0, rows)
            local_mask = observation.observed_mask.index_select(0, rows)
            clean_values = torch.where(
                local_mask, local_values, torch.zeros_like(local_values)
            )
            front_input = torch.cat(
                (clean_values, local_mask.to(dtype=clean_values.dtype)), dim=1
            )
            mean, logvar = self._encode_front(
                self.reporter_fronts[name], front_input
            )
            uni_mean, uni_logvar = gaussian_product_of_experts(
                (mean,),
                (logvar,),
                min_logvar=cfg.min_logvar,
                max_logvar=cfg.max_logvar,
            )
            unimodal = self._sample(uni_mean, uni_logvar, do_sample)
            compact.append((rows, mean, logvar))
            reporter_experts[name] = CompactModalityExpert(
                reporter=name,
                row_indices=rows,
                mean=mean,
                logvar=logvar,
                unimodal_latent=unimodal,
            )

        joint_mean, joint_logvar = rowwise_gaussian_product_of_experts(
            phase_mean,
            phase_logvar,
            compact,
            min_logvar=cfg.min_logvar,
            max_logvar=cfg.max_logvar,
        )
        latent = self._sample(joint_mean, joint_logvar, do_sample)
        biological, technical = latent.split(
            (cfg.biological_dim, cfg.technical_dim), dim=1
        )
        shared = self.shared_decoder(latent)
        phase_recon_mean, phase_recon_logvar = self.phase_decoder(shared).chunk(
            2, dim=1
        )
        reporter_recons = self._decode_reporters(shared, decode_names)
        return MIDASMultimodalOPSOutput(
            phase=GaussianReconstruction(phase_recon_mean, phase_recon_logvar),
            reporters=reporter_recons,
            joint_mean=joint_mean,
            joint_logvar=joint_logvar,
            latent=latent,
            biological_latent=biological,
            technical_latent=technical,
            phase_expert_mean=phase_mean,
            phase_expert_logvar=phase_logvar,
            phase_unimodal_latent=phase_unimodal,
            reporter_experts=reporter_experts,
        )

    def compute_loss(
        self,
        output: MIDASMultimodalOPSOutput,
        phase: torch.Tensor,
        reporter_observations: Mapping[
            str, ReporterObservation | tuple[torch.Tensor, torch.Tensor]
        ],
    ) -> dict[str, torch.Tensor]:
        self._validate_phase(phase)
        observations = self._normalise_reporter_observations(
            reporter_observations,
            batch_size=len(phase),
            device=phase.device,
        )
        cfg = self.config
        phase_nll, phase_count = masked_gaussian_nll(
            output.phase.mean,
            output.phase.logvar,
            phase,
            torch.ones_like(phase, dtype=torch.bool),
            min_logvar=cfg.min_logvar,
            max_logvar=cfg.max_logvar,
        )

        reporter_losses: OrderedDict[str, torch.Tensor] = OrderedDict()
        reporter_counts: OrderedDict[str, int] = OrderedDict()
        for name, observation in observations.items():
            count = int(observation.observed_mask.sum().item())
            reporter_counts[name] = count
            if count == 0:
                # Do not even request/use a decoder for an absent modality.
                continue
            if name not in output.reporters:
                raise ValueError(f"Output did not decode observed modality {name!r}")
            recon = output.reporters[name]
            loss, observed = masked_gaussian_nll(
                recon.mean,
                recon.logvar,
                observation.values,
                observation.observed_mask,
                min_logvar=cfg.min_logvar,
                max_logvar=cfg.max_logvar,
            )
            reporter_losses[name] = loss
            reporter_counts[name] = observed
        if not reporter_losses:
            raise ValueError("Training loss requires at least one observed reporter scalar")
        if cfg.reporter_reconstruction_reduction == "modality_mean":
            reporter_nll = torch.stack(tuple(reporter_losses.values())).mean()
        else:
            weighted = [
                reporter_losses[name] * reporter_counts[name]
                for name in reporter_losses
            ]
            reporter_nll = torch.stack(weighted).sum() / sum(
                reporter_counts[name] for name in reporter_losses
            )

        biological_mean = output.joint_mean[:, : cfg.biological_dim]
        biological_logvar = output.joint_logvar[:, : cfg.biological_dim]
        technical_mean = output.joint_mean[:, cfg.biological_dim :]
        technical_logvar = output.joint_logvar[:, cfg.biological_dim :]
        kl_biological = standard_normal_kl(biological_mean, biological_logvar)
        kl_technical_raw = standard_normal_kl(technical_mean, technical_logvar)
        kl_technical_ib = (
            cfg.technical_information_bottleneck_multiplier * kl_technical_raw
        )

        alignment = available_modality_alignment(
            output.phase_unimodal_latent,
            tuple(output.reporter_experts.values()),
        )
        total = (
            cfg.phase_reconstruction_weight * phase_nll
            + cfg.reporter_reconstruction_weight * reporter_nll
            + cfg.kl_biological_weight * kl_biological
            + cfg.kl_technical_weight * kl_technical_ib
            + cfg.modality_alignment_weight * alignment
        )
        result: dict[str, torch.Tensor] = {
            "total": total,
            "phase_nll": phase_nll,
            "reporter_nll": reporter_nll,
            "kl_biological": kl_biological,
            "kl_technical_raw": kl_technical_raw,
            "kl_technical_ib": kl_technical_ib,
            "modality_alignment": alignment,
            "observed_phase_scalars": phase_nll.new_tensor(phase_count),
            "observed_reporter_scalars": phase_nll.new_tensor(
                sum(reporter_counts.values())
            ),
            "observed_reporter_modalities": phase_nll.new_tensor(
                len(reporter_losses)
            ),
        }
        for name, value in reporter_losses.items():
            result[f"reporter_nll::{name}"] = value
        return result

    def predict_reporters(
        self,
        phase: torch.Tensor,
        reporter_names: Sequence[str],
    ) -> OrderedDict[str, torch.Tensor]:
        """Phase-only imputation through reporter-local decoder heads."""

        was_training = self.training
        try:
            self.eval()
            output = self(
                phase,
                None,
                decode_reporters=tuple(reporter_names),
                sample=False,
            )
            return OrderedDict(
                (name, recon.mean) for name, recon in output.reporters.items()
            )
        finally:
            self.train(was_training)

    def predict_reporter(self, phase: torch.Tensor, reporter: str) -> torch.Tensor:
        return self.predict_reporters(phase, (reporter,))[str(reporter)]

    def parameter_count(self, *, trainable_only: bool = False) -> int:
        parameters = self.parameters()
        if trainable_only:
            parameters = (value for value in parameters if value.requires_grad)
        return int(sum(value.numel() for value in parameters))

    def config_manifest(self) -> dict[str, Any]:
        return {
            "schema_version": MODEL_SCHEMA,
            "family": self.family,
            "source_provenance": {
                "repository": "labomics/midas",
                "frozen_commit": MIDAS_REPRODUCIBILITY_COMMIT,
                "source_file": MIDAS_SOURCE_MODELS_PATH,
                "source_file_sha256": MIDAS_SOURCE_MODELS_SHA256,
                "license_caveat": (
                    "frozen reproducibility branch contains no LICENSE file; "
                    "current upstream main declares MIT"
                ),
                "mechanisms_retained": [
                    "modality_specific_fronts",
                    "shared_posterior_encoder",
                    "unit_prior_gaussian_product_of_experts",
                    "shared_latent_decoder_trunk",
                    "modality_specific_reconstruction_blocks",
                    "biological_technical_latent_partition",
                    "unimodal_posterior_alignment",
                ],
            },
            "scientific_modality_contract": {
                "n_modalities": 1 + len(self.reporter_dimensions),
                "source_modality": {"name": "phase", "dimension": PHASE_DIM},
                "target_modalities": [
                    {"name": name, "dimension": dimension}
                    for name, dimension in self.reporter_dimensions.items()
                ],
                "target_representation": (
                    "independent_sparse_reporter_modalities"
                ),
                "panel_subset_policy": (
                    "all_52_instantiated; non-panel reporter modalities absent"
                ),
            },
            "missingness": {
                "explicit_boolean_endpoint_mask": True,
                "numeric_zero_is_observed_when_mask_true": True,
                "absent_modality_contributes_to_poe": False,
                "absent_modality_contributes_to_reconstruction": False,
            },
            "decoder": {
                "shared_latent_trunk": True,
                "reporter_local_gaussian_heads": True,
            },
            "batch_and_identifiability": {
                "batch_disentanglement_enabled": False,
                "trusted_technical_batch_available": False,
                "reporter_id_allowed_as_batch": False,
                "technical_information_bottleneck_multiplier": (
                    self.config.technical_information_bottleneck_multiplier
                ),
                "latent_split_identifiable": False,
                "batch_correction_claim": False,
                "interpretation": (
                    "latent split plus asymmetric IB are regularization only"
                ),
            },
            "config": asdict(self.config),
            "parameter_count": self.parameter_count(),
            "parameter_policy": {
                "matching_required": False,
                "explicit_limit": None,
                "role": "reporting_only",
            },
        }


__all__ = [
    "CompactModalityExpert",
    "GaussianReconstruction",
    "MIDASMultimodalOPS",
    "MIDASMultimodalOPSConfig",
    "MIDASMultimodalOPSOutput",
    "MIDAS_REPRODUCIBILITY_COMMIT",
    "MIDAS_SOURCE_MODELS_PATH",
    "MIDAS_SOURCE_MODELS_SHA256",
    "MODEL_SCHEMA",
    "PHASE_DIM",
    "ReporterObservation",
    "available_modality_alignment",
    "gaussian_product_of_experts",
    "masked_gaussian_nll",
    "rowwise_gaussian_product_of_experts",
    "standard_normal_kl",
]
