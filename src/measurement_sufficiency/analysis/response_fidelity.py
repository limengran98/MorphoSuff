"""Screen-control-relative perturbation response fidelity.

The functions deliberately operate on the canonical long prediction table.  A
response is never a raw KO average: for every model/reporter/endpoint/screen it
is the perturbation mean less the matched control mean from that same screen.

The quantities here follow the published Methods, which define them on the
*gene* response vector rather than on one endpoint at a time:

    "For each reporter and held-out gene, we calculated a control-relative
     multivariate response vector ... Response magnitude was the Euclidean norm
     of this vector. Magnitude-rank fidelity was the Spearman correlation
     between observed and predicted gene magnitudes. Amplitude fidelity was the
     ratio of predicted to observed variance across gene magnitudes. Strong-hit
     recall was the fraction of the observed top 5% of genes recovered in the
     predicted top 5%."

So the pipeline is: predictions -> screen_relative_responses (per endpoint) ->
response_magnitudes (Euclidean norm per gene) -> the three scores, all keyed by
reporter rather than by endpoint.
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
import pandas as pd

from ..adapters import AdapterCapabilities
from ..schemas import coerce_boolean_column, validate_predictions


_BASE = ["reporter_id", "endpoint_id", "split_name", "fold", "model_id", "screen_id"]
_RESPONSE_KEYS = _BASE + ["perturbation_id"]
#: Default grouping for every gene-level score: one row per reporter per fitted
#: model, matching the manuscript's "for each reporter and held-out gene".
_GENE_GROUPS = ["reporter_id", "split_name", "fold", "model_id"]

#: A column named ``top5pct_recall`` always means recall at the top 5%. The
#: frozen figure source tables use this name, and the OPS tier rule thresholds
#: (configs/ops/protocols/measurement_tiers.yaml) are calibrated against it, so
#: emitting some other fraction under this name would silently mis-tier reporters.
RESERVED_FRACTION_COLUMN = "top5pct_recall"
RESERVED_FRACTION = 0.05


def screen_relative_responses(
    predictions: pd.DataFrame, *, capabilities: AdapterCapabilities | None = None
) -> pd.DataFrame:
    """Return KO × screen responses after same-screen control subtraction.

    ``is_control`` is intentionally required here rather than inferred from a
    perturbation label.  This prevents a dataset-specific control naming
    convention from changing a scientific result.
    """
    if capabilities is not None:
        capabilities.require("screen_matched_controls", analysis="response fidelity")
    validate_predictions(predictions)
    if "is_control" not in predictions:
        raise ValueError(
            "response fidelity requires an explicit is_control column on the prediction "
            "table; controls are never inferred from a perturbation label. Both "
            "producers emit it when their metadata carries it: "
            "measurement_sufficiency.training.prediction_long_table and "
            "studies/ops/runners/common.py::canonical_predictions. For the OPS study, "
            "re-export the canonical bundle with "
            "studies/ops/preparation/export_canonical.py, which writes is_control into "
            "the inputs table. Otherwise join it from the canonical observations table "
            "on observation_id."
        )
    is_control = coerce_boolean_column(
        predictions["is_control"], column="is_control", table="predictions"
    )
    controls = predictions[is_control]
    if controls.empty:
        raise ValueError("response fidelity requires same-screen control rows")
    baseline = controls.groupby(_BASE, dropna=False)[["y_true", "y_pred"]].mean().rename(
        columns={"y_true": "control_y_true", "y_pred": "control_y_pred"}
    )
    perturbed = predictions[~is_control]
    if perturbed.empty:
        raise ValueError("response fidelity requires non-control perturbations")
    response = perturbed.groupby(_RESPONSE_KEYS, dropna=False)[["y_true", "y_pred"]].mean().join(
        baseline, on=_BASE, how="left", validate="many_to_one"
    )
    if response[["control_y_true", "control_y_pred"]].isna().any().any():
        raise ValueError("each evaluated KO × screen requires a matched control")
    response["response_true"] = response.pop("y_true") - response.pop("control_y_true")
    response["response_pred"] = response.pop("y_pred") - response.pop("control_y_pred")
    return response.reset_index()


def response_magnitudes(
    responses: pd.DataFrame,
    *,
    group_columns: list[str] | None = None,
    gene_column: str = "perturbation_id",
    across_screens: Literal["mean", "separate"] = "mean",
    standardize_endpoints: Literal["none", "observed_sd"] = "none",
    min_spread_ratio: float = 1e-3,
) -> pd.DataFrame:
    """Euclidean norm of each gene's control-relative endpoint response vector.

    This is the unit the published Methods score on.  Taking ``abs()`` of a
    single endpoint instead, as an endpoint-keyed table implicitly does, answers
    a different question: it ranks endpoint deflections rather than gene
    responses, and produces one row per endpoint where the manuscript reports one
    row per reporter.

    A norm is only meaningful over comparable coordinates, and there are two ways
    that fails.  Unequal *length* is already refused below.  Unequal *scale* is the
    other, and it is silent: if one endpoint has a much larger spread than the rest,
    the norm is that endpoint wearing the costume of a profile.  Every row therefore
    carries ``max_endpoint_scale_share``, the fraction of the summed squared endpoint
    spread held by the single largest endpoint, so the condition cannot go unseen.
    Values near ``1 / n_endpoints`` mean the endpoints contribute evenly; values near
    1 mean the norm has collapsed onto one coordinate.

    This is not hypothetical.  On the PERISCOPE anti-TOMM20 block, whose 823 endpoints
    span Granularity, Intensity, Texture and RadialDistribution families in their
    native units, ``Cells_Intensity_IntegratedIntensity_Mito`` alone holds 86% of the
    squared spread, so an unstandardised norm over 823 coordinates is effectively that
    one number.

    Standardising introduces the mirror-image failure, and it bit.  Dividing by the
    observed spread turns an endpoint that barely varies into the loudest coordinate
    in the norm, because the divisor, not the signal, is what is small.  On the
    single-cell PERISCOPE fit, ``RadialCV_mito_tubeness_8of20`` had an observed spread
    of 4.3e-06 across the 4,078 genes of one fold, which is a constant to every
    intent, and a predicted spread of 1.5e-04, equally negligible.  Their ratio is 35,
    and that one endpoint took 63% of the standardised squared norm; the fold's
    predicted response magnitudes came out 46 times more variable than the observed
    ones while the other four folds shrank normally.  Dropping only an exactly zero
    spread, as this function first did, does not catch it: the value was not zero.

    ``min_spread_ratio`` therefore drops an endpoint whose observed spread is below
    that fraction of the group's median endpoint spread, on the same reasoning that
    drops an exactly constant one: an endpoint that does not vary carries no
    information about which gene responded more, and a norm that lets it in is
    measuring the divisor.  The default of 1e-3 is three orders of magnitude below
    the typical endpoint, which admits every endpoint whose spread is resolvable and
    excludes the degenerate families; ``max_standardized_share`` is emitted so the
    condition is checked rather than assumed to have been handled.

    Args:
        standardize_endpoints: ``"none"`` keeps the raw units and is correct when the
            endpoint block is already on a common scale, as the OPS reporter blocks
            are; it is the default so that existing results are unchanged.
            ``"observed_sd"`` divides each endpoint by the standard deviation of its
            observed response within the group before taking the norm. The same
            divisor is applied to the observed and predicted vectors, so it cannot
            inflate their agreement, but it is computed from the responses in hand and
            that must be stated wherever it is used.
    """
    if standardize_endpoints not in {"none", "observed_sd"}:
        raise ValueError("standardize_endpoints must be 'none' or 'observed_sd'")
    groups = list(group_columns or _GENE_GROUPS)
    needed = set(groups + [gene_column, "endpoint_id", "screen_id", "response_true", "response_pred"])
    missing = needed.difference(responses)
    if missing:
        raise ValueError(f"responses lacks columns: {', '.join(sorted(missing))}")
    if across_screens not in {"mean", "separate"}:
        raise ValueError("across_screens must be 'mean' or 'separate'")

    keys = groups + [gene_column] + (["screen_id"] if across_screens == "separate" else [])
    collapsed = (
        responses.groupby(keys + ["endpoint_id"], dropna=False)[["response_true", "response_pred"]]
        .mean()
        .reset_index()
    )
    # A norm over a different number of endpoints is not comparable, so an
    # endpoint set that varies between genes is refused rather than ranked.
    sizes = collapsed.groupby(keys, dropna=False).endpoint_id.nunique()
    for key, frame in sizes.groupby(level=list(range(len(groups))), dropna=False):
        counts = frame.unique()
        if len(counts) > 1:
            offenders = sorted(map(str, frame[frame != counts.max()].index.get_level_values(-1)[:5]))
            raise ValueError(
                f"group {key} compares response vectors of different lengths "
                f"({sorted(counts.tolist())} endpoints); every gene must present the same "
                f"endpoint set for a Euclidean norm to be comparable. First genes with a "
                f"short vector: {offenders}"
            )
    # Endpoint spread is measured *across genes* within a group, which is the only
    # level at which it means anything: within one gene an endpoint has a single value
    # and no spread at all. It serves both the scale diagnostic and, when asked for,
    # as the divisor that puts the coordinates on a common footing.
    spread = (
        collapsed.groupby(groups + ["endpoint_id"], dropna=False)["response_true"]
        .std()
        .rename("endpoint_sd")
        .reset_index()
    )
    squared = spread.assign(squared=spread["endpoint_sd"].fillna(0.0) ** 2)
    totals = squared.groupby(groups, dropna=False)["squared"].sum().rename("total")
    largest = squared.groupby(groups, dropna=False)["squared"].max().rename("largest")
    share = pd.concat([totals, largest], axis=1)
    share["max_endpoint_scale_share"] = np.where(
        share["total"] > 0, share["largest"] / share["total"], np.nan
    )

    if standardize_endpoints == "observed_sd":
        if not 0.0 <= min_spread_ratio < 1.0:
            raise ValueError("min_spread_ratio must be in [0, 1)")
        floor = (
            spread.groupby(groups, dropna=False)["endpoint_sd"]
            .median()
            .rename("endpoint_sd_floor")
            * min_spread_ratio
        )
        collapsed = collapsed.merge(spread, on=groups + ["endpoint_id"], how="left")
        collapsed = collapsed.merge(floor.reset_index(), on=groups, how="left")
        # An endpoint that does not vary carries no information about which gene
        # responded more, so it is dropped rather than used as a divisor. Exactly
        # zero is only the visible half of that condition.
        keep = collapsed["endpoint_sd"].gt(0.0) & collapsed["endpoint_sd"].gt(
            collapsed["endpoint_sd_floor"]
        )
        collapsed = collapsed.loc[keep].copy()
        divisor = collapsed["endpoint_sd"]
        collapsed["response_true"] = collapsed["response_true"] / divisor
        collapsed["response_pred"] = collapsed["response_pred"] / divisor
        # The share that actually enters the norm after the divisor is applied.
        # Measured on the predicted vector, because the observed one has unit spread
        # per endpoint by construction and so can never show the condition. This is
        # precisely the number the raw ``max_endpoint_scale_share`` cannot see.
        per_endpoint = (
            collapsed.assign(squared=collapsed["response_pred"] ** 2)
            .groupby(groups + ["endpoint_id"], dropna=False)["squared"]
            .mean()
        )
        totals = per_endpoint.groupby(level=list(range(len(groups))), dropna=False).sum()
        largest = per_endpoint.groupby(level=list(range(len(groups))), dropna=False).max()
        share["max_standardized_share"] = (
            (largest / totals.where(totals > 0)).reindex(share.index)
        )
    else:
        share["max_standardized_share"] = np.nan

    magnitudes = (
        collapsed.groupby(keys, dropna=False)
        .agg(
            n_endpoints=("endpoint_id", "nunique"),
            magnitude_true=("response_true", lambda values: float(np.linalg.norm(values))),
            magnitude_pred=("response_pred", lambda values: float(np.linalg.norm(values))),
        )
        .reset_index()
    )
    return magnitudes.merge(
        share[["max_endpoint_scale_share", "max_standardized_share"]].reset_index(),
        on=groups,
        how="left",
    )


def _spearman(x: pd.Series, y: pd.Series) -> float:
    if len(x) < 2 or x.nunique() < 2 or y.nunique() < 2:
        return float("nan")
    return float(x.corr(y, method="spearman"))


def magnitude_spearman(
    magnitudes: pd.DataFrame,
    *,
    group_columns: list[str] | None = None,
    true_column: str = "magnitude_true",
    pred_column: str = "magnitude_pred",
) -> pd.DataFrame:
    """Spearman agreement of observed and predicted gene response magnitudes."""
    groups = list(group_columns or _GENE_GROUPS)
    missing = set(groups + [true_column, pred_column]).difference(magnitudes)
    if missing:
        raise ValueError(f"magnitudes lacks columns: {', '.join(sorted(missing))}")
    out = []
    for key, group in magnitudes.groupby(groups, dropna=False, sort=False):
        key = key if isinstance(key, tuple) else (key,)
        out.append(
            dict(
                zip(groups, key),
                n_genes=len(group),
                magnitude_spearman=_spearman(group[true_column], group[pred_column]),
            )
        )
    return pd.DataFrame(out)


def variance_ratio(
    magnitudes: pd.DataFrame,
    *,
    group_columns: list[str] | None = None,
    true_column: str = "magnitude_true",
    pred_column: str = "magnitude_pred",
) -> pd.DataFrame:
    """Predicted/observed variance ratio across gene response magnitudes."""
    groups = list(group_columns or _GENE_GROUPS)
    missing = set(groups + [true_column, pred_column]).difference(magnitudes)
    if missing:
        raise ValueError(f"magnitudes lacks columns: {', '.join(sorted(missing))}")
    out = []
    for key, group in magnitudes.groupby(groups, dropna=False, sort=False):
        key = key if isinstance(key, tuple) else (key,)
        observed = float(group[true_column].var(ddof=1))
        ratio = (
            float("nan")
            if not np.isfinite(observed) or observed <= 0
            else float(group[pred_column].var(ddof=1) / observed)
        )
        out.append(dict(zip(groups, key), n_genes=len(group), variance_ratio=ratio))
    return pd.DataFrame(out)


def _ranked_recall(
    group: pd.DataFrame,
    *,
    k: int,
    gene_column: str,
    true_column: str,
    pred_column: str,
    ties: Literal["deterministic", "inclusive", "error"],
) -> tuple[float, int, int]:
    def top_set(column: str) -> tuple[set[object], int]:
        ordered = group.sort_values([column, gene_column], ascending=[False, True], kind="stable")
        boundary = ordered[column].iloc[k - 1]
        n_tied = int((ordered[column] == boundary).sum()) if k < len(ordered) else 0
        if ties == "inclusive":
            return set(ordered.loc[ordered[column] >= boundary, gene_column]), n_tied
        if ties == "error" and k < len(ordered) and ordered[column].iloc[k] == boundary:
            raise ValueError(
                "a tie spans the selection boundary; choose ties='deterministic' for a "
                "stable gene-name tie break or ties='inclusive' to keep every tied gene"
            )
        return set(ordered[gene_column].iloc[:k]), n_tied

    actual, n_true_tied = top_set(true_column)
    predicted, n_pred_tied = top_set(pred_column)
    # The denominator is the size of the *observed* top set. Dividing a
    # tie-expanded intersection by the requested rank count is what let this
    # score exceed 1.0.
    return len(actual & predicted) / len(actual), n_true_tied, n_pred_tied


def top_fraction_recall(
    magnitudes: pd.DataFrame,
    *,
    top_fraction: float = RESERVED_FRACTION,
    group_columns: list[str] | None = None,
    gene_column: str = "perturbation_id",
    true_column: str = "magnitude_true",
    pred_column: str = "magnitude_pred",
    ties: Literal["deterministic", "inclusive", "error"] = "deterministic",
    rounding: Literal["ceil", "floor", "round"] = "ceil",
    output_column: str = RESERVED_FRACTION_COLUMN,
) -> pd.DataFrame:
    """Recall of the observed top ``top_fraction`` of genes among the predicted top.

    With ~1,000 held-out genes per reporter the three rounding modes all give
    k = 50 at 5%, so the published value is insensitive to that choice; the knob
    exists so a reader can confirm it.  Exact ties in a continuous Euclidean norm
    are effectively absent, so the tie policy is likewise not load bearing, but
    both tie counts are reported so it can be audited rather than assumed.
    """
    if not 0.0 < top_fraction <= 1.0:
        raise ValueError("top_fraction must be in (0, 1]")
    if output_column == RESERVED_FRACTION_COLUMN and top_fraction != RESERVED_FRACTION:
        raise ValueError(
            f"{RESERVED_FRACTION_COLUMN!r} is reserved for recall at the top "
            f"{RESERVED_FRACTION:.0%}; the OPS tier thresholds are calibrated against "
            "that definition. Pass an explicit output_column for another fraction."
        )
    groups = list(group_columns or _GENE_GROUPS)
    missing = set(groups + [gene_column, true_column, pred_column]).difference(magnitudes)
    if missing:
        raise ValueError(f"magnitudes lacks columns: {', '.join(sorted(missing))}")
    rounder = {"ceil": math.ceil, "floor": math.floor, "round": round}[rounding]
    rows = []
    for key, group in magnitudes.groupby(groups, dropna=False, sort=False):
        key = key if isinstance(key, tuple) else (key,)
        n_genes = len(group)
        k = max(1, min(n_genes, int(rounder(n_genes * top_fraction))))
        recall, n_true_tied, n_pred_tied = _ranked_recall(
            group,
            k=k,
            gene_column=gene_column,
            true_column=true_column,
            pred_column=pred_column,
            ties=ties,
        )
        rows.append(
            dict(
                zip(groups, key),
                n_genes=n_genes,
                k=k,
                top_fraction=top_fraction,
                n_true_tied_at_boundary=n_true_tied,
                n_pred_tied_at_boundary=n_pred_tied,
                **{output_column: recall},
            )
        )
    return pd.DataFrame(rows)


def top_k_recall(
    magnitudes: pd.DataFrame,
    *,
    k: int = 5,
    group_columns: list[str] | None = None,
    gene_column: str = "perturbation_id",
    true_column: str = "magnitude_true",
    pred_column: str = "magnitude_pred",
    ties: Literal["deterministic", "inclusive", "error"] = "deterministic",
    output_column: str = "top_k_recall",
) -> pd.DataFrame:
    """Recall at an absolute rank cut.

    Distinct from :func:`top_fraction_recall`: the OPS tier rule consumes the
    top-5% quantity, not this one.
    """
    if k < 1:
        raise ValueError("k must be positive")
    if output_column == RESERVED_FRACTION_COLUMN:
        raise ValueError(f"{RESERVED_FRACTION_COLUMN!r} is reserved for a fractional cut")
    groups = list(group_columns or _GENE_GROUPS)
    missing = set(groups + [gene_column, true_column, pred_column]).difference(magnitudes)
    if missing:
        raise ValueError(f"magnitudes lacks columns: {', '.join(sorted(missing))}")
    rows = []
    for key, group in magnitudes.groupby(groups, dropna=False, sort=False):
        key = key if isinstance(key, tuple) else (key,)
        effective = min(k, len(group))
        recall, n_true_tied, n_pred_tied = _ranked_recall(
            group,
            k=effective,
            gene_column=gene_column,
            true_column=true_column,
            pred_column=pred_column,
            ties=ties,
        )
        rows.append(
            dict(
                zip(groups, key),
                n_genes=len(group),
                k=effective,
                n_true_tied_at_boundary=n_true_tied,
                n_pred_tied_at_boundary=n_pred_tied,
                **{output_column: recall},
            )
        )
    return pd.DataFrame(rows)


def profile_fidelity(
    responses: pd.DataFrame,
    *,
    group_columns: list[str] | None = None,
    gene_column: str = "perturbation_id",
) -> pd.DataFrame:
    """Within-gene Pearson between observed and predicted endpoint profiles.

    The fifth Methods quantity: for each gene, how well the *shape* of the
    control-relative response across endpoints is recovered, independent of its
    magnitude.  Reported as the median over genes within each group.
    """
    groups = list(group_columns or _GENE_GROUPS)
    needed = set(groups + [gene_column, "endpoint_id", "response_true", "response_pred"])
    missing = needed.difference(responses)
    if missing:
        raise ValueError(f"responses lacks columns: {', '.join(sorted(missing))}")
    collapsed = (
        responses.groupby(groups + [gene_column, "endpoint_id"], dropna=False)[
            ["response_true", "response_pred"]
        ]
        .mean()
        .reset_index()
    )
    rows = []
    for key, group in collapsed.groupby(groups, dropna=False, sort=False):
        key = key if isinstance(key, tuple) else (key,)
        per_gene = [
            _pearson(gene_frame.response_true, gene_frame.response_pred)
            for _, gene_frame in group.groupby(gene_column, dropna=False, sort=False)
        ]
        usable = [value for value in per_gene if np.isfinite(value)]
        rows.append(
            dict(
                zip(groups, key),
                n_genes=len(per_gene),
                n_genes_scored=len(usable),
                profile_pearson_median=float(np.median(usable)) if usable else float("nan"),
            )
        )
    return pd.DataFrame(rows)


def _pearson(x: pd.Series, y: pd.Series) -> float:
    if len(x) < 2 or x.nunique() < 2 or y.nunique() < 2:
        return float("nan")
    return float(x.corr(y))


def response_profiles(responses: pd.DataFrame, *, profile_by: str = "perturbation_id") -> pd.DataFrame:
    """Emit aligned true/predicted response profiles for transparent plotting."""
    if profile_by not in responses:
        raise ValueError(f"profile column {profile_by!r} is absent")
    required = {"response_true", "response_pred", "endpoint_id", "screen_id"}
    if missing := required.difference(responses):
        raise ValueError(f"responses lacks columns: {', '.join(sorted(missing))}")
    columns = [profile_by, "reporter_id", "endpoint_id", "screen_id", "response_true", "response_pred"]
    present = [column for column in columns if column in responses]
    return responses.loc[:, present].sort_values(
        [column for column in columns[:4] if column in responses], kind="stable"
    ).reset_index(drop=True)


def response_fidelity(
    predictions: pd.DataFrame,
    *,
    top_fraction: float = RESERVED_FRACTION,
    top_k: int | None = None,
    across_screens: Literal["mean", "separate"] = "mean",
    capabilities: AdapterCapabilities | None = None,
) -> dict[str, pd.DataFrame]:
    """Calculate all manuscript response-fidelity quantities from one contract."""
    responses = screen_relative_responses(predictions, capabilities=capabilities)
    magnitudes = response_magnitudes(responses, across_screens=across_screens)
    output = {
        "responses": responses,
        "magnitudes": magnitudes,
        "magnitude_spearman": magnitude_spearman(magnitudes),
        "variance_ratio": variance_ratio(magnitudes),
        "strong_hit_recall": top_fraction_recall(magnitudes, top_fraction=top_fraction),
        "profile_fidelity": profile_fidelity(responses),
        "profiles": response_profiles(responses),
    }
    if top_k is not None:
        output["top_k"] = top_k_recall(magnitudes, k=top_k)
    return output
