#!/usr/bin/env python3
"""One split-half reliability estimator, for every PERISCOPE level and target block.

Reliability is the ceiling the tier rule gates on, and it decides both PERISCOPE
verdicts at the first test, before any performance quantity is consulted. It was
computed in two places, once from guide profiles and once from collapsed cells, and
audited in a third, so it is extracted here for the reason ``_ridge`` was extracted:
the levels differ in what a row is and must not differ in how the quantity is formed.

**Pooling is standardised, which is a change.** The estimate concatenates a block of
(gene, endpoint) pairs and correlates two independent halves of it. Pooling that block
in native CellProfiler units does not describe the block when the block does not share
a scale: it describes whichever endpoint has the largest units.
``analysis.response_fidelity.response_magnitudes`` has always standardised for this
reason and ``_ridge.recoverability_by_fold`` was changed to match on 2026-08-14, but
the reliability path was left in native units, so the rule was comparing a value and
its ceiling formed under different rules.

That this matters is measured rather than assumed. On the guide level a random
668-gene subset of the same screen raises the native pooled estimate from 0.050 to a
median of 0.102 across 200 draws, purely by making one endpoint a larger share of the
squared spread, while the standardised estimate stays at 0.047. A statistic that moves
when you change nothing but the number of genes is not measuring the measurement.

The verdict does not move: on the guide level the estimate goes from 0.0504 to 0.0474
against a floor of 0.30. The change is made because the two sides of a comparison have
to be formed the same way, not because a number needed rescuing.

Three properties of the estimate, each of which an earlier version got wrong:

* **The halves are responses, not profile means.** The same-screen non-targeting mean
  is subtracted, because that is what every other evidence column measures.
* **The split is inside a screen.** A gene's guides sit on nine plates here, so halves
  drawn across plates differ mostly by plate baseline rather than by reagent.
* **Pairing is per (gene, endpoint), not per gene.** Correlating each half's Euclidean
  norm collapses hundreds of coordinates to one non-negative number whose spread
  across genes is small, and two noisy estimates of a near-constant correlate at zero
  whatever the profiles do.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = [
    "ScreenHalves",
    "split_half_blocks",
    "reliability_by_block",
    "summarise_reliability",
    "spearman_brown",
]


def _nanmean(block: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """Column means over ``rows`` ignoring missing values, with 0/0 as NaN.

    ``pandas.DataFrame.mean`` skips missing values per column and yields NaN for a
    column missing throughout. ``numpy.nanmean`` does the same but warns on the
    all-missing column, and a warning that is expected on some columns is a warning
    nobody reads. The arithmetic is written out instead.
    """
    part = block[rows]
    finite = np.isfinite(part)
    count = finite.sum(axis=0)
    total = np.where(finite, part, 0.0).sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(count > 0, total / np.maximum(count, 1), np.nan)
    return mean.astype(np.float32)


@dataclass
class ScreenHalves:
    """Split-half and full gene responses for one block, control relative.

    ``half_a`` and ``half_b`` come from disjoint halves of each gene's sgRNAs, so they
    are two independent estimates of the same response. ``full`` uses every guide and
    is the analogue of the ``response_true`` that ``recoverability_by_fold``
    standardises by, which is why the divisor is taken from it rather than from either
    half: a divisor taken from one half would be correlated with that half.
    """

    block: str
    genes: np.ndarray
    half_a: np.ndarray
    half_b: np.ndarray
    full: np.ndarray
    n_guides: np.ndarray


def split_half_blocks(
    frame: pd.DataFrame,
    endpoints: list[str],
    *,
    seed: int,
    group_column: str | None = "screen_id",
    verbose: bool = True,
) -> list[ScreenHalves]:
    """Build the split-half response blocks once.

    ``group_column`` of ``None`` treats the whole table as one block, which is the
    pooled-across-screens variant reported beside the within-screen one so that the
    gap between them is visible rather than assumed.

    The gene order, the guide order and the order in which the generator is consumed
    are fixed so that a given seed gives one answer. Callers that need several gene
    subsets should build the blocks once and select rows, which is what makes a
    two-hundred-draw random control cost nothing.
    """
    rng = np.random.default_rng(seed)
    groups = (
        frame.groupby(group_column, sort=True) if group_column else [("all", frame)]
    )
    blocks: list[ScreenHalves] = []
    for name, group in groups:
        control_rows = np.flatnonzero(group["is_control"].to_numpy())
        if control_rows.size == 0:
            raise ValueError(f"block {name} has no non-targeting guides to subtract")
        values = group[endpoints].to_numpy(dtype=np.float32)
        baseline = _nanmean(values, control_rows)

        perturbed = np.flatnonzero(~group["is_control"].to_numpy())
        gene_codes = group["perturbation_id"].to_numpy()[perturbed]
        guide_codes = group["guide_id"].to_numpy()[perturbed]
        # A stable sort reproduces groupby(sort=True): groups in sorted key order and
        # rows inside a group in their original relative order, which is the order
        # Series.unique() walks when it decides which guides go in which half.
        order = np.argsort(gene_codes, kind="stable")
        gene_codes, guide_codes = gene_codes[order], guide_codes[order]
        rows_sorted = perturbed[order]
        starts = np.flatnonzero(np.r_[True, gene_codes[1:] != gene_codes[:-1]])
        bounds = np.r_[starts, len(gene_codes)]

        genes, half_a, half_b, full, counts = [], [], [], [], []
        for index in range(len(starts)):
            span = slice(bounds[index], bounds[index + 1])
            guides = pd.unique(guide_codes[span])
            if len(guides) < 2:
                continue
            permutation = rng.permutation(len(guides))
            cut = len(guides) // 2
            rows = rows_sorted[span]
            in_left = np.isin(guide_codes[span], guides[permutation[:cut]])
            genes.append(gene_codes[bounds[index]])
            half_a.append(_nanmean(values, rows[in_left]) - baseline)
            half_b.append(_nanmean(values, rows[~in_left]) - baseline)
            full.append(_nanmean(values, rows) - baseline)
            counts.append(len(guides))
        if not genes:
            continue
        blocks.append(
            ScreenHalves(
                block=str(name),
                genes=np.asarray(genes),
                half_a=np.vstack(half_a),
                half_b=np.vstack(half_b),
                full=np.vstack(full),
                n_guides=np.asarray(counts),
            )
        )
        if verbose:
            print(f"  {name}: {len(genes):,} genes with two or more sgRNAs", flush=True)
    if not blocks:
        raise ValueError("no block carried a gene with two or more sgRNAs")
    return blocks


def _pooled_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation over every finite pair, pooled with one grand mean.

    One grand mean rather than a mean per endpoint is what ``recoverability_by_fold``
    uses on its own block, and the two numbers are comparable only if formed alike.
    """
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3:
        return float("nan")
    left, right = a[mask].astype(np.float64), b[mask].astype(np.float64)
    left = left - left.mean()
    right = right - right.mean()
    denominator = np.sqrt(float(left @ left) * float(right @ right))
    return float(left @ right / denominator) if denominator > 0.0 else float("nan")


def _per_endpoint_correlation(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Correlation across genes within each endpoint, one value per column."""
    mask = np.isfinite(a) & np.isfinite(b)
    counts = mask.sum(axis=0)
    left = np.where(mask, a, 0.0).astype(np.float64)
    right = np.where(mask, b, 0.0).astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        left = np.where(mask, left - left.sum(axis=0) / np.maximum(counts, 1), 0.0)
        right = np.where(mask, right - right.sum(axis=0) / np.maximum(counts, 1), 0.0)
        numerator = (left * right).sum(axis=0)
        denominator = np.sqrt((left * left).sum(axis=0) * (right * right).sum(axis=0))
        return np.where((counts > 2) & (denominator > 0.0), numerator / denominator, np.nan)


def reliability_by_block(
    blocks: list[ScreenHalves],
    members: set[str] | None = None,
    *,
    min_spread_ratio: float = 1e-3,
) -> pd.DataFrame:
    """One row per block, under three pooling rules plus the unfiltered original.

    ``members`` of ``None`` keeps every gene; a set restricts the estimate to a
    pre-declared gene set. The columns mirror what ``recoverability_by_fold`` reports
    for the recoverability block, so a reporter's value and its ceiling can be read
    against each other rather than taken on trust:

    ``reliability``
        endpoints standardised by the observed spread of the full gene response before
        pooling. This is the quantity the rule should consume.
    ``reliability_pooled_native``
        the same kept endpoints, pooled in native units.
    ``reliability_median_endpoint``
        median of the per-endpoint correlations, which no pooling rule can distort.
    ``reliability_pooled_native_all_endpoints``
        native pooling with no endpoint dropped at all, which is what this study
        reported before 2026-08-15 and is kept so the change is checkable.
    """
    rows = []
    for block in blocks:
        keep_genes = (
            np.ones(len(block.genes), dtype=bool)
            if members is None
            else np.fromiter((gene in members for gene in block.genes), bool, len(block.genes))
        )
        if keep_genes.sum() < 3:
            continue
        half_a, half_b, full = block.half_a[keep_genes], block.half_b[keep_genes], block.full[keep_genes]
        native_all = _pooled_correlation(half_a.ravel(), half_b.ravel())

        spread = np.nanstd(full, axis=0, ddof=1)
        reference = np.nanstd(block.full, axis=0, ddof=1)
        finite = np.isfinite(spread) & (spread > 0.0)
        if not finite.any():
            continue
        keeps = finite & (spread > np.nanmedian(spread[finite]) * min_spread_ratio)
        if keeps.sum() < 2:
            continue
        kept_a, kept_b, divisor = half_a[:, keeps], half_b[:, keeps], spread[keeps]
        squared = divisor.astype(np.float64) ** 2
        usable = keeps & np.isfinite(reference) & (reference > 0.0)
        rows.append({
            "block": block.block,
            "n_genes": int(keep_genes.sum()),
            "n_endpoints": int(keeps.sum()),
            "n_paired_values": int((np.isfinite(kept_a) & np.isfinite(kept_b)).sum()),
            "median_guides_per_gene": float(np.median(block.n_guides[keep_genes])),
            "reliability": _pooled_correlation(
                (kept_a / divisor).ravel(), (kept_b / divisor).ravel()
            ),
            "reliability_pooled_native": _pooled_correlation(kept_a.ravel(), kept_b.ravel()),
            "reliability_median_endpoint": float(
                np.nanmedian(_per_endpoint_correlation(kept_a, kept_b))
            ),
            "reliability_pooled_native_all_endpoints": native_all,
            "largest_native_endpoint_share": float(squared.max() / squared.sum()),
            # An outcome, not a selection criterion: how much more response spread this
            # gene subset carries than the block as a whole, so a reliability that moved
            # can be read against whether the responses moved.
            "median_spread_vs_all_genes": (
                float(np.nanmedian(spread[usable] / reference[usable]))
                if usable.any() else float("nan")
            ),
        })
    return pd.DataFrame(rows)


def spearman_brown(half: float, *, guides_per_gene: float, floor: float) -> dict[str, float]:
    """What a split-half number implies for the full guide complement, and for the floor.

    A split-half correlation describes halves, not the whole, so reporting it against a
    floor invites the objection that it was halved by construction. ``2r / (1 + r)`` is
    the reliability of a measurement twice as long, which for a two-versus-two split is
    the gene's full guide complement.

    The same relation solved for length says how many sgRNAs per gene the reporter
    would need to reach the floor, which turns a failed gate into a design number. One
    caveat travels with it: Spearman-Brown assumes added measurements are exchangeable
    and independently noisy, while sgRNAs read on one plate share plate noise, so the
    real requirement is at least this large.
    """
    # A half-split of exactly 1.0 is admitted: the lengthened value is 1.0 and the
    # required length is zero, both of which are the right answers. Only a
    # non-positive or impossible correlation has no reading.
    if not np.isfinite(half) or half <= 0.0 or half > 1.0 or not 0.0 < floor < 1.0:
        return {
            "reliability_full_complement": float("nan"),
            "guides_per_gene_for_floor": float("nan"),
        }
    length = (floor * (1.0 - half)) / (half * (1.0 - floor))
    return {
        "reliability_full_complement": float(2.0 * half / (1.0 + half)),
        "guides_per_gene_for_floor": float(length * guides_per_gene / 2.0),
    }


def summarise_reliability(by_block: pd.DataFrame, *, floor: float) -> dict[str, float]:
    """Median over blocks, plus what the median implies for length.

    The median over screens is the study's existing reduction and is kept.

    Identifier columns are ignored by dtype rather than dropped by name. Dropping by
    name is what broke this: callers rename ``block`` to ``screen_id`` before writing
    their per-screen table, so a hard-coded drop passed in the tests that call
    :func:`reliability_by_block` directly and failed in the pipeline that does both.
    """
    if by_block.empty:
        return {"n_blocks": 0.0}
    numeric = by_block.median(numeric_only=True).to_dict()
    return (
        {key: float(value) for key, value in numeric.items()}
        | {"n_blocks": float(len(by_block))}
        | spearman_brown(
            float(numeric["reliability"]),
            guides_per_gene=float(numeric["median_guides_per_gene"]),
            floor=floor,
        )
    )
