#!/usr/bin/env python3
"""Render Figure 6 panel f: the frozen criteria applied outside the atlas.

The panel answers one question and is laid out so the answer cannot be misread:
**which gate decides**. Every row is drawn against the five evidence quantities the tier
rule consumes, each beside its own frozen threshold. Reading across a row shows what was
measured; reading down a column shows which quantity the verdict turned on. A figure
that showed only tier labels would hide the mechanism, which is the whole result.

Four structural decisions, each because the alternative would mislead.

**The funnel is part of the evidence, not decoration.** The claim that the criteria
apply broadly is not carried by the number of rows; it is carried by having judged the
public landscape against six machine-checkable capabilities and being able to say which
capability decided each rejection. The strip across the top reports that audit, and it
reports **how each judgement was reached**, because the audit itself records that eleven
Cell Painting Gallery datasets were rejected from a single sampled plate's channel
header and that this method mis-read two of them, one in each direction. Drawing an
automated pass in the same ink as an opened accession would convert a known-weak
decision into a fact.

**Three bands, and they must not share a numeric axis.** The first carries the datasets
admitted by the six capabilities. The second carries two negative controls. The third
carries a second substitution question asked of an admitted deposit, three Cell Painting
dyes standing in for the fourth: a generic organelle dye is not a targeted readout under
C2, which is the ground `cpg0000-jump-pilot` was rejected on, so those rows are not
admitted evidence and are not comparable to the rows above them. The prohibition is
recorded in ``configs/ops/figures/figure6.yaml``.

**Two rows are run on an input known to carry nothing, and both are needed.** A panel in
which every row fails says nothing about whether the criteria can tell an informative
input from an uninformative one. Each control shuffles the input block inside each
screen, leaving the targets, the perturbation labels and the control flags where they
are. On OASIS the reliability gate passes, so the rule reaches the input and the verdict
moves to `measurement required` with every performance quantity at chance, while
reliability returns bit-identical. On PERISCOPE the reliability gate fires first, so the
verdict cannot move at all. Reporting only the first would claim a detector the rule is
not; reporting only the second would understate it.

**The whiskers do not mean the same thing in every column.** The four fidelity
quantities carry one point per held-out gene fold. Reliability has no fold structure: it
is a property of the target measurement, estimated per screen for PERISCOPE and from
replicate compound wells for OASIS, where it is a single estimate and carries no ticks.

Every headline value is checked against the ``evidence.csv`` the tier rule consumed
before anything is drawn, and each plotted fold median is checked to be the value that
evidence table stores. A figure that agrees with its own recomputation but not with the
analysis it illustrates is worse than no figure.

Example:

    python studies/external/figures/figure6f_external_transfer.py \\
      --oasis ../external/oasis_analysis_v3 \\
      --oasis-batch prod_25=../external/oasis_analysis_axiom_prod_25 \\
      --periscope-cells ../external/periscope_cells_analysis_v3 \\
      --periscope-guide ../external/periscope_guide_analysis_v3 \\
      --stain-dropout ../external/periscope_stain_dropout \\
      --control ../external/periscope_guide_permuted \\
      --oasis-control ../external/oasis_analysis_permuted \\
      --output-dir ../external/figure6f
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from _common import reject_repo_paths  # noqa: E402

from measurement_sufficiency.analysis.tiers import (  # noqa: E402
    OPS_TIER_RULES,
    OPSTier,
    OPSTierRules,
    assign_ops_tier,
)
from measurement_sufficiency.metrics import ko_metrics  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
ADMISSION_AUDIT = REPO / "configs" / "ops" / "external" / "admission_audit.csv"

INK, MUTED, TRACK, BAND = "#1d2935", "#647582", "#e4e8ea", "#d7dee1"

#: The tier colours panel c already binds. Panel f reuses them for its verdict chips so
#: that a reader matching a chip here to a bar there does not need a second key. The
#: orange bound to "measurement required" is deliberately **not** reused as a failure
#: colour anywhere: it names a tier, and a reader who has just read panel c would take an
#: orange mark here to mean that tier rather than a missed threshold. A missed threshold
#: is drawn as the gap it is, in neutral ink.
TIER_COLOURS = {
    "quantitative proxy": "#167a76",
    "ranking proxy": "#4c86b6",
    "measurement required": "#e76f51",
    "not identifiable": "#8b979d",
    "unresolved": "#d7dee1",
}
#: Three sizes mapped to role, plus the panel letter, which is the documented
#: exception. An earlier version used five, which is two more than a figure can spend
#: before size stops meaning anything.
#:
#:   FS_BASE   panel title, band titles, column names, row labels, column headers
#:   FS_MID    values, verdict text, threshold subheaders, the audit line
#:   FS_SMALL  row sublabels, band notes, footnotes, in-bar labels
FS_LETTER, FS_BASE, FS_MID, FS_SMALL = 9.0, 7.0, 6.0, 5.0

#: Vertical layout, in points from the top. Written out rather than tuned by eye
#: because the panel's height is a composite decision made outside this script: the
#: caller sets ``--height-pt`` and everything here has to still fit inside it.
#: :func:`required_height` turns these into the minimum the row list needs, and the
#: renderer refuses a height below it rather than drawing rows off the bottom edge.
TOP_MATTER_PT = 84.0     # panel letter, title, audit strip and column headers
#: The same top matter without the audit strip: letter, title and column headers only.
#: The strip is evidence rather than decoration, so dropping it is a decision the
#: caller makes and the caption has to follow, not a layout convenience.
TOP_MATTER_NO_FUNNEL_PT = 42.0
BAND_HEADER_PT = 17.0    # band title and the rule under it; the note moved to the caption
#: 16, not 19. Only the deciding value is printed now, so a row carries its label, its
#: sublabel and its marks; the three numbers that used to sit beside every other mark
#: were what made the pitch a floor. Six rows shorter by 3 pt each is 18 pt back to the
#: panels above this one in the composite.
ROW_PITCH_PT = 16.0
BAND_GAP_PT = 7.0
#: No footnote, only the margin under the last row. The panel carried three lines of
#: key and then one; all of it is caption text, and the caption is where a reader of a
#: Nature figure looks for how to read a mark. What the caption must now say is the
#: reading of three marks: the stem is the value, the vertical mark is that quantity's
#: frozen threshold and is drawn in dark ink where it decided the row, and an open
#: marker means the threshold was not met. That sentence is recorded in
#: the canonical final Figure 6 source package with the rest of what left the panel.
FOOTNOTE_PT = 6.0

#: OASIS restricts every fidelity quantity to the plate-normalised readouts; the raw
#: luminescence and absorbance carry plate gain drift and would dominate a norm.
OASIS_PRIMARY_ENDPOINTS = ("mtt_normalized", "ldh_normalized")


@dataclass(frozen=True)
class Row:
    """One row of the panel, and where its numbers come from."""

    key: str          # "<source>:<reporter_id>"
    band: str         # "admitted" or "second_question"
    label: str
    sublabel: str


#: (key, band title, the sentence the caption must carry for that band).
#:
#: The third element is no longer drawn. Each was a line of methodology under its band
#: title, and methodology is caption text: what the reader needs beside the rows is
#: which class of evidence they are, not how it was produced. They are kept here rather
#: than deleted because the caption has to say them, and this is where they are written.
BANDS = (
    ("admitted",
     "Admitted",
     "admitted by the six capabilities; the frozen rule applied unchanged, with no tuning"),
    ("control",
     "Negative controls",
     "the same targets with an input known to carry nothing: inputs shuffled within "
     "screen, so reliability reads the target side and is unchanged by construction"),
    ("second_question",
     "A second substitution question",
     "asked of an admitted deposit and not admitted evidence, because a generic "
     "organelle dye is not a targeted readout under C2"),
)

ROWS = (
    # The deposit's column prefix is "mtt", but the values are a luminescence reading
    # and the published abstract says only "two cytotoxicity assays". The row is
    # labelled by what is measured, not by an assay brand inferred from a column name.
    Row("oasis:mtt_viability", "admitted",
        "metabolic activity", "OASIS cpg0037 · brightfield · per compound"),
    Row("oasis:ldh_release", "admitted",
        "LDH release", "OASIS cpg0037 · brightfield · per compound"),
    Row("periscope_cells:tomm20_mito", "admitted",
        "anti-TOMM20", "PERISCOPE cpg0021 · four dyes · per cell"),
    Row("periscope_guide:tomm20_mito", "admitted",
        "anti-TOMM20", "PERISCOPE cpg0021 · four dyes · per guide"),
    # The two controls make different points and both are needed. On OASIS the rule
    # reaches the input, so the verdict moves and names the right cause. On PERISCOPE
    # the reliability gate fires first, so the verdict cannot move at all, which is the
    # clearest statement of what that gate does.
    Row("oasis_control:mtt_viability", "control",
        "metabolic activity, shuffled", "OASIS cpg0037 · inputs shuffled"),
    Row("control:tomm20_mito", "control",
        "anti-TOMM20, shuffled", "PERISCOPE cpg0021 · inputs shuffled"),
    Row("dropout_dapi:stain_dropout_dapi", "second_question",
        "DAPI, nucleus", "predicted from ConA, phalloidin, WGA"),
    Row("dropout_cona:stain_dropout_cona", "second_question",
        "ConA, ER", "predicted from DAPI, phalloidin, WGA"),
    Row("dropout_phalloidin:stain_dropout_phalloidin", "second_question",
        "phalloidin, F-actin", "predicted from DAPI, ConA, WGA"),
    Row("dropout_wga:stain_dropout_wga", "second_question",
        "WGA, Golgi and membrane", "predicted from DAPI, ConA, phalloidin"),
)

#: Column names are panel b's names for the same quantities, which is what lets a
#: reader carry "Recovery", "Rank" and "Reliability" across the two panels without a
#: second key, and they are the only names that fit: measured at 7 pt, "Recoverability"
#: is 51.5 pt, "Amplitude rank" 54.7 and "Target reliability" 57.6, against a column
#: 53.9 pt wide on the composite.
QUANTITIES = (
    ("recoverability_r", "Recovery", "$\\geq$ 0.70", "min"),
    ("magnitude_spearman", "Rank", "$\\geq$ 0.70", "min"),
    ("variance_ratio", "Variance ratio", "0.50 to 1.50", "band"),
    # Both bars, because the rule reads recall against both, but written as two numbers
    # rather than as a phrase: at the composite's 518.74 pt this subheader collided with
    # the reliability column's, and the words it lost are restated in the footnote and
    # in the caption. The track carries both marks either way.
    ("top5pct_recall", "Top-5% recall", "$\\geq$ 0.60 / 0.50", "min"),
    ("reliability", "Reliability", "$\\geq$ 0.30", "min"),
)

#: Segments of the admission funnel, left to right. "not judged" sits last so that the
#: judged resources occupy a contiguous run and the evidence-tier bar beneath can span
#: exactly them.
FUNNEL_SEGMENTS = (
    ("rejected on C1", "no cheap broad input"),
    ("rejected on C2", "no targeted readout"),
    ("rejected on C3 or C4", "no same-cell pairing, or no perturbation"),
    ("rejected on access or economics", "proprietary, absent deposit, or no cost gap"),
    ("blocked", "passes all six, open questions"),
    ("admitted", ""),
    ("not judged", "leads unverified, or searched and no accession found"),
)
#: Evidence tiers, strongest first. Encoded by fill and hatch rather than by hue,
#: because hue is spent on the tier chips and a second hue scale would compete with it.
EVIDENCE_TIERS = (
    ("opened", INK, None),
    ("by definition", MUTED, None),
    ("automated pass", BAND, None),
    ("sampled header", BAND, "////"),
)


def thresholds() -> dict[str, tuple[float, float | None]]:
    """The frozen thresholds, read from the rule rather than restated here."""
    rules = OPS_TIER_RULES
    return {
        "recoverability_r": (rules.recoverability_min, None),
        "magnitude_spearman": (rules.magnitude_spearman_min, None),
        "variance_ratio": (rules.variance_ratio_min, rules.variance_ratio_max),
        "top5pct_recall": (rules.top5pct_quantitative_min, None),
        "reliability": (rules.reliability_min, None),
    }


def passes(value: float, bounds: tuple[float, float | None]) -> bool:
    low, high = bounds
    return float(value) >= low if high is None else low <= float(value) <= high


def oasis_recoverability_by_fold(analysis: Path) -> pd.DataFrame:
    """Per-fold recoverability for OASIS, recomputed through the library primitive.

    The OASIS workflow keeps only the fold median in ``evidence.csv``. Rather than
    invent a second definition for the whiskers, this calls the same ``ko_metrics`` the
    workflow calls and the caller checks that its median reproduces the stored value; if
    it does not, the two definitions have drifted and drawing either would be wrong.
    """
    predictions = pd.read_csv(analysis / "predictions.csv")
    scored = predictions.loc[predictions["endpoint_id"].isin(OASIS_PRIMARY_ENDPOINTS)]
    metrics = ko_metrics(scored)
    column = "pearson_r" if "pearson_r" in metrics else "r"
    if column not in metrics:
        raise SystemExit(f"ko_metrics returned no correlation column; got {list(metrics.columns)}")
    return metrics.rename(columns={column: "recoverability_r"})[
        ["reporter_id", "fold", "recoverability_r"]
    ]


def fold_tables(directory: Path) -> dict[str, pd.DataFrame]:
    """Per-fold tables, dispatching on which workflow wrote the directory.

    The two workflows name their fidelity outputs differently, and the OASIS one keeps
    no per-fold recoverability at all. Dispatching on the files that are actually there
    beats keying on the row's dataset, because the negative controls are run by both
    workflows and would otherwise need a second lookup table to stay in step.
    """
    if (directory / "fidelity_magnitude_spearman.csv").is_file():
        return {
            "recoverability_r": oasis_recoverability_by_fold(directory),
            "magnitude_spearman": pd.read_csv(directory / "fidelity_magnitude_spearman.csv"),
            "variance_ratio": pd.read_csv(directory / "fidelity_variance_ratio.csv"),
            "top5pct_recall": pd.read_csv(directory / "fidelity_strong_hit_recall.csv"),
        }
    return {
        "recoverability_r": pd.read_csv(directory / "recoverability_by_fold.csv"),
        "magnitude_spearman": pd.read_csv(directory / "magnitude_spearman.csv"),
        "variance_ratio": pd.read_csv(directory / "variance_ratio.csv"),
        "top5pct_recall": pd.read_csv(directory / "strong_hit_recall.csv"),
    }


def resolve_sources(args: argparse.Namespace) -> dict[str, Path]:
    """Map each row's source key to the analysis directory that produced it."""
    dropout = args.stain_dropout
    sources = {
        "oasis": args.oasis,
        "periscope_cells": args.periscope_cells,
        "periscope_guide": args.periscope_guide,
        "control": args.control,
        "oasis_control": args.oasis_control,
    }
    for dye in ("dapi", "cona", "phalloidin", "wga"):
        sources[f"dropout_{dye}"] = dropout / dye
    missing = [key for key, path in sources.items() if not (path / "evidence.csv").is_file()]
    if missing:
        raise SystemExit(
            f"no evidence.csv for {missing}. Every row must come from an analysis "
            "directory; the panel does not compute its own numbers."
        )
    return sources


def check_control_is_a_control(directory: Path) -> dict:
    """Refuse to draw a control row from a directory that is not a control run.

    ``analyse_guide`` records ``run_kind`` in its provenance. Pointing this argument at
    an ordinary run would draw a real result under a label saying its input carried
    nothing, which is the one mislabelling this panel cannot survive.
    """
    import json

    path = directory / "provenance.json"
    if not path.is_file():
        raise SystemExit(f"{directory} has no provenance.json, so it cannot be shown as a control")
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("run_kind") != "inputs_permuted_within_screen":
        raise SystemExit(
            f"{directory} declares run_kind={record.get('run_kind')!r}; the control row "
            "may only be drawn from a run whose inputs were permuted"
        )
    return record


def load_rows(sources: dict[str, Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the headline table and the per-replicate table behind the ticks."""
    headline, replicates = [], []
    evidence = {
        key: pd.read_csv(path / "evidence.csv").set_index("reporter_id")
        for key, path in sources.items()
    }
    per_fold, reliability_replicates = {}, {}
    for key, path in sources.items():
        per_fold[key] = fold_tables(path)
        screens = path / "reliability_within_screen.csv"
        if screens.is_file():
            reliability_replicates[key] = pd.read_csv(screens)

    bounds = thresholds()
    for row in ROWS:
        source, reporter = row.key.split(":", 1)
        if reporter not in evidence[source].index:
            raise SystemExit(
                f"{row.key}: {sources[source] / 'evidence.csv'} has no reporter {reporter!r}; "
                f"it carries {list(evidence[source].index)}"
            )
        stored = evidence[source].loc[reporter]
        for quantity, _, _, _ in QUANTITIES:
            value = float(stored[quantity])
            headline.append({
                "row_key": row.key, "band": row.band, "label": row.label,
                "sublabel": row.sublabel, "quantity": quantity, "value": value,
                "threshold_min": bounds[quantity][0], "threshold_max": bounds[quantity][1],
                "passes": passes(value, bounds[quantity]),
            })
            if quantity == "reliability":
                continue
            table = per_fold[source][quantity]
            table = table.loc[table["reporter_id"] == reporter]
            if table.empty:
                raise SystemExit(f"{row.key}: no per-fold rows for {quantity}")
            # The median over folds is what the evidence table stores. Checking it here
            # is what makes the ticks and the point the same analysis.
            recomputed = float(table[quantity].median())
            if not np.isclose(recomputed, value, rtol=0, atol=1e-9):
                raise SystemExit(
                    f"{row.key}: the fold median of {quantity} is {recomputed!r} but "
                    f"evidence.csv stores {value!r}. The panel would not be showing the "
                    "analysis it illustrates."
                )
            for _, point in table.iterrows():
                replicates.append({
                    "row_key": row.key, "quantity": quantity, "replicate_kind": "gene fold",
                    "replicate_id": point["fold"], "value": float(point[quantity]),
                })
        if source in reliability_replicates:
            screens = reliability_replicates[source]
            screens = screens.loc[screens["reporter_id"] == reporter]
            for _, point in screens.iterrows():
                replicates.append({
                    "row_key": row.key, "quantity": "reliability", "replicate_kind": "screen",
                    "replicate_id": point["screen_id"], "value": float(point["reliability"]),
                })
    return pd.DataFrame(headline), pd.DataFrame(replicates)


def batch_verdicts(batches: dict[str, Path]) -> pd.DataFrame:
    """One verdict per (production batch, reporter), for the chips beside the OASIS rows.

    Eight further rows would answer the same question the chips answer, which is whether
    the verdict is a property of the assay or of one production run, and would cost the
    panel its legibility. The five evidence quantities behind each chip are written to
    the per-panel source data and tabulated in the supplement.
    """
    rows = []
    for name, path in sorted(batches.items()):
        table = pd.read_csv(path / "evidence.csv")
        for _, record in table.iterrows():
            rows.append({
                "batch": name, "reporter_id": record["reporter_id"],
                "tier": assign_ops_tier(record).value.replace("_", " "),
                **{quantity: float(record[quantity]) for quantity, _, _, _ in QUANTITIES},
            })
    return pd.DataFrame(rows)


def admission_funnel(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Counts for the funnel strip, read from the audit table rather than restated."""
    with path.open(encoding="utf-8", newline="") as handle:
        table = list(csv.DictReader(handle))

    def bucket(record: dict[str, str]) -> str:
        if record["outcome"] in {"unverified", "absent"}:
            return "not judged"
        if record["outcome"].startswith("admitted"):
            return "admitted"
        if record["outcome"] == "blocked":
            return "blocked"
        decided = record["decided_by"]
        if decided == "C1":
            return "rejected on C1"
        if decided == "C2":
            return "rejected on C2"
        if decided in {"C3", "C4", "C5", "C6"}:
            return "rejected on C3 or C4"
        return "rejected on access or economics"

    counts = pd.Series([bucket(record) for record in table]).value_counts()
    segments = pd.DataFrame(
        [{"segment": name, "note": note, "n": int(counts.get(name, 0))}
         for name, note in FUNNEL_SEGMENTS]
    )
    if segments["n"].sum() != len(table):
        raise SystemExit(
            f"the funnel segments cover {segments['n'].sum()} of {len(table)} audit rows"
        )
    judged = [record for record in table if record["outcome"] not in {"unverified", "absent"}]
    naming = {"opened": "opened", "by_definition": "by definition", "agent": "automated pass",
              "sampled_header": "sampled header"}
    tiers = pd.Series([naming[record["evidence"]] for record in judged]).value_counts()
    evidence = pd.DataFrame(
        [{"tier": name, "n": int(tiers.get(name, 0))} for name, _, _ in EVIDENCE_TIERS]
    )
    if evidence["n"].sum() != len(judged):
        raise SystemExit("the evidence tiers do not cover every judged resource")
    return segments, evidence


#: The five frames :func:`draw` consumes, and the files ``main`` writes them to. Reading
#: them back is what lets the panel be redrawn at a different size without the external
#: analysis directories, which are produced on a compute node and are not distributed.
SOURCE_DATA_FILES = {
    "headline": "figure_source_data.csv",
    "replicates": "replicate_detail.csv",
    "segments": "admission_funnel.csv",
    "evidence": "admission_evidence_tiers.csv",
    "batches": "batch_verdicts.csv",
}
SOURCE_DATA_COLUMNS = {
    "headline": ("row_key", "band", "label", "sublabel", "quantity", "value",
                 "threshold_min", "threshold_max", "passes"),
    "replicates": ("row_key", "quantity", "replicate_kind", "replicate_id", "value"),
    "segments": ("segment", "note", "n"),
    "evidence": ("tier", "n"),
    "batches": ("batch", "reporter_id", "tier"),
}


def load_from_source_data(
    directory: Path, *, rows: tuple[Row, ...] | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Re-read the five frames this panel was drawn from, and re-check them.

    :func:`load_rows` reads the analysis directories and checks, for every quantity with
    fold structure, that the median over folds is the value ``evidence.csv`` stores.
    That check is not lost here: ``replicate_detail.csv`` holds those same per-fold
    values and ``figure_source_data.csv`` holds the same headline value, so the median
    is recomputed and compared exactly as before. What this path cannot check is that
    the two saved files still agree with the ``evidence.csv`` they came from, which is
    what the digests recorded in the package manifest are for.

    Three further checks run because a saved table can go stale in ways the original
    cannot: the stored thresholds must equal the frozen rule's thresholds today, the
    stored pass flags must follow from the stored values, and each batch verdict must be
    what :func:`assign_ops_tier` returns from the five quantities beside it.
    """
    rows = ROWS if rows is None else tuple(rows)
    frames: dict[str, pd.DataFrame] = {}
    for name, filename in SOURCE_DATA_FILES.items():
        path = directory / filename
        if not path.is_file():
            raise SystemExit(
                f"missing {path}. --from-source-data expects the five files this script "
                f"writes beside a rendered panel: {', '.join(SOURCE_DATA_FILES.values())}."
            )
        frame = pd.read_csv(path)
        missing = [c for c in SOURCE_DATA_COLUMNS[name] if c not in frame.columns]
        if missing:
            raise SystemExit(f"{path} is missing column(s) {missing}")
        frames[name] = frame

    headline, replicates, batches = frames["headline"], frames["replicates"], frames["batches"]
    bounds = thresholds()
    for row in rows:
        block = headline.loc[headline["row_key"] == row.key]
        if block.empty:
            raise SystemExit(
                f"{directory / SOURCE_DATA_FILES['headline']} has no row {row.key!r}; "
                f"it carries {sorted(headline['row_key'].unique())}"
            )
        present = set(block["quantity"])
        wanted = {quantity for quantity, _, _, _ in QUANTITIES}
        if present != wanted:
            raise SystemExit(f"{row.key}: quantities {sorted(present)}, expected {sorted(wanted)}")
        for _, cell in block.iterrows():
            quantity = cell["quantity"]
            low, high = bounds[quantity]
            stored_low = float(cell["threshold_min"])
            if not np.isclose(stored_low, low, rtol=0, atol=1e-12):
                raise SystemExit(
                    f"{row.key} {quantity}: the saved threshold is {stored_low!r} but the "
                    f"frozen rule now says {low!r}. The panel would be drawing thresholds "
                    "the analysis was not judged against."
                )
            stored_high = cell["threshold_max"]
            if (high is None) != bool(pd.isna(stored_high)):
                raise SystemExit(f"{row.key} {quantity}: upper threshold disagrees with the rule")
            recomputed = passes(float(cell["value"]), bounds[quantity])
            if bool(cell["passes"]) != recomputed:
                raise SystemExit(
                    f"{row.key} {quantity}: the saved table says passes={cell['passes']!r}, "
                    f"the frozen rule says {recomputed!r}"
                )
            if quantity == "reliability":
                continue
            spread = replicates.loc[
                (replicates["row_key"] == row.key) & (replicates["quantity"] == quantity)
            ]
            if spread.empty:
                raise SystemExit(f"{row.key}: no per-replicate rows for {quantity}")
            median = float(spread["value"].median())
            if not np.isclose(median, float(cell["value"]), rtol=0, atol=1e-9):
                raise SystemExit(
                    f"{row.key}: the replicate median of {quantity} is {median!r} but the "
                    f"headline table stores {cell['value']!r}. The panel would not be "
                    "showing the analysis it illustrates."
                )

    for _, record in batches.iterrows():
        stored = str(record["tier"])
        quantities = {q: float(record[q]) for q, _, _, _ in QUANTITIES if q in record}
        if len(quantities) != len(QUANTITIES):
            continue
        recomputed = assign_ops_tier(quantities).value.replace("_", " ")
        if stored != recomputed:
            raise SystemExit(
                f"batch {record['batch']} {record['reporter_id']}: the saved verdict is "
                f"{stored!r}, the frozen rule returns {recomputed!r}"
            )

    return headline, replicates, frames["segments"], frames["evidence"], batches


def required_height(rows: tuple[Row, ...] | None = None, *, show_funnel: bool = True) -> float:
    """The least height in points that a row list can be drawn in.

    Ten rows in three bands do not fit the 188.6 pt the first version of this panel
    used, and a height that is merely too small draws the last band off the bottom
    edge, which is the kind of failure a reader of the composite discovers and the
    author does not. The number is derived from the row list rather than asserted, so
    adding a row moves it.

    ``rows`` defaults to the module's full list, read at call time so that a caller can
    substitute one. ``show_funnel`` selects the shorter top matter used when the panel
    is drawn without the admission strip, which is a choice about what evidence the
    panel carries and never a way to make an oversized row list fit.
    """
    rows = ROWS if rows is None else tuple(rows)
    bands = {name for name, _, _ in BANDS if any(row.band == name for row in rows)}
    return (
        (TOP_MATTER_PT if show_funnel else TOP_MATTER_NO_FUNNEL_PT)
        + len(bands) * BAND_HEADER_PT
        + len(rows) * ROW_PITCH_PT
        + max(len(bands) - 1, 0) * BAND_GAP_PT
        + FOOTNOTE_PT
    )


def _tier_of(row_key: str, headline: pd.DataFrame) -> str:
    block = headline.loc[headline["row_key"] == row_key]
    return assign_ops_tier(dict(zip(block["quantity"], block["value"]))).value.replace("_", " ")


def format_value(value: float) -> str:
    """Three decimals, except where that would print a signed zero.

    ``-0.000442`` rendered as ``-0.000``, which reads as negative zero and looks like
    a formatting accident. Three decimals are kept rather than the two significant
    figures the style would otherwise ask for, because two would print the PERISCOPE
    single-cell recoverability and the DAPI dye recoverability, 0.760 and 0.764, as the
    same number in two different bands.
    """
    if value != 0.0 and abs(round(value, 3)) < 5e-4:
        return f"{value:.1g}"
    return f"{value:.3f}"


def deciding_quantities(evidence: dict[str, float], rules: OPSTierRules = OPS_TIER_RULES) -> set[str]:
    """The quantities whose failure produced this verdict, by replaying the rule.

    The panel's whole claim is that *which* threshold decides is the result, so the
    deciding cell has to be marked rather than left for the reader to reconstruct from
    five numbers and a precedence order. This walks the same precedence
    :func:`assign_ops_tier` walks and returns the minimal set: the quantities such that
    clearing them would have changed the verdict.

    Returns an empty set for a quantitative proxy, where nothing failed.
    """
    reliability = evidence.get("reliability")
    if reliability is None or pd.isna(reliability) or float(reliability) < rules.reliability_min:
        return {"reliability"}
    recoverability = evidence.get("recoverability_r")
    if recoverability is not None and not pd.isna(recoverability) and (
        float(recoverability) < rules.measurement_recoverability_max_exclusive
    ):
        return {"recoverability_r"}

    tier = assign_ops_tier(evidence, rules)
    if tier is OPSTier.QUANTITATIVE_PROXY:
        return set()
    quantitative_failures = {
        name for name, ok in (
            ("recoverability_r", float(evidence["recoverability_r"]) >= rules.recoverability_min),
            ("magnitude_spearman", float(evidence["magnitude_spearman"]) >= rules.magnitude_spearman_min),
            ("variance_ratio", rules.variance_ratio_min
             <= float(evidence["variance_ratio"]) <= rules.variance_ratio_max),
            ("top5pct_recall", float(evidence["top5pct_recall"]) >= rules.top5pct_quantitative_min),
        ) if not ok
    }
    if tier is OPSTier.RANKING_PROXY:
        # It is a proxy; what is marked is why it is not the quantitative one.
        return quantitative_failures
    # Unresolved: it also failed the ranking test, and clearing those is what would
    # have moved it. Both are marked when both fail, because neither alone suffices.
    ranking_failures = {
        name for name, ok in (
            ("magnitude_spearman", float(evidence["magnitude_spearman"]) >= rules.magnitude_spearman_min),
            ("top5pct_recall", float(evidence["top5pct_recall"]) >= rules.top5pct_ranking_min),
        ) if not ok
    }
    return ranking_failures or quantitative_failures


def draw(
    headline: pd.DataFrame,
    replicates: pd.DataFrame,
    segments: pd.DataFrame,
    evidence: pd.DataFrame,
    batches: pd.DataFrame,
    output_dir: Path,
    *,
    width_pt: float,
    height_pt: float,
    rows: tuple[Row, ...] | None = None,
    show_funnel: bool = True,
    show_letter: bool = True,
) -> list[Path]:
    """One reading grid, its bands, and the audit that produced the rows above them.

    Writes the standalone panel. The drawing itself lives in :func:`render_panel`, so
    that the figure-6 composite can host it in a sub-rectangle of the page rather than
    keeping a second copy of 200 lines that would drift from this one.
    """
    rows = ROWS if rows is None else tuple(rows)
    needed = required_height(rows, show_funnel=show_funnel)
    if height_pt < needed:
        raise SystemExit(
            f"--height-pt {height_pt:g} cannot hold {len(rows)} rows in "
            f"{len({row.band for row in rows})} bands; {needed:.0f} pt is the minimum. "
            "The panel's height is a composite decision, so this is reported rather "
            "than silently rescaled: shrinking the rows to fit would make this panel "
            "disagree with its siblings about what a row is."
        )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
        "font.size": FS_BASE, "svg.fonttype": "none", "pdf.fonttype": 42,
        "text.color": INK, "hatch.linewidth": 0.35,
    })

    fig = plt.figure(figsize=(width_pt / 72.0, height_pt / 72.0))
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
    render_panel(ax, headline, replicates, segments, evidence, batches,
                 height_pt=height_pt, rows=rows, show_funnel=show_funnel,
                 show_letter=show_letter)

    verify(fig)
    output_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for suffix in ("pdf", "svg", "png"):
        path = output_dir / f"figure6F_external_transfer.{suffix}"
        # No tight bounding box: the composite needs the panel to be exactly the
        # requested size, and a tight box silently resizes it to the content.
        fig.savefig(path, dpi=600 if suffix == "png" else None, facecolor="white")
        written.append(path)
    plt.close(fig)
    return written


def render_panel(
    ax,
    headline: pd.DataFrame,
    replicates: pd.DataFrame,
    segments: pd.DataFrame,
    evidence: pd.DataFrame,
    batches: pd.DataFrame,
    *,
    height_pt: float,
    rows: tuple[Row, ...] | None = None,
    show_funnel: bool = True,
    show_letter: bool = True,
) -> None:
    """Draw the panel into ``ax``, whose box is taken to be the panel rectangle.

    ``height_pt`` is the height of that rectangle, not of the page it sits on. Every
    vertical position is written in points from the top of the rectangle and converted
    once, so hosting the panel in a composite moves it without rescaling any type.
    """
    rows = ROWS if rows is None else tuple(rows)
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # The layout is written in points from the top and converted once, so that changing
    # the panel height rescales nothing and simply moves the floor.
    def Y(points: float) -> float:
        return 1.0 - points / height_pt

    def H(points: float) -> float:
        return points / height_pt

    # Fractions of the panel width, cut against measured text rather than guessed. At
    # 518.74 pt the longest sublabel needs a 121 pt label column once its real rendered
    # width and 5 pt breathing room are included.  The earlier 0.222 cut was based on
    # a nominal 104.3 pt measurement and let the two OASIS sublabels enter the first
    # evidence track by 5 pt in the final composite.  Moving the track start to 0.240
    # preserves the complete five-track reading grid while keeping every label clear.
    label_x, col_x0, col_x1 = 0.010, 0.240, 0.777
    col_w = (col_x1 - col_x0) / len(QUANTITIES)
    track_pad, track_frac = 0.007, 0.600
    swatch_x, swatch_w, verdict_x = 0.785, 0.009, 0.799
    batch_x0, batch_w = 0.945, 0.0104

    # Lower case, to match the sibling panels and the caption's \textbf{f,}. Omitted
    # when the panel is published on its own, where it would name a panel of one.
    if show_letter:
        ax.text(label_x - 0.004, Y(4), "f", fontsize=FS_LETTER, fontweight="bold",
                va="top", ha="left")
    # Not "a different threshold decides each verdict": several rows are decided by the
    # same gate. What varies, and what the panel is for, is which one.
    # Panels a to e title themselves in a short noun phrase with an n after a middot.
    # This one carried a 94-character claim sentence, three times any sibling. The claim
    # belongs in the caption, which states it; the panel says what it shows.
    ax.text(label_x + (0.022 if show_letter else -0.004), Y(5),
            "Frozen rules outside the atlas · 2 public screens",
            fontsize=FS_BASE, fontweight="bold", va="top", ha="left")

    # ---- the admission audit --------------------------------------------------
    # One bar, not two. The first version stacked a second bar above this one, segmented
    # by which capability decided each resource, and it could not be read: the C1 and C2
    # segments carried the same fill, as did the C3/C4 and access segments, so a reader
    # could only tell them apart by counting. Those counts are now a sentence, which is
    # what they always were. What stays as a bar is the one proportional claim a reader
    # should feel rather than read: most of the empty-set argument rests on an automated
    # pass, not on an opened accession.
    #
    # The strip is optional only in the sense that a composite may not have room for it.
    # It is the evidence for the claim that the criteria apply broadly, so a caller that
    # drops it owes the reader that evidence somewhere else.
    if show_funnel:
        total = int(segments["n"].sum())
        judged = int(segments.loc[segments["segment"] != "not judged", "n"].sum())
        counts = {row["segment"]: int(row["n"]) for _, row in segments.iterrows()}
        ax.text(label_x, Y(22),
                f"{total} public resources considered, {judged} judged against the six "
                f"capabilities, {counts['admitted']} admitted, {counts['not judged']} not judged.",
                fontsize=FS_MID, va="bottom", ha="left", color=INK)
        ax.text(label_x, Y(31),
                f"Rejected on: {counts['rejected on C1']} no cheap input, "
                f"{counts['rejected on C2']} no targeted readout, "
                f"{counts['rejected on C3 or C4']} pairing or perturbation, "
                f"{counts['rejected on access or economics']} access or economics; "
                f"{counts['blocked']} met all six and was set aside.",
                fontsize=FS_SMALL, va="bottom", ha="left", color=MUTED)

        bar_x0, bar_w = label_x, 0.800 - label_x
        cursor = bar_x0
        for name, fill, hatch in EVIDENCE_TIERS:
            count = int(evidence.loc[evidence["tier"] == name, "n"].iloc[0])
            width = bar_w * count / judged
            ax.add_patch(plt.Rectangle(
                (cursor, Y(42)), width, H(7), facecolor=fill, hatch=hatch,
                edgecolor=MUTED, linewidth=0.4, zorder=2))
            # Direct labels inside each segment, so the bar needs no key. Text colour is
            # chosen for contrast against its own fill, not by hue family: white on the two
            # dark fills reaches 14.8:1 and 4.8:1, ink on the light fill reaches 10.9:1.
            # A hatched fill is not a background text can sit on, so the label on that one
            # segment carries a white halo. The bbox check in verify() is text-against-text
            # and would not have caught it; the crop inspection did.
            ax.text(cursor + width / 2, Y(38.5), f"{count} {name}", fontsize=FS_SMALL,
                    va="center", ha="center", zorder=3,
                    color="white" if fill in (INK, MUTED) else INK,
                    bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.8}
                    if hatch else None)
            cursor += width
        ax.text(label_x, Y(52),
                f"How each of the {judged} judgements was reached.  The sampled-header "
                "method mis-read two datasets during the audit itself, which is why it is "
                "counted apart from the rest.",
                fontsize=FS_SMALL, va="bottom", ha="left", color=MUTED)

    # ---- column headers -------------------------------------------------------
    # The headers sit a fixed distance above the first band, so dropping the audit strip
    # moves them up with it rather than leaving a hole where the strip was.
    top_matter_pt = TOP_MATTER_PT if show_funnel else TOP_MATTER_NO_FUNNEL_PT
    header_pt, subheader_pt = top_matter_pt - 20.0, top_matter_pt - 11.0
    for index, (quantity, name, bar, _) in enumerate(QUANTITIES):
        left = col_x0 + index * col_w + track_pad
        ax.text(left, Y(header_pt), name, fontsize=FS_BASE, va="center", ha="left")
        ax.text(left, Y(subheader_pt), bar, fontsize=FS_MID, va="center", ha="left", color=MUTED)
    ax.text(swatch_x, Y(header_pt), "Verdict", fontsize=FS_BASE, va="center", ha="left")
    ax.text(batch_x0 + 1.9 * batch_w, Y(header_pt), "Batches", fontsize=FS_BASE,
            va="center", ha="center")


    # ---- the bands ------------------------------------------------------------
    band_rows = {name: [row for row in rows if row.band == name] for name, _, _ in BANDS}
    cursor_pt = top_matter_pt
    row_pt: dict[str, float] = {}
    for name, title, _caption_sentence in BANDS:
        if not band_rows[name]:
            continue
        ax.text(label_x, Y(cursor_pt), title, fontsize=FS_BASE, fontweight="bold",
                va="center", ha="left")
        ax.plot([label_x, 0.985], [Y(cursor_pt + 8)] * 2, color=TRACK, linewidth=0.7,
                solid_capstyle="butt")
        cursor_pt += BAND_HEADER_PT
        for row in band_rows[name]:
            row_pt[row.key] = cursor_pt
            cursor_pt += ROW_PITCH_PT
        cursor_pt += BAND_GAP_PT

    row_labels: list = []
    for row in rows:
        y = Y(row_pt[row.key])
        row_labels.append(
            ax.text(label_x, y + H(2.8), row.label, fontsize=FS_BASE, va="center", ha="left")
        )
        row_labels.append(
            ax.text(label_x, y - H(3.9), row.sublabel, fontsize=FS_SMALL, va="center",
                    ha="left", color=MUTED)
        )

        block = headline.loc[headline["row_key"] == row.key]
        deciding = deciding_quantities(dict(zip(block["quantity"], block["value"])))

        for col_index, (quantity, _, _, comparison) in enumerate(QUANTITIES):
            cell = headline.loc[
                (headline["row_key"] == row.key) & (headline["quantity"] == quantity)
            ].iloc[0]
            # 1.6 rather than 2.0 for the band column: on a 0 to 2 track the six values
            # reach 0.897 and use 44 per cent of it, which is the case the style rule
            # about wasted axis range exists for. 1.6 keeps the band's upper edge visible.
            top = 1.6 if quantity == "variance_ratio" else 1.0
            x0 = col_x0 + col_index * col_w + track_pad
            width = col_w * track_frac

            def position(value: float, _x0: float = x0, _w: float = width, _top: float = top) -> float:
                return _x0 + _w * min(max(float(value), 0.0), _top) / _top

            decides = quantity in deciding
            if comparison == "band":
                # Drawn as a band with its centre marked, not as a line to clear. The
                # rule reads this quantity two-sidedly and the four others one-sidedly,
                # so they must not look the same.
                ax.add_patch(plt.Rectangle(
                    (position(cell["threshold_min"]), y - H(4.6)),
                    position(cell["threshold_max"]) - position(cell["threshold_min"]),
                    H(9.2), facecolor=BAND, edgecolor="none", zorder=1))
                centre = position(1.0)
                ax.plot([centre, centre], [y - H(4.6), y + H(4.6)], color=MUTED,
                        linewidth=0.5, alpha=0.7, zorder=3)
            else:
                mark = position(cell["threshold_min"])
                # The threshold that decided this row is drawn in ink, the rest in the
                # neutral. That is the panel's whole claim made visible: without it a
                # reader has to reconstruct the rule's precedence from five numbers.
                ax.plot([mark, mark],
                        [y - H(5.2 if decides else 4.4), y + H(5.2 if decides else 4.4)],
                        color=INK if decides else MUTED,
                        linewidth=1.25 if decides else 0.8, zorder=5)
                # Strong-hit recall is the one quantity the rule reads against two bars.
                # Drawing only the higher one would show a row failing a threshold the
                # rule would not have failed it on.
                if quantity == "top5pct_recall":
                    lower = position(OPS_TIER_RULES.top5pct_ranking_min)
                    ax.plot([lower, lower], [y - H(3.2), y + H(3.2)], color=MUTED,
                            linewidth=0.6, alpha=0.55, zorder=5)

            here = position(cell["value"])
            # A stem from the column's zero to the value, not a full-width rail with a
            # dot floating on it. The rail was the same length whatever the value, so
            # nothing in the cell encoded magnitude and the panel read as a table. The
            # gap between the end of the stem and the threshold mark is the shortfall,
            # which an earlier version drew a second time as a dotted segment.
            ax.plot([x0, x0 + width], [y, y], color=TRACK, linewidth=0.6,
                    solid_capstyle="butt", zorder=2)
            # The stem is the same weight in every cell. Darkening the deciding one made
            # a long dark bar, and length already encodes magnitude, so the emphasis read
            # as "this value is large" rather than "this threshold decided". The ink goes
            # where the claim is: on the threshold mark, the marker edge and the number.
            ax.plot([x0, here], [y, y], color=MUTED, linewidth=1.5, alpha=0.72,
                    solid_capstyle="butt", zorder=4)

            spread = replicates.loc[
                (replicates["row_key"] == row.key) & (replicates["quantity"] == quantity)
            ]
            for _, point in spread.iterrows():
                ax.plot([position(point["value"])] * 2, [y - H(3.0), y + H(3.0)],
                        color=MUTED, linewidth=0.5, alpha=0.62, zorder=6)

            ax.plot([here], [y], marker="o", markersize=4.0 if decides else 3.2, zorder=7,
                    markerfacecolor=INK if cell["passes"] else "white",
                    markeredgecolor=INK if decides else MUTED,
                    markeredgewidth=0.9 if decides else 0.7, linewidth=0)
            # A value below the track's floor is drawn at the floor, so the mark alone
            # would put -0.057 and 0.000 in the same place. The caret says the position
            # is a clamp; the printed number, when it is printed, says by how much.
            if float(cell["value"]) < 0.0:
                ax.plot([x0 - 0.004], [y], marker=4, markersize=3.0, color=INK,
                        markeredgewidth=0.0, linewidth=0, zorder=7)
            # Only the value that decided the row is printed. Printing all five put 30
            # numbers in the panel, which is what kept the deciding one from being
            # visible at all; the other four are in Table 1 and in the source data.
            if decides:
                ax.text(col_x0 + (col_index + 1) * col_w - 0.005, y,
                        format_value(float(cell["value"])),
                        fontsize=FS_MID, va="center", ha="right",
                        fontweight="bold", color=INK)

        # A swatch and a label, not a filled pill with the label inside it. The pill was
        # unreadable and could not be fixed within the palette: measured against WCAG,
        # white text on "measurement required" reaches 3.1:1 and on "not identifiable"
        # 3.0:1, both below the 4.5:1 floor, and "ranking proxy" admits neither white
        # (3.9:1) nor ink (3.8:1). Darkening those fills would break the colour thread to
        # panel c, which binds them. Ink on the panel background is 14.8:1 for every
        # tier, and the swatch keeps the thread.
        tier = _tier_of(row.key, headline)
        ax.add_patch(FancyBboxPatch(
            (swatch_x, y - H(4.4)), swatch_w, H(8.8),
            boxstyle="round,pad=0,rounding_size=0.004",
            facecolor=TIER_COLOURS[tier], edgecolor="none", zorder=2))
        ax.text(verdict_x, y, tier, fontsize=FS_MID, va="center", ha="left",
                color=INK, zorder=3)

        # Chips belong to the admitted rows only. Matching on reporter_id alone put
        # them on the OASIS control row as well, which carries the same reporter and
        # would have claimed a per-batch verdict for a run that has none.
        reporter = row.key.split(":", 1)[1]
        chips = (
            batches.loc[batches["reporter_id"] == reporter].sort_values("batch")
            if row.band == "admitted" and row.key.startswith("oasis:")
            else batches.iloc[:0]
        )
        for index, (_, record) in enumerate(chips.iterrows()):
            ax.add_patch(plt.Rectangle(
                (batch_x0 + index * (batch_w + 0.002), y - H(4.2)), batch_w, H(8.4),
                facecolor=TIER_COLOURS[record["tier"]], edgecolor="none", zorder=2))


    # Text printed over graphics is invisible to verify(), which compares text against
    # text. Every row label sits in a column whose width is a layout constant, so a
    # label that outgrows it prints over the first track and nothing says so. The
    # boundary is known here, so the check belongs here: at 518.74 pt all five
    # sublabels overran a column cut for 643.5 pt.
    figure = ax.figure
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    limit = ax.transData.transform((col_x0 - track_pad, 0.0))[0]
    spills = [
        (artist.get_text(), (artist.get_window_extent(renderer).x1 - limit))
        for artist in row_labels
        if artist.get_window_extent(renderer).x1 > limit
    ]
    if spills:
        raise SystemExit(
            "row labels run into the first evidence track, where they print over it:\n  "
            + "\n  ".join(f"{text!r} by {over / figure.dpi * 72.0:.1f} pt" for text, over in spills)
        )


def verify(fig) -> None:
    """The geometric half of render-then-verify: no text may overlap other text."""
    import matplotlib as mpl

    # Draw first. A legend box and an annotation only take their final position when the
    # figure is drawn, and get_renderer() alone returns whatever the last draw left. The
    # standalone panel has neither, so it never noticed; on the composite this check
    # silently measured stale boxes and reported five collisions that were not there.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    # A tick whose location is outside its axis's view interval keeps a Text artist that
    # reports visible and carries a stale position, often off the page entirely. Nothing
    # draws it, so counting it as a collision reports a defect that is not there. The
    # standalone panel has no axes and never hit this; the composite has eleven.
    ignored: set[int] = set()
    for axes in getattr(fig, "axes", []):
        for axis, limits in ((axes.xaxis, axes.get_xlim()), (axes.yaxis, axes.get_ylim())):
            low, high = min(limits), max(limits)
            for tick in list(axis.get_major_ticks()) + list(axis.get_minor_ticks()):
                if not low - 1e-9 <= tick.get_loc() <= high + 1e-9:
                    ignored.update((id(tick.label1), id(tick.label2)))
    def extent(artist):
        # An Annotation's window extent is the union of its text and its leader line, so
        # two leaders crossing would be reported as text landing on text. The check is
        # about what a reader can read, so it measures the glyphs.
        if isinstance(artist, mpl.text.Annotation):
            return mpl.text.Text.get_window_extent(artist, renderer)
        return artist.get_window_extent(renderer)

    texts = [
        (t.get_text(), extent(t))
        for t in fig.findobj(mpl.text.Text)
        if t.get_text().strip() and t.get_visible() and id(t) not in ignored
    ]
    collisions = [
        (a, b) for i, (a, box_a) in enumerate(texts)
        for b, box_b in texts[i + 1:] if box_a.overlaps(box_b)
    ]
    if collisions:
        raise SystemExit(
            "panel text collides, which no amount of caption can fix:\n  "
            + "\n  ".join(f"{a!r} overlaps {b!r}" for a, b in collisions[:8])
        )
    outside = [a for a, box in texts
               if not fig.bbox.containsx(box.x0) or not fig.bbox.containsx(box.x1)
               or box.y0 < 0 or box.y1 > fig.bbox.y1]
    if outside:
        raise SystemExit(f"text runs outside the figure: {outside[:5]}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--from-source-data", type=Path, default=None, metavar="DIR",
        help="redraw from the five per-panel tables in DIR instead of from the "
             "analysis directories, which are produced on a compute node and are not "
             "distributed. The thresholds, pass flags, replicate medians and batch "
             "verdicts are all re-checked against the frozen rule on the way in",
    )
    parser.add_argument("--oasis", type=Path)
    parser.add_argument("--oasis-batch", action="append", default=[], metavar="NAME=PATH",
                        help="a per-batch OASIS analysis; repeatable. Drawn as verdict "
                             "chips beside the pooled rows rather than as further rows")
    parser.add_argument("--periscope-cells", type=Path)
    parser.add_argument("--periscope-guide", type=Path)
    parser.add_argument("--stain-dropout", type=Path,
                        help="directory holding one sub-directory per held-out dye")
    parser.add_argument("--control", type=Path,
                        help="a PERISCOPE guide analysis whose provenance declares permuted inputs")
    parser.add_argument("--oasis-control", type=Path,
                        help="an OASIS analysis whose provenance declares permuted inputs. "
                             "This is the control that can move a verdict: cpg0021 is "
                             "stopped at the reliability gate before the input is read")
    parser.add_argument("--admission-audit", type=Path, default=ADMISSION_AUDIT)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--width-pt", type=float, default=643.5,
        help="panel width in points; the sibling panels of Figure 6 are 643.5 pt",
    )
    parser.add_argument(
        "--height-pt", type=float, default=384.0,
        help="panel height in points. required_height() derives the minimum from the "
             "row list and the top matter, and the renderer refuses anything below it "
             "rather than drawing the last band off the bottom edge. The default suits "
             "the full ten-row panel with its audit strip",
    )
    parser.add_argument("--no-render", action="store_true",
                        help="write the source data and run the checks but skip matplotlib")
    parser.add_argument(
        "--band", action="append", default=[], choices=[name for name, _, _ in BANDS],
        help="draw only these bands; repeatable. Omitting a band drops evidence from "
             "the panel, so whatever is dropped has to appear somewhere else",
    )
    parser.add_argument(
        "--no-letter", action="store_true",
        help="omit the panel letter. Use when the panel is published on its own rather "
             "than as panel f of Figure 6",
    )
    parser.add_argument(
        "--no-funnel", action="store_true",
        help="omit the admission strip. It is the evidence for the claim that the "
             "criteria apply broadly, so a caller that drops it owes the reader that "
             "evidence elsewhere",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = reject_repo_paths(args.output_dir)
    rows = tuple(row for row in ROWS if row.band in args.band) if args.band else ROWS
    if not rows:
        raise SystemExit(f"--band {args.band} selects no row")

    controls: dict[str, dict] = {}
    if args.from_source_data is not None:
        conflicting = [
            name for name, value in (
                ("--oasis", args.oasis), ("--periscope-cells", args.periscope_cells),
                ("--periscope-guide", args.periscope_guide),
                ("--stain-dropout", args.stain_dropout), ("--control", args.control),
                ("--oasis-control", args.oasis_control),
            ) if value is not None
        ]
        if conflicting:
            raise SystemExit(
                f"--from-source-data reads the saved tables, so {', '.join(conflicting)} "
                "would be ignored. Pass one or the other, not both."
            )
        headline, replicates, segments, evidence, batch_table = load_from_source_data(
            args.from_source_data, rows=rows
        )
        print(f"redrawn from the saved tables in {args.from_source_data}")
    else:
        missing = [
            name for name, value in (
                ("--oasis", args.oasis), ("--periscope-cells", args.periscope_cells),
                ("--periscope-guide", args.periscope_guide),
                ("--stain-dropout", args.stain_dropout), ("--control", args.control),
                ("--oasis-control", args.oasis_control),
            ) if value is None
        ]
        if missing:
            raise SystemExit(
                f"missing {', '.join(missing)}. Pass the analysis directories, or "
                "--from-source-data to redraw from the saved per-panel tables."
            )
        sources = resolve_sources(args)
        controls = {
            "PERISCOPE guide": check_control_is_a_control(args.control),
            "OASIS": check_control_is_a_control(args.oasis_control),
        }
        batches = {}
        for item in args.oasis_batch:
            if "=" not in item:
                raise SystemExit(f"--oasis-batch expects NAME=PATH, got {item!r}")
            name, _, path = item.partition("=")
            batches[name] = Path(path)

        headline, replicates = load_rows(sources)
        segments, evidence = admission_funnel(args.admission_audit)
        batch_table = batch_verdicts(batches) if batches else pd.DataFrame(
            columns=["batch", "reporter_id", "tier"]
        )

    source_data = output / "source_data"
    source_data.mkdir(parents=True, exist_ok=True)
    headline.to_csv(source_data / "figure_source_data.csv", index=False)
    replicates.to_csv(source_data / "replicate_detail.csv", index=False)
    segments.to_csv(source_data / "admission_funnel.csv", index=False)
    evidence.to_csv(source_data / "admission_evidence_tiers.csv", index=False)
    batch_table.to_csv(source_data / "batch_verdicts.csv", index=False)

    print("panel 6f, headline values against the frozen thresholds:\n")
    wide = headline.pivot(index="row_key", columns="quantity", values="value")
    wide = wide.reindex([row.key for row in rows])[[q for q, _, _, _ in QUANTITIES]]
    print(wide.to_string(float_format=lambda v: f"{v:.4f}"))
    print("\nverdicts:")
    for row in rows:
        print(f"  {row.key:<44} {_tier_of(row.key, headline)}")
    if not batch_table.empty:
        print("\nper-batch verdicts:")
        print(batch_table[["batch", "reporter_id", "tier"]].to_string(index=False))
    for name, record in controls.items():
        permutation = record["permutation"]
        moved = permutation.get("n_rows_moved", permutation.get("n_wells_moved"))
        total = permutation.get("n_rows", permutation.get("n_wells"))
        groups = permutation.get("n_groups", permutation.get("n_screens"))
        print(f"\n{name} control: {moved:,} of {total:,} units moved across {groups} screens")
    print("\nthresholds not met, which is what decides each verdict:")
    for _, row in headline.loc[~headline["passes"]].iterrows():
        print(f"  {row['row_key']:<44} {row['quantity']:<20} "
              f"{row['value']:.4f} against {row['threshold_min']}")

    if args.no_render:
        print(f"\nwrote source data to {source_data}; rendering skipped")
        return 0
    written = draw(headline, replicates, segments, evidence, batch_table, output,
                   width_pt=args.width_pt, height_pt=args.height_pt,
                   rows=rows, show_funnel=not args.no_funnel,
                   show_letter=not args.no_letter)
    print("\nwrote:")
    for path in written:
        print(f"  {path}")
    for name in ("figure_source_data.csv", "replicate_detail.csv", "admission_funnel.csv",
                 "admission_evidence_tiers.csv", "batch_verdicts.csv"):
        print(f"  {source_data / name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
