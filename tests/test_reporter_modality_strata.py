"""The reporter registry must carry the source atlas's own live/fixed marker split.

The source atlas states "39 live, 13 fixed markers" (Liu et al., *A multimodal
perturbation atlas defines the phenotypic resolution of cellular morphology*,
bioRxiv 2026, DOI 10.64898/2026.06.01.728087; abstract read via Europe PMC
PPR1244083).  The registry always encoded that split, but only inside free-text
reporter names, so no analysis could stratify on it and nothing checked that the
repository's view of the markers still matched the source's.

Two independent facts pin the fixed stratum, and this file asserts both:

  * six reporters carry a ``(4i)`` name suffix, and
  * ``build_exact_caches.FOURI_ALIASES`` names exactly those six and gives them a
    *different endpoint-selection rule* from every other reporter (marker-matched
    CellProfiler single-object intensity, rather than OrganelleProfiler intensity).

That second point is why the stratification is necessary rather than decorative:
the 4i reporters' endpoints are not constructed the same way as the other 46, so a
tier distribution that differs across modality could be a property of the endpoint
definition rather than of the measurement.

``FOURI_ALIASES`` is read with :mod:`ast` rather than imported, because
``build_exact_caches`` imports h5py at module level and these assertions need no
optional dependency to be true.
"""

from __future__ import annotations

import ast
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "configs" / "ops" / "data" / "reporter_registry.csv"
CACHE_BUILDER = ROOT / "studies" / "ops" / "preparation" / "build_exact_caches.py"

#: Declared by the source atlas abstract.  Not a threshold and not tunable.
SOURCE_LIVE = 39
SOURCE_FIXED = 13

FIXATION_STATES = {"live", "fixed"}
TARGET_MODALITIES = {
    "immunofluorescence_4i",
    "cell_painting_stain",
    "live_dye",
    "endogenous_tag",
    "transcriptional_reporter",
    "unresolved",
}

#: Columns ``export_canonical.load_registry`` requires.  Listed here so that adding
#: a stratification column can never be paid for by dropping one it needs.
LOAD_REGISTRY_REQUIRED = {
    "reporter_slug",
    "reporter_name",
    "short_name",
    "biological_system",
}


def registry() -> list[dict[str, str]]:
    with REGISTRY.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def fouri_aliases() -> set[str]:
    """The reporter slugs ``build_exact_caches`` treats as 4i, read without importing."""
    tree = ast.parse(CACHE_BUILDER.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            getattr(target, "id", None) == "FOURI_ALIASES" for target in node.targets
        ):
            return set(ast.literal_eval(node.value))
    raise AssertionError("build_exact_caches no longer defines FOURI_ALIASES")


def test_registry_declares_both_stratification_columns() -> None:
    rows = registry()
    assert rows, "reporter registry is empty"
    for column in ("fixation_state", "target_modality"):
        assert column in rows[0], f"reporter registry lacks {column}"
    assert LOAD_REGISTRY_REQUIRED.issubset(rows[0]), (
        "reporter registry lost a column export_canonical.load_registry requires: "
        f"{sorted(LOAD_REGISTRY_REQUIRED.difference(rows[0]))}"
    )


def test_every_reporter_carries_a_declared_value() -> None:
    for row in registry():
        assert row["fixation_state"] in FIXATION_STATES, row
        assert row["target_modality"] in TARGET_MODALITIES, row


def test_live_fixed_split_matches_the_source_atlas() -> None:
    """39 live and 13 fixed, as the source atlas abstract declares.

    A drift here means the repository's marker taxonomy has diverged from the
    resource it analyses, which would silently mis-assign reporters in any
    modality-stratified result.  Do not adjust the constants to match the data;
    find out which marker changed.
    """
    rows = registry()
    n_fixed = sum(row["fixation_state"] == "fixed" for row in rows)
    n_live = len(rows) - n_fixed
    assert (n_live, n_fixed) == (SOURCE_LIVE, SOURCE_FIXED), (
        f"registry declares {n_live} live / {n_fixed} fixed; "
        f"the source atlas declares {SOURCE_LIVE} / {SOURCE_FIXED}"
    )


def test_the_4i_stratum_is_exactly_what_the_cache_builder_treats_as_4i() -> None:
    """The registry stratum and the pipeline's special-cased set must not drift apart."""
    declared = {
        row["reporter_slug"]
        for row in registry()
        if row["target_modality"] == "immunofluorescence_4i"
    }
    assert declared == fouri_aliases(), (
        "registry 4i stratum and build_exact_caches.FOURI_ALIASES disagree:\n"
        f"  registry only: {sorted(declared - fouri_aliases())}\n"
        f"  builder only:  {sorted(fouri_aliases() - declared)}"
    )


def test_fixed_stratum_is_only_4i_and_cell_painting() -> None:
    """Every fixed reporter is evidenced by a name suffix, none is assigned by hand."""
    for row in registry():
        if row["fixation_state"] != "fixed":
            continue
        if row["target_modality"] == "immunofluorescence_4i":
            assert "(4i)" in row["reporter_name"], row
        elif row["target_modality"] == "cell_painting_stain":
            assert "(cp)" in row["reporter_name"], row
        else:  # pragma: no cover - only on a regression
            raise AssertionError(f"fixed reporter with unexpected modality: {row}")


def test_cell_painting_stratum_is_the_seven_canonical_reagents() -> None:
    """These seven are a Cell Painting panel inside the atlas.

    They are the stratum that is directly comparable to external Cell Painting
    resources, so their count is worth pinning: TOMM20 in particular is the single
    antibody target of the PERISCOPE atlas, which makes it a same-target external
    comparison rather than a generic one.
    """
    stains = {
        row["short_name"].rstrip("*")
        for row in registry()
        if row["target_modality"] == "cell_painting_stain"
    }
    assert stains == {"Hoechst", "NPM1", "ConA", "TOMM20", "Phalloidin", "Tubulin", "WGA"}


def test_unresolved_stratum_is_not_silently_pooled() -> None:
    """MKI67 has no evidence for its reagent class and must stay flagged.

    The 39/13 arithmetic places it on the live side, but nothing in the registry
    says what it is.  It is recorded as ``unresolved`` so a stratified analysis
    reports it separately.  Assigning it to a real modality requires evidence from
    the source, not a tidier table.
    """
    unresolved = [row for row in registry() if row["target_modality"] == "unresolved"]
    assert [row["reporter_slug"] for row in unresolved] == [
        "cell_proliferation_marker_mki67"
    ]
