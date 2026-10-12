"""Contracts for the external-dataset preparations, checked without network access.

Two public datasets carry the external validation: `cpg0037-oasis`, where a genuine
brightfield channel predicts an MTT and LDH plate assay, and `cpg0021-periscope`,
where four Cell Painting dyes predict an anti-TOMM20 channel in A549. They test
different halves of the claim and declare different capabilities, and both of those
facts have to stay true as the scripts change.

The tests that matter most here are the feature-partition ones. Deciding which
columns may serve as the cheap input is the single place where the expensive
measurement can leak into the model, and every trap below was found in the real
column lists rather than invented:

* ``Cells_Correlation_RWC_Brightfield_Mito`` names an input and a target channel in
  one column, so a substring match on the input channel imports the target;
* ``Image_Threshold_FinalThreshold_mito_bw`` spells the target channel in lower
  case, so a case-sensitive matcher misses it;
* ``Image_ImageQuality_Correlation_OrigAGP_10`` prefixes the channel with ``Orig``,
  so a token match that requires an exact word misses it;
* ``Cells_AreaShape_Area`` names no channel at all, yet is computed inside a mask
  drawn on the expensive channels.
"""

from __future__ import annotations

import contextlib
import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
EXTERNAL = ROOT / "studies" / "external"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    # Registered before execution: a module holding a dataclass resolves its own
    # annotations through sys.modules, so loading one this way without registering it
    # raises inside @dataclass rather than in anything the test wrote.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


common = _load("external_common", EXTERNAL / "_common.py")
oasis = _load("external_oasis_prepare", EXTERNAL / "oasis" / "prepare.py")
periscope = _load("external_periscope_prepare", EXTERNAL / "periscope" / "prepare.py")


# --------------------------------------------------------------------------- #
# Channel attribution
# --------------------------------------------------------------------------- #

OASIS_CHANNELS = ("Brightfield", "DNA", "Mito", "AGP", "RNA", "ER")


@pytest.mark.parametrize(
    ("column", "expected"),
    [
        ("Image_Granularity_10_Brightfield", {"Brightfield"}),
        # Cross-channel: names both an input and a fluorescence channel.
        ("Cells_Correlation_RWC_Brightfield_Mito", {"Brightfield", "Mito"}),
        # Lower case, and the channel is glued to a suffix.
        ("Image_Threshold_FinalThreshold_mito_bw", {"Mito"}),
        # CellProfiler's Orig prefix.
        ("Image_ImageQuality_Correlation_OrigAGP_10", {"AGP"}),
        # No channel: segmentation-derived.
        ("Cells_AreaShape_Area", set()),
        # A token boundary is required: ER must not match inside these words.
        ("Cells_AreaShape_Center_X", set()),
        ("Nuclei_Number_Object_Number", set()),
    ],
)
def test_channel_attribution_of_real_column_shapes(column: str, expected: set[str]) -> None:
    channels = common.CellProfilerChannels(OASIS_CHANNELS)
    assert set(channels.of(column)) == expected


def test_partition_is_an_exact_split_and_admits_only_positive_attribution() -> None:
    channels = common.CellProfilerChannels(OASIS_CHANNELS)
    columns = [
        "Image_Granularity_10_Brightfield",
        "Cells_Intensity_MeanIntensity_Brightfield",
        "Cells_Correlation_RWC_Brightfield_Mito",
        "Image_Threshold_FinalThreshold_mito_bw",
        "Image_ImageQuality_Correlation_OrigAGP_10",
        "Cells_AreaShape_Area",
    ]
    result = common.partition_features(columns, channels=channels, input_channels=("Brightfield",))
    assert result["input"] == [
        "Image_Granularity_10_Brightfield",
        "Cells_Intensity_MeanIntensity_Brightfield",
    ]
    assert result["excluded_cross_channel"] == ["Cells_Correlation_RWC_Brightfield_Mito"]
    assert set(result["excluded_other_channel"]) == {
        "Image_Threshold_FinalThreshold_mito_bw",
        "Image_ImageQuality_Correlation_OrigAGP_10",
    }
    assert result["excluded_channel_free"] == ["Cells_AreaShape_Area"]
    assert sum(len(value) for value in result.values()) == len(columns)


def test_no_input_column_ever_mentions_a_target_channel() -> None:
    """The property the whole partition exists to guarantee."""
    channels = common.CellProfilerChannels(("DAPI", "ConA", "Mito", "Phalloidin", "WGA"))
    columns = [
        "Cells_Granularity_10_Mito",
        "Cells_Granularity_10_ConA",
        "Cells_Correlation_Correlation_ConA_Mito",
        "Cells_Intensity_MeanIntensity_mito_tubeness",
        "Cells_AreaShape_Zernike_0_0",
    ]
    result = common.partition_features(
        columns, channels=channels, input_channels=("DAPI", "ConA", "Phalloidin", "WGA"),
        target_channels=("Mito",),
    )
    for column in result["input"]:
        assert "Mito" not in channels.of(column), column
    # The derived mito_tubeness channel belongs to the target, not to the
    # channel-free bucket, or 222 target columns would be silently discarded.
    assert "Cells_Intensity_MeanIntensity_mito_tubeness" in result["target"]


def test_overlapping_input_and_target_channels_are_refused() -> None:
    channels = common.CellProfilerChannels(("A", "B"))
    with pytest.raises(ValueError, match="overlap"):
        common.partition_features(["x_A"], channels=channels, input_channels=("A",), target_channels=("A",))


def test_partition_refuses_a_table_with_no_attributable_input() -> None:
    """A wrong channel vocabulary must fail loudly, not yield an empty model."""
    channels = common.CellProfilerChannels(OASIS_CHANNELS)
    with pytest.raises(ValueError, match="no column is positively attributable"):
        common.partition_features(
            ["Cells_AreaShape_Area"], channels=channels, input_channels=("Brightfield",)
        )


# --------------------------------------------------------------------------- #
# Location handling
# --------------------------------------------------------------------------- #


def test_output_inside_the_repository_is_refused() -> None:
    with pytest.raises(SystemExit, match="may not point inside"):
        common.reject_repo_paths(ROOT / "configs")
    with pytest.raises(SystemExit, match="may not point inside"):
        common.reject_repo_paths(ROOT)


def test_output_outside_the_repository_is_accepted(tmp_path: Path) -> None:
    assert common.reject_repo_paths(tmp_path / "prepared") == (tmp_path / "prepared").resolve()


def test_a_missing_location_names_both_the_flag_and_the_variable(monkeypatch) -> None:
    monkeypatch.delenv("SOME_EXTERNAL_BASE", raising=False)
    with pytest.raises(SystemExit) as excinfo:
        common.resolve_source(None, what="the archive", env_var="SOME_EXTERNAL_BASE", flag="--base")
    assert "--base" in str(excinfo.value) and "SOME_EXTERNAL_BASE" in str(excinfo.value)


# --------------------------------------------------------------------------- #
# Declared capabilities
# --------------------------------------------------------------------------- #


def test_the_two_datasets_answer_complementary_questions() -> None:
    """OASIS carries the label-free half, PERISCOPE the perturbation and transfer half.

    These are declarations about what the accessions physically contain, so they are
    pinned. MTT and LDH are plate assays and cannot become cell-level; PERISCOPE has
    no transmitted-light channel and cannot become label-free evidence.
    """
    assert oasis.INPUT_CHANNELS == ("Brightfield",)
    assert set(oasis.REPORTERS) == {"mtt_viability", "ldh_release"}
    assert periscope.INPUT_CHANNELS == ("DAPI", "ConA", "Phalloidin", "WGA")
    assert periscope.TARGET_CHANNELS == ("Mito",)
    assert "Brightfield" not in periscope.CHANNELS.names


def _oasis_profile() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Metadata_well_id": ["p_r1_c1", "p_r1_c2", "p_r1_c3"],
            "Metadata_compound_name": ["DMSO", "CompoundA", "CompoundB"],
            "Metadata_control_type": ["negcon", None, None],
            "Metadata_Well": ["A01", "A02", "A03"],
            "Metadata_mtt_lumi": [1.0, 2.0, 3.0],
            "Metadata_mtt_normalized": [0.9, 1.1, 1.2],
            "Metadata_ldh_abs": [0.1, 0.2, 0.3],
            "Metadata_ldh_normalized": [0.01, 0.02, None],
            "Image_Granularity_1_Brightfield": [1.0, 2.0, 3.0],
        }
    )


def _consequential() -> list[str]:
    """Columns the OASIS preparation actually consumes."""
    return (
        ["Image_Granularity_1_Brightfield"]
        + [f"Metadata_{endpoint}" for spec in oasis.REPORTERS.values() for endpoint in spec["endpoints"]]
        + ["Metadata_compound_name", "Metadata_control_type", "Metadata_Well"]
    )


def test_byte_identical_duplicate_wells_are_collapsed_and_counted() -> None:
    """Released plates repeat wells; a benign repeat must not cost the plate.

    `plate_41002689` carries 388 rows for 384 wells. Three of the repeats are
    byte-identical rows and are collapsed; nothing is dropped silently, the counts
    reach `provenance.json`.
    """
    profile = _oasis_profile()
    doubled = pd.concat([profile, profile.iloc[[0]]], ignore_index=True)
    collapsed, record = oasis.collapse_benign_duplicates(
        doubled, plate="plate_1", consequential=_consequential()
    )
    assert len(collapsed) == 3
    assert record["n_exact_duplicate_rows"] == 1
    assert record["n_wells_collapsed"] == 0


def test_a_repeat_that_disagrees_only_on_an_unused_annotation_is_collapsed() -> None:
    """The real case on `plate_41002689`: two rows differing in Metadata_OASIS_ID alone.

    Every feature, both assay readouts, the compound and the concentration agree, so
    the repeat provably cannot change a result and refusing the plate over it would
    discard data for a duplicated catalogue entry.
    """
    profile = _oasis_profile().assign(Metadata_OASIS_ID=["x", "y", "z"])
    variant = profile.iloc[[0]].assign(Metadata_OASIS_ID="x2")
    collapsed, record = oasis.collapse_benign_duplicates(
        pd.concat([profile, variant], ignore_index=True), plate="plate_1",
        consequential=_consequential(),
    )
    assert len(collapsed) == 3
    assert record["n_wells_collapsed"] == 1
    assert record["disagreeing_columns"] == ["Metadata_OASIS_ID"]


def test_a_repeat_that_disagrees_on_a_measurement_is_refused() -> None:
    """A well whose readout disagrees is a real ambiguity, not a tidying problem."""
    profile = _oasis_profile()
    variant = profile.iloc[[0]].assign(Metadata_mtt_normalized=99.0)
    with pytest.raises(ValueError, match="must not be resolved by choosing a row"):
        oasis.collapse_benign_duplicates(
            pd.concat([profile, variant], ignore_index=True), plate="plate_1",
            consequential=_consequential(),
        )


def test_a_repeat_that_disagrees_on_a_feature_is_refused() -> None:
    profile = _oasis_profile()
    variant = profile.iloc[[0]].assign(Image_Granularity_1_Brightfield=99.0)
    with pytest.raises(ValueError, match="must not be resolved by choosing a row"):
        oasis.collapse_benign_duplicates(
            pd.concat([profile, variant], ignore_index=True), plate="plate_1",
            consequential=_consequential(),
        )


def test_oasis_marks_only_declared_negative_controls() -> None:
    observations, _, _ = oasis.canonical_rows(
        _oasis_profile(), source="axiom", batch="prod_25", plate="plate_1",
        feature_columns=["Image_Granularity_1_Brightfield"],
    )
    assert observations["is_control"].tolist() == [True, False, False]


def test_the_screen_is_the_plate_because_the_plate_carries_the_controls() -> None:
    """`screen_id` is the unit control-relative response subtracts within.

    In OPS a screen is both the control-matching unit and the environment, so the two
    coincide. Here they do not: DMSO controls sit on every plate, while a batch spans
    roughly seventeen plates. Setting `screen_id` to the batch would pool controls over
    all of them and throw away the plate-matched precision the design provides, which
    would corrupt the response itself rather than merely add noise. The batch is kept
    separately for environment-transfer analyses.
    """
    observations, _, _ = oasis.canonical_rows(
        _oasis_profile(), source="axiom", batch="prod_25", plate="plate_1",
        feature_columns=["Image_Granularity_1_Brightfield"],
    )
    assert observations["screen_id"].unique().tolist() == ["plate_1"]
    assert observations["batch_id"].unique().tolist() == ["axiom_prod_25"]
    # A well-level table has no imaging field, and the plate is already the screen, so
    # the field unit is the plate row.
    assert observations["field_id"].str.startswith("plate_1|row").all()


def test_oasis_keeps_an_unmeasured_endpoint_as_null_rather_than_dropping_it() -> None:
    """A missing assay value is missing, not zero, and not an absent row."""
    _, _, targets = oasis.canonical_rows(
        _oasis_profile(), source="axiom", batch="prod_25", plate="plate_1",
        feature_columns=["Image_Granularity_1_Brightfield"],
    )
    assert len(targets) == 3 * 4  # three wells, two reporters, two endpoints each
    missing = targets.loc[targets.endpoint_id.eq("ldh_normalized") & targets.y_true.isna()]
    assert len(missing) == 1


def test_oasis_rejects_a_plate_whose_wells_repeat() -> None:
    profile = _oasis_profile()
    profile.loc[2, "Metadata_well_id"] = "p_r1_c1"
    with pytest.raises(ValueError, match="not unique"):
        oasis.canonical_rows(profile, source="axiom", batch="prod_25", plate="plate_1",
                             feature_columns=["Image_Granularity_1_Brightfield"])


def _periscope_chunk() -> pd.DataFrame:
    return pd.DataFrame(
        {
            periscope.GENE_COLUMN: ["nontargeting", "TP53", "KRAS"],
            periscope.GUIDE_COLUMN: ["AAAA", "CCCC", "GGGG"],
            periscope.PLATE_COLUMN: ["CP186A"] * 3,
            periscope.WELL_COLUMN: ["A1", "A1", "A2"],
            periscope.SITE_COLUMN: [1, 1, 2],
            periscope.CELL_COLUMN: [1, 2, 1],
            "Cells_Granularity_1_ConA": [1.0, 2.0, 3.0],
            "Cells_Granularity_1_Mito": [4.0, 5.0, 6.0],
        }
    )


def test_periscope_marks_non_targeting_guides_as_controls() -> None:
    observations, _, _ = periscope.canonical_rows(
        _periscope_chunk(), plate="CP186A",
        input_columns=["Cells_Granularity_1_ConA"], target_columns=["Cells_Granularity_1_Mito"],
    )
    assert observations["is_control"].tolist() == [True, False, False]
    assert observations["observation_id"].is_unique
    # The imaging site is the field unit; the plate is the screen unit.
    assert observations["field_id"].tolist() == ["CP186A|A1|1", "CP186A|A1|1", "CP186A|A2|2"]
    assert observations["screen_id"].unique().tolist() == ["CP186A"]


def test_periscope_targets_are_wide_one_row_per_cell() -> None:
    """Wide, not the canonical long layout, and the reason is arithmetic.

    The long layout exists to represent *sparse* availability, which the OPS reporter
    blocks need. Here every cell carries all 823 endpoints, so long form encodes
    nothing extra and costs 823 times the rows. Measured on the same 30,000 cells:
    0.20 GB wide against 2.52 GB long, a factor of 12.7, which puts the full 11-million
    cell arm at roughly 940 GB in long form and 37 GB wide.

    The four required canonical tables are unaffected, because ``targets`` is optional,
    so the adapter still validates the dataset.
    """
    _, inputs, targets = periscope.canonical_rows(
        _periscope_chunk(), plate="CP186A",
        input_columns=["Cells_Granularity_1_ConA"], target_columns=["Cells_Granularity_1_Mito"],
    )
    assert list(targets.columns) == ["observation_id", "Cells_Granularity_1_Mito"]
    assert len(targets) == 3, "one row per cell, not one row per cell and endpoint"
    assert "Cells_Granularity_1_Mito" not in inputs.columns, "the target must never enter the input"


def test_reliability_pairs_use_disjoint_halves_and_exclude_what_cannot_replicate() -> None:
    """Reliability is the evidence the tier rule gates on, so its construction is pinned.

    A low prediction score has two incompatible explanations: the cheap measurement
    does not carry the information, or the expensive measurement is not reproducible
    and nothing could predict it. The rule refuses to call the first without ruling
    out the second, so a reliability estimate built from overlapping halves, or from
    controls whose expected response is zero by construction, would corrupt the tier
    rather than merely add noise.
    """
    analyse = _load("external_oasis_analyse", EXTERNAL / "oasis" / "analyse.py")
    observations = pd.DataFrame(
        {
            "observation_id": [f"o{index}" for index in range(7)],
            "reporter_id": ["mtt_viability"] * 7,
            "perturbation_id": ["A", "A", "A", "A", "B", "DMSO", "DMSO"],
            "is_control": [False] * 5 + [True, True],
        }
    )
    targets = pd.DataFrame(
        {
            "observation_id": [f"o{index}" for index in range(7)],
            "endpoint_id": ["mtt_normalized"] * 7,
            "y_true": [1.0, 1.1, 0.9, 1.2, 5.0, 1.0, 1.0],
        }
    )
    pairs = analyse.replicate_pairs(observations, targets, seed=0)

    # Only the compound measured four times can be split into two halves.
    assert pairs["perturbation_id"].tolist() == ["A"]
    assert int(pairs["n_measurements"].iloc[0]) == 4
    # The halves are disjoint: their means must average back to the overall mean.
    assert pytest.approx((pairs["value_a"].iloc[0] + pairs["value_b"].iloc[0]) / 2) == 1.05
    # Controls contribute nothing: their expected response is zero by construction.
    assert "DMSO" not in set(pairs["perturbation_id"])


def test_reliability_pairing_is_reproducible_under_a_seed() -> None:
    analyse = _load("external_oasis_analyse", EXTERNAL / "oasis" / "analyse.py")
    observations = pd.DataFrame(
        {
            "observation_id": [f"o{index}" for index in range(6)],
            "reporter_id": ["mtt_viability"] * 6,
            "perturbation_id": ["A"] * 6,
            "is_control": [False] * 6,
        }
    )
    targets = pd.DataFrame(
        {
            "observation_id": [f"o{index}" for index in range(6)],
            "endpoint_id": ["mtt_normalized"] * 6,
            "y_true": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }
    )
    first = analyse.replicate_pairs(observations, targets, seed=7)
    again = analyse.replicate_pairs(observations, targets, seed=7)
    pd.testing.assert_frame_equal(first, again)


def test_declared_capabilities_are_what_the_accessions_physically_support() -> None:
    """The capability block is what every downstream analysis gates on.

    These are not defaults to be relaxed if an analysis is inconvenient. MTT and LDH
    are plate assays, so OASIS can never declare ``exact_pairing``; PERISCOPE has no
    transmitted-light channel, so it can never carry label-free evidence however its
    capabilities read.
    """
    assert oasis.CAPABILITIES == {
        "exact_pairing": False,
        "raw_images": False,
        "guides": False,
        "screen_matched_controls": True,
        "repeated_screens": True,
        "covariates": False,
    }
    assert periscope.CAPABILITIES == {
        "exact_pairing": True,
        "raw_images": False,
        "guides": True,
        "screen_matched_controls": True,
        "repeated_screens": True,
        "covariates": False,
    }


def test_the_panel_covers_both_halves_and_neither_alone(tmp_path: Path) -> None:
    """Together the two datasets span the claim; separately each refuses part of it."""
    from measurement_sufficiency.adapters import AdapterCapabilities, EligibilityError

    oasis_caps = AdapterCapabilities(**oasis.CAPABILITIES)
    periscope_caps = AdapterCapabilities(**periscope.CAPABILITIES)

    # Only PERISCOPE can support the same-cell falsification arm.
    with pytest.raises(EligibilityError):
        oasis_caps.require("exact_pairing", analysis="same-cell falsification")
    periscope_caps.require("exact_pairing", analysis="same-cell falsification")

    # Both support the control-relative and environment-transfer arms, which is what
    # makes either usable for the response-fidelity and transfer analyses.
    for caps in (oasis_caps, periscope_caps):
        caps.require("screen_matched_controls", analysis="control-relative response")
        caps.require("repeated_screens", analysis="environment transfer")

    # Neither claims covariates, because in both the shape features come from masks
    # drawn on the expensive channel.
    for caps in (oasis_caps, periscope_caps):
        with pytest.raises(EligibilityError):
            caps.require("covariates", analysis="covariate-matched derangement")


def test_partitions_are_written_with_one_dtype_per_measurement_column(tmp_path: Path) -> None:
    """Independently written partitions must still read back as one dataset.

    Each plate is written on its own, so pandas infers dtypes per plate: a column
    holding only integral values on one plate becomes int64 there and float64
    elsewhere, and pyarrow then refuses to read the directory, with
    ``Float value 0.000079 was truncated converting to int64``. It surfaced on the
    guide-level arm after nine plates and five hours of downloading, in 35 of 824
    target columns, all RadialDistribution mito_tubeness fractions that are constant
    on some plates. They are continuous measurements; an integer column is an artefact
    of the values happening to be round.
    """
    import pytest

    pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    directory = tmp_path / "targets"
    directory.mkdir()
    identifiers = periscope._IDENTIFIER_COLUMNS
    assert "observation_id" in identifiers

    # Plate one: the measurement happens to be integral. Plate two: it is not.
    integral = pd.DataFrame({"observation_id": ["a", "b"], "m": [0, 0]})
    fractional = pd.DataFrame({"observation_id": ["c", "d"], "m": [0.000079, 0.5]})
    for name, frame in (("plate=P1", integral), ("plate=P2", fractional)):
        measurement = [c for c in frame.columns if c not in identifiers]
        frame = frame.copy()
        frame[measurement] = frame[measurement].astype("float32")
        frame.to_parquet(directory / f"{name}.part000.parquet", index=False)

    types = {
        str(pq.ParquetFile(path).schema_arrow.field("m").type)
        for path in directory.glob("*.parquet")
    }
    assert types == {"float"}, f"partitions disagree on the measurement dtype: {types}"
    combined = pd.read_parquet(directory)
    assert len(combined) == 4, "the partitioned directory must read back as one dataset"


def test_reliability_is_measured_on_the_quantity_it_is_meant_to_bound() -> None:
    """The ceiling must be in the same units as the thing it bounds.

    ``recoverability_r`` is the correlation between observed and predicted
    control-relative responses, pooled over (gene, endpoint). Reliability is supposed
    to bound it, so it has to be the correlation between two independent estimates of
    that same observed response, on that same axis.

    An earlier version correlated the Euclidean norm of each half's **raw** profile
    mean, pooled across screens. Three mismatches at once: no control subtraction, so
    the paired value tracked the plate's signal level rather than the response;
    halves drawn across the eight or nine plates a gene appears on, so they differed
    by baseline rather than by reagent; and a norm, which collapses 823 coordinates to
    one non-negative number whose spread across genes is small, so two noisy estimates
    of a near-constant correlate at zero however well the profiles agree. It returned
    0.007 and would have had the frozen rule declare a published atlas not identifiable
    on the strength of a unit mismatch.

    This pins the construction, not the value.
    """
    analyse = _load("external_periscope_analyse_guide", EXTERNAL / "periscope" / "analyse_guide.py")

    # The three properties were once pinned by reading the function's source for
    # "baseline", "within_screen" and "np.linalg.norm". That stopped meaning anything
    # the moment the estimator moved into _reliability, and it never checked behaviour
    # in the first place. Each is pinned by construction below instead.

    # Pairing is per (gene, endpoint). Every gene here has the same response norm, so
    # a norm-based estimate has no spread across genes and cannot correlate, while a
    # per-coordinate estimate sees three perfectly agreeing halves.
    same_norm = pd.DataFrame(
        [
            {"screen_id": "P1", "perturbation_id": gene, "guide_id": f"{gene}{guide}",
             "is_control": False, "e1": values[0], "e2": values[1]}
            for gene, values in (("g1", (3.0, 4.0)), ("g2", (4.0, 3.0)), ("g3", (-5.0, 0.0)))
            for guide in ("a", "b")
        ]
        + [
            {"screen_id": "P1", "perturbation_id": "nontargeting", "guide_id": guide,
             "is_control": True, "e1": 0.0, "e2": 0.0}
            for guide in ("c1", "c2")
        ]
    )
    norm_blind = analyse.reliability_from_guides(same_norm, ["e1", "e2"], seed=0)
    assert pytest.approx(norm_blind["reliability"].iloc[0], abs=1e-9) == 1.0, (
        "every gene has response norm 5, so a per-gene norm carries no signal; "
        "reaching 1.0 requires pairing per (gene, endpoint)"
    )

    # The split can be taken inside a screen, and that is a different reduction from
    # pooling screens: one row per screen against a single pooled row.
    two_screens = pd.concat([same_norm, same_norm.assign(screen_id="P2")], ignore_index=True)
    per_screen = analyse.reliability_from_guides(two_screens, ["e1", "e2"], seed=0)
    pooled = analyse.reliability_from_guides(two_screens, ["e1", "e2"], seed=0, within_screen=False)
    assert sorted(per_screen["screen_id"]) == ["P1", "P2"]
    assert list(pooled["screen_id"]) == ["all"]

    # A gene whose two guides agree perfectly on their response must reach r = 1,
    # and the control must actually be subtracted: shifting the baseline alone must
    # not change the estimate.
    endpoints = ["e1", "e2"]
    def frame(shift: float) -> pd.DataFrame:
        rows = []
        for gene, values in (("g1", (1.0, 2.0)), ("g2", (3.0, 5.0)), ("g3", (-2.0, 0.5))):
            for guide in ("a", "b"):
                rows.append({"screen_id": "P1", "perturbation_id": gene, "guide_id": f"{gene}{guide}",
                             "is_control": False,
                             "e1": values[0] + shift, "e2": values[1] + shift})
        for guide in ("c1", "c2"):
            rows.append({"screen_id": "P1", "perturbation_id": "nontargeting", "guide_id": guide,
                         "is_control": True, "e1": shift, "e2": shift})
        return pd.DataFrame(rows)

    plain = analyse.reliability_from_guides(frame(0.0), endpoints, seed=0)
    shifted = analyse.reliability_from_guides(frame(100.0), endpoints, seed=0)
    assert pytest.approx(plain["reliability"].iloc[0], abs=1e-9) == 1.0
    assert pytest.approx(shifted["reliability"].iloc[0], abs=1e-9) == 1.0, (
        "a constant shift of both the perturbed wells and the control must cancel; "
        "if it does not, the control is not being subtracted"
    )


def test_reliability_is_not_decided_by_the_endpoint_with_the_largest_units() -> None:
    """The ceiling is pooled the way the value it bounds is pooled.

    ``response_magnitudes`` standardises each endpoint before taking a norm and
    ``_ridge.recoverability_by_fold`` was changed to match, because pooling endpoints
    in native CellProfiler units reports whichever endpoint has the largest units
    rather than the block. Reliability was pooling the same block in native units, so
    the rule was comparing a value and its ceiling formed under different rules.

    ``loud`` is all noise at a large scale: no gene-to-gene signal, per-guide noise of
    1000. ``quiet`` is all signal at a small scale: gene-to-gene spread of 1 and
    per-guide noise of 0.001. So the halves of ``loud`` cannot agree and the halves of
    ``quiet`` agree almost exactly.

    Four guides per gene, because the divisor is the spread of the full gene response
    and the point only holds when a noisy endpoint is noisy in the full response too.
    With four guides the noise enters ``full`` at half the amplitude it enters a
    two-guide half, which is the real relationship; an endpoint whose noise cancelled
    exactly in ``full`` would not be down-weighted by this divisor, and no real
    endpoint behaves that way.

    The predicted values follow from the construction. Native pooling weights by
    native squared spread, so ``loud`` takes essentially all of it and the pooled value
    is ``loud``'s own correlation, near zero. Standardised pooling divides by the
    spread of the *full* response, which for ``loud`` is the four-guide noise at 500
    against a two-guide half at 707, so ``loud`` still carries 2/3 of the standardised
    variance and the pooled value lands near 1/3 rather than at the midpoint. That is
    the point: the divisor bounds an endpoint's influence, it does not erase it. The
    data seed is fixed only to make the sampling error of ``loud``'s near-zero
    correlation reproducible.
    """
    numpy = pytest.importorskip("numpy")
    rng = numpy.random.default_rng(20260815)
    n_genes, n_guides = 400, 4
    signal = rng.normal(0.0, 1.0, n_genes)
    rows = []
    for index in range(n_genes):
        gene = f"g{index}"
        for guide in range(n_guides):
            rows.append({
                "screen_id": "P1", "perturbation_id": gene, "guide_id": f"{gene}_{guide}",
                "is_control": False,
                "loud": float(rng.normal(0.0, 1000.0)),
                "quiet": float(signal[index] + rng.normal(0.0, 0.001)),
            })
    rows += [
        {"screen_id": "P1", "perturbation_id": "nontargeting", "guide_id": f"c{guide}",
         "is_control": True, "loud": 0.0, "quiet": 0.0}
        for guide in range(n_guides)
    ]
    analyse = _load("external_periscope_analyse_guide", EXTERNAL / "periscope" / "analyse_guide.py")
    table = analyse.reliability_from_guides(pd.DataFrame(rows), ["loud", "quiet"], seed=0)

    native = float(table["reliability_pooled_native_all_endpoints"].iloc[0])
    standardised = float(table["reliability"].iloc[0])
    share = float(table["largest_native_endpoint_share"].iloc[0])
    assert share > 0.99, f"loud must hold the native squared spread, got {share}"
    assert abs(native) < 0.20, f"native pooling should follow loud's own r, got {native}"
    assert 0.25 < standardised < 0.45, (
        f"standardised pooling should land near the predicted 1/3, got {standardised}"
    )
    assert float(table["reliability_median_endpoint"].iloc[0]) > 0.45, (
        "the per-endpoint median is the pooling-free reading and must see quiet's agreement"
    )


# --------------------------------------------------------------------------- #
# Single-cell level: the estimator, the fold blocks, and the missing-value fill
# --------------------------------------------------------------------------- #
#
# The single-cell script does not call scikit-learn. 1.08 million cells by 2,508
# features cannot be centred and scaled five times, so the ridge is solved from
# uncentred second moments accumulated once per fold block. That is an estimator
# written by hand in the middle of a scientific claim, so the first test below pins it
# to the reference implementation rather than to its own output.


def _cells_module():
    return _load("external_periscope_analyse_cells", EXTERNAL / "periscope" / "analyse_cells.py")


def _ridge_module():
    """The estimator both levels share, so neither can drift into its own."""
    return _load("external_periscope_ridge", EXTERNAL / "periscope" / "_ridge.py")


def test_moment_ridge_reproduces_sklearn_on_standardised_features() -> None:
    numpy = pytest.importorskip("numpy")
    linear_model = pytest.importorskip("sklearn.linear_model")
    cells = _ridge_module()

    rng = numpy.random.default_rng(7)
    x = rng.normal(size=(240, 9))
    # A constant column: the training block has no spread, and the reference and the
    # moment solve must agree that it contributes nothing rather than dividing by zero.
    x[:, 4] = 3.5
    y = x @ rng.normal(size=(9, 3)) + rng.normal(scale=0.4, size=(240, 3))
    alpha = 1.7

    mean, scale, coefficients, intercept = cells.ridge_via_moments(
        len(x), x.sum(axis=0), y.sum(axis=0), x.T @ x, x.T @ y, alpha=alpha
    )
    ours = ((x - mean) / scale) @ coefficients + intercept

    reference_scale = x.std(axis=0)
    reference_scale[reference_scale <= 0.0] = 1.0
    standardised = (x - x.mean(axis=0)) / reference_scale
    reference = linear_model.Ridge(alpha=alpha, fit_intercept=True).fit(standardised, y)

    assert numpy.allclose(coefficients, reference.coef_.T, atol=1e-8), (
        "the moment solve must be the same estimator as Ridge on standardised "
        "features, not merely a similar one"
    )
    assert numpy.allclose(ours, reference.predict(standardised), atol=1e-8)


def test_fold_statistics_are_the_totals_minus_the_held_out_block() -> None:
    """The one-pass trick is only valid because the moments are additive over rows."""
    numpy = pytest.importorskip("numpy")
    cells = _ridge_module()

    rng = numpy.random.default_rng(3)
    x = rng.normal(size=(60, 5)).astype(numpy.float32)
    y = rng.normal(size=(60, 2)).astype(numpy.float32)
    block_of_row = numpy.array([index % 3 for index in range(60)])

    stats = cells.block_moments(x, y, block_of_row, 3, chunk=7)
    train_rows = numpy.flatnonzero(block_of_row != 0)
    direct = cells.block_moments(x[train_rows], y[train_rows], numpy.zeros(len(train_rows), int), 1)[0]

    assert direct["n"] == stats[1]["n"] + stats[2]["n"]
    for key in ("gram", "cross", "sum_x", "sum_y", "sumsq_y"):
        assert numpy.allclose(direct[key], stats[1][key] + stats[2][key], atol=1e-6), key


def test_missing_inputs_are_filled_from_controls_only() -> None:
    """Controls train every fold, so a control-only statistic cannot leak a held-out gene."""
    numpy = pytest.importorskip("numpy")
    cells = _ridge_module()

    is_control = numpy.array([True, True, True, False, False])
    base = numpy.array(
        [[1.0, 0.0], [3.0, 0.0], [5.0, 0.0], [900.0, 0.0], [numpy.nan, 0.0]], dtype=numpy.float32
    )
    fill, columns, filled = cells.impute_from_controls(base.copy(), is_control)
    assert (columns, filled) == (1, 1)
    assert fill[0] == pytest.approx(3.0), "the fill must be the control median"

    moved = base.copy()
    moved[3, 0] = -900.0  # a perturbed cell, far away in the other direction
    assert cells.impute_from_controls(moved, is_control)[0][0] == pytest.approx(3.0), (
        "a perturbed cell must not move the fill value; if it does, the value is no "
        "longer train-only for the fold that holds that gene out"
    )


def test_a_column_missing_in_every_control_is_refused_rather_than_guessed() -> None:
    numpy = pytest.importorskip("numpy")
    cells = _ridge_module()
    is_control = numpy.array([True, True, False])
    values = numpy.array([[numpy.nan], [numpy.nan], [2.0]], dtype=numpy.float32)
    with pytest.raises(SystemExit, match="no.*train-only fill"):
        cells.impute_from_controls(values, is_control)


def test_controls_are_never_held_out_and_every_gene_lands_in_one_fold() -> None:
    numpy = pytest.importorskip("numpy")
    cells = _ridge_module()

    genes = numpy.array(["g1", "g2", "g3", "g4", "nontargeting", "nontargeting"])
    is_control = numpy.array([False, False, False, False, True, True])
    assignment = cells.gene_folds(genes[~is_control], n_folds=2, seed=0)
    fold_of_row = numpy.array([assignment.get(gene, -1) for gene in genes])
    block_of_row = numpy.where(is_control, 2, fold_of_row)

    assert set(block_of_row[is_control]) == {2}, "controls must sit in their own block"
    assert (block_of_row[~is_control] >= 0).all()
    assert sorted(assignment.values()).count(0) == 2 and sorted(assignment.values()).count(1) == 2
    assert len(set(assignment)) == 4, "a gene may not appear in two folds"


def test_the_penalty_is_chosen_without_the_held_out_fold(monkeypatch) -> None:
    """Nested selection is only leakage-free if the outer fold enters neither inner half."""
    numpy = pytest.importorskip("numpy")
    cells = _ridge_module()

    rng = numpy.random.default_rng(11)
    n_rows = 300
    x = rng.normal(size=(n_rows, 6)).astype(numpy.float32)
    y = (x[:, :2] @ rng.normal(size=(2, 3))).astype(numpy.float32)
    block_of_row = numpy.array([index % 4 for index in range(n_rows)])
    stats = cells.block_moments(x, y, block_of_row, 4)
    total = {key: sum(s[key] for s in stats) for key in stats[0]}

    seen = []
    original = cells.validation_error

    def spy(x_all, y_all, rows, fitted, scale, **kwargs):
        seen.append(numpy.asarray(rows))
        return original(x_all, y_all, rows, fitted, scale, **kwargs)

    monkeypatch.setattr(cells, "validation_error", spy)
    # The fixture is noise free, so the smallest penalty wins and the boundary guard
    # fires. That is the guard doing its job; what this test asserts is which rows
    # were scored on the way there.
    with contextlib.suppress(SystemExit):
        cells.select_alpha(
            x, y, block_of_row, stats, total,
            outer_fold=0, inner_fold=1, ratios=(1e-6, 1e-3, 1.0),
        )
    assert seen, "penalty selection never scored a validation fold"

    outer_rows = set(numpy.flatnonzero(block_of_row == 0).tolist())
    for rows in seen:
        assert not outer_rows.intersection(rows.tolist()), (
            "the outer held-out fold was scored during penalty selection"
        )
        assert set(rows.tolist()) == set(numpy.flatnonzero(block_of_row == 1).tolist())


def test_a_penalty_at_the_edge_of_the_grid_is_an_error_not_a_result() -> None:
    numpy = pytest.importorskip("numpy")
    cells = _ridge_module()

    rng = numpy.random.default_rng(5)
    x = rng.normal(size=(200, 4)).astype(numpy.float32)
    y = (x @ rng.normal(size=(4, 2))).astype(numpy.float32)
    block_of_row = numpy.array([index % 4 for index in range(200)])
    stats = cells.block_moments(x, y, block_of_row, 4)
    total = {key: sum(s[key] for s in stats) for key in stats[0]}

    # Noise-free and well conditioned, so the smallest penalty always wins and the
    # grid cannot bracket an interior optimum.
    with pytest.raises(SystemExit, match="edge of the grid"):
        cells.select_alpha(
            x, y, block_of_row, stats, total,
            outer_fold=0, inner_fold=1, ratios=(1e-8, 1e-4, 1.0),
        )


def test_rank_deficiency_counts_the_collinear_directions() -> None:
    """The diagnostic that would have caught the unregularised first run."""
    numpy = pytest.importorskip("numpy")
    cells = _ridge_module()

    rng = numpy.random.default_rng(2)
    base = rng.normal(size=(400, 5))
    # Three exact copies: the correlation matrix must show three null directions.
    x = numpy.column_stack([base, base[:, 0], base[:, 1], base[:, 2]])
    stats = cells.block_moments(
        x.astype(numpy.float32), numpy.zeros((400, 1), numpy.float32), numpy.zeros(400, int), 1
    )[0]
    deficient, smallest = cells.rank_deficiency(stats["gram"], stats["sum_x"], stats["n"])
    assert deficient == 3
    assert smallest < 1e-8


def test_recoverability_is_a_block_statistic_not_its_largest_endpoint() -> None:
    """The same defect the magnitude path was already guarded against, one path over.

    A reporter block whose endpoints do not share a scale makes a pooled native
    correlation a report on whichever endpoint has the largest units. On the
    single-cell PERISCOPE fit one endpoint held 60% to 98% of the native squared
    spread with its own r of 0.13 to 0.17, while the median endpoint correlated at
    0.83, and the pooled native number swung from 0.121 to 0.439 across folds.
    """
    numpy = pytest.importorskip("numpy")
    ridge = _ridge_module()

    rng = numpy.random.default_rng(19)
    rows = []
    for gene in range(60):
        # e_loud is a thousand times the scale of the others and is predicted badly;
        # the two quiet endpoints are predicted well.
        truth = rng.normal(size=3)
        noise = rng.normal(size=3)
        for endpoint, scale, true_value, pred_value in (
            ("e_loud", 1000.0, truth[0], noise[0]),
            ("e_quiet_a", 1.0, truth[1], truth[1] * 0.95 + 0.05 * noise[1]),
            ("e_quiet_b", 1.0, truth[2], truth[2] * 0.95 + 0.05 * noise[2]),
        ):
            rows.append({
                "fold": 0, "endpoint_id": endpoint,
                "response_true": true_value * scale, "response_pred": pred_value * scale,
            })
    table = ridge.recoverability_by_fold(pd.DataFrame(rows))

    assert table["largest_native_endpoint_share"].iloc[0] > 0.99, (
        "the fixture must reproduce the condition; if it stops, the test proves nothing"
    )
    assert abs(table["recoverability_r_pooled_native"].iloc[0]) < 0.3, (
        "native pooling must be shown to report the badly predicted loud endpoint"
    )
    assert table["recoverability_r"].iloc[0] > 0.6, (
        "standardised pooling must report the block, in which two of three endpoints "
        "are recovered well"
    )
    assert table["recoverability_r_median_endpoint"].iloc[0] > 0.6
    assert table["n_endpoints"].iloc[0] == 3


def test_recoverability_drops_an_endpoint_with_no_observed_spread() -> None:
    """Standardising by a spread of zero is the failure the magnitude path already had."""
    numpy = pytest.importorskip("numpy")
    ridge = _ridge_module()

    rng = numpy.random.default_rng(23)
    rows = []
    for gene in range(40):
        value = rng.normal()
        rows.append({"fold": 0, "endpoint_id": "e_ok",
                     "response_true": value, "response_pred": value * 0.9})
        rows.append({"fold": 0, "endpoint_id": "e_constant",
                     "response_true": 1e-9, "response_pred": rng.normal()})
    table = ridge.recoverability_by_fold(pd.DataFrame(rows))
    assert table["n_endpoints"].iloc[0] == 1, "the constant endpoint must not be a divisor"
    assert table["recoverability_r"].iloc[0] > 0.95


# --------------------------------------------------------------------------- #
# The negative control, and the second substitution question
# --------------------------------------------------------------------------- #
#
# A panel in which every row fails says nothing about whether the criteria can tell an
# informative input from an uninformative one, so one row is run on an input that is
# known to carry nothing. And a second question is asked of the same deposit, three
# dyes standing in for the fourth, which is the only arm here that could reach a proxy
# tier and therefore the arm whose feature split has to be checked hardest.


def _stain_dropout_module():
    return _load(
        "external_periscope_stain_dropout", EXTERNAL / "periscope" / "analyse_stain_dropout.py"
    )


def test_permutation_shuffles_inside_a_group_and_never_across_one() -> None:
    """The control must destroy the association without moving a row between screens.

    Shuffling across screens would additionally destroy the plate structure, and the
    response is defined relative to same-plate controls, so a collapse could then be
    blamed on the baseline rather than on the permutation. Inside a screen there is no
    such escape.
    """
    numpy = pytest.importorskip("numpy")
    ridge = _ridge_module()

    groups = numpy.array(["P1"] * 60 + ["P2"] * 40)
    values = numpy.arange(100, dtype=numpy.float32).reshape(100, 1)
    shuffled = values.copy()
    record = ridge.permute_rows_within(shuffled, groups, seed=0)

    assert record["n_groups"] == 2 and record["n_rows"] == 100
    assert record["fraction_moved"] > 0.9, "a shuffle that moves almost nothing is not one"
    for name, low, high in (("P1", 0, 60), ("P2", 60, 100)):
        before = set(values[low:high, 0].tolist())
        after = set(shuffled[low:high, 0].tolist())
        assert before == after, f"{name} exchanged rows with another screen"


def test_permutation_leaves_the_target_side_untouched() -> None:
    """Reliability reads only the target block, so the control must not move it.

    This is what makes the control readable: recoverability and reliability come from
    different blocks, so a permutation of the input block is expected to collapse one
    and leave the other exactly where it was. If both moved, the row would not separate
    the two explanations the tier rule exists to separate.
    """
    numpy = pytest.importorskip("numpy")
    ridge = _ridge_module()
    analyse = _load("external_periscope_analyse_guide", EXTERNAL / "periscope" / "analyse_guide.py")

    frame = pd.DataFrame(
        [
            {"screen_id": "P1", "perturbation_id": gene, "guide_id": f"{gene}{guide}",
             "is_control": False, "e1": value, "e2": value * 2.0}
            for gene, value in (("g1", 1.0), ("g2", 3.0), ("g3", -2.0), ("g4", 0.5))
            for guide in ("a", "b")
        ]
        + [
            {"screen_id": "P1", "perturbation_id": "nontargeting", "guide_id": guide,
             "is_control": True, "e1": 0.0, "e2": 0.0}
            for guide in ("c1", "c2")
        ]
    )
    before = analyse.reliability_from_guides(frame, ["e1", "e2"], seed=0)

    inputs = numpy.arange(len(frame), dtype=numpy.float32).reshape(-1, 1)
    ridge.permute_rows_within(inputs, frame["screen_id"].to_numpy(), seed=0)
    after = analyse.reliability_from_guides(frame, ["e1", "e2"], seed=0)
    assert before["reliability"].tolist() == after["reliability"].tolist()


def test_the_control_cannot_reselect_its_own_penalty() -> None:
    """Selection and a fixed ratio are alternatives, not a precedence to be guessed.

    With the association destroyed the best penalty is the largest on the grid, which
    ``select_alpha`` refuses to return because a boundary choice is an artefact of where
    the grid was cut. The control therefore fixes the ratio at what the observed run
    selected, and a run that passed both would silently be a different experiment.
    """
    analyse = _load("external_periscope_analyse_guide", EXTERNAL / "periscope" / "analyse_guide.py")
    ridge = _ridge_module()
    with pytest.raises(ValueError, match="not both"):
        ridge.fit_folds(
            pd.DataFrame({"screen_id": [], "perturbation_id": [], "is_control": []}),
            None, None, [], n_folds=2, seed=0, alpha=1.0, alpha_ratio=0.01,
            reporter_id="x",
        )
    parser = analyse.build_parser()
    args = parser.parse_args(
        ["--prepared", ".", "--output-dir", "/nonexistent", "--permute-inputs", "within_screen"]
    )
    assert args.permute_inputs == "within_screen" and args.alpha_ratio is None, (
        "the control's penalty is not defaulted; the run script passes it explicitly"
    )


def test_no_input_column_is_attributable_to_the_held_out_dye() -> None:
    """The one guarantee the stain-dropout arm rests on, asserted where it is relied on.

    A cross-channel ``Correlation`` column names two dyes, so holding out WGA must drop
    every column that mentions WGA at all, not only the WGA-only ones. These column
    names are taken from the real 2,508-column guide block.
    """
    dropout = _stain_dropout_module()
    columns = [
        "Cells_Granularity_10_WGA",
        "Cells_Correlation_Correlation_ConA_WGA",
        "Cells_Correlation_Correlation_ConA_Cycle01_DAPI",
        "Cells_Intensity_MeanIntensity_Phalloidin",
        "Nuclei_Texture_Contrast_DAPI_3_00_256",
        "Cells_Correlation_Correlation_ConA_DAPI_Painting",
        "Cells_Intensity_MeanIntensity_ConA",
    ]
    partition = dropout.split_for("WGA", columns)
    assert partition["target"] == ["Cells_Granularity_10_WGA"]
    assert "Cells_Correlation_Correlation_ConA_WGA" in partition["excluded_cross_channel"]
    for column in partition["input"]:
        assert "WGA" not in dropout.CHANNELS.of(column), column


def test_a_dye_target_with_no_columns_of_its_own_is_an_error() -> None:
    """Refusing beats fitting a reporter that has no endpoints."""
    dropout = _stain_dropout_module()
    with pytest.raises(SystemExit, match="no column is attributable"):
        dropout.split_for(
            "WGA",
            ["Cells_Intensity_MeanIntensity_ConA", "Nuclei_Texture_Contrast_DAPI_3_00_256",
             "Cells_Intensity_MeanIntensity_Phalloidin"],
        )


def test_the_reliability_summary_survives_the_rename_its_caller_does() -> None:
    """The composition, not the parts, is what the analysis scripts run.

    ``reliability_from_guides`` renames the estimator's ``block`` column to
    ``screen_id`` before writing its per-screen table, and ``summarise_reliability``
    then reduces that same table. Both halves were tested separately and passed while
    the composition raised ``KeyError: ['block'] not found in axis`` twenty-five minutes
    into a fit. Identifier columns are now ignored by dtype, and this test runs the two
    in the order the scripts run them.
    """
    analyse = _load("external_periscope_analyse_guide", EXTERNAL / "periscope" / "analyse_guide.py")
    reliability = _load("external_periscope_reliability", EXTERNAL / "periscope" / "_reliability.py")

    frame = pd.DataFrame(
        [
            {"screen_id": screen, "perturbation_id": gene, "guide_id": f"{gene}{guide}",
             "is_control": False, "e1": value, "e2": value * 2.0}
            for screen in ("P1", "P2")
            for gene, value in (("g1", 1.0), ("g2", 3.0), ("g3", -2.0), ("g4", 0.5))
            for guide in ("a", "b")
        ]
        + [
            {"screen_id": screen, "perturbation_id": "nontargeting", "guide_id": guide,
             "is_control": True, "e1": 0.0, "e2": 0.0}
            for screen in ("P1", "P2") for guide in ("c1", "c2")
        ]
    )
    within = analyse.reliability_from_guides(frame, ["e1", "e2"], seed=0)
    assert "screen_id" in within and "block" not in within
    summary = reliability.summarise_reliability(within, floor=analyse.RELIABILITY_FLOOR)
    assert summary["n_blocks"] == 2.0
    assert summary["reliability"] == pytest.approx(1.0, abs=1e-9)
    # A perfectly reliable target needs no more guides than it has.
    assert summary["guides_per_gene_for_floor"] < summary["median_guides_per_gene"]


def test_the_oasis_permutation_moves_whole_wells_inside_their_plate() -> None:
    """Three invariants, each of which a plausible implementation breaks.

    The control is only readable on this dataset, because cpg0021's reliability gate
    fires before the input is consulted at all and its verdict cannot move. Here
    reliability is 0.72 and 0.64, so the rule does reach the input.

    A well must keep one feature vector across its two reporter rows, or the same well
    would carry two different profiles. Features must not cross plates, or the collapse
    could be attributed to the plate baseline rather than to the permutation, since
    response is defined against same-plate controls. And the vector must move as a unit
    rather than column by column, or the control would be testing feature scrambling
    instead of broken pairing.
    """
    numpy = pytest.importorskip("numpy")
    analyse = _load("external_oasis_analyse", EXTERNAL / "oasis" / "analyse.py")

    inputs = pd.DataFrame([
        {"observation_id": f"{screen}w{well}|{reporter}", "input_cell_id": f"{screen}w{well}",
         "screen_id": screen, "f1": float(well), "f2": float(well) * 10.0}
        for screen in ("P1", "P2") for well in range(6) for reporter in ("mtt", "ldh")
    ])
    permuted, record = analyse.permute_inputs_within_screen(inputs, ["f1", "f2"], seed=0)

    assert record["n_screens"] == 2 and record["n_wells"] == 12
    assert record["fraction_moved"] > 0.5
    assert (permuted.groupby("input_cell_id")[["f1", "f2"]].nunique() == 1).all().all(), (
        "a well received two different feature sets for its two reporters"
    )
    for screen in ("P1", "P2"):
        before = set(map(tuple, inputs.loc[inputs.screen_id == screen, ["f1", "f2"]].to_numpy()))
        after = set(map(tuple, permuted.loc[permuted.screen_id == screen, ["f1", "f2"]].to_numpy()))
        assert before == after, f"{screen} exchanged features with another plate"
    assert numpy.allclose(permuted["f2"], permuted["f1"] * 10.0), (
        "the feature vector must move as a unit; column-wise shuffling is a different control"
    )
    # Identifiers, labels and the target side are untouched.
    assert permuted["observation_id"].tolist() == inputs["observation_id"].tolist()
    assert permuted["input_cell_id"].tolist() == inputs["input_cell_id"].tolist()


def test_a_batch_restriction_filters_all_three_tables_together() -> None:
    """Filtering observations alone would leave orphan target rows in the reliability pairs."""
    analyse = _load("external_oasis_analyse", EXTERNAL / "oasis" / "analyse.py")

    observations = pd.DataFrame([
        {"observation_id": f"{batch}|{well}", "batch_id": batch, "is_control": False,
         "perturbation_id": f"c{well}", "input_cell_id": f"{batch}w{well}",
         "screen_id": batch, "reporter_id": "mtt"}
        for batch in ("b1", "b2") for well in range(3)
    ])
    inputs = observations.assign(f1=1.0)
    targets = pd.DataFrame([
        {"observation_id": row, "endpoint_id": "e", "y_true": 1.0}
        for row in observations["observation_id"]
    ])
    kept_obs, kept_in, kept_tg = analyse.restrict_to_batches(
        observations, inputs, targets, ["b1"]
    )
    assert set(kept_obs["batch_id"]) == {"b1"}
    assert len(kept_in) == len(kept_tg) == 3
    assert set(kept_tg["observation_id"]) == set(kept_obs["observation_id"]), (
        "a target row survived whose observation was dropped"
    )
    with pytest.raises(SystemExit, match="unknown batch"):
        analyse.restrict_to_batches(observations, inputs, targets, ["b3"])


def test_a_non_finite_validation_score_is_refused_rather_than_minimised() -> None:
    """``min`` over NaN scores returns the first element, which is the smallest ratio.

    That is how a missing value in the target block presented itself as "the selected
    penalty ratio is at the edge of the grid, so the grid does not bracket the optimum".
    The statement was true of a meaningless list, the grid was widened twice on the
    strength of it, and the real cause was a column of NaN. A score that is not a number
    is a broken evaluation, not a bad penalty.
    """
    numpy = pytest.importorskip("numpy")
    ridge = _ridge_module()

    # Enough features relative to rows, and enough noise, that a middling penalty wins.
    # Pure noise would select the largest penalty and pure signal the smallest, and
    # either would trip the boundary guard and hide what this test is about.
    rng = numpy.random.default_rng(4)
    x = rng.normal(size=(300, 80))
    y = x @ rng.normal(size=(80, 2)) + rng.normal(scale=1.0, size=(300, 2))
    block_of_row = numpy.arange(300) % 3

    clean = ridge.block_moments(x, y, block_of_row, 3)
    total = {key: sum(s[key] for s in clean) for key in clean[0]}
    ratio, scored = ridge.select_alpha(
        x, y, block_of_row, clean, total, outer_fold=0, inner_fold=1
    )
    assert all(numpy.isfinite(error) for _, error in scored)
    assert ratio not in (ridge.ALPHA_RATIOS[0], ridge.ALPHA_RATIOS[-1])

    # One missing target value is all it takes: it enters sumsq_y and the cross moments
    # and makes every penalty's error non-finite at once.
    y[7, 1] = numpy.nan
    broken = ridge.block_moments(x, y, block_of_row, 3)
    total = {key: sum(s[key] for s in broken) for key in broken[0]}
    with pytest.raises(SystemExit, match="not finite"):
        ridge.select_alpha(x, y, block_of_row, broken, total, outer_fold=0, inner_fold=1)
