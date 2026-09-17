#!/usr/bin/env python3
"""Prepare cpg0037-oasis as a canonical measurement-sufficiency dataset.

Ewald et al., *Cell Systems* 17(5):101566 (2026), doi:10.1016/j.cels.2026.101566,
image 1,085 compounds at eight concentrations in primary human hepatocytes across five
model systems, and run **two cytotoxicity assays on the same wells they image**. That
combination is what makes the deposit usable here: the cheap input is a genuine
brightfield channel and the expensive readout is a biochemical plate assay rather than
another image, so it tests whether measurement sufficiency holds across a change of
measurement *kind*, not only across a change of platform.

The two readouts are named here by what the deposit contains and what it measures, not
by assay brand. The published abstract says only "two cytotoxicity assays"; the
released per-plate ``biochem.parquet`` carries ``mtt_lumi`` / ``mtt_normalized`` and
``ldh_abs`` / ``ldh_normalized``. The first pair is a **luminescence** reading, which
the classical colorimetric MTT assay is not, so the ``mtt`` prefix is treated as the
deposit's column name rather than as an identification of the assay. This preparation
therefore calls the reporters a luminescent metabolic-activity readout and an
LDH-release readout. Naming a third party's assay on the strength of a column prefix
is the kind of claim that survives into a manuscript and is wrong there.

Two properties must be stated wherever a result from this dataset is reported.

**Pairing is well-level, not cell-level.** MTT and LDH are plate assays and cannot
be resolved to single cells. The canonical schema names its identifiers
``input_cell_id`` and ``target_cell_id``; here both carry the well id. The adapter
manifest therefore declares ``exact_pairing: false``, and
``AdapterCapabilities.require("exact_pairing")`` will refuse the same-cell
falsification analyses rather than run them on a unit that cannot support them.

**The input is brightfield only, by positive attribution.** Of 5,640 feature columns
in a plate profile, 808 are attributable to Brightfield alone and become the input.
The other 4,832 are excluded and counted in the provenance record: 120 cross-channel
columns that name Brightfield together with a fluorescence channel (the
``Correlation_K``, ``Correlation_Manders`` and ``Correlation_RWC`` families, which
would import the fluorescence signal directly), 4,512 attributable to fluorescence
channels, and 200 attributable to no channel at all. That last group is excluded on
purpose: ``AreaShape`` and ``Threshold`` features are computed inside masks drawn on
the fluorescence channels, so they are not obtainable from brightfield alone and
admitting them would overstate what a label-free measurement can do.

Nothing is downloaded into the repository. ``--output-dir`` is refused if it
resolves inside the checkout.

Example, writing to a caller-owned directory outside the checkout:

    python studies/external/oasis/prepare.py \\
      --output-dir ../external/oasis_canonical \\
      --source axiom --max-plates 2
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Iterator

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common import (  # noqa: E402
    CellProfilerChannels,
    frame_feature_columns,
    partition_features,
    reject_repo_paths,
    resolve_source,
)

DATASET_ID = "cpg0037_oasis"
PUBLIC_BASE = "https://cellpainting-gallery.s3.amazonaws.com"
BUCKET_PREFIX = "cpg0037-oasis"
ENV_VAR = "CPG0037_OASIS_BASE"

#: Channel vocabulary of the OASIS CellProfiler tables, read from a plate profile.
CHANNELS = CellProfilerChannels(("Brightfield", "DNA", "Mito", "AGP", "RNA", "ER"))
INPUT_CHANNELS = ("Brightfield",)

#: The five model systems are top-level prefixes of the deposit, but only ``axiom``
#: releases a uniform per-plate profile layer. Measured against the bucket on
#: 2026-08-13:
#:
#:   axiom      66 plates as profiles/<batch>/<plate>/<plate>_augmented.csv.gz, 0.39 GB
#:              over four production batches: prod_25 (15), prod_26 (17), prod_27 (17),
#:              prod_30 (17)
#:   broad      publishes profiles_anonymized/ rather than profiles/
#:   hepatopac  one profile, named HepatoPAC_Preciscan_* rather than per plate
#:   insphero   no profiles/ prefix at all
#:   xellar     profiles named OASIS--20Xvs40X--*, a magnification comparison
#:
#: The default is therefore axiom alone. The other four are reachable with --source
#: but do not follow this layout, and ``list_profiles`` raises rather than returning an
#: empty inventory, so a partial panel cannot be mistaken for a complete one.
SOURCES = ("axiom", "broad", "hepatopac", "insphero", "xellar")
DEFAULT_SOURCES = ("axiom",)

#: Targeted readouts. Two independent assays, so two reporters rather than one
#: reporter with four endpoints: measurement sufficiency is decided per measurement,
#: and pooling a viability assay with a membrane-integrity assay would hide that.
REPORTERS: dict[str, dict[str, object]] = {
    "mtt_viability": {
        "reporter_name": "metabolic activity, luminescent readout",
        "biological_system": "Cell viability",
        "target_modality": "biochemical_plate_assay",
        "endpoints": ("mtt_lumi", "mtt_normalized"),
    },
    "ldh_release": {
        "reporter_name": "LDH release",
        "biological_system": "Membrane integrity",
        "target_modality": "biochemical_plate_assay",
        "endpoints": ("ldh_abs", "ldh_normalized"),
    },
}

#: Value of ``Metadata_control_type`` marking an in-plate negative control. Verified
#: at 69 of 384 wells on plate_41002690, matching its DMSO well count exactly.
NEGATIVE_CONTROL = "negcon"

#: What this dataset can and cannot support. These are statements about what the
#: accession physically contains, not tuning knobs: MTT and LDH are plate assays and
#: can never become cell-level, so ``exact_pairing`` cannot be promoted.
CAPABILITIES = {
    # Well-level pairing. The same-cell analyses must refuse to run.
    "exact_pairing": False,
    "raw_images": False,
    "guides": False,
    # 69 negcon wells per 384-well compound plate, plus whole reference plates.
    "screen_matched_controls": True,
    # Four independent production batches within the axiom model system:
    # prod_25, prod_26, prod_27, prod_30.
    "repeated_screens": True,
    # Cell counts exist but come from fluorescence-driven segmentation, so they are
    # not covariates a label-free measurement could supply.
    "covariates": False,
}


def _get(url: str, timeout: int = 120) -> bytes:
    with urllib.request.urlopen(url, timeout=timeout) as handle:  # noqa: S310 - public https only
        return handle.read()


def list_profiles(base: str, source: str) -> list[str]:
    """List the augmented plate-profile keys for one model system.

    Uses the anonymous S3 REST listing, paginated. There is deliberately no
    fallback to a hard-coded plate list: a silently truncated inventory would make
    a partial panel look complete.
    """
    prefix = f"{BUCKET_PREFIX}/{source}/workspace/profiles/"
    keys: list[str] = []
    token = ""
    while True:
        url = f"{base}/?list-type=2&prefix={prefix}&max-keys=1000"
        if token:
            url += f"&continuation-token={urllib.parse.quote(token, safe='')}"
        body = _get(url).decode("utf-8")
        keys.extend(re.findall(r"<Key>(.*?)</Key>", body))
        truncated = "<IsTruncated>true</IsTruncated>" in body
        token_match = re.search(r"<NextContinuationToken>(.*?)</NextContinuationToken>", body)
        if not (truncated and token_match):
            break
        token = token_match.group(1)
    return sorted(key for key in keys if key.endswith("_augmented.csv.gz"))


def read_profile(base: str, key: str) -> pd.DataFrame:
    """Read one gzipped augmented plate profile."""
    return pd.read_csv(io.BytesIO(gzip.decompress(_get(f"{base}/{key}"))), low_memory=False)


def collapse_benign_duplicates(
    profile: pd.DataFrame, *, plate: str, consequential: list[str]
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Collapse repeated wells, but only where the repetition cannot change a result.

    Some released plates carry a well more than once. On ``plate_41002689`` four of
    384 wells repeat: three are byte-identical rows, and one carries two rows that
    differ in ``Metadata_OASIS_ID`` alone while agreeing on every feature, both assay
    readouts, the compound and the concentration.

    Dropping duplicates outright would be wrong, because a well whose *measurements*
    disagree is a genuine ambiguity that a preparation must not resolve by picking a
    row. Refusing outright would also be wrong, because it would discard a plate over
    a duplicated catalogue annotation. So the rule is narrow and stated: collapse a
    repeated well only when every column this preparation actually consumes agrees
    across its rows, and raise, naming the wells and the disagreeing columns, when any
    of them does not.

    Returns:
        The collapsed profile and a record of what was collapsed, which the caller
        writes into ``provenance.json``. Nothing is dropped silently.
    """
    wells = profile["Metadata_well_id"].astype(str)
    if not wells.duplicated().any():
        return profile, {"n_exact_duplicate_rows": 0, "n_wells_collapsed": 0, "disagreeing_columns": []}

    exact = profile.drop_duplicates()
    n_exact = len(profile) - len(exact)

    repeated = exact["Metadata_well_id"].astype(str)
    still = set(repeated[repeated.duplicated(keep=False)])
    disagreeing: set[str] = set()
    blocking: list[str] = []
    for well in sorted(still):
        rows = exact.loc[repeated.eq(well)]
        columns = [column for column in rows.columns if rows[column].astype(str).nunique(dropna=False) > 1]
        disagreeing.update(columns)
        if set(columns) & set(consequential):
            blocking.append(well)
    if blocking:
        raise ValueError(
            f"{plate}: {len(blocking)} well(s) repeat with disagreeing values in columns this "
            f"preparation consumes: {sorted(set(disagreeing) & set(consequential))}. "
            f"First wells: {blocking[:3]}. A repeated well whose measurements disagree is a real "
            "ambiguity and must not be resolved by choosing a row."
        )
    collapsed = exact.drop_duplicates(subset="Metadata_well_id", keep="first")
    return collapsed, {
        "n_exact_duplicate_rows": int(n_exact),
        "n_wells_collapsed": int(len(exact) - len(collapsed)),
        "disagreeing_columns": sorted(disagreeing),
    }


def batch_of(key: str) -> str:
    """Batch identifier from a profile key, for example ``prod_25``."""
    match = re.search(r"/profiles/([^/]+)/", key)
    if not match:
        raise ValueError(f"cannot read a batch from profile key: {key}")
    return match.group(1)


def plate_of(key: str) -> str:
    match = re.search(r"/([^/]+)_augmented\.csv\.gz$", key)
    if not match:
        raise ValueError(f"cannot read a plate from profile key: {key}")
    return match.group(1)


def canonical_rows(
    profile: pd.DataFrame, *, source: str, batch: str, plate: str, feature_columns: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Convert one plate profile into observation, input and target rows.

    Returns ``(observations, inputs, targets)``. Targets are long form and carry one
    row per (well, endpoint); a well whose assay value is absent yields a row with a
    null ``y_true`` rather than being dropped, so the availability mask stays honest.
    """
    required = {"Metadata_well_id", "Metadata_compound_name", "Metadata_control_type"}
    missing = required.difference(profile.columns)
    if missing:
        raise ValueError(f"{plate}: profile lacks {sorted(missing)}")

    wells = profile["Metadata_well_id"].astype(str)
    if wells.duplicated().any():
        raise ValueError(f"{plate}: well_id is not unique within the plate")

    # The plate is the screen: it is the unit that carries its own DMSO controls, and
    # screen_id is what control-relative response subtracts within. Using the batch
    # here would pool controls over roughly seventeen plates and discard the plate-
    # matched precision the design provides. The batch is kept as its own column for
    # environment-transfer analyses, which is the other thing a screen means in OPS
    # but which here is a coarser grouping than control matching.
    screen_id = plate
    control = profile["Metadata_control_type"].astype("string").str.strip().str.casefold()
    is_control = control.eq(NEGATIVE_CONTROL).fillna(False)

    observations = pd.DataFrame(
        {
            "observation_id": wells,
            # Well-level pairing. Both identifiers carry the well because MTT and LDH
            # cannot be resolved to a cell. Declared as exact_pairing=false.
            "input_cell_id": wells,
            "target_cell_id": wells,
            "assay_id": "",  # filled per reporter below
            "perturbation_id": profile["Metadata_compound_name"].astype(str),
            "guide_id": pd.NA,
            "screen_id": screen_id,
            "batch_id": f"{source}_{batch}",
            "well_id": profile.get("Metadata_Well", pd.Series([""] * len(profile))).astype(str),
            # There is no imaging field in a well-level table, and the plate is already
            # the screen, so the field unit is the plate row: wells in one row share a
            # dispensing and imaging pass.
            "field_id": plate + "|row" + profile.get("Metadata_row", pd.Series([0] * len(profile))).astype(str),
            "is_control": is_control.to_numpy(),
        }
    )

    inputs = profile[feature_columns].copy()
    inputs.insert(0, "input_cell_id", wells)

    frames = []
    for reporter_id, spec in REPORTERS.items():
        for endpoint in spec["endpoints"]:  # type: ignore[union-attr]
            column = f"Metadata_{endpoint}"
            values = profile[column] if column in profile.columns else pd.Series([pd.NA] * len(profile))
            frames.append(
                pd.DataFrame(
                    {
                        "observation_id": wells,
                        "reporter_id": reporter_id,
                        "endpoint_id": endpoint,
                        "y_true": pd.to_numeric(values, errors="coerce").to_numpy(),
                    }
                )
            )
    targets = pd.concat(frames, ignore_index=True)
    return observations, inputs, targets


def iter_plates(base: str, sources: tuple[str, ...], max_plates: int | None) -> Iterator[tuple[str, str]]:
    """Yield ``(source, key)`` for every plate profile, capped by ``max_plates`` per source."""
    for source in sources:
        keys = list_profiles(base, source)
        if not keys:
            raise SystemExit(
                f"no augmented plate profiles found under {BUCKET_PREFIX}/{source}/workspace/profiles/. "
                "The deposit layout may have changed; refusing to report an empty inventory as success."
            )
        for key in keys[:max_plates] if max_plates else keys:
            yield source, key


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", required=True, help="destination outside the repository")
    parser.add_argument("--base", default=None, help=f"object-store base URL (env {ENV_VAR})")
    parser.add_argument("--source", action="append", choices=SOURCES, default=None,
                        help="model system; repeatable; default is axiom, the only one with a "
                             "uniform per-plate profile layer (see SOURCES)")
    parser.add_argument("--max-plates", type=int, default=None,
                        help="cap plates per model system, for a smoke run")
    parser.add_argument("--format", choices=("parquet", "csv"), default="parquet")
    parser.add_argument("--dry-run", action="store_true", help="list what would be read and exit")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    base = resolve_source(args.base, what="the Cell Painting Gallery", env_var=ENV_VAR,
                          flag="--base", default=PUBLIC_BASE)
    output = reject_repo_paths(args.output_dir)
    sources = tuple(args.source) if args.source else DEFAULT_SOURCES

    plates = list(iter_plates(base, sources, args.max_plates))
    print(json.dumps({"plates": len(plates), "sources": list(sources), "base": base}, indent=2))
    if args.dry_run:
        return 0

    observations, inputs, targets = [], [], []
    duplicate_records: list[dict[str, object]] = []
    partition: dict[str, list[str]] | None = None
    for index, (source, key) in enumerate(plates, start=1):
        profile = read_profile(base, key)
        if partition is None:
            partition = partition_features(
                frame_feature_columns(profile), channels=CHANNELS, input_channels=INPUT_CHANNELS
            )
            print(json.dumps({stage: len(value) for stage, value in partition.items()}, indent=2))
        consequential = (
            partition["input"]
            + [f"Metadata_{endpoint}" for spec in REPORTERS.values() for endpoint in spec["endpoints"]]
            + ["Metadata_compound_name", "Metadata_control_type", "Metadata_Well"]
        )
        profile, collapsed = collapse_benign_duplicates(profile, plate=plate_of(key), consequential=consequential)
        if collapsed["n_wells_collapsed"]:
            duplicate_records.append({"plate": plate_of(key), **collapsed})
        missing = set(partition["input"]).difference(profile.columns)
        if missing:
            raise SystemExit(
                f"{plate_of(key)}: profile lacks {len(missing)} of the input feature columns fixed by "
                "the first plate. Feature sets must not vary across plates; prepare each differing "
                "layout separately rather than silently intersecting them."
            )
        rows = canonical_rows(
            profile, source=source, batch=batch_of(key), plate=plate_of(key),
            feature_columns=partition["input"],
        )
        observations.append(rows[0])
        inputs.append(rows[1])
        targets.append(rows[2])
        print(f"  [{index}/{len(plates)}] {plate_of(key)}  wells={len(rows[0])}", flush=True)

    assert partition is not None
    write(output, observations, inputs, targets, partition, sources, base, args.format, duplicate_records)
    return 0


def write(
    output: Path,
    observations: list[pd.DataFrame],
    inputs: list[pd.DataFrame],
    targets: list[pd.DataFrame],
    partition: dict[str, list[str]],
    sources: tuple[str, ...],
    base: str,
    format_: str,
    duplicate_records: list[dict[str, object]],
) -> None:
    """Write canonical tables, the adapter manifest and the provenance record."""
    output.mkdir(parents=True, exist_ok=True)
    observation_table = pd.concat(observations, ignore_index=True)
    input_table = pd.concat(inputs, ignore_index=True)
    target_table = pd.concat(targets, ignore_index=True)

    screens = sorted(observation_table["screen_id"].unique())
    assays = pd.DataFrame(
        [
            {
                "assay_id": f"{reporter_id}|{screen}",
                "reporter_id": reporter_id,
                "screen_id": screen,
                "endpoint_ids": ",".join(spec["endpoints"]),  # type: ignore[arg-type]
            }
            for reporter_id, spec in REPORTERS.items()
            for screen in screens
        ]
    )
    reporters = pd.DataFrame(
        [
            {
                "reporter_id": reporter_id,
                "reporter_name": spec["reporter_name"],
                "biological_system": spec["biological_system"],
                "target_modality": spec["target_modality"],
                "endpoint_schema_id": f"{DATASET_ID}_{reporter_id}_v1",
            }
            for reporter_id, spec in REPORTERS.items()
        ]
    )
    # One observation row per (well, reporter), because assay_id is reporter-specific.
    # reporter_id is carried explicitly even though the canonical schema does not
    # require it: training.prediction_long_table reads it from the inputs metadata,
    # and without it every prediction is labelled "unspecified" and the downstream
    # per-reporter analyses cannot group.
    expanded = pd.concat(
        [
            observation_table.assign(
                reporter_id=reporter_id,
                assay_id=reporter_id + "|" + observation_table["screen_id"],
                observation_id=observation_table["observation_id"] + "|" + reporter_id,
            )
            for reporter_id in REPORTERS
        ],
        ignore_index=True,
    )
    target_table = target_table.assign(
        observation_id=target_table["observation_id"] + "|" + target_table["reporter_id"]
    ).drop(columns="reporter_id")
    # pairing_key is the value that establishes the linkage, not the name of the
    # key, and it must be unique within an assay. Here that value is the well: the
    # brightfield profile and the biochemical readout are joined because they come
    # from the same physical well, and each well appears once per assay.
    pairing = expanded[["observation_id", "input_cell_id", "target_cell_id", "assay_id"]].assign(
        pairing_key=expanded["input_cell_id"], pairing_confidence=1.0
    )

    suffix = "parquet" if format_ == "parquet" else "csv"
    written = {}
    for name, table in {
        "observations": expanded,
        "reporters": reporters,
        "assays": assays,
        "pairing": pairing,
        "inputs": input_table,
        "targets": target_table,
    }.items():
        path = output / f"{name}.{suffix}"
        table.to_parquet(path, index=False) if format_ == "parquet" else table.to_csv(path, index=False)
        written[name] = path.name

    manifest = {
        "dataset_id": DATASET_ID,
        "capabilities": dict(CAPABILITIES),
        "input_schema": {
            "schema_id": f"{DATASET_ID}_brightfield_v1",
            "representation": "tabular",
            "feature_names": partition["input"],
        },
        "tables": {name: filename for name, filename in written.items()},
    }
    (output / "canonical_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    provenance = {
        "dataset_id": DATASET_ID,
        "source_base": base,
        "bucket_prefix": BUCKET_PREFIX,
        "model_systems": list(sources),
        "screens": screens,
        "n_observations": int(len(expanded)),
        "n_wells": int(observation_table["observation_id"].nunique()),
        "n_control_observations": int(expanded["is_control"].sum()),
        "feature_partition": {stage: len(value) for stage, value in partition.items()},
        "input_channels": list(INPUT_CHANNELS),
        "channel_vocabulary": list(CHANNELS.names),
        "excluded_cross_channel_examples": partition["excluded_cross_channel"][:10],
        "excluded_channel_free_examples": partition["excluded_channel_free"][:10],
        "collapsed_duplicate_wells": duplicate_records,
        "collapsed_duplicate_note": (
            "Some released plates carry a well more than once. A repeated well is collapsed "
            "only when every column this preparation consumes agrees across its rows; a "
            "disagreement in any of them raises. Nothing is dropped silently."
        ),
        "pairing_unit": "well",
        "pairing_note": (
            "input_cell_id and target_cell_id both carry the well id. MTT and LDH are "
            "plate assays and cannot be resolved to single cells, so exact_pairing is "
            "declared false and same-cell falsification is ineligible."
        ),
        "excluded_channel_free_note": (
            "Channel-free features (AreaShape, Threshold, object counts) are excluded "
            "because their masks and thresholds are computed on fluorescence channels "
            "and are therefore not obtainable from brightfield alone."
        ),
    }
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
