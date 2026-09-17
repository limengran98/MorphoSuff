#!/usr/bin/env python3
"""Prepare cpg0021-periscope as a canonical measurement-sufficiency dataset.

PERISCOPE (Ramezani et al., *Nat Methods* 2025) is a genome-wide optical pooled
screen with a Cell Painting readout. Its A549 arm matches this study's own cell
line, its perturbation library is genome scale, and every well carries
non-targeting guides by construction because the library is pooled. That makes it
the strongest available external test of the *perturbation and environment* half
of the claim.

**It cannot test the label-free half, and this script does not pretend otherwise.**
The cheap input here is four fluorescent dyes (DAPI/Painting, ConA, Phalloidin,
WGA) and the expensive readout is the fifth channel, an anti-TOMM20 antibody. There
is no transmitted-light channel. The manifest records ``cheap_input_is_label_free:
false`` so that a result from this dataset cannot be quoted as label-free evidence.

Why it is worth using anyway: TOMM20 is not an arbitrary external target. This
study measures ``mitochondria, TOMM20 (cp)`` as one of its own 52 reporters, so the
comparison is same-target on a different platform rather than a generic transfer,
and the question becomes whether TOMM20 lands in the same tier in both.

**One confound must be controlled before any number from this dataset is reported.**
Anti-TOMM20 is Alexa Fluor 594 (615/24 nm), ConA is Cy5-like (680/42 nm) and
phalloidin is 530/30 nm. The published destaining addresses phenotype-versus-ISS
overlap and reports no phenotype-to-phenotype unmixing, so "four dyes predict
TOMM20" may in part be measuring spectral bleed-through rather than biology. Two
controls are required and neither is performed here: permute cells within a site
and confirm the score collapses, and check that predictive power survives in cells
where TOMM20 itself is knocked out. ``provenance.json`` records this as an
outstanding obligation rather than leaving it to be rediscovered.

Feature attribution is positive, as in the OASIS preparation. Of 3,766 feature
columns, 2,508 are attributable to the four input dyes and 823 to Mito. Sixty
cross-channel ``Correlation`` columns name Mito together with an input dye and are
excluded because they would import the target directly; 375 channel-free columns
(``AreaShape``, thresholds, counts) are excluded because their masks are drawn on
the phenotypic images, which include the target channel.

Example, writing to a caller-owned directory outside the checkout:

    python studies/external/periscope/prepare.py \\
      --output-dir ../external/periscope_canonical \\
      --plate CP186A --max-cells 20000
"""

from __future__ import annotations

import argparse
import gzip
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
    partition_features,
    reject_repo_paths,
    resolve_source,
)

DATASET_ID = "cpg0021_periscope_a549"
PUBLIC_BASE = "https://cellpainting-gallery.s3.amazonaws.com"
SINGLE_CELL_PREFIX = "cpg0021-periscope/broad/workspace/profiles/A549/single_cell/"
PROFILE_PREFIX = "cpg0021-periscope/broad/workspace/profiles/A549/"

#: Aggregation levels the deposit publishes, and what each costs and supports.
#:
#:   single_cell  125 GB over nine plates. One row per segmented cell, so it is the
#:                only level that supports the same-cell arm, and it is the level whose
#:                recoverability is comparable to the published study, which predicts
#:                per cell and aggregates the predictions into knockout responses.
#:   guide        about 8 GB. One row per (gene, sgRNA) per plate, median four guides
#:                per gene, which is what a guide-half reliability estimate needs and
#:                what analysis.stability.guide_half_consistency consumes.
#:   gene         about 2 GB. One row per gene per plate; no replicate structure left,
#:                so reliability is not estimable from it at all.
#:
#: Aggregating before fitting is an easier task than fitting per cell and aggregating
#: the predictions, because it removes noise from both sides. A guide-level
#: recoverability is therefore not comparable to the published number and must be
#: labelled as its own quantity rather than quoted against it.
LEVELS = ("single_cell", "guide", "gene")
LEVEL_PATTERNS = {
    "single_cell": "single_cell_profiles",
    "guide": "_guide_ALLBATCHES___",
    "gene": "_gene_ALLBATCHES___",
}
ENV_VAR = "CPG0021_PERISCOPE_BASE"

CHANNELS = CellProfilerChannels(("DAPI", "ConA", "Mito", "Phalloidin", "WGA"))
#: The four generic dyes. Not label-free; see the module docstring.
INPUT_CHANNELS = ("DAPI", "ConA", "Phalloidin", "WGA")
#: Anti-TOMM20. Includes the derived ``mito_tubeness`` columns, which the
#: case-insensitive channel matcher attributes to Mito rather than losing to the
#: channel-free bucket.
TARGET_CHANNELS = ("Mito",)

REPORTER_ID = "tomm20_mito"
REPORTER_NAME = "mitochondria, anti-TOMM20 (Cell Painting)"

#: Per-cell metadata columns this preparation depends on.
GENE_COLUMN = "Metadata_Foci_Barcode_MatchedTo_GeneCode"
GUIDE_COLUMN = "Metadata_Foci_Barcode_MatchedTo_Barcode"
PLATE_COLUMN = "Metadata_Foci_plate"
WELL_COLUMN = "Metadata_Foci_well"
SITE_COLUMN = "Metadata_Foci_site"
CELL_COLUMN = "Metadata_Cells_ObjectNumber"

#: Value of the gene code marking a non-targeting guide, that is, an in-well control.
NON_TARGETING = "nontargeting"

#: Columns that identify a row rather than measure it, and so keep their own type.
_IDENTIFIER_COLUMNS = frozenset({"observation_id", "input_cell_id", "target_cell_id"})

#: What this dataset can and cannot support. Statements about the accession, not
#: tuning knobs. ``exact_pairing`` is true because rows are segmented cells and every
#: channel is measured on the same cell; there is no transmitted-light channel here,
#: so this dataset can never carry the label-free half of the claim.
CAPABILITIES = {
    "exact_pairing": True,
    "raw_images": False,
    # sgRNA identity is read per cell by in-situ sequencing.
    "guides": True,
    # Pooled library: non-targeting guides are in every well by construction.
    "screen_matched_controls": True,
    # Nine plates across three independent viral transductions.
    "repeated_screens": True,
    # Cell size and shape are computed inside masks drawn on the phenotypic images,
    # which include the target channel.
    "covariates": False,
}


def _open(url: str, timeout: int = 300):
    return urllib.request.urlopen(url, timeout=timeout)  # noqa: S310 - public https only


def list_plate_profiles(base: str, level: str) -> list[str]:
    """List the nine per-plate profile keys at one aggregation level."""
    prefix = SINGLE_CELL_PREFIX if level == "single_cell" else PROFILE_PREFIX
    pattern = LEVEL_PATTERNS[level]
    keys: list[str] = []
    token = ""
    while True:
        url = f"{base}/?list-type=2&prefix={prefix}&max-keys=1000"
        if token:
            url += f"&continuation-token={urllib.parse.quote(token, safe='')}"
        body = _open(url).read().decode("utf-8")
        keys.extend(re.findall(r"<Key>(.*?)</Key>", body))
        match = re.search(r"<NextContinuationToken>(.*?)</NextContinuationToken>", body)
        if not ("<IsTruncated>true</IsTruncated>" in body and match):
            break
        token = match.group(1)
    found = sorted(
        key for key in keys
        if pattern in key and key.endswith(".csv.gz") and "normalized" not in key
    )
    if not found:
        raise SystemExit(
            f"no {level} profile objects matching {pattern!r} under {prefix}. The deposit "
            "layout may have changed; refusing to report an empty inventory as success."
        )
    return found


def plate_of(key: str) -> str:
    match = re.search(r"___(CP\d+[A-Z]?)___", key)
    if not match:
        raise ValueError(f"cannot read a plate from key: {key}")
    return match.group(1)


def read_header(base: str, key: str, *, probe_bytes: int = 8 << 20) -> list[str]:
    """Read the column header without downloading the whole multi-gigabyte object."""
    request = urllib.request.Request(
        f"{base}/{urllib.parse.quote(key)}", headers={"Range": f"bytes=0-{probe_bytes - 1}"}
    )
    raw = urllib.request.urlopen(request, timeout=300).read()  # noqa: S310
    import zlib

    text = zlib.decompressobj(16 + zlib.MAX_WBITS).decompress(raw)
    line = text.split(b"\n", 1)[0]
    return line.decode("utf-8").split(",")


def stream_chunks(base: str, key: str, *, usecols: list[str], chunk_rows: int) -> Iterator[pd.DataFrame]:
    """Stream a gzipped per-cell profile in row chunks, without materialising the file."""
    with _open(f"{base}/{urllib.parse.quote(key)}") as response:
        with gzip.open(response, mode="rt", encoding="utf-8") as handle:
            for chunk in pd.read_csv(handle, usecols=usecols, chunksize=chunk_rows, low_memory=False):
                yield chunk


def canonical_rows(
    chunk: pd.DataFrame, *, plate: str, input_columns: list[str], target_columns: list[str],
    level: str = "single_cell",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Convert one chunk of profile rows into observation, input and target rows.

    At ``single_cell`` the unit is a segmented cell and the well and site are known.
    At ``guide`` the deposit publishes only the gene and the sgRNA, so the unit is
    (gene, guide, plate) and there is no spatial structure below the plate: the field
    split is unavailable and ``field_id`` carries the plate.
    """
    gene = chunk[GENE_COLUMN].astype(str)
    if level == "single_cell":
        cell_id = (
            plate + "|" + chunk[WELL_COLUMN].astype(str)
            + "|" + chunk[SITE_COLUMN].astype(str)
            + "|" + chunk[CELL_COLUMN].astype(str)
        )
        well = chunk[WELL_COLUMN].astype(str)
        field = plate + "|" + well + "|" + chunk[SITE_COLUMN].astype(str)
    else:
        cell_id = plate + "|" + gene + "|" + chunk[GUIDE_COLUMN].astype(str)
        well = pd.Series([""] * len(chunk), index=chunk.index)
        field = pd.Series([plate] * len(chunk), index=chunk.index)
    is_control = gene.str.strip().str.casefold().eq(NON_TARGETING)

    observations = pd.DataFrame(
        {
            "observation_id": cell_id,
            "input_cell_id": cell_id,
            "target_cell_id": cell_id,
            "assay_id": f"{REPORTER_ID}|{plate}",
            "reporter_id": REPORTER_ID,
            "perturbation_id": gene,
            "guide_id": chunk[GUIDE_COLUMN].astype(str),
            # The plate is the independent batch: nine plates, three viral
            # transductions, which is what a whole-screen transfer holds out.
            "screen_id": plate,
            "well_id": well,
            "field_id": field,
            "is_control": is_control.to_numpy(),
        }
    )
    inputs = chunk[input_columns].copy()
    inputs.insert(0, "input_cell_id", cell_id)
    # Targets stay wide: one row per cell, one column per endpoint. The canonical
    # long layout exists to represent *sparse* availability, which is what the OPS
    # reporter blocks need. Here every cell carries all 823 endpoints, so long form
    # encodes nothing extra and costs 823x the rows: the full arm would be about nine
    # billion of them, which no machine makes reasonable. The four required canonical
    # tables are unaffected, because ``targets`` is an optional table.
    targets = chunk[target_columns].copy()
    targets.insert(0, "observation_id", cell_id)
    return observations, inputs, targets


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", required=True, help="destination outside the repository")
    parser.add_argument("--base", default=None, help=f"object-store base URL (env {ENV_VAR})")
    parser.add_argument("--plate", action="append", default=None, help="plate id such as CP186A; repeatable")
    parser.add_argument("--level", choices=LEVELS, default="single_cell",
                        help="aggregation level. single_cell (125 GB) is the only one comparable "
                             "to the published recoverability and the only one supporting the "
                             "same-cell arm; guide (about 8 GB) supports the whole tier chain and "
                             "gives a guide-half reliability; see LEVELS.")
    parser.add_argument("--max-cells", type=int, default=None,
                        help="cap cells per plate, for a smoke run. The full arm is 125 GB across nine plates.")
    parser.add_argument("--chunk-rows", type=int, default=20000)
    parser.add_argument("--rows-per-part", type=int, default=500000,
                        help="flush a partition file once this many cells have accumulated. "
                             "A whole plate is 1.2-1.9 million cells by 2,508 features, so "
                             "holding one before writing peaks near 40 GB during the concat.")
    parser.add_argument("--format", choices=("parquet", "csv"), default="parquet")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    base = resolve_source(args.base, what="the Cell Painting Gallery", env_var=ENV_VAR,
                          flag="--base", default=PUBLIC_BASE)
    output = reject_repo_paths(args.output_dir)

    keys = list_plate_profiles(base, args.level)
    if args.plate:
        wanted = set(args.plate)
        keys = [key for key in keys if plate_of(key) in wanted]
        missing = wanted.difference(plate_of(key) for key in keys)
        if missing:
            raise SystemExit(f"requested plates not present in the deposit: {sorted(missing)}")
    print(json.dumps({"level": args.level, "plates": [plate_of(key) for key in keys], "base": base}, indent=2))
    if args.dry_run:
        return 0

    header = read_header(base, keys[0])
    features = [column for column in header if not column.startswith("Metadata_")]
    partition = partition_features(
        features, channels=CHANNELS, input_channels=INPUT_CHANNELS, target_channels=TARGET_CHANNELS
    )
    print(json.dumps({stage: len(value) for stage, value in partition.items()}, indent=2))

    metadata_columns = (
        [GENE_COLUMN, GUIDE_COLUMN, PLATE_COLUMN, WELL_COLUMN, SITE_COLUMN, CELL_COLUMN]
        if args.level == "single_cell"
        else [GENE_COLUMN, GUIDE_COLUMN]
    )
    missing = set(metadata_columns).difference(header)
    if missing:
        raise SystemExit(f"single-cell profile lacks required metadata columns: {sorted(missing)}")
    usecols = metadata_columns + partition["input"] + partition["target"]

    # Written plate by plate rather than accumulated. Holding the full arm in memory
    # would be 11 million cells by 2,508 input features before anything is analysed.
    output.mkdir(parents=True, exist_ok=True)
    for name in ("inputs", "targets"):
        (output / name).mkdir(exist_ok=True)
    observations: list[pd.DataFrame] = []
    for key in keys:
        plate = plate_of(key)
        taken = 0
        plate_inputs: list[pd.DataFrame] = []
        plate_targets: list[pd.DataFrame] = []
        pending, part = 0, 0

        def flush(part: int) -> int:
            for name, frames in (("inputs", plate_inputs), ("targets", plate_targets)):
                block = pd.concat(frames, ignore_index=True)
                # Force one float type for every measurement column before writing.
                # Each partition is written independently, so pandas infers dtypes per
                # plate: a column that happens to hold only integral values on one plate
                # becomes int64 there and float64 elsewhere, and the partitions then
                # cannot be read as one dataset. Measured on the guide-level arm, 35 of
                # 824 target columns diverged this way, all RadialDistribution
                # mito_tubeness fractions that are constant on some plates. These are
                # continuous measurements; an integer column is an artefact of the
                # values happening to be round, never a statement about the quantity.
                measurement = [c for c in block.columns if c not in _IDENTIFIER_COLUMNS]
                block[measurement] = block[measurement].astype("float32")
                block.to_parquet(
                    output / name / f"plate={plate}.part{part:03d}.parquet", index=False
                )
            plate_inputs.clear()
            plate_targets.clear()
            print(f"  {plate}: part {part} written", flush=True)
            return part + 1

        for chunk in stream_chunks(base, key, usecols=usecols, chunk_rows=args.chunk_rows):
            if args.max_cells is not None and taken >= args.max_cells:
                break
            if args.max_cells is not None:
                chunk = chunk.head(args.max_cells - taken)
            rows = canonical_rows(
                chunk, plate=plate, input_columns=partition["input"],
                target_columns=partition["target"], level=args.level,
            )
            observations.append(rows[0])
            plate_inputs.append(rows[1])
            plate_targets.append(rows[2])
            taken += len(chunk)
            pending += len(chunk)
            if pending >= args.rows_per_part:
                part = flush(part)
                pending = 0
            print(f"  {plate}: {taken} cells", flush=True)
        if plate_inputs:
            part = flush(part)
        if part == 0:
            raise SystemExit(f"{plate}: streamed zero rows; refusing to write an empty partition")

    write(output, observations, partition, base, keys, args.format, args.level)
    return 0


def write(
    output: Path,
    observations: list[pd.DataFrame],
    partition: dict[str, list[str]],
    base: str,
    keys: list[str],
    format_: str,
    level: str,
) -> None:
    """Write the small canonical tables, the adapter manifest and the provenance.

    ``inputs`` and ``targets`` were already written per plate by the caller; only the
    tables small enough to hold whole arrive here.
    """
    observation_table = pd.concat(observations, ignore_index=True)
    screens = sorted(observation_table["screen_id"].unique())
    assays = pd.DataFrame(
        [
            {
                "assay_id": f"{REPORTER_ID}|{screen}",
                "reporter_id": REPORTER_ID,
                "screen_id": screen,
                "endpoint_ids": ",".join(partition["target"]),
            }
            for screen in screens
        ]
    )
    reporters = pd.DataFrame(
        [
            {
                "reporter_id": REPORTER_ID,
                "reporter_name": REPORTER_NAME,
                "biological_system": "Mitochondria",
                "target_modality": "cell_painting_stain",
                "endpoint_schema_id": f"{DATASET_ID}_{REPORTER_ID}_v1",
            }
        ]
    )
    pairing = observation_table[["observation_id", "input_cell_id", "target_cell_id", "assay_id"]].assign(
        pairing_key=observation_table["input_cell_id"], pairing_confidence=1.0
    )

    suffix = "parquet" if format_ == "parquet" else "csv"
    written = {"inputs": "inputs", "targets": "targets"}
    for name, table in {
        "observations": observation_table,
        "reporters": reporters,
        "assays": assays,
        "pairing": pairing,
    }.items():
        path = output / f"{name}.{suffix}"
        table.to_parquet(path, index=False) if format_ == "parquet" else table.to_csv(path, index=False)
        written[name] = path.name

    manifest = {
        "dataset_id": DATASET_ID,
        # At guide level a row is an average over many cells, so there is no cell to
        # pair: exact_pairing is demoted regardless of what the deposit could support.
        "capabilities": dict(CAPABILITIES) if level == "single_cell"
        else {**CAPABILITIES, "exact_pairing": False},
        "input_schema": {
            "schema_id": f"{DATASET_ID}_four_dye_v1",
            "representation": "tabular",
            "feature_names": partition["input"],
        },
        # inputs and targets are directories of per-plate parquet files. pandas and
        # pyarrow read a directory as one dataset, so the adapter needs no change, and
        # a consumer that cannot hold the arm whole can read one plate at a time.
        "tables": {name: filename for name, filename in written.items()},
        "target_layout": "wide",
        "aggregation_level": level,
        "target_endpoint_columns": partition["target"],
        "layout_note": (
            "targets are wide: one row per cell, one column per endpoint. The canonical "
            "long layout represents sparse availability, which this dataset does not "
            "have: every cell carries all "
            f"{len(partition['target'])} endpoints, so long form would cost that many "
            "times the rows for no information. The four required canonical tables are "
            "unaffected because targets is an optional table."
        ),
    }
    (output / "canonical_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    provenance = {
        "dataset_id": DATASET_ID,
        "source_base": base,
        "source_keys": keys,
        "screens": screens,
        "n_observations": int(len(observation_table)),
        "n_control_observations": int(observation_table["is_control"].sum()),
        "n_perturbations": int(observation_table["perturbation_id"].nunique()),
        "n_guides": int(observation_table["guide_id"].nunique()),
        "feature_partition": {stage: len(value) for stage, value in partition.items()},
        "input_channels": list(INPUT_CHANNELS),
        "target_channels": list(TARGET_CHANNELS),
        "target_layout": "wide",
        "aggregation_level": level,
        "aggregation_note": (
            "single_cell predicts per cell and is the level comparable to the published "
            "recoverability. guide fits on profiles already averaged within a (gene, sgRNA), "
            "which removes noise from both the input and the target and therefore yields a "
            "higher number that is its own quantity, not the published one."
        ),
        "cheap_input_is_label_free": False,
        "cheap_input_note": (
            "The input is four fluorescent dyes, not a transmitted-light channel. This "
            "dataset tests the perturbation and environment half of the claim. A result "
            "from it must not be quoted as label-free evidence."
        ),
        "same_target_note": (
            "The target is anti-TOMM20. This study measures 'mitochondria, TOMM20 (cp)' as "
            "one of its own 52 reporters, so this is a same-target comparison on a different "
            "platform rather than a generic external transfer."
        ),
        "endpoint_scale_note": (
            "The 823 endpoints are in native CellProfiler units and are not on a common "
            "scale: measured on plate CP186N the Granularity family has a median response "
            "standard deviation of 2.53 against 0.0053 for Intensity, and "
            "Cells_Intensity_IntegratedIntensity_Mito alone holds 86% of the summed squared "
            "spread. A Euclidean response magnitude over the raw block is therefore that one "
            "endpoint. Pass standardize_endpoints='observed_sd' to response_magnitudes, and "
            "read the max_endpoint_scale_share it reports."
        ),
        "outstanding_controls": [
            "Spectral bleed-through was tested and does not explain the prediction: DAPI "
            "alone, 165 nm from the AF594 target where a leak is not physically available, "
            "reaches median r = 0.701 against 0.807 for all four channels. See "
            "studies/external/README.md#periscope-diagnostics.",
            "Segmentation is driven by the phenotypic images, which include the target "
            "channel, so every feature is computed inside a mask partly defined by the "
            "target. Re-segmenting from the cheap channels is the strict control and is "
            "not done.",
        ],
        "excluded_cross_channel_examples": partition["excluded_cross_channel"][:10],
        "excluded_channel_free_examples": partition["excluded_channel_free"][:10],
    }
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in provenance.items() if key != "source_keys"}, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
