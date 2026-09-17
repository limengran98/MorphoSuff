#!/usr/bin/env python3
"""Synchronize canonical figure sources to a manuscript release.

The canonical source artwork remains in its declared main and supplementary
source packages. This helper copies the ten formal composites, then refreshes
their digest records and the publication entries in the release registry.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path


HERE = Path(__file__).resolve().parent
FINAL = HERE / "final"
REFRESHED = {
    "Figure1.pdf": FINAL / "figure1/composite/Figure1.pdf",
    "Figure2.pdf": FINAL / "figure2/composite/Figure2.pdf",
    "Figure3.pdf": FINAL / "figure3/composite/Figure3.pdf",
    "Figure4.pdf": FINAL / "figure4/composite/Figure4.pdf",
    "Figure5.pdf": FINAL / "figure5/composite/Figure5.pdf",
    "Figure6.pdf": FINAL / "figure6/published/Figure6.pdf",
    "SupplementaryFigure1.pdf": HERE / "supplementary_figure1/published/SupplementaryFigure1.pdf",
    "SupplementaryFigure2.pdf": HERE / "supplementary_figure2/composite/SupplementaryFigure2.pdf",
    "SupplementaryFigure3.pdf": HERE / "supplementary_figure3/composite/SupplementaryFigure3.pdf",
    "SupplementaryFigure4.pdf": HERE / "supplementary_figure4/SupplementaryFigure4.pdf",
}

SOURCE_METADATA = {
    "Figure6.pdf": {
        "code_root": "final/figure6/code",
        "source_data_root": "final/figure6",
        "notes": "Measurement decisions and use-specific external validation, including held-out hepatocyte metabolic-loss prioritization.",
    },
    "SupplementaryFigure2.pdf": {
        "code_root": "supplementary_figure2/code",
        "source_data_root": "supplementary_figure2",
        "notes": "Label-efficiency calibration and phase-representation sensitivity.",
    },
    "SupplementaryFigure3.pdf": {
        "code_root": "supplementary_figure3/code",
        "source_data_root": "../supplementary_data/SupplementaryData1",
        "notes": "Complete predictor-specific decision map and LAMP1 criterion-level comparison.",
    },
    "SupplementaryFigure4.pdf": {
        "code_root": "supplementary_figure4/code",
        "source_data_root": "supplementary_figure4",
        "notes": "Supporting utility prediction, screen-context stability, repeated-measurement comparisons and external selection-budget analyses.",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=destination.parent, prefix=f".{destination.name}.", delete=False
    ) as handle:
        tmp = Path(handle.name)
    try:
        shutil.copyfile(source, tmp)
        shutil.copystat(source, tmp)
        os.replace(tmp, destination)
    finally:
        tmp.unlink(missing_ok=True)


def refresh_release(root: Path) -> None:
    figure_dir = root / "figures"
    digest_path = root / "figures.sha256.json"
    source_map_path = root / "figure_sources/source_map.csv"
    if not figure_dir.is_dir() or not digest_path.is_file() or not source_map_path.is_file():
        raise SystemExit(f"not a compatible manuscript root: {root}")

    for name, source in REFRESHED.items():
        if not source.is_file():
            raise SystemExit(f"missing canonical source figure: {source}")
        atomic_copy(source, figure_dir / name)

    payload = json.loads(digest_path.read_text(encoding="utf-8"))
    payload["note"] = (
        "Digests of the six main-figure PDFs and four supplementary-figure PDFs. "
        "Supplementary Figure 1 contains the complete three-context decision audit; "
        "Supplementary Figure 2 contains label-efficiency and representation-sensitivity analyses; "
        "Supplementary Figure 3 contains the predictor-specific decision map and criterion-level example; "
        "Figure 6 includes held-out external metabolic-loss prioritization; "
        "Supplementary Figure 4 contains supporting utility, screen-context, repeat-reference and selection-budget analyses."
    )
    figures = payload.setdefault("figures", {})
    for name in REFRESHED:
        figures[name] = sha256(figure_dir / name)
    digest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    with source_map_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
        fields = list(rows[0].keys()) if rows else []
    for row in rows:
        figure_pdf = f"{row.get('figure', '')}.pdf"
        if figure_pdf in REFRESHED:
            row["sha256"] = sha256(figure_dir / figure_pdf)
            if figure_pdf in SOURCE_METADATA:
                row.update(SOURCE_METADATA[figure_pdf])
    present = {f"{row.get('figure', '')}.pdf" for row in rows}
    for name, metadata in SOURCE_METADATA.items():
        if name in present:
            continue
        figure = Path(name).stem
        rows.append({
            "figure": figure,
            "published_pdf": f"paper/figures/{name}",
            "sha256": sha256(figure_dir / name),
            "code_root": metadata["code_root"],
            "source_data_root": metadata["source_data_root"],
            "notes": metadata["notes"],
        })
    with source_map_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    # These PDFs are the same local publication outputs, not an external upload.
    # Leave workflow source binaries and their separately reviewed pins untouched.
    scope_path = root.parent / "release/publication_scope.json"
    if scope_path.is_file():
        scope = json.loads(scope_path.read_text(encoding="utf-8"))
        scope["note"] = [line.replace("two supplementary figures", "four supplementary figures")
                         .replace("three supplementary figures", "four supplementary figures")
                         for line in scope.get("note", [])]
        binaries = scope.setdefault("repository_binaries", [])
        indexed = {entry["path"]: entry for entry in binaries}
        outputs = [figure_dir / name for name in REFRESHED]
        outputs.extend(root / name for name in ("manuscript.pdf", "supplementary.pdf"))
        # Pin this compact provenance table individually, not an unrestricted
        # binary-data root for the numerical supplement.
        outputs.append(root / "supplementary_data/SupplementaryData1/provenance/frozen_input_manifest.csv.gz")
        outputs.extend(root / "supplementary_data/SupplementaryData3/data" / name for name in (
            "heldout_predictions.parquet", "correspondence_null_distribution.csv.gz",
        ))
        for path in outputs:
            if not path.is_file():
                continue
            relative = path.relative_to(root.parent).as_posix()
            entry = indexed.get(relative)
            if entry is None:
                entry = {"path": relative}
                binaries.append(entry)
            entry.update(sha256=sha256(path), bytes=path.stat().st_size)
        scope_path.write_text(json.dumps(scope, indent=2) + "\n", encoding="utf-8")

    print(f"synced {root}")
    for name in REFRESHED:
        print(f"  {name}\t{sha256(figure_dir / name)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--paper-root", action="append", type=Path, required=True,
        help="manuscript root containing figures/, figures.sha256.json and figure_sources/source_map.csv",
    )
    args = parser.parse_args()
    for root in args.paper_root:
        refresh_release(root.resolve())


if __name__ == "__main__":
    main()
