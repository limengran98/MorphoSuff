#!/usr/bin/env python3
"""Assemble the former Figure 6d/e analyses as Supplementary Figure 2."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import tempfile

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = ROOT / "composite"
PAPER = ROOT.parent.parent
MM = 1 / 25.4
WIDTH_MM = 183.0
HEIGHT_MM = 158.0
WHITE = "#FFFFFF"
INK = "#202A33"
SCRIPTS = {
    "a": ROOT / "a_low_label_calibration" / "build_panel_d.py",
    "b": ROOT / "b_representation_boundary" / "build_panel_e.py",
}
SOURCES = {
    "a": [
        ROOT / "a_low_label_calibration" / "figure_source_data.csv",
        ROOT / "a_low_label_calibration" / "figure_source_label_efficiency.csv",
    ],
    "b": [
        ROOT / "b_representation_boundary" / "figure_source_representation_matrix.csv",
        ROOT / "b_representation_boundary" / "figure_source_paired_contrasts.csv",
        ROOT / "b_representation_boundary" / "figure_source_phase_crops.csv",
        ROOT / "b_representation_boundary" / "source_images" / "early_endosome_eea1_phase_crop.npy",
        ROOT / "b_representation_boundary" / "source_images" / "mitochondria_tomm20_phase_crop.npy",
        ROOT / "b_representation_boundary" / "source_images" / "nucleoli_npm1_phase_crop.npy",
    ],
}


def load_module(letter: str):
    path = SCRIPTS[letter]
    spec = importlib.util.spec_from_file_location(f"suppfig2_{letter}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
        temporary = Path(handle.name)
    try:
        shutil.copy2(source, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def write_manifest() -> Path:
    path = OUT / "figure_source_manifest.csv"
    rows = []
    for panel, sources in SOURCES.items():
        for source in sources:
            if not source.is_file():
                raise FileNotFoundError(source)
            n = None
            if source.suffix.lower() == ".csv":
                with source.open("r", encoding="utf-8", newline="") as handle:
                    n = max(0, sum(1 for _ in csv.reader(handle)) - 1)
            rows.append({
                "panel": panel,
                "source": source.relative_to(ROOT).as_posix(),
                "rows": n,
                "sha256": sha256(source),
            })
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["panel", "source", "rows", "sha256"])
        writer.writeheader(); writer.writerows(rows)
    return path


def write_artifact_manifest() -> Path:
    path = ROOT / "artifact_manifest.csv"
    rows = []
    for artifact in sorted(ROOT.rglob("*")):
        if (
            artifact.is_file()
            and artifact.name != path.name
            and "__pycache__" not in artifact.parts
        ):
            rows.append({
                "path": artifact.relative_to(ROOT).as_posix(),
                "bytes": artifact.stat().st_size,
                "sha256": sha256(artifact),
            })
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "bytes", "sha256"])
        writer.writeheader()
        writer.writerows(rows)
    return path


def build() -> None:
    modules = {letter: load_module(letter) for letter in "ab"}
    modules["a"].configure(); modules["b"].configure()
    mpl.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42,
                         "svg.fonttype": "none", "figure.facecolor": WHITE,
                         "savefig.facecolor": WHITE})
    fig = plt.figure(figsize=(WIDTH_MM * MM, HEIGHT_MM * MM), facecolor=WHITE)
    rows = fig.subfigures(2, 1, height_ratios=[80, 78], hspace=.025)
    modules["a"].draw_panel_d(rows[0], add_letter=False, compact=False)
    modules["b"].draw_panel_e(rows[1], add_letter=False, compact=False)
    for subfig, letter in zip(rows, "ab"):
        subfig.text(.004, .995, letter, ha="left", va="top",
                    fontsize=8.5, fontweight="bold", color=INK)

    OUT.mkdir(parents=True, exist_ok=True)
    stem = OUT / "SupplementaryFigure2"
    fig.savefig(stem.with_suffix(".svg"), bbox_inches=None, pad_inches=0)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches=None, pad_inches=0)
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches=None, pad_inches=0)
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches=None, pad_inches=0)
    manifest = write_manifest()

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    width, height = fig.canvas.get_width_height()
    overflow = []
    for text in fig.findobj(mpl.text.Text):
        if not text.get_visible() or not text.get_text().strip():
            continue
        box = text.get_window_extent(renderer)
        if box.x0 < -1 or box.y0 < -1 or box.x1 > width + 1 or box.y1 > height + 1:
            overflow.append(text.get_text())
    report = {
        "status": "PASS" if not overflow else "FAIL",
        "canvas_mm": {"width": WIDTH_MM, "height": HEIGHT_MM},
        "texts_outside_canvas": sorted(set(overflow)),
        "source_manifest": manifest.name,
        "source_manifest_sha256": sha256(manifest),
        "claim_boundary": "Exploratory label-efficiency and representation analyses; neither replaces the frozen reporter-level utility decision.",
    }
    (OUT / "figure_qa_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    atomic_copy(stem.with_suffix(".pdf"), PAPER / "figures" / "SupplementaryFigure2.pdf")
    write_artifact_manifest()
    plt.close(fig)


if __name__ == "__main__":
    build()
