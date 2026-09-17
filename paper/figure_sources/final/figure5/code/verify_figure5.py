#!/usr/bin/env python3
"""Validate the complete Figure 5 bundle after all panels are rendered.

This verifier is intentionally independent of the panel builders.  It reads only
frozen source tables and exported artefacts, writes a machine-readable check table
plus Markdown/JSON reports, and exits non-zero if any required check fails.

Run only after panels a--e and the composite are complete::

    python code/verify_figure5.py
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from xml.etree import ElementTree as ET

import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "source_data"
QA_ROOT = ROOT / "qa"

EXPECTED_DPI = 600.0
DPI_TOLERANCE = 1.0
COMPOSITE_WIDTH_MM = 183.0
COMPOSITE_WIDTH_TOLERANCE_MM = 0.35
COMPOSITE_HEIGHT_MM = 185.0
COMPOSITE_HEIGHT_TOLERANCE_MM = 0.35
PT_TO_MM = 25.4 / 72.0

ARTEFACT_STEMS = {
    "a": ROOT / "a_stability_constraints" / "figure" / "Figure5-a",
    "b": ROOT / "b_low_stability_cases" / "figure" / "Figure5-b",
    "c": ROOT / "c_amplitude_spectrum" / "figure" / "Figure5-c_amplitude_spectrum",
    "d": ROOT / "d_environment_landscape" / "figure" / "Figure5-d_environment_landscape",
    "e": ROOT / "e_conditional_ambiguity" / "figure" / "Figure5-e_conditional_ambiguity",
    "composite": ROOT / "composite" / "Figure5",
}


@dataclass
class Check:
    section: str
    check_id: str
    status: str
    expected: str
    observed: str
    details: str = ""


class Verifier:
    def __init__(self) -> None:
        self.checks: list[Check] = []

    def add(
        self,
        section: str,
        check_id: str,
        passed: bool,
        *,
        expected: Any,
        observed: Any,
        details: str = "",
    ) -> None:
        self.checks.append(
            Check(
                section=section,
                check_id=check_id,
                status="PASS" if passed else "FAIL",
                expected=_display(expected),
                observed=_display(observed),
                details=details,
            )
        )

    def guarded(
        self,
        section: str,
        check_id: str,
        expected: Any,
        action: Callable[[], tuple[bool, Any, str]],
    ) -> None:
        try:
            passed, observed, details = action()
            self.add(
                section,
                check_id,
                passed,
                expected=expected,
                observed=observed,
                details=details,
            )
        except Exception as exc:  # keep the complete audit instead of stopping early
            self.add(
                section,
                check_id,
                False,
                expected=expected,
                observed=f"ERROR: {type(exc).__name__}: {exc}",
            )

    @property
    def failed(self) -> list[Check]:
        return [check for check in self.checks if check.status != "PASS"]


def _display(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.12g}"
    if isinstance(value, (list, tuple, set, dict)):
        return json.dumps(value, sort_keys=True, ensure_ascii=False)
    return str(value)


def _close(a: float, b: float, tolerance: float = 1e-10) -> bool:
    return math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=tolerance)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pdf_page_mm(path: Path) -> tuple[float, float]:
    raw = path.read_bytes()
    match = re.search(
        rb"/MediaBox\s*\[\s*([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+"
        rb"([-+0-9.eE]+)\s+([-+0-9.eE]+)\s*\]",
        raw,
    )
    if match is None:
        raise ValueError("PDF MediaBox not found")
    x0, y0, x1, y1 = map(float, match.groups())
    return (x1 - x0) * PT_TO_MM, (y1 - y0) * PT_TO_MM


def _pdf_font_audit(path: Path) -> tuple[set[str], bool]:
    raw = path.read_bytes()
    names = {
        item.decode("latin-1", errors="replace")
        for item in re.findall(rb"/(?:BaseFont|FontName)\s*/([^\s/<>{}\[\]()]+)", raw)
    }
    has_type3 = re.search(rb"/Subtype\s*/Type3\b", raw) is not None
    return names, has_type3


def _png_mm_and_dpi(path: Path) -> tuple[float, float, float, float]:
    with Image.open(path) as image:
        dpi = image.info.get("dpi")
        if not dpi or len(dpi) < 2:
            raise ValueError("PNG has no embedded DPI metadata")
        xdpi, ydpi = float(dpi[0]), float(dpi[1])
        width_mm = image.width / xdpi * 25.4
        height_mm = image.height / ydpi * 25.4
        return width_mm, height_mm, xdpi, ydpi


def _svg_root(path: Path) -> ET.Element:
    return ET.parse(path).getroot()


def _svg_live_text(path: Path) -> tuple[int, bool]:
    root = _svg_root(path)
    nodes = [node for node in root.iter() if node.tag.rsplit("}", 1)[-1] == "text"]
    raw = path.read_text(encoding="utf-8")
    return len(nodes), "DejaVu Sans" in raw


def _parse_svg_length_mm(value: str) -> float:
    match = re.fullmatch(r"\s*([-+0-9.eE]+)\s*([A-Za-z]*)\s*", value)
    if match is None:
        raise ValueError(f"Unsupported SVG length {value!r}")
    number = float(match.group(1))
    unit = match.group(2) or "px"
    factors = {
        "mm": 1.0,
        "cm": 10.0,
        "in": 25.4,
        "pt": PT_TO_MM,
        "px": 25.4 / 96.0,
    }
    if unit not in factors:
        raise ValueError(f"Unsupported SVG unit {unit!r}")
    return number * factors[unit]


def _svg_page_mm(path: Path) -> tuple[float, float]:
    root = _svg_root(path)
    width = root.attrib.get("width")
    height = root.attrib.get("height")
    if width is None or height is None:
        raise ValueError("SVG width/height missing")
    return _parse_svg_length_mm(width), _parse_svg_length_mm(height)


def check_artefacts(v: Verifier) -> None:
    for panel, stem in ARTEFACT_STEMS.items():
        for suffix in (".png", ".pdf", ".svg"):
            path = stem.with_suffix(suffix)
            v.add(
                "artefacts",
                f"{panel}_{suffix[1:]}_exists",
                path.is_file() and path.stat().st_size > 0,
                expected="non-empty file",
                observed=str(path.relative_to(ROOT)) if path.exists() else "missing",
            )

        png = stem.with_suffix(".png")
        if png.is_file():
            v.guarded(
                "export",
                f"{panel}_png_600dpi",
                f"{EXPECTED_DPI:.0f}±{DPI_TOLERANCE:.0f} dpi",
                lambda p=png: _check_png_dpi(p),
            )

        pdf = stem.with_suffix(".pdf")
        if pdf.is_file():
            v.guarded(
                "export",
                f"{panel}_pdf_dejavu_no_type3",
                "DejaVuSans font present; no Type3 fonts",
                lambda p=pdf: _check_pdf_fonts(p),
            )

        svg = stem.with_suffix(".svg")
        if svg.is_file():
            v.guarded(
                "export",
                f"{panel}_svg_live_dejavu_text",
                "at least one live text node using DejaVuSans",
                lambda p=svg: _check_svg_text(p),
            )


def _check_png_dpi(path: Path) -> tuple[bool, Any, str]:
    width_mm, height_mm, xdpi, ydpi = _png_mm_and_dpi(path)
    passed = (
        abs(xdpi - EXPECTED_DPI) <= DPI_TOLERANCE
        and abs(ydpi - EXPECTED_DPI) <= DPI_TOLERANCE
    )
    observed = {"x_dpi": xdpi, "y_dpi": ydpi}
    return passed, observed, f"physical size {width_mm:.2f} × {height_mm:.2f} mm"


def _check_pdf_fonts(path: Path) -> tuple[bool, Any, str]:
    names, has_type3 = _pdf_font_audit(path)
    has_dejavu = any("DejaVuSans" in name for name in names)
    passed = has_dejavu and not has_type3
    return passed, {"fonts": sorted(names), "type3": has_type3}, "PDF dictionaries inspected"


def _check_svg_text(path: Path) -> tuple[bool, Any, str]:
    n_text, has_dejavu = _svg_live_text(path)
    return n_text > 0 and has_dejavu, {"text_nodes": n_text, "dejavu": has_dejavu}, "SVG XML inspected"


def check_composite_geometry(v: Verifier) -> None:
    stem = ARTEFACT_STEMS["composite"]
    measurements: dict[str, tuple[float, float]] = {}
    readers = {
        "pdf": _pdf_page_mm,
        "png": lambda p: _png_mm_and_dpi(p)[:2],
        "svg": _svg_page_mm,
    }
    for suffix, reader in readers.items():
        path = stem.with_suffix(f".{suffix}")
        if not path.is_file():
            continue
        try:
            width, height = reader(path)
            measurements[suffix] = (float(width), float(height))
            passed = (
                abs(width - COMPOSITE_WIDTH_MM) <= COMPOSITE_WIDTH_TOLERANCE_MM
                and abs(height - COMPOSITE_HEIGHT_MM) <= COMPOSITE_HEIGHT_TOLERANCE_MM
            )
            v.add(
                "geometry",
                f"composite_{suffix}_page_size",
                passed,
                expected=(
                    f"width {COMPOSITE_WIDTH_MM}±{COMPOSITE_WIDTH_TOLERANCE_MM} mm; "
                    f"height {COMPOSITE_HEIGHT_MM}±{COMPOSITE_HEIGHT_TOLERANCE_MM} mm"
                ),
                observed={"width_mm": round(width, 4), "height_mm": round(height, 4)},
            )
        except Exception as exc:
            v.add(
                "geometry",
                f"composite_{suffix}_page_size",
                False,
                expected=(
                    f"width {COMPOSITE_WIDTH_MM}±{COMPOSITE_WIDTH_TOLERANCE_MM} mm; "
                    f"height {COMPOSITE_HEIGHT_MM}±{COMPOSITE_HEIGHT_TOLERANCE_MM} mm"
                ),
                observed=f"ERROR: {type(exc).__name__}: {exc}",
            )
    if len(measurements) == 3:
        widths = [value[0] for value in measurements.values()]
        heights = [value[1] for value in measurements.values()]
        v.add(
            "geometry",
            "composite_formats_same_size",
            max(widths) - min(widths) <= 0.35 and max(heights) - min(heights) <= 0.35,
            expected="PNG/PDF/SVG width and height agree within 0.35 mm",
            observed={key: [round(x, 4) for x in value] for key, value in measurements.items()},
        )


def check_panel_letters(v: Verifier) -> None:
    svg = ARTEFACT_STEMS["composite"].with_suffix(".svg")
    if not svg.is_file():
        v.add(
            "layout",
            "composite_panel_letters_a_to_e_unique",
            False,
            expected={letter: 1 for letter in "abcde"},
            observed="composite SVG missing",
        )
        return
    root = _svg_root(svg)
    text_values = [
        "".join(node.itertext()).strip()
        for node in root.iter()
        if node.tag.rsplit("}", 1)[-1] == "text"
    ]
    counts = {letter: text_values.count(letter) for letter in "abcde"}
    v.add(
        "layout",
        "composite_panel_letters_a_to_e_unique",
        all(value == 1 for value in counts.values()),
        expected={letter: 1 for letter in "abcde"},
        observed=counts,
        details="Exact live-text matches in the composite SVG",
    )


def check_scientific_sources(v: Verifier) -> None:
    # Panel a: frozen reporter table, four associations, and joint-model result.
    canonical = pd.read_csv(SOURCE / "panel_a/canonical_reporter_stability_median10.csv")
    v.add(
        "science_a",
        "panel_a_52_unique_reporters",
        len(canonical) == 52 and canonical["reporter_slug"].nunique() == 52,
        expected={"rows": 52, "reporters": 52},
        observed={"rows": len(canonical), "reporters": canonical["reporter_slug"].nunique()},
    )
    aggregations = sorted(canonical["consensus_aggregation"].dropna().unique().tolist())
    v.add(
        "science_a",
        "panel_a_frozen_median10",
        aggregations == ["unweighted_median_10_models"],
        expected=["unweighted_median_10_models"],
        observed=aggregations,
    )
    assoc = pd.read_csv(SOURCE / "panel_a/stability_associations_median10.csv")
    expected_rho = {
        "within_screen_split_half_pearson_median": 0.5468283104243148,
        "guide_half_consistency": 0.37591036414565826,
        "cross_screen_pearson_median": 0.3485103132161956,
        "log10_exact_links": 0.4821755083295821,
    }
    observed_rho = dict(zip(assoc["predictor"], assoc["rho"]))
    rho_ok = len(assoc) == 4 and set(observed_rho) == set(expected_rho) and all(
        _close(observed_rho[key], value, 1e-12) for key, value in expected_rho.items()
    )
    v.add(
        "science_a",
        "panel_a_four_frozen_associations",
        rho_ok,
        expected=expected_rho,
        observed=observed_rho,
    )
    expected_n = {
        "within_screen_split_half_pearson_median": 52,
        "guide_half_consistency": 35,
        "cross_screen_pearson_median": 34,
        "log10_exact_links": 52,
    }
    observed_n = dict(zip(assoc["predictor"], assoc["n_reporters"].astype(int)))
    v.add(
        "science_a",
        "panel_a_frozen_association_sample_sizes",
        observed_n == expected_n,
        expected=expected_n,
        observed=observed_n,
    )
    joint = pd.read_csv(SOURCE / "panel_a/joint_constraint_model_median10.csv")
    joint_row = joint.iloc[0]
    joint_ok = (
        len(joint) == 1
        and int(joint_row["n_reporters"]) == 35
        and _close(joint_row["r2_in_sample"], 0.3871269911021159, 1e-12)
        and _close(joint_row["permutation_p"], 0.000999666777740753, 1e-15)
        and int(joint_row["permutation_draws"]) == 3000
    )
    v.add(
        "science_a",
        "panel_a_joint_model_frozen",
        joint_ok,
        expected={"n": 35, "R2": 0.3871269911021159, "P": 0.000999666777740753, "draws": 3000},
        observed={
            "n": int(joint_row["n_reporters"]),
            "R2": float(joint_row["r2_in_sample"]),
            "P": float(joint_row["permutation_p"]),
            "draws": int(joint_row["permutation_draws"]),
        },
    )

    # Panel b: two cases × three stability tests with frozen sample sizes.
    summary = pd.read_csv(SOURCE / "panel_b/low_stability_case_summary.csv")
    observed_n = summary["n"].astype(int).tolist()
    v.add(
        "science_b",
        "panel_b_six_case_test_rows",
        len(summary) == 6 and sorted(observed_n) == [3, 3, 3, 3, 100, 100],
        expected={"rows": 6, "sorted_n": [3, 3, 3, 3, 100, 100]},
        observed={"rows": len(summary), "sorted_n": sorted(observed_n)},
    )
    selection = pd.read_csv(SOURCE / "microscopy/selection_manifest.csv")
    image_counts = selection["target_key"].value_counts().to_dict()
    v.add(
        "science_b_d",
        "microscopy_frozen_3_prb_2_ferhonox",
        image_counts == {"prb": 3, "ferhonox": 2},
        expected={"prb": 3, "ferhonox": 2},
        observed=image_counts,
    )

    # Panel c: the frozen amplitude-compression spectrum.
    amplitude = pd.read_csv(SOURCE / "panel_c/amplitude_fidelity_spectrum.csv")
    compressed = amplitude["magnitude_variance_ratio"] < 0.5
    retained = compressed & (amplitude["magnitude_spearman"] > 0.5)
    v.add(
        "science_c",
        "panel_c_52_11_10",
        len(amplitude) == 52
        and amplitude["reporter_slug"].nunique() == 52
        and int(compressed.sum()) == 11
        and int(retained.sum()) == 10,
        expected={"rows": 52, "reporters": 52, "compressed": 11, "rank_retained": 10},
        observed={
            "rows": len(amplitude),
            "reporters": amplitude["reporter_slug"].nunique(),
            "compressed": int(compressed.sum()),
            "rank_retained": int(retained.sum()),
        },
    )

    # Panel d: 81 directed reporter-to-destination-screen evaluations.
    environment = pd.read_csv(SOURCE / "panel_d/canonical_environment_transfer_display.csv")
    env_shape = {
        "rows": len(environment),
        "reporters": environment["reporter_slug"].nunique(),
        "destination_screens": environment["destination_screen"].nunique(),
    }
    v.add(
        "science_d",
        "panel_d_81_directions_34_reporters_66_screens",
        env_shape == {"rows": 81, "reporters": 34, "destination_screens": 66},
        expected={"rows": 81, "reporters": 34, "destination_screens": 66},
        observed=env_shape,
    )
    sign_flip = np.allclose(
        environment["gene_minus_strict_screen_pearson"].to_numpy(float),
        -environment["consensus10_screen_penalty_gene"].to_numpy(float),
        rtol=0,
        atol=1e-12,
        equal_nan=False,
    )
    v.add(
        "science_d",
        "panel_d_transfer_loss_sign_contract",
        bool(sign_flip),
        expected="gene_minus_strict_screen_pearson == -consensus10_screen_penalty_gene",
        observed=bool(sign_flip),
    )
    associations = pd.read_csv(SOURCE / "panel_d/association_summary.csv")
    v.add(
        "science_d",
        "panel_d_association_cluster_contract",
        len(associations) == 4
        and set(associations["n_rows"].astype(int)) == {81}
        and set(associations["n_clusters"].astype(int)) == {34},
        expected={"rows": 4, "n_rows": [81], "n_clusters": [34]},
        observed={
            "rows": len(associations),
            "n_rows": sorted(associations["n_rows"].astype(int).unique().tolist()),
            "n_clusters": sorted(associations["n_clusters"].astype(int).unique().tolist()),
        },
    )
    mapping = pd.read_csv(SOURCE / "panel_d/ferhonox_point_to_image_mapping.csv")
    mapping_screens = sorted(mapping["destination_screen"].tolist())
    v.add(
        "science_d",
        "panel_d_two_ferhonox_directions",
        len(mapping) == 2 and mapping_screens == ["Biohub_OPS0047", "Biohub_OPS0067"],
        expected=["Biohub_OPS0047", "Biohub_OPS0067"],
        observed=mapping_screens,
    )

    # Panel e: six target/screen/representation summaries from 5,990 gene rows.
    ambiguity = pd.read_csv(SOURCE / "panel_e/conditional_ambiguity_per_gene.csv")
    ambiguity_summary = pd.read_csv(SOURCE / "panel_e/conditional_ambiguity_summary.csv")
    e_shape = {
        "per_gene_rows": len(ambiguity),
        "summary_rows": len(ambiguity_summary),
        "summary_n_genes_sum": int(ambiguity_summary["n_genes"].sum()),
    }
    v.add(
        "science_e",
        "panel_e_5990_gene_rows_six_summaries",
        e_shape == {"per_gene_rows": 5990, "summary_rows": 6, "summary_n_genes_sum": 5990},
        expected={"per_gene_rows": 5990, "summary_rows": 6, "summary_n_genes_sum": 5990},
        observed=e_shape,
    )


def check_provenance(v: Verifier) -> None:
    provenance_path = SOURCE / "upstream_provenance.csv"
    provenance = pd.read_csv(provenance_path)
    mismatches: list[str] = []
    missing: list[str] = []
    for row in provenance.itertuples(index=False):
        relative = str(row.destination)
        prefix = "figure5/"
        if relative.startswith(prefix):
            relative = relative[len(prefix):]
        destination = ROOT / relative
        if not destination.is_file():
            missing.append(str(destination.relative_to(ROOT)))
            continue
        digest = _sha256(destination)
        if digest != str(row.destination_sha256):
            mismatches.append(str(destination.relative_to(ROOT)))
    v.add(
        "provenance",
        "frozen_source_hashes_match",
        not missing and not mismatches,
        expected={"missing": 0, "sha256_mismatch": 0},
        observed={"rows": len(provenance), "missing": len(missing), "sha256_mismatch": len(mismatches)},
        details=("missing=" + ",".join(missing) + "; mismatch=" + ",".join(mismatches))
        if missing or mismatches
        else "All destination SHA-256 values match upstream_provenance.csv",
    )


def write_reports(v: Verifier, qa_root: Path) -> None:
    qa_root.mkdir(parents=True, exist_ok=True)
    csv_path = qa_root / "qa_checks.csv"
    json_path = qa_root / "qa_report.json"
    md_path = qa_root / "qa_report.md"

    fields = ["section", "check_id", "status", "expected", "observed", "details"]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(asdict(check) for check in v.checks)

    passed = sum(check.status == "PASS" for check in v.checks)
    failed = len(v.checks) - passed
    payload = {
        "figure": "Figure 5",
        "status": "PASS" if failed == 0 else "FAIL",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_checks": len(v.checks),
        "n_passed": passed,
        "n_failed": failed,
        "checks": [asdict(check) for check in v.checks],
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# Figure 5 quality assurance",
        "",
        f"Status: **{payload['status']}** ({passed}/{len(v.checks)} checks passed).",
        "",
        "This report audits frozen scientific counts, source provenance, required artefacts, physical size, typography, vector text and panel lettering. It does not rerun or alter any scientific analysis.",
        "",
    ]
    current_section: str | None = None
    for check in v.checks:
        if check.section != current_section:
            current_section = check.section
            lines.extend([f"## {current_section}", ""])
        marker = "x" if check.status == "PASS" else " "
        detail = f" — {check.details}" if check.details else ""
        lines.append(
            f"- [{marker}] `{check.check_id}`: expected {check.expected}; observed {check.observed}{detail}"
        )
    lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--qa-root",
        type=Path,
        default=QA_ROOT,
        help="Output directory for qa_checks.csv and qa_report.{md,json}",
    )
    args = parser.parse_args(argv)

    verifier = Verifier()
    check_artefacts(verifier)
    check_composite_geometry(verifier)
    check_panel_letters(verifier)
    check_scientific_sources(verifier)
    check_provenance(verifier)
    write_reports(verifier, args.qa_root)

    n_failed = len(verifier.failed)
    print(
        json.dumps(
            {
                "status": "PASS" if n_failed == 0 else "FAIL",
                "checks": len(verifier.checks),
                "failed": n_failed,
                "qa_root": str(args.qa_root),
            },
            indent=2,
        )
    )
    return 0 if n_failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
