#!/usr/bin/env python3
"""Deterministic source, geometry and publication-output QA for Figure 4."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path
import xml.etree.ElementTree as ET

import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "source_data"
QA = ROOT / "qa"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def check(name: str, passed: bool, detail: str) -> dict[str, object]:
    return {"check": name, "status": "PASS" if passed else "FAIL", "detail": detail}


def main() -> None:
    QA.mkdir(parents=True, exist_ok=True)
    checks: list[dict[str, object]] = []

    means = pd.read_csv(SOURCE / "panel_a" / "reporter_arm_means.csv")
    residual = pd.read_csv(SOURCE / "panel_b" / "exact_vs_residual.csv")
    reconstruction = pd.read_csv(SOURCE / "panel_c" / "reconstruction_summary.csv")
    directions = pd.read_csv(SOURCE / "panel_d" / "all10_strict_direction_metrics.csv")
    cases = pd.read_csv(SOURCE / "panel_d" / "transfer_case_selection.csv")
    roster = pd.read_csv(SOURCE / "method_roster.csv")

    arm_counts = means.groupby("arm").reporter_slug.nunique()
    checks.append(check(
        "panel_a_complete_52_by_6",
        len(arm_counts) == 6 and bool((arm_counts == 52).all()),
        f"arm reporter counts={arm_counts.to_dict()}",
    ))
    checks.append(check(
        "panel_b_complete_pairs",
        len(residual) == 52 and residual.reporter_slug.nunique() == 52
        and not residual[["exact_pair", "within_condition_residual"]].isna().any().any(),
        f"rows={len(residual)}, reporters={residual.reporter_slug.nunique()}",
    ))

    c_ok = len(reconstruction) == 3
    c_details: list[str] = []
    for slug in reconstruction.reporter_slug:
        base = SOURCE / "panel_c" / slug
        profiles = pd.read_csv(base / "response_profiles.csv.gz")
        genes = pd.read_csv(base / "gene_magnitude_scatter.csv.gz")
        c_ok &= len(profiles) == 12 and len(genes) == 1000
        c_details.append(f"{slug}: profiles={len(profiles)}, genes={len(genes)}")
    checks.append(check("panel_c_frozen_examples", c_ok, "; ".join(c_details)))

    per_method = directions.groupby("method_id").size()
    checks.append(check(
        "panel_d_all_methods_all_directions",
        len(directions) == 810 and len(per_method) == 10 and bool((per_method == 81).all())
        and len(roster) == 10 and len(cases) == 2,
        f"rows={len(directions)}, per_method={per_method.to_dict()}, cases={len(cases)}",
    ))

    expected_method_colors = [
        "#3E6F9C", "#5E88AD", "#79A8C9", "#7B6597", "#A783AD",
        "#CC79A7", "#4F9A70", "#4FA7B8", "#C85A3C", "#DDA06A",
    ]
    style_text = (ROOT / "code" / "figure4_style.py").read_text(encoding="utf-8")
    panel_c_text = (ROOT / "code" / "build_panel_c.py").read_text(encoding="utf-8")
    palette_ok = (
        roster.sort_values("method_order").method_color.astype(str).tolist()
        == expected_method_colors
        and all(token in style_text for token in ("#3E6F9C", "#D56C75", "#79A8C9", "#A783AD"))
        and all(token in panel_c_text for token in ("#E8943A", "#D56C75", "#7B6597", "#4FA7B8"))
        and all(token not in style_text + panel_c_text for token in ("#16868C", "#D64F61", "#F57C00", "#C43D4B", "#8E5CC2"))
    )
    checks.append(check(
        "approved_cplus_palette",
        palette_ok,
        f"method_colors={roster.sort_values('method_order').method_color.astype(str).tolist()}",
    ))

    manifest = pd.read_csv(ROOT / "source_manifest_sha256.csv")
    mismatches: list[str] = []
    for row in manifest.itertuples(index=False):
        # Frozen records created on Windows remain readable on every platform.
        local = ROOT / str(row.relative_path).replace("\\", "/")
        if not local.exists() or local.stat().st_size != int(row.bytes) or sha256(local) != str(row.sha256):
            mismatches.append(str(row.relative_path))
    checks.append(check(
        "frozen_source_identity",
        len(manifest) >= 18 and not mismatches,
        f"{len(manifest)} released source files checked; mismatches={mismatches}",
    ))

    required_stems = [
        ROOT / "a_intervention_terrain" / "figure4a_intervention_terrain",
        ROOT / "b_residual_barcode" / "figure4b_residual_barcode",
        ROOT / "c_reconstruction_modes" / "Figure4-c",
        ROOT / "d_screen_transfer" / "figure4d_screen_transfer",
        ROOT / "composite" / "Figure4",
    ]
    missing_outputs = [str(stem.with_suffix(ext)) for stem in required_stems
                       for ext in (".png", ".pdf", ".svg") if not stem.with_suffix(ext).exists()]
    checks.append(check("required_outputs", not missing_outputs, f"missing={missing_outputs}"))

    png = ROOT / "composite" / "Figure4.png"
    with Image.open(png) as image:
        width, height = image.size
        dpi = image.info.get("dpi", (0, 0))
    checks.append(check(
        "composite_png_geometry",
        abs(width - 183 / 25.4 * 600) <= 2 and abs(height - 225 / 25.4 * 600) <= 2 and min(dpi) >= 590,
        f"pixels={width}x{height}, dpi={dpi}",
    ))

    pdf_bytes = (ROOT / "composite" / "Figure4.pdf").read_bytes()
    media = re.search(rb"/MediaBox\s*\[\s*0\s+0\s+([0-9.]+)\s+([0-9.]+)\s*\]", pdf_bytes)
    if media:
        w_mm = float(media.group(1)) * 25.4 / 72
        h_mm = float(media.group(2)) * 25.4 / 72
    else:
        w_mm = h_mm = float("nan")
    checks.append(check(
        "composite_pdf_geometry",
        media is not None and abs(w_mm - 183) < 0.3 and abs(h_mm - 225) < 0.3,
        f"media_box_mm={w_mm:.2f}x{h_mm:.2f}",
    ))
    checks.append(check(
        "pdf_embedded_text_no_type3",
        b"/Type3" not in pdf_bytes,
        f"contains_Type3={b'/Type3' in pdf_bytes}",
    ))

    svg_path = ROOT / "composite" / "Figure4.svg"
    svg_text = svg_path.read_text(encoding="utf-8")
    tree = ET.parse(svg_path)
    svg_live_text = len(tree.findall(".//{http://www.w3.org/2000/svg}text"))
    checks.append(check(
        "svg_live_text",
        svg_live_text >= 100,
        f"text_nodes={svg_live_text}",
    ))
    forbidden = ["ten-method mean", "pilot", "queue", "v1 source"]
    # Search displayed text only, never arbitrary matches inside base64 images.
    displayed_text = " ".join("".join(node.itertext())
                              for node in tree.findall(".//{http://www.w3.org/2000/svg}text"))
    found = [term for term in forbidden if term.lower() in displayed_text.lower()]
    checks.append(check("no_internal_display_terms", not found, f"found={found}"))

    overall = "PASS" if all(row["status"] == "PASS" for row in checks) else "FAIL"
    report = {
        "figure": "Figure 4",
        "overall_status": overall,
        "manual_visual_review": {
            "status": "REQUIRED_AFTER_EACH_REBUILD",
            "review_record": None,
            "review_instructions": "paper/BUILD.md",
            "notes": "Automatic checks do not certify human review of a new rendering.",
        },
        "checks": checks,
    }
    (QA / "figure4_qa_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )

    md = ["# Figure 4 QA", "", f"Overall status: **{overall}**", ""]
    md += [f"- **{row['status']}** — {row['check']}: {row['detail']}" for row in checks]
    md += ["", "Inspect the complete rebuilt figure at manuscript size; see paper/BUILD.md."]
    (ROOT / "qa_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    artifacts = sorted(
        p for p in ROOT.rglob("*") if p.is_file()
        and "__pycache__" not in p.parts
        and p.name not in {
            "artifact_manifest.csv",
            "PACKAGE_MANIFEST.sha256",
            "package_source_manifest_sha256.csv",
            "figure4_archive_audit.json",
            "figure4_archive_audit.md",
        }
    )
    with (ROOT / "artifact_manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["relative_path", "bytes", "sha256"])
        for path in artifacts:
            writer.writerow([path.relative_to(ROOT), path.stat().st_size, sha256(path)])

    # This is verification, not a new approval of the frozen numerical inputs.
    # Never rewrite source_manifest_sha256.csv here, including after a failure.

    print(json.dumps(report, indent=2))
    if overall != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
