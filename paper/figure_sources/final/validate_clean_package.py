#!/usr/bin/env python3
"""Validate and seal the clean Figure 1–6 source release."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PAPER = ROOT.parents[1]
PUBLICATION_DIR = PAPER / "figures"
FIGURE_DIGESTS = PAPER / "figures.sha256.json"
FIGURES = [ROOT / f"figure{i}" for i in range(1, 7)]
GENERATED_OUTPUTS = {
    1: ["composite/Figure1.pdf", "composite/Figure1.png"],
    2: ["composite/Figure2.pdf", "composite/Figure2.svg", "composite/Figure2.png"],
    3: ["composite/Figure3.pdf", "composite/Figure3.svg", "composite/Figure3.png", "composite/Figure3.tiff"],
    4: ["composite/Figure4.pdf", "composite/Figure4.svg", "composite/Figure4.png"],
    5: ["composite/Figure5.pdf", "composite/Figure5.svg", "composite/Figure5.png"],
    6: ["published/Figure6.pdf", "published/Figure6.svg", "published/Figure6.png"],
}
PUBLICATION_PDFS = {index: f"Figure{index}.pdf" for index in range(1, 7)}
ALLOWED_TOP = {
    *(f"figure{i}" for i in range(1, 7)),
    "README.md",
    "build_all.ps1",
    "validate_clean_package.py",
    "CLEAN_PACKAGE_REPORT.json",
    "PACKAGE_MANIFEST.sha256",
    "__pycache__",
}
FORBIDDEN_LEGACY_DIRS = {
    ROOT / "figure6" / "d_low_label_calibration",
    ROOT / "figure6" / "e_representation_boundary",
}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    errors: list[str] = []
    shared_style = ROOT.parent / 'publication_style.py'
    if not shared_style.is_file():
        errors.append('missing shared presentation dependency: ../publication_style.py')
    try:
        expected_digests = json.loads(FIGURE_DIGESTS.read_text(encoding="utf-8"))["figures"]
    except (FileNotFoundError, KeyError, json.JSONDecodeError) as exc:
        expected_digests = {}
        errors.append(f"cannot read publication digest registry: {exc}")
    top = {path.name for path in ROOT.iterdir()}
    extra = sorted(top - ALLOWED_TOP)
    if extra:
        errors.append(f"unexpected top-level entries: {extra}")
    missing_top = sorted({f"figure{i}" for i in range(1, 7)} - top)
    if missing_top:
        errors.append(f"missing figure folders: {missing_top}")
    legacy = sorted(path.relative_to(ROOT).as_posix() for path in FORBIDDEN_LEGACY_DIRS if path.exists())
    if legacy:
        errors.append(
            "superseded Figure 6 panel packages remain in the active main-figure tree: "
            + ", ".join(legacy)
            + "; their sole current copies belong in supplementary_figure2"
        )

    per_figure: dict[str, object] = {}
    for index, folder in enumerate(FIGURES, start=1):
        missing_source = []
        for relative in ["README.md", "build.ps1"]:
            path = folder / relative
            if not path.is_file() or path.stat().st_size == 0:
                missing_source.append(relative)
        python_sources = sorted(folder.rglob("*.py"))
        if not python_sources:
            missing_source.append("*.py")
        publication_name = PUBLICATION_PDFS[index]
        publication_pdf = PUBLICATION_DIR / publication_name
        publication_sha256 = (
            digest(publication_pdf)
            if publication_pdf.is_file() and publication_pdf.stat().st_size > 0
            else None
        )
        expected_sha256 = expected_digests.get(publication_name)
        if publication_sha256 is None:
            errors.append(f"figure{index} publication PDF missing: paper/figures/{publication_name}")
        elif publication_sha256 != expected_sha256:
            errors.append(
                f"figure{index} publication PDF digest mismatch: "
                f"expected {expected_sha256}, got {publication_sha256}"
            )
        cache_entries = [
            path.relative_to(folder).as_posix()
            for path in folder.rglob("*")
            if path.name == "__pycache__" or path.suffix == ".pyc" or "backup" in path.name.lower()
        ]
        obsolete = [
            path.relative_to(folder).as_posix()
            for path in folder.rglob("*")
            if path.is_file() and any(token in path.name for token in ("Cplus_preview", "Figure3_v2", "Figure4_v2", "Figure5_v2"))
        ]
        if missing_source:
            errors.append(f"figure{index} source package missing: {missing_source}")
        if cache_entries:
            errors.append(f"figure{index} cache/backup entries: {cache_entries}")
        if obsolete:
            errors.append(f"figure{index} obsolete outputs: {obsolete}")
        per_figure[f"figure{index}"] = {
            "files": sum(1 for path in folder.rglob("*") if path.is_file()),
            "missing_source": missing_source,
            "cache_or_backup_entries": cache_entries,
            "obsolete_outputs": obsolete,
            "local_generated_outputs_present": [
                relative
                for relative in GENERATED_OUTPUTS[index]
                if (folder / relative).is_file()
            ],
            "publication_pdf": f"paper/figures/{publication_name}",
            "publication_pdf_sha256": publication_sha256,
            "publication_digest_verified": publication_sha256 == expected_sha256,
        }

    report = {"status": "PASS" if not errors else "FAIL", "errors": errors, "figures": per_figure}
    (ROOT / "CLEAN_PACKAGE_REPORT.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    manifest_rows = []
    if shared_style.is_file():
        manifest_rows.append(f'{digest(shared_style)}  ../publication_style.py')
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or path.name == "PACKAGE_MANIFEST.sha256":
            continue
        manifest_rows.append(f"{digest(path)}  {path.relative_to(ROOT).as_posix()}")
    (ROOT / "PACKAGE_MANIFEST.sha256").write_text("\n".join(manifest_rows) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
