#!/usr/bin/env python3
"""Verify frozen portable sources; seal local render outputs separately.

Only an explicit --freeze-source updates source_manifest_sha256.csv. Ordinary
builds verify it and cannot silently accept changed inputs. Text sources are
hashed after LF normalization; binary inputs remain byte-exact.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "MANIFEST.sha256"
SOURCE_MANIFEST = ROOT / "source_manifest_sha256.csv"
GENERATED_DIRS = {
    "build", "composite", "figure", "panels", "main_panels", "published",
    "qa", "__pycache__",
}
GENERATED_NAMES = {
    "CHANGELOG.md", "MANIFEST.sha256", "PACKAGE_MANIFEST.sha256",
    "package_source_manifest_sha256.csv", "CLEAN_PACKAGE_REPORT.json",
    "publication_output_audit.json", "qa_checks.csv", "qa_report.md",
    "artifact_manifest.csv", "build_manifest.json",
}
BINARY_SUFFIXES = {".pdf", ".pptx", ".npz", ".npy", ".gz"}
RENDER_SUFFIXES = {".png", ".svg", ".tif", ".tiff", ".jpg", ".jpeg", ".pdf", ".pptx"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def is_source(path: Path) -> bool:
    relative = path.relative_to(ROOT)
    if path == SOURCE_MANIFEST or GENERATED_DIRS.intersection(relative.parts):
        return False
    if path.name in GENERATED_NAMES or path.suffix == ".pyc":
        return False
    if path.suffix == ".json" and "qa" in path.name.lower():
        return False
    if "source_data" in relative.parts and (
        "code" in relative.parts or path.name in {"reproduction.json", "source_manifest.json"}
    ):
        return False
    if path.suffix.lower() in RENDER_SUFFIXES:
        return relative.as_posix() in {
            f"a_workflow/source/Figure{ROOT.name.removeprefix('figure')}a.pdf",
            f"a_workflow/source/Figure{ROOT.name.removeprefix('figure')}a.pptx",
        }
    return True


def source_files() -> list[Path]:
    """Include tracked inputs and new, non-ignored sources; never local renders."""
    try:
        listing = subprocess.run(
            ["git", "-C", str(ROOT), "ls-files", "--cached", "--others",
             "--exclude-standard", "-z", "--", "."],
            check=True, capture_output=True,
        ).stdout.decode()
        candidates = [ROOT / name for name in listing.split("\0") if name]
    except (OSError, subprocess.CalledProcessError):
        candidates = list(ROOT.rglob("*"))
    sources = {path for path in candidates if path.is_file() and is_source(path)}
    # Shared typography is an explicit source dependency, not an untracked font
    # assumption. Scientific inputs remain entirely within this figure package.
    shared_style = ROOT.parents[1] / "publication_style.py"
    if shared_style.is_file():
        sources.add(shared_style)
    return sorted(sources, key=source_path)


def source_path(path: Path) -> str:
    return Path(os.path.relpath(path, ROOT)).as_posix()


def source_record(path: Path) -> tuple[str, int, str]:
    payload = path.read_bytes()
    normalization = "byte-exact" if path.suffix.lower() in BINARY_SUFFIXES else "LF"
    if normalization == "LF":
        payload = payload.replace(b"\r\n", b"\n")
    return hashlib.sha256(payload).hexdigest(), len(payload), normalization


def freeze_source() -> None:
    with SOURCE_MANIFEST.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["relative_path", "sha256", "bytes", "normalization"])
        for path in source_files():
            writer.writerow([source_path(path), *source_record(path)])
    print(f"froze {SOURCE_MANIFEST.name}; this is an explicit author-approved source update")


def verify_source() -> None:
    if not SOURCE_MANIFEST.is_file():
        raise SystemExit(f"missing source manifest: {SOURCE_MANIFEST.name}")
    with SOURCE_MANIFEST.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    expected = {row["relative_path"]: row for row in rows}
    observed = {source_path(path): path for path in source_files()}
    failures = []
    if len(expected) != len(rows):
        failures.append("duplicate source-manifest paths")
    for relative in sorted(set(expected) - set(observed)):
        failures.append(f"missing or noncanonical source: {relative}")
    for relative in sorted(set(observed) - set(expected)):
        failures.append(f"unregistered source: {relative}")
    for relative in sorted(set(expected) & set(observed)):
        row = expected[relative]
        digest, size, normalization = source_record(observed[relative])
        if (
            digest != row["sha256"]
            or size != int(row["bytes"])
            or normalization != row.get("normalization")
        ):
            failures.append(f"changed source: {relative}")
    if failures:
        raise SystemExit(
            "source verification failed (review changes before --freeze-source):\n"
            + "\n".join(failures)
        )
    print(f"source verification PASS ({len(expected)} portable inputs)")


def local_files() -> list[Path]:
    return sorted(
        (path for path in ROOT.rglob("*")
         if path.is_file() and path != MANIFEST
         and "__pycache__" not in path.parts and path.suffix != ".pyc"),
        key=lambda path: path.relative_to(ROOT).as_posix(),
    )


def write_local_manifest() -> None:
    lines = [f"{sha256(path)}  {path.relative_to(ROOT).as_posix()}" for path in local_files()]
    MANIFEST.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote local {MANIFEST.name} ({len(lines)} files); source manifest unchanged")


def verify_local_manifest() -> None:
    if not MANIFEST.is_file():
        raise SystemExit(f"missing local archive manifest: {MANIFEST.name}")
    failures = []
    entries = 0
    for raw in MANIFEST.read_text(encoding="utf-8").splitlines():
        expected, relative = raw.split("  ", 1)
        path = ROOT / relative
        entries += 1
        if not path.is_file():
            failures.append(f"missing: {relative}")
        elif sha256(path) != expected:
            failures.append(f"changed: {relative}")
    if failures:
        raise SystemExit("archive verification failed:\n" + "\n".join(failures))
    print(f"local archive verification PASS ({entries} files)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--freeze-source", action="store_true", help="explicitly approve current portable sources")
    mode.add_argument("--verify-source", action="store_true", help="verify portable sources without writing")
    mode.add_argument("--verify", action="store_true", help="verify sources and the full local archive")
    args = parser.parse_args()
    if args.freeze_source:
        freeze_source()
    else:
        verify_source()
        if args.verify:
            verify_local_manifest()
        elif not args.verify_source:
            write_local_manifest()


if __name__ == "__main__":
    main()
