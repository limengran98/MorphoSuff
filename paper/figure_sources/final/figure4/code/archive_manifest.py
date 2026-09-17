#!/usr/bin/env python3
"""Create or verify integrity manifests for the sealed Figure 4 package."""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FULL_MANIFEST = ROOT / "PACKAGE_MANIFEST.sha256"
SOURCE_MANIFEST = ROOT / "package_source_manifest_sha256.csv"
MANIFEST_NAMES = {FULL_MANIFEST.name, SOURCE_MANIFEST.name}
GENERATED_DIRS = {"composite", "qa"}
GENERATED_NAMES = {
    "artifact_manifest.csv",
    "qa_report.md",
    "figure_qa_report.json",
    "composite_build_report.json",
}
GENERATED_SUFFIXES = {".pdf", ".png", ".svg", ".tif", ".tiff"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def package_files() -> list[Path]:
    return sorted(
        path for path in ROOT.rglob("*")
        if path.is_file()
        and path.name not in MANIFEST_NAMES
        and "__pycache__" not in path.parts
        and path.suffix.lower() not in {".pyc", ".pyo"}
    )


def source_files() -> list[Path]:
    files: list[Path] = []
    for path in package_files():
        relative = path.relative_to(ROOT)
        if any(part in GENERATED_DIRS for part in relative.parts):
            continue
        if path.name in GENERATED_NAMES or path.suffix.lower() in GENERATED_SUFFIXES:
            continue
        files.append(path)
    return files


def write_manifests() -> None:
    with SOURCE_MANIFEST.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "bytes", "sha256"])
        writer.writeheader()
        for path in source_files():
            writer.writerow({
                "path": path.relative_to(ROOT).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            })

    files = sorted([*package_files(), SOURCE_MANIFEST])
    with FULL_MANIFEST.open("w", encoding="utf-8", newline="\n") as handle:
        for path in files:
            handle.write(f"{sha256(path)}  {path.relative_to(ROOT).as_posix()}\n")


def verify_manifest() -> None:
    failures: list[str] = []
    entries = 0
    for line in FULL_MANIFEST.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        entries += 1
        expected, rel = line.split("  ", 1)
        path = ROOT / Path(rel)
        if not path.is_file():
            failures.append(f"missing: {rel}")
        elif sha256(path) != expected:
            failures.append(f"hash mismatch: {rel}")
    if failures:
        raise SystemExit("Manifest verification failed:\n" + "\n".join(failures))
    print(f"PASS: verified {entries} archived files")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        verify_manifest()
    else:
        write_manifests()
        print(f"Wrote {FULL_MANIFEST.name} and {SOURCE_MANIFEST.name}")


if __name__ == "__main__":
    main()
