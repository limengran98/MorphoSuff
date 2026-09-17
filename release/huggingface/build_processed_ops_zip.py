#!/usr/bin/env python3
"""Build one ZIP64 archive containing the training-ready processed OPS layer."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import zipfile
from pathlib import Path


PREFIX = "processed_ops/"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--readme", type=Path, default=Path(__file__).with_name("OPS_TRAINING_DATA.md"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = json.loads(args.manifest.read_text(encoding="utf-8"))
    rows = sorted(
        (row for row in payload["files"] if str(row["repo_path"]).startswith(PREFIX)),
        key=lambda row: row["repo_path"],
    )
    if not rows:
        raise RuntimeError("Manifest contains no processed_ops files")
    if len({row["repo_path"] for row in rows}) != len(rows):
        raise RuntimeError("Manifest contains duplicate processed_ops destinations")
    for row in rows:
        source = Path(row["local_path"])
        if not source.is_file() or source.stat().st_size != int(row["size_bytes"]):
            raise RuntimeError(f"Missing or size-mismatched source: {source}")

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".partial")
    if output.exists() or partial.exists():
        raise FileExistsError(f"Refusing to replace existing archive or partial output: {output}")

    manifest_buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        manifest_buffer,
        fieldnames=("path", "size_bytes", "sha256", "kind", "license", "source_identity"),
        delimiter="\t",
        lineterminator="\n",
    )
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {
                "path": row["repo_path"],
                "size_bytes": row["size_bytes"],
                "sha256": row["sha256"],
                "kind": row["kind"],
                "license": row["license"],
                "source_identity": row["source_identity"],
            }
        )

    try:
        with zipfile.ZipFile(
            partial,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=1,
            allowZip64=True,
            strict_timestamps=False,
        ) as archive:
            archive.writestr("README.md", args.readme.read_text(encoding="utf-8"))
            archive.writestr("ARCHIVE_MANIFEST.tsv", manifest_buffer.getvalue())
            total = len(rows)
            for index, row in enumerate(rows, start=1):
                source = Path(row["local_path"])
                print(
                    f"adding\t{index}/{total}\t{row['repo_path']}\t{row['size_bytes']}",
                    flush=True,
                )
                archive.write(source, arcname=row["repo_path"])
        os.replace(partial, output)
    except BaseException:
        if partial.exists():
            partial.unlink()
        raise

    with zipfile.ZipFile(output, mode="r") as archive:
        bad_member = archive.testzip()
        names = set(archive.namelist())
    if bad_member is not None:
        raise RuntimeError(f"ZIP CRC validation failed at {bad_member}")
    expected = {row["repo_path"] for row in rows} | {"README.md", "ARCHIVE_MANIFEST.tsv"}
    if names != expected:
        raise RuntimeError(f"ZIP membership mismatch: missing={expected - names}, extra={names - expected}")

    archive_sha256 = sha256_file(output)
    checksum = output.with_suffix(output.suffix + ".sha256")
    checksum.write_text(f"{archive_sha256}  {output.name}\n", encoding="utf-8")
    summary = {
        "status": "PASS",
        "archive": str(output),
        "files": len(rows),
        "source_bytes": sum(int(row["size_bytes"]) for row in rows),
        "archive_bytes": output.stat().st_size,
        "sha256": archive_sha256,
        "checksum_file": str(checksum),
    }
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
