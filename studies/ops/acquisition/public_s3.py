#!/usr/bin/env python3
"""Inventory and download public OPS processed single-cell assets.

The Biohub OPS Explorer exposes processed phase and reporter H5AD files from
an unsigned public S3 bucket.  This utility records the remote object size,
downloads selected files with resumable range requests, verifies the final
byte count and writes a local SHA-256 manifest.  No project-specific storage
path is assumed.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import base64
import csv
import hashlib
import json
import re
import threading
import time
from pathlib import Path
from typing import Any


DEFAULT_BUCKET = "ops-explorer-public"
DEFAULT_PREFIX = (
    "leonetti_ops/ops_data_portal_submission/v1.0.20260521/"
    "atlas/cell_features/"
)


def _client() -> Any:
    try:
        import boto3
        from botocore import UNSIGNED
        from botocore.config import Config
    except ImportError as error:  # pragma: no cover - depends on optional extra
        raise RuntimeError(
            "Public OPS acquisition requires boto3; install .[ops-data]."
        ) from error
    return boto3.client(
        "s3",
        config=Config(
            signature_version=UNSIGNED,
            connect_timeout=20,
            read_timeout=60,
            retries={"max_attempts": 4, "mode": "standard"},
        ),
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inventory(bucket: str, prefix: str) -> list[dict[str, object]]:
    client = _client()
    rows: list[dict[str, object]] = []
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for item in page.get("Contents", []):
            key = str(item["Key"])
            if not key.lower().endswith(".h5ad"):
                continue
            name = Path(key).name
            if name == "all_cells_phase.h5ad":
                kind = "phase"
            elif name.startswith("all_cells_fluor_"):
                kind = "reporter"
            else:
                kind = "other"
            rows.append(
                {
                    "kind": kind,
                    "filename": name,
                    "bucket": bucket,
                    "key": key,
                    "s3_uri": f"s3://{bucket}/{key}",
                    "https_url": f"https://{bucket}.s3.amazonaws.com/{key}",
                    "bytes": int(item["Size"]),
                    "last_modified": item["LastModified"].isoformat(),
                }
            )
    return sorted(rows, key=lambda row: (str(row["kind"]), str(row["filename"])))


ACQUISITION_MANIFEST_SCHEMA_VERSION = 1


def published_checksum(client: object, bucket: str, key: str) -> dict[str, object]:
    """Resolve an upstream-published digest for one object.

    Three tiers, most authoritative first. An S3 additional checksum is a real
    published digest. A single-part ETag is the object MD5. A multipart ETag is
    md5-of-md5s and is *not* the object digest, so it is reported as unavailable
    rather than recorded as one; these H5ADs are multi-GB and almost always
    multipart, which is why the verifier must tolerate an absent published digest
    instead of failing forever.
    """
    try:
        head = client.head_object(Bucket=bucket, Key=key, ChecksumMode="ENABLED")
    except TypeError:  # pragma: no cover - older botocore without ChecksumMode
        head = client.head_object(Bucket=bucket, Key=key)
    encoded = head.get("ChecksumSHA256")
    if encoded:
        return {
            "published_checksum_algorithm": "sha256",
            "published_checksum": base64.b64decode(encoded).hex(),
            "published_checksum_source": "s3_object_checksum",
        }
    etag = str(head.get("ETag", "")).strip('"')
    if re.fullmatch(r"[0-9a-f]{32}", etag):
        return {
            "published_checksum_algorithm": "md5",
            "published_checksum": etag,
            "published_checksum_source": "s3_etag_single_part",
        }
    return {
        "published_checksum_algorithm": None,
        "published_checksum": None,
        "published_checksum_source": "unavailable",
        "published_checksum_reason": (
            "multipart ETag is md5-of-md5s, and the object carries no S3 additional "
            "checksum; supply one with --published-checksums if upstream publishes it"
        ),
    }


def acquisition_manifest(records: list[dict[str, object]]) -> dict[str, object]:
    """Shape download records into the manifest verify_acquisition.py reads.

    The downloader previously wrote a bare JSON list, which the verifier could not
    read at all; see reproducibility/ops/manifests/acquisition_manifest.example.json
    for the contract.
    """
    assets = []
    for row in records:
        assets.append(
            {
                "logical_name": f"{row.get('kind', 'asset')}:{row.get('filename')}",
                "local_path": str(row.get("path", "")),
                "file_name": row.get("filename"),
                "bytes": row.get("bytes"),
                "local_sha256": row.get("sha256"),
                "upstream_locator": row.get("s3_uri"),
                "published_checksum_algorithm": row.get("published_checksum_algorithm"),
                "published_checksum": row.get("published_checksum"),
                "published_checksum_source": row.get("published_checksum_source", "unavailable"),
            }
        )
    return {
        "schema_version": ACQUISITION_MANIFEST_SCHEMA_VERSION,
        "source": "ops-explorer-public-s3",
        "assets": assets,
    }


def write_inventory(rows: list[dict[str, object]], output: Path) -> None:
    if not rows:
        raise SystemExit(
            "refusing to write an empty inventory: an empty result means the pinned "
            "prefix matched nothing, which is a failure rather than an inventory of "
            "zero objects. Pass --allow-empty to record it deliberately."
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() == ".json":
        output.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        return
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)


def read_inventory(path: Path) -> list[dict[str, object]]:
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError("inventory JSON must contain a list")
        return [dict(row) for row in payload]
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def resumable_download(
    *,
    bucket: str,
    key: str,
    output: Path,
    expected_bytes: int,
    chunk_mib: int,
    workers: int,
    max_attempts: int,
) -> dict[str, object]:
    client = _client()
    remote_bytes = int(client.head_object(Bucket=bucket, Key=key)["ContentLength"])
    if remote_bytes != expected_bytes:
        raise RuntimeError(
            f"remote byte count changed for {key}: {remote_bytes} != {expected_bytes}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".part")
    if output.exists():
        if output.stat().st_size != remote_bytes:
            raise RuntimeError(f"existing file has the wrong size: {output}")
        return {
            "path": str(output.resolve()),
            "bytes": remote_bytes,
            "sha256": sha256_file(output),
            "status": "reused",
        }
    completed = partial.stat().st_size if partial.exists() else 0
    if completed > remote_bytes:
        raise RuntimeError(f"partial file is larger than remote object: {partial}")
    chunk = int(chunk_mib) * 1024 * 1024
    lock = threading.Lock()

    def fetch(start: int, end: int) -> tuple[int, bytes]:
        expected = end - start + 1
        for attempt in range(1, max_attempts + 1):
            try:
                response = client.get_object(
                    Bucket=bucket, Key=key, Range=f"bytes={start}-{end}"
                )
                body = response["Body"]
                try:
                    payload = body.read()
                finally:
                    body.close()
                if len(payload) != expected:
                    raise IOError(f"short range read: {len(payload)} != {expected}")
                return start, payload
            except Exception:
                if attempt == max_attempts:
                    raise
                time.sleep(min(30, 2**attempt))
        raise AssertionError("unreachable")

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        while completed < remote_bytes:
            ranges: list[tuple[int, int]] = []
            cursor = completed
            for _ in range(workers):
                if cursor >= remote_bytes:
                    break
                end = min(remote_bytes - 1, cursor + chunk - 1)
                ranges.append((cursor, end))
                cursor = end + 1
            futures = [pool.submit(fetch, start, end) for start, end in ranges]
            payloads = dict(future.result() for future in futures)
            with lock, partial.open("ab") as handle:
                for start, _ in ranges:
                    if start != completed:
                        raise RuntimeError("non-contiguous ranged download")
                    payload = payloads[start]
                    handle.write(payload)
                    completed += len(payload)
            print(
                f"{output.name}: {completed / 1024**3:.2f}/"
                f"{remote_bytes / 1024**3:.2f} GiB "
                f"({100 * completed / remote_bytes:.1f}%)",
                flush=True,
            )
    partial.replace(output)
    return {
        "path": str(output.resolve()),
        "bytes": remote_bytes,
        "sha256": sha256_file(output),
        "status": "downloaded",
    }


def select_rows(
    rows: list[dict[str, object]], kinds: set[str], reporters: set[str]
) -> list[dict[str, object]]:
    selected = []
    for row in rows:
        if str(row["kind"]) not in kinds:
            continue
        if reporters and str(row["kind"]) == "reporter":
            token = str(row["filename"]).removeprefix("all_cells_fluor_").removesuffix(
                ".h5ad"
            )
            if token.lower() not in reporters:
                continue
        selected.append(row)
    return selected


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    inv = sub.add_parser("inventory", help="list public processed H5AD assets")
    inv.add_argument("--allow-empty", action="store_true", help="record an empty inventory instead of failing")
    inv.add_argument("--bucket", default=DEFAULT_BUCKET)
    inv.add_argument("--prefix", default=DEFAULT_PREFIX)
    inv.add_argument("--output", type=Path, required=True)

    get = sub.add_parser("download", help="download selected assets from an inventory")
    get.add_argument("--inventory", type=Path, required=True)
    get.add_argument("--output-dir", type=Path, required=True)
    get.add_argument(
        "--kind", action="append", choices=["phase", "reporter", "other"], default=[]
    )
    get.add_argument(
        "--reporter",
        action="append",
        default=[],
        help="reporter filename token, for example lysosome_lamp1",
    )
    get.add_argument("--chunk-mib", type=int, default=64)
    get.add_argument("--workers", type=int, default=4)
    get.add_argument("--max-attempts", type=int, default=8)
    get.add_argument("--manifest", type=Path)
    get.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "inventory":
        rows = inventory(args.bucket, args.prefix)
        if not rows and not args.allow_empty:
            raise SystemExit(
                f"no .h5ad objects under s3://{args.bucket}/{args.prefix}; the prefix is "
                "wrong or the release moved. Pass --allow-empty to record this anyway."
            )
        write_inventory(rows, args.output)
        print(f"wrote {len(rows)} public objects to {args.output}")
        return 0

    rows = read_inventory(args.inventory)
    kinds = set(args.kind or ["phase", "reporter"])
    reporters = {str(value).lower() for value in args.reporter}
    selected = select_rows(rows, kinds, reporters)
    if not selected:
        raise SystemExit("no objects matched the requested selection")
    if args.dry_run:
        print(json.dumps(selected, indent=2))
        return 0
    records = []
    for row in selected:
        records.append(
            {
                **row,
                **resumable_download(
                    bucket=str(row["bucket"]),
                    key=str(row["key"]),
                    output=args.output_dir / str(row["filename"]),
                    expected_bytes=int(row["bytes"]),
                    chunk_mib=args.chunk_mib,
                    workers=args.workers,
                    max_attempts=args.max_attempts,
                ),
            }
        )
    manifest = args.manifest or args.output_dir / "acquisition_manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(acquisition_manifest(records), indent=2), encoding="utf-8")
    print(f"wrote verified local manifest: {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
