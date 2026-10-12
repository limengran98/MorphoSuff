#!/usr/bin/env python3
"""Verify an untracked OPS acquisition manifest without persisting local paths."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def published_digest(path: Path, algorithm: str) -> str:
    if algorithm not in {"md5", "sha256"}:
        raise ValueError(f"unsupported published checksum algorithm: {algorithm}")
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


LEGACY_KEYS = {"filename", "path", "sha256"}


def normalise(payload: object) -> tuple[list[dict], list[str]]:
    """Accept both the current manifest and the legacy bare-list download record.

    The released downloader wrote a JSON list, which this verifier could not read
    at all. Upgrading that shape in memory means a user who already downloaded
    terabytes can verify them without downloading again.
    """
    notes: list[str] = []
    if isinstance(payload, dict) and "assets" in payload:
        return list(payload["assets"]), notes
    if isinstance(payload, list) and all(isinstance(item, dict) for item in payload):
        if payload and not LEGACY_KEYS.intersection(payload[0]):
            raise SystemExit(
                "manifest is a list but does not look like a download record; see "
                "reproducibility/ops/manifests/acquisition_manifest.example.json"
            )
        notes.append("legacy inventory-style manifest upgraded in memory")
        return [
            {
                "logical_name": f"{item.get('kind', 'asset')}:{item.get('filename')}",
                "local_path": item.get("path", ""),
                "bytes": item.get("bytes"),
                "local_sha256": item.get("sha256", ""),
                "published_checksum_algorithm": item.get("published_checksum_algorithm"),
                "published_checksum": item.get("published_checksum"),
                "published_checksum_source": item.get("published_checksum_source", "unavailable"),
            }
            for item in payload
        ], notes
    raise SystemExit(
        "unrecognized acquisition manifest shape; expected an object with an 'assets' "
        "list, as in reproducibility/ops/manifests/acquisition_manifest.example.json"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--published-checksum-policy",
        choices=("require", "require-when-published", "none"),
        default="require-when-published",
        help=(
            "require: every asset must carry an upstream digest (use before tagging a "
            "release). require-when-published: verify it when upstream publishes one, "
            "since a multipart S3 ETag is not an object digest. none: local digests only."
        ),
    )
    args = parser.parse_args()
    assets, notes = normalise(json.loads(args.manifest.read_text()))
    if not assets:
        raise SystemExit(
            "acquisition manifest declares no assets; nothing was verified. An empty "
            "manifest is a failed acquisition, not a successful verification."
        )
    failures = []
    n_published = 0
    n_unavailable = 0
    for asset in assets:
        path = Path(asset["local_path"])
        algorithm = str(asset.get("published_checksum_algorithm") or "").lower()
        published = str(asset.get("published_checksum") or "").lower()
        source = str(asset.get("published_checksum_source") or "").lower()
        local_sha = str(asset.get("local_sha256", "")).lower()
        has_published = algorithm in {"md5", "sha256"} and len(published) == {"md5": 32, "sha256": 64}[algorithm]
        if not path.is_file():
            failures.append(f"missing: {asset['logical_name']}")
            continue
        if not has_published:
            if args.published_checksum_policy == "require" or source not in {"unavailable", ""}:
                failures.append(f"missing published checksum: {asset['logical_name']}")
                continue
            n_unavailable += 1
        elif published_digest(path, algorithm) != published:
            failures.append(f"published checksum mismatch: {asset['logical_name']}")
            continue
        else:
            n_published += 1
        if len(local_sha) != 64 or local_sha.startswith("replace-"):
            failures.append(f"missing locally recorded SHA-256: {asset['logical_name']}")
        elif sha256(path) != local_sha:
            failures.append(f"local SHA-256 mismatch: {asset['logical_name']}")
        elif asset.get("bytes") and path.stat().st_size != int(asset["bytes"]):
            failures.append(f"byte-count mismatch: {asset['logical_name']}")
    if failures:
        raise SystemExit("OPS acquisition verification failed: " + "; ".join(failures))
    print(
        json.dumps(
            {
                "status": "PASS",
                "n_assets": len(assets),
                "n_published_checksum_verified": n_published,
                "n_published_checksum_unavailable": n_unavailable,
                "n_local_sha256_verified": len(assets),
                "published_checksum_policy": args.published_checksum_policy,
                "notes": notes,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
