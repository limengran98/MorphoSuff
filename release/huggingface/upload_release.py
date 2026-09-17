#!/usr/bin/env python3
"""Upload a no-copy MorphoSuff release manifest to Hugging Face."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--manifest", type=Path, required=True)
    value.add_argument("--repo-id", default="Amanda1998/MorphoSuff")
    value.add_argument("--repo-type", default="dataset", choices=("dataset", "model", "space"))
    value.add_argument("--revision", default="main")
    value.add_argument("--batch-size", type=int, default=32)
    value.add_argument(
        "--large-file-threshold-gib",
        type=float,
        default=1.0,
        help="Upload files at least this large in their own resumable commit.",
    )
    value.add_argument("--dry-run", action="store_true")
    value.add_argument("--state", type=Path, default=Path(__file__).with_name("upload_state.json"))
    return value


def main() -> int:
    args = parser().parse_args()
    payload = json.loads(args.manifest.read_text(encoding="utf-8"))
    files = payload["files"]
    api = None
    if not args.dry_run:
        try:
            from huggingface_hub import CommitOperationAdd, HfApi, get_token
        except ImportError as exc:
            raise SystemExit(
                "Install huggingface_hub or run this script with /usr/bin/python3 on the study system"
            ) from exc
        token = os.environ.get("HF_TOKEN") or get_token()
        if not token:
            raise SystemExit(
                "No Hugging Face credential found. Export HF_TOKEN with write access to "
                "Amanda1998/MorphoSuff or run `huggingface-cli login`."
            )
        api = HfApi(token=token)
        account = api.whoami()
        print(f"huggingface_login\t{account.get('name', 'unknown')}", flush=True)
    for item in files:
        path = Path(item["local_path"])
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size != item["size_bytes"] or sha256_file(path) != item["sha256"]:
            raise RuntimeError(f"Source changed after manifest creation: {path}")
    if args.dry_run:
        print(json.dumps({"status": "dry_run_pass", "files": len(files), "bytes": sum(item["size_bytes"] for item in files)}, indent=2))
        return 0

    assert api is not None
    api.create_repo(repo_id=args.repo_id, repo_type=args.repo_type, exist_ok=True)
    completed: set[str] = set()
    if args.state.is_file():
        state_payload = json.loads(args.state.read_text(encoding="utf-8"))
        if state_payload.get("repo_id") not in (None, args.repo_id):
            raise RuntimeError(
                f"Upload state belongs to {state_payload.get('repo_id')}, not {args.repo_id}"
            )
        completed = set(state_payload.get("completed", []))

    metadata_root = Path(__file__).resolve().parent
    metadata = [
        (metadata_root / "README.md", "README.md"),
        (metadata_root / "OPS_TRAINING_DATA.md", "processed_ops/README.md"),
        (metadata_root / "MODEL_METADATA.md", "models/README.md"),
        (metadata_root / "MANIFEST.tsv", "MANIFEST.tsv"),
    ]
    queue = [(str(path), destination, None, path.stat().st_size) for path, destination in metadata]
    queue.extend(
        (item["local_path"], item["repo_path"], item["sha256"], item["size_bytes"])
        for item in files
    )
    pending = []
    for local_path, repo_path, digest, size_bytes in queue:
        file_identity = f"{repo_path}:{digest or sha256_file(Path(local_path))}"
        if file_identity not in completed:
            pending.append((local_path, repo_path, file_identity, size_bytes))
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if args.large_file_threshold_gib <= 0:
        raise ValueError("--large-file-threshold-gib must be positive")
    large_threshold = int(args.large_file_threshold_gib * 1024**3)
    batches = []
    current = []
    for item in pending:
        if item[3] >= large_threshold:
            if current:
                batches.append(current)
                current = []
            batches.append([item])
        else:
            current.append(item)
            if len(current) == args.batch_size:
                batches.append(current)
                current = []
    if current:
        batches.append(current)

    completed_count = len(queue) - len(pending)
    for batch_index, batch in enumerate(batches, start=1):
        operations = [
            CommitOperationAdd(path_in_repo=repo_path, path_or_fileobj=local_path)
            for local_path, repo_path, _, _ in batch
        ]
        batch_bytes = sum(size_bytes for _, _, _, size_bytes in batch)
        first = completed_count + 1
        last = completed_count + len(batch)
        print(
            f"uploading_batch\t{batch_index}/{len(batches)}\tfiles={len(batch)}\t"
            f"bytes={batch_bytes}\tpaths={batch[0][1]}..{batch[-1][1]}",
            flush=True,
        )
        api.create_commit(
            repo_id=args.repo_id,
            repo_type=args.repo_type,
            revision=args.revision,
            operations=operations,
            commit_message=f"Add MorphoSuff release files {first}-{last}",
            num_threads=min(8, len(batch)),
        )
        completed.update(file_identity for _, _, file_identity, _ in batch)
        args.state.write_text(json.dumps({"repo_id": args.repo_id, "completed": sorted(completed)}, indent=2) + "\n", encoding="utf-8")
        completed_count = last
        print(f"uploaded_batch\t{first}-{last}", flush=True)
    print(json.dumps({"status": "complete", "repo_id": args.repo_id, "files": len(queue)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
