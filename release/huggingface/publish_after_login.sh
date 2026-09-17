#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
manifest="$repo_root/release/huggingface/local_upload_manifest.json"

if [[ ! -f "$manifest" ]] || ! python3 - "$manifest" <<'PY'
import json
import sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
paths = {row["repo_path"] for row in payload.get("files", [])}
raise SystemExit(0 if "processed_ops/phase172/phase172.npy" in paths else 1)
PY
then
  "$repo_root/release/huggingface/prepare_local_release.sh"
fi

/usr/bin/python3 "$repo_root/release/huggingface/upload_release.py" \
  --manifest "$manifest" \
  --repo-id Amanda1998/MorphoSuff \
  --repo-type dataset
