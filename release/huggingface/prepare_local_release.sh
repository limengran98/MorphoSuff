#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
canonical_root=${MORPHOSUFF_CANONICAL_ROOT:-$repo_root/../reports/figure_packages/ops_manuscript_measurement_sufficiency_canonical_final}
scpair_root=${MORPHOSUFF_SCPAIR_ROOT:-$repo_root/../results/ops_low_label_scpair_gene5_v1}
cytoland_root=${MORPHOSUFF_CYTOLAND_ROOT:-$repo_root/../results/ops_raw_phase_cytoland_finetune_pilot6_gene5_v1}
phase_cache=${MORPHOSUFF_PHASE_CACHE:-$repo_root/../data/processed/ops_phase172_indexed}
exact_cache_root=${MORPHOSUFF_EXACT_CACHE_ROOT:-$repo_root/../data/processed/ops_full_reporter_exact/reporters}
asset_audit_root=${MORPHOSUFF_ASSET_AUDIT_ROOT:-$repo_root/../results/ops_phase0_asset_audit}

python3 "$repo_root/release/huggingface/build_release_manifest.py" \
  --canonical-root "$canonical_root" \
  --scpair-root "$scpair_root" \
  --cytoland-root "$cytoland_root" \
  --phase-cache "$phase_cache" \
  --exact-cache-root "$exact_cache_root" \
  --asset-audit-root "$asset_audit_root" \
  --private-manifest "$repo_root/release/huggingface/local_upload_manifest.json" \
  --public-manifest "$repo_root/release/huggingface/MANIFEST.tsv"

/usr/bin/python3 "$repo_root/release/huggingface/upload_release.py" \
  --manifest "$repo_root/release/huggingface/local_upload_manifest.json" \
  --dry-run
