# Public OPS data acquisition

The processed single-cell phase and reporter tables used by the benchmark are
publicly readable from the Biohub OPS Explorer S3 release. No cloud account or
signed URL is required.

Install the acquisition dependencies and create a remote inventory:

```bash
python -m pip install -e ".[ops-data]"
python studies/ops/acquisition/public_s3.py inventory \
  --output local/ops_public_h5ad_inventory.csv
```

Download the phase table and selected reporter tables:

```bash
python studies/ops/acquisition/public_s3.py download \
  --inventory local/ops_public_h5ad_inventory.csv \
  --output-dir local/ops_h5ad \
  --kind phase --kind reporter \
  --reporter lysosome_lamp1 \
  --reporter stress_response_5xupre
```

Omit `--reporter` to download all processed reporter H5AD files. Downloads are
resumable, checked against the public object byte count and recorded with a
local SHA-256. Use `--dry-run` to inspect the exact selected objects without
downloading them.

Continue directly to exact pairing and canonical training assets:

```bash
python studies/ops/preparation/prepare_public_ops.py \
  --h5ad-dir local/ops_h5ad \
  --output-dir local/ops_prepared \
  --reporter-registry configs/ops/data/reporter_registry.csv
```

The compact source archive and upstream analysis notebooks remain available
from Zenodo (`10.5281/zenodo.20495192`) and the pinned
`czbiohub-sf/ops-paper-analysis` repository. Raw image runners use the same
public `ops-explorer-public` bucket and accept a user-selected image root.
