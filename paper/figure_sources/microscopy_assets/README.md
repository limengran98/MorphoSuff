# Canonical raw microscopy assets for manuscript and supplementary figures

This folder is the portable, lossless source bundle for every real microscopy crop
currently displayed in the final six-figure manuscript and its Supplementary Figure 2,
together with direct
same-reporter control/counterpart crops retained for later editing. It is intentionally
separate from rendered figure PDFs/PNGs so a later layout change never requires copying
pixels from a screenshot or a compressed preview.

## Contents

* `raw_assets/` — canonical native arrays. Each `.npz` preserves phase and targeted
  fluorescence as `float32`; Figure 4/5 assets also retain the integer masks used in
  their original analysis. Supplementary Figure 2b representation-boundary examples are phase-only
  `float32` `.npy` arrays.
* `metadata/` — one machine-readable provenance record per raw asset.
* `asset_manifest.csv` — asset identity, use status, figure/panel use, reporter,
  perturbation, screen, cell/crop identifiers, source information and checksums.
* `render_spec.csv` — frozen per-panel display windows, crop orientation, physical
  pixel size and scale-bar policy. It records display only; it never changes source
  pixel values.
* `deduplication_map.csv` — source-package aliases whose raw pixels are already held
  by a canonical asset in `raw_assets/`; this prevents silent loss of potential cases
  while avoiding duplicate copies.
* `asset_checksums.sha256` — integrity hashes for all canonical raw arrays, metadata
  and manifest tables.
* `build_asset_bundle.py` — deterministic copier/auditor. It rebuilds this bundle from
  the frozen `final/` and `supplementary_figure2/` source packages without rebuilding
  or changing a figure.

## Image integrity and rendering policy

The assets are native 384 × 384 level-0 crops at 0.325 µm per pixel where the OPS
source recorded this value. They are **not** rendered PNG/JPEG thumbnails and have not
been denoised, interpolated, registered, enhanced, cropped further or content-edited.
The source-file SHA-256 and an array-content SHA-256 are both stored: the latter remains
stable even if a future lossless NPZ container is repacked.

For a manuscript redraw, load the raw array directly and use the matching row in
`render_spec.csv` for the phase and fluorescence display ranges. Preserve native
orientation and square aspect ratio. Any display interpolation used solely for page
rendering must be recorded in the figure-specific code; it must not be mistaken for a
new microscopy measurement or used to fabricate resolution.

## Scope and provenance boundary

`use_status=rendered` marks assets actively present in the final manuscript figures;
`potential` marks paired or reusable cases retained because they are direct controls or
future-edit counterparts. Shared figure use is deduplicated rather than copied.

Most raw crops trace to the public OPS level-0 OME-Zarr acquisition, with screen, well,
cell identifiers and channel mapping recorded in the manifest. A small number of legacy
Fig. 1 diagnostic crops were preserved as frozen, lossless paired arrays; their final
figure manifests retain the biological/cell registration and display details but not a
complete local staging-shard path. This is a provenance-note only: the stored channels
are the original raw float32 arrays used by the current figures.

## Verification

Run from this directory:

```bash
python build_asset_bundle.py
sha256sum --check asset_checksums.sha256
```

The builder only writes within this `microscopy_assets/` folder.
