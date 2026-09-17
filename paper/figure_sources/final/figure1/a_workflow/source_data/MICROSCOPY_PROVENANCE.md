# Original microscopy — retained, not redrawn

The two microscopy objects were retained from the supplied base PowerPoint, repositioned and resized together. Neither image was regenerated or content-edited. Separate PNG copies and the underlying arrays are provided for provenance.

- Asset: `lamp1_borcs7_ko`.
- Reporter: LAMP1; perturbation: BORCS7 knockout.
- Screen: Biohub_OPS0011; well: A/1/0; segmentation ID: 5095123; phase row index: 255214.
- Native stored shape: 384 × 384 pixels per channel; 0.325 µm/pixel; native orientation.
- Data: `raw_data/lamp1_borcs7_ko.npz`; metadata: same basename `.json`.
- Raw NPZ SHA256: `81925e7524f4e97bb096b67d5135c674ae3c3b194ee0d0e29e903b11aac12152`.
- Original source: `s3://ops-explorer-public/leonetti_ops/ops_data_portal_submission/v1.0.20260521/datasets/Biohub_OPS0011/Biohub_OPS0011.zarr`.

Fixed display windows:

| Channel | Low | High | Display |
| --- | ---: | ---: | --- |
| Phase | -0.3490651994943619 | 0.6395067185163511 | Grayscale |
| Fluorescence | 106.61254043579102 | 211.3301994323731 | Original 256-entry linear gradient: #020609 → #173947 → #4FA7B8 → #DDF1F3 |

No new contrast fitting, denoising, registration, rotation, crop or pixel synthesis was performed.

In the revised slide each image is 64 CSS px wide. The common external 20 µm bar is `64 × 20 / (384 × 0.325) = 10.2564102564` CSS px long, equivalent to 7.6923 pt. The same scale applies to both images.

The neighbouring generated cell-segmentation illustration is a separate explanatory schematic, not a computed segmentation of these measured crops.
