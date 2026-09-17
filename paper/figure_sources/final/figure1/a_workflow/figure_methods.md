# Figure 1a methods

The editable source is `source/Figure1a.pptx`. Native labels, arrows,
sequencing symbols, schema blocks and scale bar remain editable. Generated
biology and measured microscopy remain separate embedded picture objects.

This schematic describes the public source study, not newly collected samples:
1,000 knockout genes, 73 screens and 52 reporters across screens. Each screen
has 1–7 reporters, giving 99 acquired reporter–screen assays; all reporters
are not jointly observed in every cell. Approximately 4,000 denotes targeting
sgRNAs, supported by 3,991–4,000 guides in the frozen reporter-order table,
not an exact guide total including controls.

Segmentation supplies cell boundaries and features; in situ sequencing supplies
sgRNA identity. Separate branches join a shared cell ID. No physical extraction
or morphology-based guide identification is depicted. A/C/G/T are schematic
alphabet symbols, not observed reads. The sgRNA, culture, screen and segmentation
illustrations were newly generated, not cut from the reference, and are not
measured evidence. The segmentation cartoon is not a computed microscopy mask.

The real LAMP1–BORCS7 pair from OPS0011 retains the 384 × 384-pixel crops,
0.325 µm/pixel calibration, frozen display windows and orientation. The common
external 20 µm bar applies to both images. Raw arrays and exact-cell metadata
are in `source_data/raw_data`; `source_data/MICROSCOPY_PROVENANCE.md` records
the display specification. Reporter-specific cell-by-endpoint blocks are
editable schema drawings, not numeric heatmaps: row counts, widths and row
fills are illustrative. Targets comprise 20–72 endpoints per reporter and
1,604 heterogeneous endpoints overall; downstream KO aggregation is not shown.

## Re-export

Run `source/export_to_pdf.ps1 -Python <python.exe>` with Windows PowerPoint and
PyMuPDF. It opens only this PPT read-only, exports a temporary native PDF,
restores the exact embedded image bytes at the PDF's unchanged placements,
refreshes hashes and image records in `source/export_manifest.json`, then
regenerates previews. It leaves the source unchanged and closes only the
presentation it opened.

The versioned export is `source/Figure1a.pdf`, not a disposable file under
`figure/`. `code/build_figure1a.py` verifies source/PDF hashes, copies the PDF
to `figure/` and rebuilds previews; an edited PPT must first be re-exported.
Source-note paths are project-relative. Native dimensions are
183.091667 × 63.5 mm; PNG is 600 dpi, PDF retains native text and vectors, and
SVG previews preserve text where supported by extraction. Restoration rejects
cropped, rotated or ambiguous picture mappings instead of guessing. No pixels,
contrast or layout are edited during export.
