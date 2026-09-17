# Figure 2a methods

The editable source is `source/Figure2a.pptx`, with its bound PDF export
and image records in the same directory.

Ten methods predict 20–72 reporter-specific fluorescence-derived phenotype
endpoints from 172 engineered phase-morphology features. The generic f(x)
does not imply a particular neural architecture. Feature/target strips depict
dimensions, not numeric values. The task is endpoint prediction, not
fluorescence-image synthesis.

The real same-cell pRb–FECH pair from OPS0077 retains 384 × 384-pixel crops,
0.325 µm/pixel calibration, frozen display windows, native orientation and
20 µm bars. The Field examples use real pRb–SALL3 phase cell-centred crops from
OPS0077 wells A1/A2/A3 as illustrative views, not complete imaging fields or
actual train/test assignments. The held-out unit is the entire screen–well–tile
field. Borders convey illustrative split membership, not observed biology.
The visible label "Example cell crops" distinguishes the displayed images from
the whole-field holdout unit; the main legend also states that they are not
the actual partition assignments.

Gene holdout excludes all cells of a test knockout gene. KO A/B/C are generic
labels; the same generated neutral cell is repeated deliberately so its
appearance does not imply measured genotype effects. Field and gene evaluation
use five folds with a separate validation fold, omitted from the compact
train/test drawing. Whole-screen transfer excludes the complete destination
screen and uses source screens carrying the same reporter. The generated
multiwell plate is a screen symbol, not a definition of single-plate holdout.
The experimental hierarchy remains screen → well → field → cell.

Labels, arrows, scale bars, split borders and feature/target strips are native
editable PowerPoint objects. Microscopy and generated biology are independent
picture objects. Raw data, display specifications and generation prompts
accompany the panel.

## Re-export

Run `source/export_to_pdf.ps1 -Python <python.exe>` with Windows PowerPoint and
PyMuPDF. It opens only the authoritative PPT read-only, exports a temporary
native PDF, restores exact embedded image bytes at unchanged PDF placements,
refreshes file hashes and image records in `source/export_manifest.json`,
then regenerates previews. It leaves the PPT unchanged and closes only the
presentation it opened.

The versioned export is `source/Figure2a.pdf`, not a disposable file under
`figure/`. `code/build_figure2a.py` verifies the accepted source/PDF pair, copies
the PDF to `figure/` and rebuilds previews; edited slides require re-export first. Native dimensions
are 183.091667 × 63.5 mm. PNG is 600 dpi; PDF retains native text/vectors; SVG
previews retain text where supported by extraction. Image restoration counters
PowerPoint downsampling without pixel, contrast or layout edits and rejects
cropped, rotated or ambiguous source-picture mappings.

The crop clarification is editable PowerPoint text and is retained in native
PDF export. `source/export_manifest.json` binds the source and export hashes;
source notes use project-relative paths.
