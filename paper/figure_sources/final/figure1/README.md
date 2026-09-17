# Figure 1 source package

This package contains the numerical tables, registered microscopy inputs,
editable workflow and plotting code for Figure 1.

| Panel | Content | Source directory |
| --- | --- | --- |
| a | OPS acquisition, exact-cell linkage and reporter-specific targets | `a_workflow/` |
| b | Screen and assay coverage; reporter endpoint dimensions | `b_exact_linkage_scale/` |
| c | Sparse reporter–screen acquisition topology | `c_sparse_assay_topology/` |
| d | Knockout-response atlas | `d_ko_response_atlas/` |
| e | Same-cell phase and targeted-response examples | `e_same_cell_response_maps/` |

Panel methods and legends specify statistical units, counting rules, selected
examples and display transforms. Data totals are 73 screens, 99 reporter–screen
assays, 52 reporters, 7,344,374 distinct linked phase cells, 9,996,286 paired
reporter observations and 1,604 targeted endpoints.

## Build

From this directory, using the environment in `requirements.txt`:

```powershell
./build.ps1
```

Equivalent Python steps:

```bash
python -B code/archive_manifest.py --verify-source
python -B build_figure1.py
python -B code/verify_figure1.py
python -B code/archive_manifest.py
```

Outputs include each panel's `figure/` exports, the complete
`composite/Figure1.{pdf,svg,png}` and local verification reports in `qa/`.
The publication PDF is stored at `paper/figures/Figure1.pdf`.

## Workflow and microscopy sources

The editable workflow is `a_workflow/source/Figure1a.pptx`; its PDF export
and binding hashes are beside it. Ordinary builds verify and assemble these
inputs. To export an edited slide, use the adjacent `export_to_pdf.ps1`
helper with native PowerPoint.

Measured microscopy is stored separately from schematic illustrations.
Raw arrays, cell identities, display windows and scale calibration are
documented under `a_workflow/source_data/` and the corresponding panel
source directories. Generation records identify the schematic assets.

## Layout and integrity

The composite is 183 × 239.468 mm. Panels b and c are 183 × 30 mm and
183 × 85 mm, respectively. The shared `../../publication_style.py`
defines DejaVu Sans, 9 pt bold composite letters and a 6 pt ordinary-text floor.
Standalone panels are unlabelled.

The verifier checks PDF geometry, resolved text size and the manuscript's
actual placement. For a separate manuscript copy:

```bash
python -B code/verify_figure1.py --manuscript path/to/manuscript.tex
```

An intentional source update is recorded with
`python -B code/archive_manifest.py --freeze-source` after review.
Normal builds verify the source manifest. After rendering, inspect the
complete figure at manuscript size for label, mark and image readability.
