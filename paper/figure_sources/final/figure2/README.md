# Figure 2 source package

This package contains the ten-method benchmark, workflow artwork, frozen
fold/direction scores and reproducible summary calculations for Figure 2.

| Panel | Statistical unit | Display |
| --- | --- | --- |
| a | Schematic | Prediction task and held-out partitions |
| b | Reporter | KO-response and cell-phenotype recovery |
| c | Physical screen | Consensus recovery and method spread |
| d | Reporter–screen assay | KO-response and cell-phenotype recovery |

Each quantitative panel places KO-response recovery on the left and
cell-phenotype recovery on the right. Numerical source IDs B/C map to panel b,
D/E to c, and F/G to d; source identifiers are distinct from panel letters.

## Build

From this directory, using `requirements.txt` and SciPy for the numerical
summary and inference checks:

```powershell
./build.ps1
```

Equivalent Python steps:

```bash
python -B code/archive_manifest.py --verify-source
python -B code/rebuild_benchmark_sources.py
python -B code/fig2_data.py --verify-inference
python -B code/fig2_composite.py
python -B code/verify_printscale.py
python -B code/archive_manifest.py
```

Outputs are `composite/Figure2.{pdf,svg,png}`, standalone panels under
`panels/a/` through `panels/d/`, and local reports in `qa/`.
The publication PDF is stored at `paper/figures/Figure2.pdf`.

## Numerical definitions

Within each method, reporter scores are arithmetic means over five equal-weight
folds or all eligible destination directions. Cross-method consensus is the
unweighted median of the ten aligned method scores.

The compact input `source_data/frozen_partition_scores.csv.gz` contains
5,200 fold rows and 810 destination-direction rows. The builder checks coverage,
reconstructs reporter summaries, derives the corresponding cell-level
screen/assay views and calculates 54 reporter-paired method comparisons.
Independent KO screen/assay tables have their own aggregation definition.

For field and gene cell-level views, screen/assay membership reweights
reporter-level scores: the 99 assay marks therefore resolve 52 distinct
reporter values. Whole-screen cell scores are evaluated directly in the
destination cells. Between-screen spread includes both target composition
and acquisition context.

## Method comparisons

`source_data/statistics/reporter_level_primary_inference.csv` records
paired comparisons with the prespecified MLP. Two-sided Wilcoxon signed-rank
tests use Holm correction across all 54 comparisons and an absolute median
paired-effect criterion of 0.02.

Eleven comparisons meet both criteria. The MLP exceeds scButterfly and MIDAS
for both metrics under field/gene holdout, and exceeds Ridge for field
KO/cell and gene cell recovery. For held-out-gene KO responses, median paired
MLP gains over Ridge, scButterfly and MIDAS are 0.017603, 0.080716 and
0.133383. The Ridge contrast falls below the 0.02 effect criterion.
A marginal median difference is not a paired comparison, and an undetected
advantage does not establish equivalence. Screen and assay summaries are
descriptive.

The read-only `--verify-inference` command recomputes these comparisons
from the deposited scores without writing predictions or changing fitted models.

## Artwork and integrity

Panel a uses `a_workflow/source/Figure2a.pptx` and its bound PDF export.
Microscopy cell crops are illustrative views, not whole fields or actual
partition assignments; source metadata and display specifications accompany
them. Native PowerPoint objects remain editable. Quantitative PDF/SVG exports
retain live text.

The shared `../../publication_style.py` defines DejaVu Sans and 9 pt
bold composite-only letters. Quantitative text is at least 6.5 pt at authored
size. Standalone panels have no letters.

`code/verify_printscale.py` reads the PDF's resolved geometry and the
manuscript's placement; a separate manuscript can be supplied with
`--manuscript path/to/manuscript.tex`. Visual review remains necessary
after rendering.

Source hashes cover the scientific inputs, code, shared style and bound
workflow files. Record an intentional reviewed update with
`python -B code/archive_manifest.py --freeze-source`; ordinary builds
verify the existing record.
