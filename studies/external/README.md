# External dataset workflows

These workflows apply measurement-sufficiency assessment to paired measurements
outside the OPS atlas. They prepare public data, fit held-out predictors, estimate
response fidelity and target reliability, and export the evidence and decisions
used in the external applications in Fig. 6f.

The reusable component is the assessment logic. Pairing, statistical units,
reliability estimators and input representations follow each experimental design.
For your own resource, start with the [new-dataset guide](../../docs/adapt_new_dataset.md).

## Dataset comparison

| | OASIS (`cpg0037-oasis`) | PERISCOPE (`cpg0021-periscope`) |
| --- | --- | --- |
| Biological system | Primary human hepatocytes, `axiom` arm | A549 cells |
| Predictor input | Brightfield; 808 features | Four Cell Painting dyes; 2,508 features |
| Target measurement | Metabolic activity and LDH release | Anti-TOMM20; 823 endpoints |
| Pairing unit | Well | Cell; also deposited guide-level profiles |
| Perturbations | 1,085 compounds, eight concentrations | Genome-wide CRISPR; 20,393 genes |
| Acquisition contexts | 66 plates, four production batches | Nine plates, three viral transductions |
| Controls | Plate-matched DMSO wells | Plate-matched nontargeting guides |
| Approximate source size | 0.39 GB of well-level profiles | 8 GB guide profiles; 125 GB single-cell profiles |

OASIS tests a brightfield-to-biochemical measurement question in a different
biological system. PERISCOPE tests fluorescent-input prediction of a targeted
readout across platforms in A549, the same cell line as the OPS atlas.

## Installation and locations

From the repository root:

```bash
python -m pip install -e '.[sklearn,parquet]'
```

Preparation reads anonymously from the public Cell Painting Gallery; credentials
are not required. Data are downloaded or streamed when a preparation command runs.
All preparation and analysis output directories must be outside the checkout;
the scripts enforce this requirement. The examples use `../external/`.

Keep the default Parquet format for these workflows. PERISCOPE analysis loaders
expect Parquet metadata and partitioned input/target tables.

## OASIS: prepare, fit and assess

```bash
python studies/external/oasis/prepare.py \
  --source axiom \
  --output-dir ../external/oasis_canonical

python studies/external/oasis/analyse.py \
  --prepared ../external/oasis_canonical \
  --output-dir ../external/oasis_analysis \
  --folds 5 --seed 0
```

The `axiom` arm provides uniform per-plate profiles. The analysis holds out
complete compounds, fits Ridge models with training-only preprocessing, and
evaluates the plate-normalized target endpoints.
Replicate-well partitions provide the target-reliability estimate.

For a batch-specific assessment, add `--batch axiom_prod_25` to the analysis command
and use a separate output directory. The other batches are `axiom_prod_26`,
`axiom_prod_27` and `axiom_prod_30`.

To run the input-permutation control, repeat the analysis with a new output
directory and `--permute-inputs within_screen`. This keeps targets and controls
fixed while shuffling brightfield inputs within plate.

The [Supplementary Information](../../paper/supplementary.pdf) reports
batch-specific results and the permutation comparison. Full-precision main
and batch evidence, with threshold-sensitivity results, is deposited in
[Supplementary Data 2](../../paper/supplementary_data/SupplementaryData2/README.md).

## OASIS: prioritize metabolic-activity loss

The complementary use-specific analysis asks how well brightfield predictions
select compounds with the largest decreases in measured metabolic activity.
It uses 768 released brightfield DINO features, paired development data from
868 compounds and a fixed evaluation on 217 held-out compounds. The primary
top-5% budget recovered 10 of the 11 largest measured losses. Dose-matched
measurements from two production sources provide a repeat-measurement reference
on 94 compounds.

[Supplementary Data 3](../../paper/supplementary_data/SupplementaryData3/README.md)
contains frozen predictions, all fixed model and budget comparisons, compound
selection records and a no-refit scoring workflow. This is a worked study analysis,
using the same top-fraction recall definition as the reusable package. It does
not change the preparation or fitting commands above: the input representation,
signed loss and dose aggregation differ from the original response-fidelity
analysis in Fig. 6f.

## PERISCOPE: guide-level workflow

```bash
python studies/external/periscope/prepare.py \
  --level guide \
  --output-dir ../external/periscope_guide

python studies/external/periscope/analyse_guide.py \
  --prepared ../external/periscope_guide \
  --output-dir ../external/periscope_guide_analysis \
  --folds 5 --seed 0 --alpha auto
```

This route uses deposited gene–sgRNA–plate profiles. Genes are held out together,
the Ridge penalty is selected inside each outer training set, and response
summaries are computed within the fold loop. Reliability uses disjoint guide
partitions within plate, with endpoint standardization before pooling.

The guide-level permutation control also supports `--permute-inputs within_screen`.
The reported control uses `--alpha-ratio 0.01` with a separate output directory;
this fixed-penalty control is distinct from the nested-selected main analysis.

## PERISCOPE: cell-level workflow

```bash
python studies/external/periscope/prepare.py \
  --level single_cell --max-cells 120000 \
  --chunk-rows 40000 --rows-per-part 200000 \
  --output-dir ../external/periscope_cells

python studies/external/periscope/analyse_cells.py \
  --prepared ../external/periscope_cells \
  --output-dir ../external/periscope_cells_analysis \
  --folds 5 --seed 0 --alpha auto
```

The manuscript preparation retains the first 120,000 streamed cells per plate,
for 1.08 million cells across nine plates. The analysis holds out genes, predicts
individual cells and then aggregates perturbation responses. Guide-level
prediction instead starts with averaged input and target profiles, so the two
routes have different evaluation units. The cell-level fit loads roughly 15 GB
of input and target arrays before working-memory overhead.

## PERISCOPE diagnostics

| Question | Analysis |
| --- | --- |
| How does prediction depend on the input channel? | [Channel ablation and within-site pairing controls](periscope/bleedthrough_control.py) on a single plate with held-out imaging sites |
| Does reliability depend on endpoint scaling or gene selection? | [Reliability analysis](periscope/reliability_audit.py) with predefined gene sets and size-matched random subsets |
| Can three Cell Painting dyes predict the fourth? | [Leave-one-dye-out analysis](periscope/analyse_stain_dropout.py) on guide profiles |

Channel ablation addresses optical cross-talk separately from the dependence
of the deposited segmentation masks on the target channel. The dye analyses
evaluate additional measurement questions within PERISCOPE, not additional
independent datasets. Results and interpretation are documented in the
[Supplementary Information](../../paper/supplementary.pdf); the complete
external evidence display is in the
[Supplementary Figure 1 source package](../../paper/figure_sources/supplementary_figure1/README.md).

<details>
<summary><b>Diagnostic commands</b></summary>

Single-plate channel and pairing controls:

```bash
python studies/external/periscope/prepare.py \
  --level single_cell --plate CP186N --max-cells 150000 \
  --output-dir ../external/periscope_ctrl

python studies/external/periscope/bleedthrough_control.py \
  --prepared ../external/periscope_ctrl \
  --output-dir ../external/periscope_control --max-cells 40000
```

For gene-set sensitivity, prepare one human gene symbol per line from the
CEGv2 and MitoCarta 3.0 collections cited in the Supplementary Information.
The paths below refer to those user-supplied lists:

```bash
python studies/external/periscope/reliability_audit.py \
  --prepared ../external/periscope_guide \
  --output-dir ../external/reliability_audit_guide \
  --gene-set CEGv2=../external/genesets/cegv2.symbols.txt \
  --gene-set MitoCarta3=../external/genesets/mitocarta3.symbols.txt \
  --n-random 200
```

Use the cell-level preparation and a separate output directory to assess
that sampling level. Each gene set is compared with size-matched random sets.

Leave-one-dye-out assessment:

```bash
python studies/external/periscope/analyse_stain_dropout.py \
  --prepared ../external/periscope_guide \
  --output-dir ../external/periscope_stain_dropout --folds 5
```

The last command writes a directory per dye plus combined evidence and tier
tables. Target-dye and cross-channel features are excluded from each input.

</details>

## Outputs and inspection

Each preparation writes `canonical_manifest.json`, `provenance.json`, the four
canonical metadata tables (`observations`, `reporters`, `assays`, `pairing`), and
input/target payload tables. Inspect the declared contract with:

```bash
measurement-sufficiency validate-dataset \
  ../external/oasis_canonical/canonical_manifest.json
```

Validation checks the canonical metadata and reports declared capabilities.
The analysis scripts handle dataset-specific payload loading, including the wide
target tables in PERISCOPE.

| Analysis output | Contents |
| --- | --- |
| `evidence.csv` | Recovery, magnitude rank, variance ratio, top-5% recall and reliability |
| `tiers.csv` | Evidence with the published rule's decision and record count |
| `reliability.csv` | Target-reliability estimates and available diagnostics |
| OASIS: `split.csv`, `predictions.csv` | Frozen fold assignments and long held-out predictions |
| OASIS: `replicate_pairs.csv`, `fidelity_*.csv` | Reliability pairs and control-relative response summaries |
| PERISCOPE: `penalty_selection.csv`, `recoverability_by_fold.csv` | Model-selection and fold-level recovery summaries |
| PERISCOPE: `magnitudes.csv`, `magnitude_spearman.csv`, `variance_ratio.csv`, `strong_hit_recall.csv` | Response-fidelity quantities |

PERISCOPE also exports within-screen and pooled-screen reliability tables.
Guide analysis writes `responses.parquet`; cell analysis writes it only with
`--write-responses`. OASIS and guide-level analysis also write run provenance.

## Interpreting and reusing the assessment

The external applications retain the manuscript's reference operating points.
These are shared settings for this comparison, rather than universal thresholds.
Target reliability is assessed first: a value below 0.30 or an unavailable estimate
gives `not_identifiable`. With sufficient reliability, prediction fidelity can
support a quantitative or ranking proxy, require measurement, or leave the case
`unresolved`. Each decision refers to the evaluated target, task and context.

OASIS pairs wells; its compound-level responses and reliability are dose-averaged
because concentration is not retained in the perturbation key. PERISCOPE evaluates
fluorescence-to-fluorescence prediction using segmentation informed by images
that include the target channel. Feature preparation selects measurements
attributable to the declared input channels. Segmentation and optical cross-talk
are considered separately in the accompanying analyses.

Replicate wells, guide partitions and cell halves capture different sources of
variation. For a new application, specify the intended use, input measurements,
holdout and replicate units, and acceptance criteria. The shared analysis
functions and evidence-table interface then support the same assessment workflow.

The CellProfiler-based OASIS features also use fluorescence-derived segmentation
masks. The separate metabolic-loss example uses released brightfield-channel
DINO features and dose-matched aggregation, as described in
[Supplementary Data 3](../../paper/supplementary_data/SupplementaryData3/README.md).
Dataset-selection criteria and retained resource assessments are in
Supplementary Table 1 and the [resource audit table](../../configs/ops/external/admission_audit.csv).

Use `--dry-run` on preparation commands to inspect source selection; small
`--max-plates` or `--max-cells` runs check preparation, not a complete assessment.
