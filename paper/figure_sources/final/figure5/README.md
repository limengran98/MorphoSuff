# Figure 5 | Distinct constraints limit morphology-based measurement sufficiency

This directory is a self-contained, plotting-only manuscript-figure bundle. It
uses the frozen ten-model median consensus and does not retrain a predictor or
re-estimate a scientific result.

## Evidence structure

- **a** links reporter recoverability to overlapping measurement-stability and
  coverage constraints, and shows the prespecified joint-model permutation
  null.
- **b** resolves two low-stability reporters into well, guide and random-cell
  components and anchors the pRb example in registered same-guide microscopy.
- **c** separates KO-response ranking from response-amplitude fidelity across
  all 52 reporters.
- **d** orders all 81 directed whole-screen transfers by performance loss,
  aligns the two NTC-shift tracks and maps the frozen FeRhoNox failure case to
  registered screen-specific images.
- **e** displays all 5,990 gene-level conditional-ambiguity ratios for the
  engineered 172-dimensional morphology representation and the raw-thumbnail
  sensitivity representation.

## Rebuild

```powershell
./build.ps1
```

The adopted composite is `composite/Figure5.{pdf,svg,png}`. PDF and SVG retain live editable text;
the PNG is exported at 600 dpi. The machine-readable QA
report is in `qa/`.

Scientific inputs are contained in this `figure5` directory; rendering also uses
`../../publication_style.py`. The composite is 183 × 185 mm, with 9 pt bold
DejaVu Sans composite-only letters and a 6 pt text floor. All five standalone
panels are unlabelled. Explanatory panel titles are
removed. Microscopy channel labels are outside the images, and statistical
units and interval definitions are retained in the manuscript caption. Generated
composites and verification reports remain local; the publication PDF is
tracked once at `paper/figures/Figure5.pdf`.
