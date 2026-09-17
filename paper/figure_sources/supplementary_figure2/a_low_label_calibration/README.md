# Supplementary Figure 2a

Rebuild both publication layouts with:

```bash
export MPLCONFIGDIR="${TMPDIR:-.}/mpl_fig6d"
python build_panel_d.py
```

`draw_panel_d(container, add_letter=False, compact=False)` draws the 183 × 78 mm full-width grammar into a Matplotlib `SubFigure`. Set `compact=True` for the native 105 × 65 mm asymmetric-composite grammar. The function never imports a pre-rendered panel image.

The script reads only the local frozen tables, writes the two derived source tables, and exports PNG, PDF and SVG for both layouts. `upstream_source_provenance.csv` records the hashes of the original deposited tables from which these local copies were derived.
