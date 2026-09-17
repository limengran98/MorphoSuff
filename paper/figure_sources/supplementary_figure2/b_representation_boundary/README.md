# Supplementary Figure 2b

Rebuild both publication layouts with:

```bash
export MPLCONFIGDIR="${TMPDIR:-.}/mpl_fig6e"
python build_panel_e.py
```

`draw_panel_e(container, add_letter=False, compact=False)` draws the 183 × 76 mm full-width grammar into a Matplotlib `SubFigure`. Set `compact=True` for the native 73 × 105 mm vertical grammar. Both modes load the registered phase arrays directly and render the quantitative panels as vectors.

The script reads only the local frozen tables and image arrays, writes three derived source tables, and exports PNG, PDF and SVG for both layouts. `upstream_source_provenance.csv` records the hashes of the original deposited inputs from which these local copies were derived.
