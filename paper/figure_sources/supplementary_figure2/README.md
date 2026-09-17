# Supplementary Figure 2 source package

This source package contains label-efficiency calibration in panel a and the representation-boundary analysis in panel b. The compositor renders both panels natively from the deposited tables and registered image arrays.

Run `./build.ps1 -Python <python.exe>` from this directory. The script rebuilds both standalone panels and the 183-mm composite, exports editable SVG/PDF plus 600-dpi PNG/TIFF, and synchronizes the publication PDF to `paper/figures/SupplementaryFigure2.pdf`.

The source-data manifest and QA record are written to `composite/`. Files named `upstream_source_provenance.csv` preserve hashes of the inputs from which the local self-contained copies were derived; the build itself reads only files in this package.

Git tracks one copy of the code, frozen tables, compact phase-crop arrays,
contracts and provenance. Generated panel/composite PDF, SVG, PNG and TIFF files
are ignored here; the publication PDF is tracked once at
`paper/figures/SupplementaryFigure2.pdf`.
