### Figure 1c methods

The panel was generated from the frozen OPS assay-edge, reporter-node and screen-node tables. A reporter-screen assay was included when a unique `reporter_slug::screen_id` key was present in the assay table after coordinate matching, OrganelleProfiler matching and gene-guide concordance filtering. Marker area was mapped linearly to `gene_guide_concordant_exact_links`; no log transform, normalization, imputation, statistical test or uncertainty interval was applied.

Reporter rows were ordered without using prediction outcomes: first by the manuscript's predeclared biological-system order, then by decreasing number of available screens, and finally by stable reporter display name. Screens were partitioned into single-reporter and multi-reporter experiments. Within each partition, screens were ordered by the mean row position of their assayed reporters, then by occupancy and physical screen number. The ordering is therefore descriptive of the acquisition topology and does not depend on recoverability.

The screen-occupancy track is aligned exactly to the matrix columns. Each unit-height stacked segment represents one assayed reporter and is coloured by its biological system; stack height therefore remains the untransformed reporter count. The two group headings share a common baseline. OPS0043 and OPS0077 labels occupy a structurally empty region at the top of the multi-reporter matrix, with cross-axes connectors terminating at the side of their seven- and six-reporter stacks rather than at individual assay glyphs.

All 52 reporters are retained as matrix rows. To preserve legibility at the final 183-mm width, 26 row names are displayed using an outcome-independent rule fixed before rendering: reporters recurring as examples elsewhere in the manuscript, reporters observed in at least three screens, one high-coverage representative of each biological system and alternating remaining rows in the frozen order. The long ChromaLIVE 488 name is suppressed because TOMM20 already labels the mitochondrial block; its assay row remains fully plotted. The complete selection and its reason are stored in `reporter_label_manifest.tsv`.

Two registered same-cell examples were embedded in structurally empty regions of the single-reporter matrix: LAMP1 under BORCS7 knockout and FeRhoNox under ALG12 knockout. Phase and fluorescence arrays were loaded from the frozen 384-by-384 crops, shown with their recorded per-image display windows and not spatially altered. Both use a pixel size of 0.325 micrometres and a 20-micrometre scale bar. The crops define what a reporter-screen assay contains and are connected to their corresponding assay glyphs; they do not constitute evidence of cross-screen reproducibility.

The microscopy channels were rendered with identical horizontal and vertical physical dimensions so that the original square crop geometry was preserved. Centred titles and channel names occupy a separate header band, and scale-bar labels are inset from the image boundary. Programmatic final-canvas checks require each title to remain inside its card and each scale label to remain inside its fluorescence crop. Card positions were tested against the complete assay-edge table; neither card covers a plotted reporter-screen assay. LAMP1, FeRhoNox and pRb are highlighted according to the same predeclared manuscript-example set used in panel b. The assay-level coverage spectrum and its ordering are reported in panel b, leaving panel c dedicated to acquisition topology and registered examples.

The 9,996,286 plotted exact-linked observations are assay-level reporter observations. In multi-reporter screens, a phase cell can contribute to more than one reporter assay; these values must not be summed within a screen and interpreted as unique cells. Blank positions are structurally unassayed combinations.

The final native canvas is 183 × 85 mm, with a 6-pt minimum live-text size.
The same 26 selected reporter labels are separated deterministically by at
least 2.05 row units and retain leaders to their unchanged source rows.
System-label, recurrence-label and microscopy-header corridors are allocated
independently. Every visible text artist is checked for pairwise overlap and
canvas clipping; the recurrence header is additionally restricted to its own
track so that it cannot cover the neighbouring occupancy bars. The final PDF
and 600-dpi preview are visually reviewed for text–mark and leader-line clashes.
