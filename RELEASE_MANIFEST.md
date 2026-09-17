# Public release contents

This repository is a compact, current release of MorphoSuff. It contains one
reusable Python package, the analysis workflows supporting the paper, the
submitted manuscript PDFs and the final source data and plotting code for six
main and four Supplementary figures.

| Component | Included |
| --- | --- |
| Reusable software | `src/measurement_sufficiency/` |
| Dataset workflows | `studies/` and `configs/` |
| Usage documentation | `README.md` and `docs/` |
| Numerical reproduction | `reproducibility/` |
| Manuscript PDFs | `paper/manuscript.pdf` and `paper/supplementary.pdf` |
| Figure code and source tables | `paper/figure_sources/` |
| Supplementary Data 1–3 | `paper/supplementary_data/` |
| Large processed study data | External companion dataset linked in the README |
| Registered binary figure inputs | Tagged GitHub release asset |

Not included are manuscript LaTeX sources, submission materials, editor-facing forms,
internal reports, build logs, caches, historical figure versions or duplicated
rendered figures. This keeps the Git checkout below 50 MiB while preserving the
scientific source layer.

Study-owned code is MIT licensed. Dataset attribution and third-party software
terms are documented in `THIRD_PARTY_NOTICES.md`.
