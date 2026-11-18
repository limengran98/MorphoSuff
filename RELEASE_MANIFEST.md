# Public release contents

This repository is a compact, current release of MorphoSuff. It contains one
reusable Python package, the analysis workflows supporting the paper, the
v3 manuscript and Supplementary Information PDFs, and six main and nine
Supplementary figure PDFs.

| Component | Included |
| --- | --- |
| Reusable software | `src/measurement_sufficiency/` |
| Dataset workflows | `studies/` and `configs/` |
| Usage documentation | `README.md` and `docs/` |
| Numerical reproduction | `reproducibility/` |
| Manuscript PDFs | `paper/manuscript.pdf` and `paper/supplementary.pdf` |
| Figure PDFs | `paper/figures/` |
| Project website | `website/` (static HTML, styles, scripts and figure previews) |
| Processed OPS training data and selected checkpoints | [Hugging Face companion](https://huggingface.co/datasets/Amanda1998/MorphoSuff) · [inventory](release/huggingface/README.md) |
| Supplementary Data | Separate manuscript-submission attachment; not part of the Hub data/model deposit |

The `paper/` directory contains only final PDFs. LaTeX, plotting source files,
figure source data and submission materials remain outside this public checkout.

Study-owned code is MIT licensed. Dataset attribution and third-party software
terms are documented in `THIRD_PARTY_NOTICES.md`.
