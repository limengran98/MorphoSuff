# Third-party notices

This repository does not vendor third-party model source code or model
parameters. Optional adapters import separately installed packages or invoke
separately pinned external runners. Those components retain their original
licences and citation requirements.

## OASIS profiles and metadata

Supplementary Data 3 contains derived measurement tables and predictions based on
Jessica Ewald's [Axiom OASIS profiles and metadata](https://doi.org/10.5281/zenodo.17067683),
released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
The source files are `dino_raw.parquet` and `metadata.parquet`; the publication
bundle records their provenance and the derived-input hashes. Brightfield-channel
selection, control-relative prediction, aggregation and prioritization are study
analyses, not unchanged upstream results. Cite both the dataset and
[Ewald et al., Cell Systems (2026)](https://doi.org/10.1016/j.cels.2026.101566).
The derived source-data tables retain CC BY 4.0 attribution; study-authored
scoring and figure code is covered by this repository's MIT License.

## TabM

- Project: `yandex-research/tabm`
- Frozen revision: `28e47ae301c92ec37787dde1ce923a0793f405b4`
- Frozen package version: `0.0.3`
- Frozen `tabm.py` SHA-256:
  `fc654af6a16bac53d893a8265c79d7af4ebddcb95ad0d600cc6b6bc6b7317ade`
- Licence: Apache License 2.0
- Copyright: Yandex LLC

The public adapter loads TabM only after checking the declared wheel version
or an explicitly supplied source-file digest. TabM source is not copied into
this repository. The `tabm` optional extra installs only the official 0.0.3
runtime requirements (`torch>=1.12,<3`,
`rtdl_num_embeddings>=0.0.12,<0.1`, and
`typing_extensions>=4.6.0,<5`); it does not bypass the adapter's explicit
wheel-version or source-digest declaration.

## CatBoost

- Project: CatBoost
- Licence: Apache License 2.0

The optional adapter imports `CatBoostRegressor` from a separately installed
CatBoost distribution and disables training-file output.

## scikit-learn

- Project: scikit-learn
- Licence: BSD 3-Clause

The built-in Ridge and gradient-boosted-tree reference estimators use
scikit-learn. Any generic scikit-learn multilayer-perceptron adapter is not a
numerical substitute for the study-frozen PyTorch MLP unless a protocol
explicitly declares it as such.

## External study runners

The following upstream identities were recovered; no third-party code is
copied into this repository:

- scButterfly: `BioX-NKU/scButterfly` at
  `eb31e04bb8c4abdf85c4cbecd044fcd359105caa`, version `0.0.9`, MIT; companion
  source repository at `36941cb97f9de21705ce3ec12d3294d4519df82e`.
- MIDAS: `labomics/midas` reproducibility branch at
  `3ef7847c88c90583c05147cafef496d862986dd3`. No licence file was present at
  that pinned commit, so the upstream checkout is recorded as `NOASSERTION`.
  The OPS runner uses separately authored adapter code and does not copy the
  upstream repository. Users obtaining upstream MIDAS source should consult
  the repository owner for the applicable terms.
- scPair: `quon-titative-biology/scPair` at
  `c585949ca8ea1314f5e68b260e3d9c5b2dabe61c`, version `0.1.0`, MIT.

The exact-byte, study-authored scButterfly and MIDAS OPS model/training
components are stored under `studies/ops/external_runners/`; they contain no
third-party source. The common execution path and scPair source bridge are in
`studies/ops/runners/`. They enforce preprocessing, frozen split and endpoint
contracts and produce canonical observation-level prediction exports. The
scPair bridge imports only a caller-supplied checkout after verifying the
pinned public `model.py` digest.
