# OPS model components

This directory stores the study-owned architecture and optimizer components
used by the executable biological-model runners:

- `scbutterfly/`: exact-pair reporter specialist and staged optimizer;
- `midas/`: phase-plus-sparse-reporter multimodal model and training bridge.

The complete public data/split/training/validation/prediction workflow is in
`studies/ops/runners/run_external_fold.py`.  Upstream repository identities,
revisions, licences and source digests are recorded in
`studies/ops/model_sources.yaml`.

