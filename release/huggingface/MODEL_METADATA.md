# Representative checkpoint metadata

## scPair: NPM1, held-out-gene fold 0, 1% labels

- Destination: `models/scpair_npm1_gene_fold0_1pct/checkpoint_average.pt`
- Method identity: scPair, pinned revision
  `c585949ca8ea1314f5e68b260e3d9c5b2dabe61c`
- Protocol: target-only, gene holdout, fold 0, label fraction 0.01
- Checkpoint selection: validation-state average; the outer test was loaded
  after checkpoint selection and was not used for selection
- SHA-256: `69749e07748f411aaf0fec14c89a31e059f3bf99518ad59a103c253ff5c8f09c`
- Licence: MIT

## Cytoland VSCyto2D: LAMP1, held-out-gene fold 0

- Destination: `models/cytoland_lamp1_gene_fold0/best_trainable_state.pt`
- Protocol: partial supervised fine-tuning of encoder stages 2–3 plus endpoint
  head; validation-best standardized endpoint MSE
- Base checkpoint SHA-256:
  `1abdbc1c727aba33dad0d174dba57d128a12019b712db946ce06012ecd64c1fe`
- Reporter/fold state SHA-256:
  `0c5baaf0723c3db90636198ceb2e1585e1a165a2ee3aff36dc880a93295720fe`
- Reporter: lysosome LAMP1; gene holdout fold 0; 24 endpoints
- Selection: best epoch 6, validation fold 1; the released state contains only
  the fitted trainable components and endpoint normalization metadata
- Base model: <https://virtualcellmodels.cziscience.com/model/cytoland>
- Licence: upstream Cytoland terms apply

These checkpoints are compact reproducibility examples. Canonical numerical
results are defined by the released source tables, not by re-evaluating a
single example checkpoint.
