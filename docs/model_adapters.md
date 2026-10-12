# Model adapter boundary

The frozen ten-method roster contains three implementation classes.

## Portable sparse-predictor adapters

- **Ridge and GBDT** use scikit-learn specialists and observed rows only.
- The portable **MLP** reference uses scikit-learn. It is useful for adapting
  new datasets but is not the frozen OPS PyTorch MLP identity.
- **CatBoost** uses one official `CatBoostRegressor` per endpoint with file
  writing disabled; install `measurement-sufficiency[catboost]`.
- **TabM** uses one unmodified official specialist per reporter. It loads only
  from an exact declared wheel version or an explicitly supplied source file
  whose SHA-256 matches the declaration. The frozen source is
  `yandex-research/tabm@28e47ae301c92ec37787dde1ce923a0793f405b4`,
  version `0.0.3`, `tabm.py` SHA-256
  `fc654af6a16bac53d893a8265c79d7af4ebddcb95ad0d600cc6b6bc6b7317ade`
  (Apache-2.0).

The generic portable ResMLP, MultiTab and TabM adapters accept an `endpoint_groups`
mapping from reporter ID
to its endpoint IDs. This preserves the heterogeneous reporter blocks and
ensures that only observed endpoints contribute to the loss. Training uses
deterministic cyclic sampling with equal reporter exposure; loss is averaged
first over available cells for each endpoint, then equally over endpoints and
active reporters. Reporter abundance therefore does not define the objective.
The local ResMLP and MultiTab implementations in this factory are reusable
architecture adapters only. They do not have checkpoint-layout or forward
parity with the frozen OPS classes and must not be used to regenerate an OPS
benchmark result under the same method identity.

## OPS-native frozen identities

`create_ops_model()` exposes the study's PyTorch MLP, ResMLP and MultiTab
architectures. `ops_training.py` supplies explicit reporter-aware train and
validation batches, deterministic reporter-balanced sampling, endpoint- and
reporter-balanced observed-only loss, AdamW fitting, validation-best restore,
RNG capture and strict checkpoint restore. MultiTab attention is confined to
feature columns within each cell; cells never attend to one another.

These OPS-native entry points are separate from the generic
`SparsePredictor` factory because the frozen protocol must be given declared
validation data and reporter-specific ragged endpoint blocks. It must not
silently create a validation subset from a flat sparse matrix.

The study-native architecture provenance is retained by source checksum:

- candidate architecture source SHA-256:
  `eeb80b7838ade5ab111b0aa41827b8b72d270910519e67d1992d5c7bdbdeda3d`;
- masked multi-task ResMLP source SHA-256:
  `edde55c07d4266aaa6ced494fddabda3cce2206d3dcf1013485979b61f87b0e1`.

State and forward parity tests compare all three architectures against the
checksummed, repository-owned frozen study sources. The package does not
bundle checkpoints or claim identical numerical results without the same
data, preprocessing, split, optimization configuration and training state.
The study-authored implementations are distributed under the repository's MIT
License; `docs/ops_model_code_provenance.md` records their audit anchors.

## Pinned biological-model runners

scButterfly, MIDAS and scPair use explicit study runner identities rather than
similarly named portable substitutes. Recovered upstream commits and
study-adapter hashes are pinned in `studies/ops/model_sources.yaml`; executable
contract-check, smoke and full-fold commands are in
`docs/external_model_runners.md`. These methods are launched through
`studies/ops/runners/run_external_fold.py`, outside the generic
`create_model()` factory, and export the same canonical prediction table as
the built-in estimators.

Model names are estimator identities, not biological replicates. Consensus is
computed only after predictions align on the same held-out rows and endpoints.
