# External biological model execution

The public OPS layer includes executable study-owned training paths for
scButterfly, MIDAS and scPair in `studies/ops/runners`.  These paths consume the
same canonical input, target and frozen assignment tables as the built-in
models and write the same observation-level prediction schema.

The distinction between *upstream source* and *OPS execution code* is explicit:

- upstream repositories supply the published model mechanisms;
- the study-owned adapters bind those mechanisms to continuous 172D phase
  inputs and heterogeneous continuous reporter blocks; and
- `run_external_fold.py` supplies data loading, split enforcement, train-only
  preprocessing, optimization, validation checkpoint selection and prediction
  export.

## Common export

Every completed fold writes a validated long table containing:

```text
observation_id,reporter_id,endpoint_id,split_name,fold,model_id,
y_true,y_pred,screen_id,perturbation_id
```

The runner also records the adapter/upstream provenance, input paths, feature
schema, fold roles, preprocessing values, validation trajectory and selected
checkpoint digest.

## scButterfly

Pinned upstream identities:

- `BioX-NKU/scButterfly` at
  `eb31e04bb8c4abdf85c4cbecd044fcd359105caa` (version 0.0.9);
- `BioX-NKU/scButterfly_source` at
  `36941cb97f9de21705ce3ec12d3294d4519df82e`; and
- MIT licence in both pinned checkouts.

The benchmark uses one exact-pair specialist per reporter.  Its two encoders,
two decoders, translator, latent discriminators, modality pretraining and
alternating joint-training controller are included under
`studies/ops/external_runners/scbutterfly`.  No external checkout is needed at
runtime because the OPS adapter is self-contained study code; the upstream
pins document method provenance.

## MIDAS

The model provenance is `labomics/midas`, reproducibility branch, commit
`3ef7847c88c90583c05147cafef496d862986dd3`.  OPS represents phase plus the
sparse reporter blocks as distinct modalities, retaining modality-specific
fronts, product-of-experts fusion, a shared latent decoder and reporter-local
Gaussian reconstruction heads.  Reporter identity is a modality key and is
not used as a technical-batch label.

The executable study adapter is included under
`studies/ops/external_runners/midas`.  The runner does not redistribute the
upstream repository; the upstream commit and source hash are retained for
scientific provenance.

## scPair

Pinned upstream identity: `quon-titative-biology/scPair` at
`c585949ca8ea1314f5e68b260e3d9c5b2dabe61c`, version 0.1.0, MIT.

```bash
git clone https://github.com/quon-titative-biology/scPair.git external/scPair
git -C external/scPair checkout --detach c585949ca8ea1314f5e68b260e3d9c5b2dabe61c
```

`scpair_core.py` verifies the pinned `scpair/model.py` digest and instantiates
the released input, output and cross-modality modules at each reporter's
observed endpoint width.  `run_external_fold.py` then performs the independent
reporter training and validation selection used by this benchmark.

## Reproduction commands

The unified ten-method command, strict whole-screen builder and common fold
materializer are documented in `studies/ops/runners/full_label/README.md`;
biological-runner-specific commands are also provided in
`studies/ops/runners/README.md`.  Paths are supplied by the caller and can
point to any local copy prepared from the public OPS release.  CPU is supported
for compact smoke testing; the reported benchmark configurations are intended
for CUDA execution.
