# OPS study layer

This directory contains the OPS/A549 study contract and executable study
workflows. The reusable package owns general computation; this layer maps the
public OPS resource to exact caches, canonical tables, model runners and
publication analyses.

## Scope

OPS is a sparse paired-measurement atlas: phase-morphology inputs are linked to
reporter-specific targeted phenotype blocks at exact-cell resolution.  The
study inventory is **10 biological systems, 52 reporters, 99 reporter-screen
assays, and 73 physical screens**.  A reporter-screen assay is an observed
edge, not a dense 52-target measurement at every screen.

Start with `data_sources.yaml`. Inventory and download the unsigned public
processed H5ADs with `acquisition/public_s3.py`, then run
`preparation/prepare_public_ops.py`. The preparation command establishes exact
pairing, reporter-specific endpoint masks, 172D inputs and all frozen split
assignments before training.

The publication consensus is the element-wise unweighted median of the ten
publicly named registry methods.  Methods are sensitivity axes, not biological
replicates; uncertainty is resampled at the declared biological unit.

## Public boundary

Committed artifacts may include small schemas, source locators, SHA-256
digests, sampling/split manifests, and software provenance.  Keep the
following outside version control: full raw microscopy, H5/H5AD stores,
embedding and training matrices, complete prediction archives and model
checkpoints. Compact manuscript results and figure inputs are included under
`paper/`. All paths supplied at execution time are relative to the invocation
directory or user configuration; none is stored here.

See `preparation/README.md`, `../../docs/ops_data_and_preparation.md` and
`../../docs/ops_protocols.md` for the operational contract.
