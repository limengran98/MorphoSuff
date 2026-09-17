# Data and code availability

## Software and manuscript bundle

This repository contains the reusable MorphoSuff package, dataset-specific
workflows and the [manuscript bundle](../paper/README.md), including compact
figure sources and publication assets. Study-owned code is released under the
[MIT License](../LICENSE). See [software citation metadata](../CITATION.cff)
and [code ownership and licensing](../CODE_OWNERSHIP_AND_LICENSE.md).

## Processed study data

The [companion dataset card](../release/huggingface/README.md) provides the
public repository locator, inventory, loading example and provenance. The
[OPS training guide](../release/huggingface/OPS_TRAINING_DATA.md) documents the
Figure 2 download and training workflow.

The processed release comprises the shared 172D phase matrix, frozen
field/gene assignments, 52 exact-linked sparse reporter target blocks,
canonical Figure 1–6 source tables, protocol metadata, composite figures and
two representative fitted states. Raw microscopy, image crops, source H5AD
objects, per-cell image embeddings and the full checkpoint collection remain
outside this processed release.

The [release selection policy](../release/huggingface/release_spec.json)
defines its scope. Release manifests record a SHA-256 digest and byte count
for each selected file; public manifests omit local filesystem paths.

## Upstream resource

The [public OPS source archive](https://doi.org/10.5281/zenodo.20495192)
provides the upstream resource. Public processed single-cell objects are
acquired from the unsigned S3 source recorded in the
[source manifest](../studies/ops/data_sources.yaml); see the
[acquisition and preparation guide](ops_data_and_preparation.md). Upstream
data terms remain applicable.

## Archival DOIs

Repository locators provide access to the code and processed data.
Release-specific archival DOIs will be recorded separately in
[CITATION.cff](../CITATION.cff), this page and the dataset card when available.
The upstream OPS DOI identifies the original resource, not a MorphoSuff
software or processed-data release.
