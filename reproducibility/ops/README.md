# OPS reproducibility layer

This directory maps each scientific stage to an executable public entry point,
its frozen contract and its canonical outputs. Machine-specific scheduler
configuration is not part of the protocol.

1. Inventory and resumably download the processed public H5ADs with
   `studies/ops/acquisition/public_s3.py`.
2. Run `studies/ops/preparation/prepare_public_ops.py` to generate exact
   same-cell caches, the frozen 172D representation, split assignments and
   canonical reporter bundles.
3. Materialize common field, gene or strict whole-screen folds with
   `studies/ops/runners/full_label/`.
4. Execute the full-label, low-label, same-cell or raw-image protocol through
   the corresponding included runner under `studies/ops/runners/`.
5. Export the common long prediction contract and derive metrics from aligned
   held-out rows.
6. Rebuild the publication figures from their
   [source packages](../../paper/figure_sources/README.md).

Upstream method source and weights are resolved from the pinned public
locators and verified at runtime. Observations, images, weights, predictions
and fitted checkpoints are downloaded or generated in caller-selected paths;
they are normal runtime artifacts rather than source-code dependencies.

`source_map.tsv` records the entry point, frozen contract, canonical inputs and
outputs, statistical unit and execution status for every stage.
