# OPS low-label implementation

`scripts/` contains the local modules used by the OPS low-label sampling
builder and the classical, neural, biological, sciPENN and scPair entry points.
They implement reporter-aware sampling, held-out partitions, model fitting,
checkpoint selection and evaluation.

Use [run.py](../run.py) through the [low-label workflow](../README.md).
The facade supplies data paths, target tables, interpreter and module paths.
The pinned TabM backend is provided separately at its declared version.
Third-party upstream sources are obtained from their own distributions.
