# OPS-native model code provenance

The package-facing `ops_models.py` and `ops_training.py` modules implement the
study-authored OPS models and training protocol. Their source identities are:

| Source identity | SHA-256 |
| --- | --- |
| `ops_reporter_candidate_models.py` | `eeb80b7838ade5ab111b0aa41827b8b72d270910519e67d1992d5c7bdbdeda3d` |
| `ops_reporter_masked_multitask_v2_lib.py` | `edde55c07d4266aaa6ced494fddabda3cce2206d3dcf1013485979b61f87b0e1` |

The implementation covers independent GELU/LayerNorm MLP specialists, shared
pre-normalized ResMLP, within-cell column-only MultiTab, deterministic balanced
reporter sampling, observed-only balanced loss, and training-state/RNG restore.
It deliberately excludes MMoE and endpoint-query models.

The OPS-native MLP, ResMLP and MultiTab implementations are study-authored
code distributed under this repository's MIT License. The SHA-256 anchors
record the study identities and allow the package implementations to
be checked against the repository-owned frozen source copies. No additional
runtime or release authorization is required.
