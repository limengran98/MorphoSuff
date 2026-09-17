"""Source-identity anchors for the frozen OPS study implementations.

This module is deliberately dependency-free: it imports nothing beyond the
standard library.  The SHA-256 anchors below identify the historical
study-authored files that :mod:`measurement_sufficiency.ops_models` migrates,
and verifying them is a provenance check, not a modelling operation.  Keeping
them out of ``ops_models`` means the check runs in every environment rather
than only where PyTorch happens to be installed.

See ``docs/ops_model_code_provenance.md`` for the redistribution boundary.
"""

from __future__ import annotations

import hashlib
from pathlib import Path


CANDIDATE_MODEL_SOURCE_SHA256 = (
    "eeb80b7838ade5ab111b0aa41827b8b72d270910519e67d1992d5c7bdbdeda3d"
)
MASKED_MULTITASK_SOURCE_SHA256 = (
    "edde55c07d4266aaa6ced494fddabda3cce2206d3dcf1013485979b61f87b0e1"
)
DEFAULT_PHASE_INPUT_DIM = 172

#: Repository-relative paths of the frozen study sources, mapped to the digest
#: each one must have.  ``reproducibility/ops/validate_public_layer.py`` keeps a
#: parallel entry for the first file; ``tests/test_ops_frozen_source_anchors.py``
#: asserts the two copies agree so they cannot drift apart.
FROZEN_STUDY_SOURCES: dict[str, str] = {
    "studies/ops/runners/low_label/legacy_exact/scripts/ops_reporter_candidate_models.py": (
        CANDIDATE_MODEL_SOURCE_SHA256
    ),
    "studies/ops/runners/low_label/legacy_exact/scripts/ops_reporter_masked_multitask_v2_lib.py": (
        MASKED_MULTITASK_SOURCE_SHA256
    ),
}


def frozen_source_digest(path: str | Path) -> str:
    """Return the SHA-256 hex digest of a frozen study source file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


__all__ = [
    "CANDIDATE_MODEL_SOURCE_SHA256",
    "DEFAULT_PHASE_INPUT_DIM",
    "FROZEN_STUDY_SOURCES",
    "MASKED_MULTITASK_SOURCE_SHA256",
    "frozen_source_digest",
]
