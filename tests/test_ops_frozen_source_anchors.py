"""Provenance anchors for the frozen study sources."""

from __future__ import annotations

from pathlib import Path

import pytest

from measurement_sufficiency.ops_provenance import (
    FROZEN_STUDY_SOURCES,
    frozen_source_digest,
)


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("relative", sorted(FROZEN_STUDY_SOURCES))
def test_frozen_study_source_matches_its_declared_digest(relative: str) -> None:
    path = ROOT / relative
    assert path.is_file(), f"repository-owned frozen study source is missing: {relative}"
    assert frozen_source_digest(path) == FROZEN_STUDY_SOURCES[relative]
