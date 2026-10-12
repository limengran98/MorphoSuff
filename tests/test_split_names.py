"""One canonical spelling for each split.

Before this, the strict whole-screen split had four spellings across the tree:
``strict_screen`` in configs, ``strict_whole_screen`` in the OPS runners,
``whole-screen`` in the portable CLI, and ``whole_screen`` written into the
manifest by ``make_whole_screen_splits``. A table produced under one spelling
matched no consumer expecting another, silently.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import pytest

from measurement_sufficiency.splits import (
    SPLIT_NAMES,
    canonical_split_name,
    make_whole_screen_splits,
)


ROOT = Path(__file__).resolve().parents[1]


def _observations(n: int = 30) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "observation_id": [f"o{index}" for index in range(n)],
            "screen_id": [f"s{index % 5}" for index in range(n)],
        }
    )


@pytest.mark.parametrize(
    "spelling", ["whole-screen", "whole_screen", "strict_screen", "strict-screen", "strict_whole_screen"]
)
def test_every_historical_spelling_resolves(spelling: str) -> None:
    assert canonical_split_name(spelling) == "strict_whole_screen"


def test_unknown_split_name_is_rejected_rather_than_passed_through() -> None:
    with pytest.raises(ValueError, match="unknown split name"):
        canonical_split_name("screen")


def test_manifest_records_the_canonical_name() -> None:
    manifest = make_whole_screen_splits(_observations(), n_folds=5, seed=0)
    assert manifest.split_name == "strict_whole_screen"
    assert manifest.split_name in SPLIT_NAMES


@pytest.mark.parametrize(
    "relative",
    [
        "configs/ops/splits/strict_screen.yaml",
        "configs/ops/protocols/full_label.yaml",
        "configs/ops/figures/figure2.yaml",
        "configs/ops/figures/figure3.yaml",
    ],
)
def test_config_split_literals_are_canonical(relative: str) -> None:
    text = (ROOT / relative).read_text(encoding="utf-8")
    # An alias may still be listed under an explicit `aliases:` key; anywhere else
    # it would name a split that the producers no longer write.
    body = re.sub(r"^aliases:.*$", "", text, flags=re.M)
    for alias in ("strict_screen", "whole-screen"):
        assert not re.search(rf"(?<!_){re.escape(alias)}", body), (
            f"{relative} still names the split {alias!r} outside an aliases: entry"
        )
