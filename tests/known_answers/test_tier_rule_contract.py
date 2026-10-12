"""Known-answer checks for the OPS tier implementation and protocol configuration."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import pytest

from measurement_sufficiency.analysis.tiers import (
    OPS_TIER_RULES,
    OPSTier,
    OPSTierRules,
    tier_table,
)
from measurement_sufficiency import sufficiency


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "configs" / "ops" / "protocols" / "measurement_tiers.yaml"

#: Transcribed by hand from the manuscript Methods paragraph that states the rule:
#: "A quantitative proxy required held-out-gene recoverability >=0.70, magnitude
#: Spearman >=0.70, variance ratio from 0.50 to 1.50, top-5% recall >=0.60 and
#: within-screen reliability >=0.30. A ranking proxy required magnitude Spearman
#: >=0.70 and top-5% recall >=0.50 ... Measurement required applied when
#: reliability was >=0.30 and recoverability was <0.60."
METHODS_THRESHOLDS = {
    "recoverability_min": 0.70,
    "magnitude_spearman_min": 0.70,
    "variance_ratio_min": 0.50,
    "variance_ratio_max": 1.50,
    "top5pct_quantitative_min": 0.60,
    "top5pct_ranking_min": 0.50,
    "reliability_min": 0.30,
    "measurement_recoverability_max_exclusive": 0.60,
}


def _protocol_numbers() -> dict[str, float]:
    """Read the YAML mirror without adding a parser dependency to the package."""
    text = PROTOCOL.read_text(encoding="utf-8")
    section = {}
    current = None
    for line in text.splitlines():
        if line and not line[0].isspace() and line.rstrip().endswith(":"):
            current = line.split(":", 1)[0].strip()
            section[current] = {}
        elif current and ":" in line and line.startswith(" ") and not line.strip().startswith("#"):
            key, _, value = line.strip().partition(":")
            section[current][key.strip()] = value.strip()
    quantitative = section["quantitative_proxy"]
    ranking = section["ranking_proxy"]
    required = section["measurement_required"]
    low, high = re.findall(r"-?\d+\.?\d*", quantitative["variance_ratio"])
    return {
        "recoverability_min": float(quantitative["recoverability_min"]),
        "magnitude_spearman_min": float(quantitative["magnitude_spearman_min"]),
        "variance_ratio_min": float(low),
        "variance_ratio_max": float(high),
        "top5pct_quantitative_min": float(quantitative["top5pct_recall_min"]),
        "top5pct_ranking_min": float(ranking["top5pct_recall_min"]),
        "reliability_min": float(quantitative["reliability_min"]),
        "measurement_recoverability_max_exclusive": float(required["recoverability_max_exclusive"]),
    }


@pytest.mark.parametrize("field", sorted(METHODS_THRESHOLDS))
def test_dataclass_matches_the_published_methods(field: str) -> None:
    assert getattr(OPS_TIER_RULES, field) == METHODS_THRESHOLDS[field]


@pytest.mark.parametrize("field", sorted(METHODS_THRESHOLDS))
def test_protocol_yaml_mirrors_the_dataclass(field: str) -> None:
    assert _protocol_numbers()[field] == getattr(OPS_TIER_RULES, field)


def test_protocol_records_the_reliability_missing_branch() -> None:
    """The published rule tiers missing reliability as not identifiable."""
    text = PROTOCOL.read_text(encoding="utf-8")
    assert "reliability_missing" in text
    from measurement_sufficiency.analysis.tiers import assign_ops_tier

    evidence = {
        "recoverability_r": 0.9,
        "magnitude_spearman": 0.9,
        "variance_ratio": 1.0,
        "top5pct_recall": 0.9,
        "reliability": float("nan"),
    }
    assert assign_ops_tier(evidence) is OPSTier.NOT_IDENTIFIABLE


def test_portable_thresholds_carry_no_default_calibration() -> None:
    """sufficiency.TierThresholds must not read as a second OPS calibration."""
    with pytest.raises(TypeError):
        sufficiency.TierThresholds()
    assert set(sufficiency.REQUIRED_EVIDENCE) != set(
        ("recoverability_r", "magnitude_spearman", "variance_ratio", "top5pct_recall", "reliability")
    ), "the portable rule and the OPS rule are different rules over different evidence"


def test_tier_table_reduces_replicates_to_one_row_per_reporter() -> None:
    rows = []
    for reporter, recoverability, spearman in (("r1", 0.9, 0.9), ("r2", 0.2, 0.3)):
        for model in range(10):
            rows.append(
                {
                    "reporter_id": reporter,
                    "model_id": f"m{model}",
                    "recoverability_r": recoverability,
                    "magnitude_spearman": spearman,
                    "variance_ratio": 1.0,
                    "top5pct_recall": 0.9,
                    "reliability": 0.9,
                }
            )
    table = tier_table(pd.DataFrame(rows), group_columns=["reporter_id"])
    assert len(table) == 2
    assert set(table.n_records) == {10}
    assert dict(zip(table.reporter_id, table.tier)) == {
        "r1": OPSTier.QUANTITATIVE_PROXY.value,
        "r2": OPSTier.MEASUREMENT_REQUIRED.value,
    }


def test_all_missing_reliability_survives_aggregation_as_not_identifiable() -> None:
    rows = [
        {
            "reporter_id": "r1",
            "model_id": f"m{index}",
            "recoverability_r": 0.9,
            "magnitude_spearman": 0.9,
            "variance_ratio": 1.0,
            "top5pct_recall": 0.9,
            "reliability": float("nan"),
        }
        for index in range(10)
    ]
    table = tier_table(pd.DataFrame(rows), group_columns=["reporter_id"])
    assert table.tier.iloc[0] == OPSTier.NOT_IDENTIFIABLE.value


def test_ranking_beats_measurement_required_when_a_reporter_satisfies_both() -> None:
    """PSMB7's own evidence, which is the only row in the atlas where the order shows."""
    from measurement_sufficiency.analysis.tiers import assign_ops_tier

    evidence = {
        "recoverability_r": 0.596075,
        "magnitude_spearman": 0.798745,
        "variance_ratio": 0.798561,
        "top5pct_recall": 0.8,
        "reliability": 0.430941,
    }
    assert assign_ops_tier(evidence) is OPSTier.RANKING_PROXY


def test_the_protocol_records_the_precedence() -> None:
    text = PROTOCOL.read_text(encoding="utf-8")
    assert "precedence:" in text, "the YAML must record the rule order, not only the thresholds"
    order = text.split("precedence:", 1)[1]
    assert order.index("ranking_proxy") < order.index("measurement_required")
