import pandas as pd
import pytest

from measurement_sufficiency.consensus import FINAL_TEN_MODELS, consensus_median
from measurement_sufficiency.metrics import cell_metrics, ko_metrics


def predictions(models=("ridge",)):
    rows = []
    for model_i, model in enumerate(models):
        for obs, perturbation, screen, truth in [("a", "p1", "s1", 1.0), ("b", "p1", "s1", 3.0), ("c", "p2", "s2", 2.0), ("d", "p2", "s2", 6.0)]:
            rows.append({"observation_id": obs, "reporter_id": "r", "endpoint_id": "e", "split_name": "gene", "fold": 0, "model_id": model, "y_true": truth, "y_pred": truth + model_i, "screen_id": screen, "perturbation_id": perturbation})
    return pd.DataFrame(rows)


def test_cell_and_ko_metrics_come_from_long_contract():
    pred = predictions()
    assert cell_metrics(pred).n.iloc[0] == 4
    assert ko_metrics(pred).n.iloc[0] == 2


def test_consensus_is_a_median_not_a_mean():
    """Guard: swapping median for mean in consensus.py must turn this red.

    The other fixture uses offsets 0..9, whose mean and median are both 4.5, so
    the mutation is invisible there. One extreme offset separates them.
    """
    rows = []
    offsets = [0.0] * 9 + [100.0]
    for model, offset in zip(FINAL_TEN_MODELS, offsets):
        rows.append(
            {
                "observation_id": "a", "reporter_id": "r", "endpoint_id": "e",
                "split_name": "gene", "fold": 0, "model_id": model,
                "y_true": 1.0, "y_pred": 1.0 + offset,
                "screen_id": "s1", "perturbation_id": "p1",
            }
        )
    result = consensus_median(pd.DataFrame(rows))
    assert result.y_pred.iloc[0] == 1.0, "an unweighted median ignores the single outlier"
    assert result.y_pred.iloc[0] != pytest.approx(11.0), "this is the mean, not the median"


def test_ten_model_median_is_unweighted_and_strict():
    pred = predictions(FINAL_TEN_MODELS)
    result = consensus_median(pred)
    assert set(result.model_id) == {"ten_model_median"}
    assert result.n_models.eq(10).all()
    with pytest.raises(ValueError, match="frozen"):
        consensus_median(pred[pred.model_id != "midas"])
