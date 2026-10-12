import numpy as np
import pandas as pd
import pytest

from measurement_sufficiency.preprocessing import FeaturePreprocessor, control_relative_response


def test_preprocessor_is_fit_on_training_values_only():
    train = pd.DataFrame({"x": [1.0, np.nan, 3.0], "category": ["a", "b", "c"]})
    test = pd.DataFrame({"x": [100.0]})
    processor = FeaturePreprocessor().fit(train, ["x"])
    transformed = processor.transform(test)
    assert transformed.x.iloc[0] > 90  # it did not refit to the test value
    assert processor.transform(train).x.isna().sum() == 0


def test_screen_matched_control_response():
    data = pd.DataFrame({"perturbation_id": ["control", "control", "ko", "ko"], "screen_id": ["s1", "s2", "s1", "s2"], "is_control": [True, True, False, False], "value": [1.0, 10.0, 4.0, 15.0]})
    out = control_relative_response(data, value_columns=["value"])
    ko = out[out.perturbation_id == "ko"].sort_values("screen_id")
    assert ko.value.tolist() == [3.0, 5.0]
    # Three distinct gates. The previous single assertion used match="control",
    # which the missing-column message at the top of the function already
    # satisfies, so the two screen-matching gates below were never exercised.
    with pytest.raises(ValueError, match="lacks columns"):
        control_relative_response(data[data.screen_id == "s1"], value_columns=["missing"])
    with pytest.raises(ValueError, match="requires at least one control"):
        control_relative_response(data.assign(is_control=False), value_columns=["value"])
    unmatched = pd.concat(
        [data[data.is_control & data.screen_id.eq("s1")], data[~data.is_control]]
    )
    with pytest.raises(ValueError, match="missing screen-matched control"):
        control_relative_response(unmatched, value_columns=["value"])


def test_control_flag_is_parsed_strictly_not_by_truthiness():
    """A CSV round trip writes False as the string "False", which is truthy."""
    data = pd.DataFrame(
        {
            "perturbation_id": ["control", "ko"],
            "screen_id": ["s1", "s1"],
            "is_control": ["True", "False"],
            "value": [1.0, 4.0],
        }
    )
    out = control_relative_response(data, value_columns=["value"])
    assert out.loc[out.perturbation_id.eq("ko"), "value"].iloc[0] == 3.0
    with pytest.raises(Exception, match="unrecognized boolean token"):
        control_relative_response(data.assign(is_control=["True", "maybe"]), value_columns=["value"])
