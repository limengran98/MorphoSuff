import pandas as pd
import pytest

from measurement_sufficiency.schemas import SchemaError, validate_canonical_tables, validate_predictions


def canonical_tables():
    return {
        "observations": pd.DataFrame({"observation_id": ["o1", "o2"], "input_cell_id": ["i1", "i2"], "target_cell_id": ["t1", "t2"], "assay_id": ["a1", "a1"], "perturbation_id": ["p1", "p2"], "guide_id": [None, "g2"], "screen_id": ["s1", "s1"], "well_id": ["w1", "w2"], "field_id": ["f1", "f2"], "is_control": [True, False]}),
        "reporters": pd.DataFrame({"reporter_id": ["r1"], "reporter_name": ["R"], "biological_system": ["cell"], "target_modality": ["fluorescence"], "endpoint_schema_id": ["e1"]}),
        "assays": pd.DataFrame({"assay_id": ["a1"], "reporter_id": ["r1"], "screen_id": ["s1"], "endpoint_ids": ["e1,e2"]}),
        "pairing": pd.DataFrame({"observation_id": ["o1", "o2"], "input_cell_id": ["i1", "i2"], "target_cell_id": ["t1", "t2"], "assay_id": ["a1", "a1"], "pairing_key": ["x1", "x2"], "pairing_confidence": [1.0, 1.0]}),
    }


def test_canonical_tables_validate():
    validate_canonical_tables(canonical_tables())


def test_one_phase_cell_can_pair_once_in_each_of_multiple_assays():
    tables = canonical_tables()
    tables["reporters"] = pd.concat(
        [
            tables["reporters"],
            pd.DataFrame(
                {
                    "reporter_id": ["r2"],
                    "reporter_name": ["R2"],
                    "biological_system": ["cell"],
                    "target_modality": ["fluorescence"],
                    "endpoint_schema_id": ["e2"],
                }
            ),
        ],
        ignore_index=True,
    )
    tables["assays"] = pd.concat(
        [
            tables["assays"],
            pd.DataFrame(
                {
                    "assay_id": ["a2"],
                    "reporter_id": ["r2"],
                    "screen_id": ["s1"],
                    "endpoint_ids": ["e3"],
                }
            ),
        ],
        ignore_index=True,
    )
    tables["observations"] = pd.concat(
        [
            tables["observations"],
            pd.DataFrame(
                {
                    "observation_id": ["o3"],
                    "input_cell_id": ["i1"],
                    "target_cell_id": ["t1"],
                    "assay_id": ["a2"],
                    "perturbation_id": ["p1"],
                    "guide_id": [None],
                    "screen_id": ["s1"],
                    "well_id": ["w1"],
                    "field_id": ["f1"],
                    "is_control": [True],
                }
            ),
        ],
        ignore_index=True,
    )
    tables["pairing"] = pd.concat(
        [
            tables["pairing"],
            pd.DataFrame(
                {
                    "observation_id": ["o3"],
                    "input_cell_id": ["i1"],
                    "target_cell_id": ["t1"],
                    "assay_id": ["a2"],
                    "pairing_key": ["x1"],
                    "pairing_confidence": [1.0],
                }
            ),
        ],
        ignore_index=True,
    )
    validate_canonical_tables(tables)


def test_pairing_remains_one_to_one_within_an_assay():
    tables = canonical_tables()
    tables["observations"].loc[1, "input_cell_id"] = "i1"
    tables["pairing"].loc[1, "input_cell_id"] = "i1"
    with pytest.raises(SchemaError, match="input_cell_id more than once within an assay"):
        validate_canonical_tables(tables)


def test_observation_assay_fk_and_screen_context_are_enforced():
    tables = canonical_tables()
    tables["observations"].loc[0, "assay_id"] = "missing"
    with pytest.raises(SchemaError, match="unknown assay_id"):
        validate_canonical_tables(tables)

    tables = canonical_tables()
    tables["observations"].loc[0, "screen_id"] = "wrong"
    with pytest.raises(SchemaError, match="must match the declared assay"):
        validate_canonical_tables(tables)


def test_pairing_must_match_the_observation_assay_context():
    tables = canonical_tables()
    tables["pairing"].loc[0, "target_cell_id"] = "other"
    with pytest.raises(SchemaError, match="assay-qualified linkage"):
        validate_canonical_tables(tables)


def test_predictions_require_unique_rows():
    row = {"observation_id": "o", "reporter_id": "r", "endpoint_id": "e", "split_name": "field", "fold": 0, "model_id": "ridge", "y_true": 1.0, "y_pred": 1.2, "screen_id": "s", "perturbation_id": "p"}
    with pytest.raises(SchemaError, match="unique"):
        validate_predictions(pd.DataFrame([row, row]))
