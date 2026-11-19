"""Input-contract tests: every required failure mode is exercised."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nids.errors import InputValidationError
from nids.validation import read_csv_bytes, validate_flow_frame


def test_valid_frame_is_accepted_and_ordered(valid_many_rows, artifacts):
    outcome = validate_flow_frame(valid_many_rows, artifacts)
    assert list(outcome.frame.columns) == artifacts.feature_columns
    assert len(outcome.frame) == len(valid_many_rows)
    assert outcome.warnings == ()
    for column in artifacts.numerical_columns:
        assert outcome.frame[column].dtype == np.float64


def test_missing_required_columns_are_listed(valid_one_row, artifacts):
    frame = valid_one_row.drop(columns=["duration", "service"])
    with pytest.raises(InputValidationError) as error:
        validate_flow_frame(frame, artifacts)
    assert "duration" in str(error.value)
    assert "service" in str(error.value)
    assert len(error.value.details) == 2


def test_extra_columns_are_ignored_with_warning(valid_one_row, artifacts):
    frame = valid_one_row.copy()
    frame["commentaire"] = "test"
    outcome = validate_flow_frame(frame, artifacts)
    assert "commentaire" not in outcome.frame.columns
    assert any("supplémentaires" in warning for warning in outcome.warnings)


def test_label_columns_are_ignored_with_warning(valid_one_row, artifacts):
    frame = valid_one_row.copy()
    frame["attack_type"] = "neptune"
    frame["attack_category"] = "DoS"
    outcome = validate_flow_frame(frame, artifacts)
    assert any("étiquettes" in warning for warning in outcome.warnings)
    assert "attack_category" not in outcome.frame.columns


def test_missing_values_are_rejected(valid_one_row, artifacts):
    frame = valid_one_row.copy()
    frame.loc[frame.index[0], "src_bytes"] = np.nan
    with pytest.raises(InputValidationError) as error:
        validate_flow_frame(frame, artifacts)
    assert any("src_bytes" in detail for detail in error.value.details)


def test_blank_categorical_value_is_rejected(valid_one_row, artifacts):
    frame = valid_one_row.copy()
    frame.loc[frame.index[0], "protocol_type"] = "   "
    with pytest.raises(InputValidationError):
        validate_flow_frame(frame, artifacts)


def test_non_numeric_value_is_rejected_with_column_name(valid_one_row, artifacts):
    frame = valid_one_row.copy()
    frame["src_bytes"] = frame["src_bytes"].astype(object)
    frame.loc[frame.index[0], "src_bytes"] = "abc"
    with pytest.raises(InputValidationError) as error:
        validate_flow_frame(frame, artifacts)
    assert "src_bytes" in str(error.value)


def test_infinite_value_is_rejected(valid_one_row, artifacts):
    frame = valid_one_row.copy()
    frame["duration"] = frame["duration"].astype(float)
    frame.loc[frame.index[0], "duration"] = np.inf
    with pytest.raises(InputValidationError) as error:
        validate_flow_frame(frame, artifacts)
    assert "infinies" in str(error.value)


def test_unknown_category_is_accepted_with_warning(valid_one_row, artifacts):
    frame = valid_one_row.copy()
    frame.loc[frame.index[0], "service"] = "service_jamais_vu"
    outcome = validate_flow_frame(frame, artifacts)
    assert any("inconnue" in warning for warning in outcome.warnings)


def test_empty_dataframe_is_rejected(artifacts):
    with pytest.raises(InputValidationError):
        validate_flow_frame(pd.DataFrame(columns=artifacts.feature_columns), artifacts)


def test_empty_csv_is_rejected():
    with pytest.raises(InputValidationError):
        read_csv_bytes(b"", "empty.csv")


def test_header_only_csv_is_rejected(artifacts, valid_one_row):
    content = valid_one_row.head(0).to_csv(index=False).encode("utf-8")
    frame = read_csv_bytes(content, "header_only.csv")
    with pytest.raises(InputValidationError):
        validate_flow_frame(frame, artifacts)


def test_wrong_extension_is_rejected(valid_one_row):
    content = valid_one_row.to_csv(index=False).encode("utf-8")
    with pytest.raises(InputValidationError):
        read_csv_bytes(content, "flows.txt")


def test_duplicate_columns_are_rejected(valid_one_row, artifacts):
    frame = pd.concat([valid_one_row, valid_one_row[["duration"]]], axis=1)
    with pytest.raises(InputValidationError):
        validate_flow_frame(frame, artifacts)


def test_column_order_does_not_change_result(valid_many_rows, artifacts):
    shuffled = valid_many_rows[list(reversed(artifacts.feature_columns))]
    outcome = validate_flow_frame(shuffled, artifacts)
    assert list(outcome.frame.columns) == artifacts.feature_columns

    from nids.inference import predict_flows

    ordered_prediction = predict_flows(valid_many_rows, artifacts)
    shuffled_prediction = predict_flows(shuffled, artifacts)
    assert ordered_prediction.results["predicted_class"].tolist() == (
        shuffled_prediction.results["predicted_class"].tolist()
    )
    assert np.allclose(
        ordered_prediction.probabilities.to_numpy(),
        shuffled_prediction.probabilities.to_numpy(),
    )


def test_duplicate_rows_are_allowed(valid_one_row, artifacts):
    frame = pd.concat([valid_one_row, valid_one_row], ignore_index=True)
    outcome = validate_flow_frame(frame, artifacts)
    assert len(outcome.frame) == 2
