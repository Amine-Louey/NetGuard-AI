"""Inference pipeline: classes, probabilities, batching and consistency."""

from __future__ import annotations

import numpy as np
import pytest

from nids.errors import InputValidationError, PredictionError
from nids.inference import predict_flows, predict_single_flow


def test_predictions_match_recorded_fixture(raw_flows, expected_frame, artifacts):
    prediction = predict_flows(raw_flows, artifacts)
    assert prediction.results["predicted_class"].tolist() == expected_frame["expected_class"].tolist()
    assert np.allclose(
        prediction.results["confidence"].to_numpy(),
        expected_frame["expected_confidence"].to_numpy(),
        atol=1e-12,
    )


def test_probability_columns_match_fixture(raw_flows, expected_frame, artifacts):
    prediction = predict_flows(raw_flows, artifacts)
    for class_name in artifacts.class_names:
        assert np.allclose(
            prediction.probabilities[class_name].to_numpy(),
            expected_frame[f"p_{class_name}"].to_numpy(),
            atol=1e-12,
        )


def test_probabilities_are_valid(raw_flows, artifacts):
    prediction = predict_flows(raw_flows, artifacts)
    values = prediction.probabilities.to_numpy()
    assert np.isfinite(values).all()
    assert (values >= 0).all()
    assert (values <= 1).all()
    assert np.allclose(values.sum(axis=1), 1.0)


def test_single_row_is_supported(valid_one_row, artifacts):
    prediction = predict_flows(valid_one_row, artifacts)
    assert len(prediction) == 1
    assert prediction.results.iloc[0]["flow_id"] == 1
    assert prediction.results.iloc[0]["predicted_class"] in artifacts.class_names


def test_single_flow_helper(valid_one_row, artifacts):
    result = predict_single_flow(valid_one_row, artifacts)
    assert set(result["probabilities"]) == set(artifacts.class_names)
    assert 0.0 <= result["confidence"] <= 1.0
    assert abs(sum(result["probabilities"].values()) - 1.0) < 1e-9


def test_single_flow_helper_rejects_many_rows(valid_many_rows, artifacts):
    with pytest.raises(PredictionError):
        predict_single_flow(valid_many_rows, artifacts)


def test_many_rows_are_supported(demo_sample, artifacts):
    prediction = predict_flows(demo_sample, artifacts)
    assert len(prediction) == len(demo_sample)
    assert prediction.results["flow_id"].tolist() == list(range(1, len(demo_sample) + 1))


def test_pipeline_matches_direct_frozen_preprocessing(raw_flows, artifacts):
    prediction = predict_flows(raw_flows, artifacts)
    transformed = artifacts.preprocessor.transform(raw_flows)
    direct_classes = artifacts.model.predict(transformed)
    expected_labels = artifacts.label_encoder.inverse_transform(direct_classes)
    assert prediction.results["predicted_class"].tolist() == list(expected_labels)


def test_input_dataframe_is_not_mutated(valid_many_rows, artifacts):
    before = valid_many_rows.copy(deep=True)
    predict_flows(valid_many_rows, artifacts)
    np.testing.assert_array_equal(before.to_numpy(), valid_many_rows.to_numpy())


def test_missing_columns_fail_before_prediction(valid_one_row, artifacts):
    with pytest.raises(InputValidationError):
        predict_flows(valid_one_row.drop(columns=["flag"]), artifacts)


def test_all_five_classes_have_a_probability(valid_one_row, artifacts):
    prediction = predict_flows(valid_one_row, artifacts)
    assert list(prediction.probabilities.columns) == list(artifacts.class_names)
    assert prediction.probabilities.shape == (1, 5)


def test_status_is_consistent_with_predicted_class(raw_flows, artifacts):
    prediction = predict_flows(raw_flows, artifacts)
    normal = prediction.results["predicted_class"] == "Normal"
    assert (prediction.results.loc[normal, "status"] == "Normal").all()
    assert (prediction.results.loc[~normal, "status"] == "Alerte").all()
