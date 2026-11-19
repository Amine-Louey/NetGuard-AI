"""Attack Analytics tests: every statistic must derive from predictions."""

from __future__ import annotations

import pandas as pd
import pytest

from nids.analytics import build_attack_statistics
from nids.errors import InputValidationError
from nids.inference import predict_flows


def test_statistics_are_consistent_with_predictions(raw_flows, artifacts):
    results = predict_flows(raw_flows, artifacts).results
    stats = build_attack_statistics(results)
    assert stats.total == len(results)
    assert stats.normal == int((results["predicted_class"] == "Normal").sum())
    assert stats.malicious == stats.total - stats.normal
    assert stats.distribution["count"].sum() == stats.total
    assert pytest.approx(stats.distribution["share"].sum()) == 1.0
    assert set(stats.distribution["category"]) == {"Normal", "DoS", "Probe", "R2L", "U2R"}


def test_alerts_contain_only_attacks_and_are_sorted(valid_many_rows, artifacts):
    results = predict_flows(valid_many_rows, artifacts).results
    stats = build_attack_statistics(results)
    assert set(stats.alerts["predicted_class"]).issubset({"DoS", "Probe", "R2L", "U2R"})
    confidences = stats.alerts["confidence"].tolist()
    assert confidences == sorted(confidences, reverse=True)


def test_all_normal_batch(artifacts):
    frame = pd.DataFrame(
        {
            "flow_id": [1, 2],
            "predicted_class": ["Normal", "Normal"],
            "confidence": [0.99, 0.75],
            "status": ["Normal", "Normal"],
        }
    )
    stats = build_attack_statistics(frame)
    assert stats.malicious == 0
    assert stats.malicious_rate == 0
    assert stats.alerts.empty
    assert stats.mean_alert_confidence is None


def test_empty_results_are_rejected():
    with pytest.raises(InputValidationError):
        build_attack_statistics(pd.DataFrame())


def test_missing_columns_are_rejected():
    with pytest.raises(InputValidationError):
        build_attack_statistics(pd.DataFrame({"flow_id": [1], "predicted_class": ["DoS"]}))
