"""Artifact loading and consistency tests."""

from __future__ import annotations

import pytest

from nids.artifacts import load_artifacts
from nids.errors import ArtifactError


def test_artifacts_load_and_are_consistent(artifacts):
    assert len(artifacts.feature_columns) == 41
    assert len(artifacts.numerical_columns) == 38
    assert len(artifacts.categorical_columns) == 3
    assert len(artifacts.class_names) == 5
    assert len(artifacts.preprocessor.get_feature_names_out()) == 118
    assert artifacts.model.n_features_in_ == 118


def test_class_names_are_mapped_through_label_encoder(artifacts):
    assert artifacts.class_names == ("DoS", "Normal", "Probe", "R2L", "U2R")
    assert set(artifacts.class_names) == {"Normal", "DoS", "Probe", "R2L", "U2R"}


def test_known_categories_are_recorded(artifacts):
    assert set(artifacts.known_categories) == {"protocol_type", "service", "flag"}
    assert "tcp" in artifacts.known_categories["protocol_type"]
    assert "http" in artifacts.known_categories["service"]


def test_metrics_summary_contains_all_models(artifacts):
    assert "Accuracy" in artifacts.metrics_summary
    assert set(artifacts.metrics_summary["Macro F1"]) == {
        "Random Forest (Test)",
        "MLP (Test)",
        "1D-CNN (Test)",
    }


def test_missing_artifact_directory_raises(tmp_path):
    with pytest.raises(ArtifactError):
        load_artifacts(tmp_path / "does_not_exist")
