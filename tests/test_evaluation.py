"""Evaluation report integrity and scientific consistency tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nids.evaluation import (
    comparison_table,
    confusion_matrix_table,
    load_model_evaluation,
    normalized_confusion_matrix_table,
    per_class_table,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def evaluation():
    return load_model_evaluation(PROJECT_ROOT / "reports")


def test_selected_model_and_splits(evaluation):
    assert evaluation["metadata"]["selected_model"] == "Random Forest"
    assert evaluation["test_set"]["size"] == 7161
    assert evaluation["validation_set"]["size"] == 7160
    assert set(evaluation["test_set"]["models"]) == {"Random Forest", "MLP", "1D-CNN"}


def test_test_metrics_match_the_saved_week3_summary(evaluation):
    with (PROJECT_ROOT / "models" / "metrics_summary.json").open(encoding="utf-8") as handle:
        summary = json.load(handle)
    mapping = {
        "Random Forest": "Random Forest (Test)",
        "MLP": "MLP (Test)",
        "1D-CNN": "1D-CNN (Test)",
    }
    for model, key in mapping.items():
        metrics = evaluation["test_set"]["models"][model]["metrics"]
        assert metrics["accuracy"] == pytest.approx(summary["Accuracy"][key], abs=1e-12)
        assert metrics["macro_f1"] == pytest.approx(summary["Macro F1"][key], abs=1e-12)
        assert metrics["weighted_f1"] == pytest.approx(summary["Weighted F1"][key], abs=1e-12)
        assert metrics["recall_r2l"] == pytest.approx(summary["Recall R2L"][key], abs=1e-12)
        assert metrics["recall_u2r"] == pytest.approx(summary["Recall U2R"][key], abs=1e-12)


def test_reproduction_is_explicitly_verified(evaluation):
    reproduction = evaluation["reproduction"]
    assert reproduction["test_verified"] is True
    assert reproduction["validation_verified"] is True
    assert reproduction["random_forest_max_abs_difference_test"] < 1e-12


def test_per_class_support_sums_to_test_size(evaluation):
    table = per_class_table(evaluation)
    assert table is not None
    assert table["Support"].sum() == evaluation["test_set"]["size"]
    assert set(table["Classe"]) == {"Normal", "DoS", "Probe", "R2L", "U2R"}


def test_confusion_matrix_is_consistent_with_recall(evaluation):
    matrix = confusion_matrix_table(evaluation)
    per_class = per_class_table(evaluation).set_index("Classe")
    assert matrix is not None
    for class_name in matrix.index:
        support = matrix.loc[class_name].sum()
        assert support == per_class.loc[class_name, "Support"]
        recall = matrix.loc[class_name, class_name] / support
        assert recall == pytest.approx(per_class.loc[class_name, "Recall"], abs=1e-9)


def test_normalized_rows_sum_to_one(evaluation):
    matrix = normalized_confusion_matrix_table(evaluation)
    assert matrix is not None
    for _, row in matrix.iterrows():
        assert row.sum() == pytest.approx(1.0, abs=1e-6)


def test_comparison_table_has_all_models(evaluation):
    table = comparison_table(evaluation)
    assert table["Modèle"].tolist() == ["Random Forest", "MLP", "1D-CNN"]
    assert table.loc[table["Modèle"] == "Random Forest", "Macro F1"].iloc[0] == pytest.approx(
        0.9355841102217399
    )


def test_validation_and_test_are_not_confused(evaluation):
    assert evaluation["validation_set"]["size"] != evaluation["test_set"]["size"]
    assert "Validation set" in (
        evaluation["validation_set"]["models"]["Random Forest"]["source"]
    )
