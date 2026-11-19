"""Read-only access to the experimental evaluation saved by Week 3 assets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from nids import config
from nids.errors import ArtifactError

_METRIC_LABELS = {
    "accuracy": "Accuracy",
    "macro_f1": "Macro F1",
    "weighted_f1": "Weighted F1",
    "recall_r2l": "Recall R2L",
    "recall_u2r": "Recall U2R",
    "fpr": "FPR",
    "latency_ms": "Latence (ms)",
}


def load_model_evaluation(reports_dir: str | Path | None = None) -> dict[str, Any]:
    directory = Path(reports_dir) if reports_dir is not None else config.REPORTS_DIR
    path = directory / config.MODEL_EVALUATION_PATH
    if not path.is_file():
        raise ArtifactError(
            f"Rapport d'évaluation introuvable : {path}. "
            "Exécutez `python scripts/generate_evaluation_report.py`."
        )
    try:
        with path.open(encoding="utf-8") as handle:
            report = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"Rapport d'évaluation illisible : {exc}") from exc
    if "test_set" not in report or "models" not in report["test_set"]:
        raise ArtifactError("Rapport d'évaluation incomplet.")
    return report


def comparison_table(report: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for name, model in report["test_set"]["models"].items():
        metrics = model["metrics"]
        rows.append(
            {
                "Modèle": name,
                "Type": "Machine Learning (arbres)" if name == "Random Forest" else "Deep Learning",
                "Accuracy": metrics["accuracy"],
                "Macro F1": metrics["macro_f1"],
                "Weighted F1": metrics["weighted_f1"],
                "Recall R2L": metrics["recall_r2l"],
                "Recall U2R": metrics["recall_u2r"],
                "FPR": metrics["fpr"],
                "Latence (ms)": metrics.get("latency_ms", float("nan")),
            }
        )
    return pd.DataFrame(rows)


def metric_cards(report: dict[str, Any], model_name: str = "Random Forest") -> dict[str, float]:
    return dict(report["test_set"]["models"][model_name]["metrics"])


def per_class_table(
    report: dict[str, Any], split: str = "test_set", model_name: str = "Random Forest"
) -> pd.DataFrame | None:
    section = report.get(split, {})
    model = section.get("models", {}).get(model_name)
    if not model or not model.get("detail_available", False):
        return None
    detail = model["per_class"]
    rows = []
    for class_name in report["dataset"]["classes"]:
        values = detail[class_name]
        rows.append(
            {
                "Classe": class_name,
                "Precision": values["precision"],
                "Recall": values["recall"],
                "F1-score": values["f1"],
                "Support": values["support"],
            }
        )
    return pd.DataFrame(rows)


def confusion_matrix_table(
    report: dict[str, Any], split: str = "test_set", model_name: str = "Random Forest"
) -> pd.DataFrame | None:
    section = report.get(split, {})
    model = section.get("models", {}).get(model_name)
    if not model or "confusion_matrix" not in model:
        return None
    names = report["dataset"]["classes"]
    matrix = model["confusion_matrix"]
    return pd.DataFrame(matrix, index=names, columns=names)


def normalized_confusion_matrix_table(
    report: dict[str, Any], split: str = "test_set", model_name: str = "Random Forest"
) -> pd.DataFrame | None:
    section = report.get(split, {})
    model = section.get("models", {}).get(model_name)
    if not model or "confusion_matrix_normalized" not in model:
        return None
    names = report["dataset"]["classes"]
    matrix = model["confusion_matrix_normalized"]
    return pd.DataFrame(matrix, index=names, columns=names)


def aggregate_table(report: dict[str, Any], split: str = "test_set") -> pd.DataFrame:
    rows = []
    for name, model in report[split]["models"].items():
        metrics = model["metrics"]
        row = {"Modèle": name}
        for key, label in _METRIC_LABELS.items():
            if key in metrics:
                row[label] = metrics[key]
        rows.append(row)
    return pd.DataFrame(rows)
