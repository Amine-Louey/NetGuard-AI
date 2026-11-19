"""Reproducibility checks and construction of the saved evaluation report.

The Week 3 notebook is the source of truth for the experimental metrics.  This
module rebuilds the deterministic split, reloads the frozen models and verifies
that the recomputed test metrics match ``metrics_summary.json`` before writing
any asset consumed by the application.
"""

from __future__ import annotations

import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from nids import config
from nids.dataset import DatasetSplit, reproduce_split
from nids.errors import ArtifactError

METRIC_KEYS = {
    "Accuracy": "accuracy",
    "Macro F1": "macro_f1",
    "Weighted F1": "weighted_f1",
    "Recall R2L": "recall_r2l",
    "Recall U2R": "recall_u2r",
    "FPR": "fpr",
}


def calculate_fpr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean one-vs-rest false-positive rate, exactly as in the notebook."""

    matrix = confusion_matrix(y_true, y_pred)
    rates: list[float] = []
    for index in range(len(matrix)):
        false_positive = matrix[:, index].sum() - matrix[index, index]
        true_negative = matrix.sum() - (
            matrix[index, :].sum() + matrix[:, index].sum() - matrix[index, index]
        )
        rates.append(
            false_positive / (false_positive + true_negative)
            if (false_positive + true_negative) > 0
            else 0.0
        )
    return float(np.mean(rates))


def _class_detail(
    y_true: np.ndarray, y_pred: np.ndarray, class_names: tuple[str, ...]
) -> dict[str, Any]:
    labels = list(range(len(class_names)))
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=list(class_names),
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    normalised = matrix.astype(float) / matrix.sum(axis=1, keepdims=True)
    per_class = {
        name: {
            "precision": float(report[name]["precision"]),
            "recall": float(report[name]["recall"]),
            "f1": float(report[name]["f1-score"]),
            "support": int(report[name]["support"]),
        }
        for name in class_names
    }
    return {
        "per_class": per_class,
        "macro_precision": float(report["macro avg"]["precision"]),
        "macro_recall": float(report["macro avg"]["recall"]),
        "weighted_precision": float(report["weighted avg"]["precision"]),
        "weighted_recall": float(report["weighted avg"]["recall"]),
        "confusion_matrix": matrix.astype(int).tolist(),
        "confusion_matrix_normalized": np.round(normalised, 6).tolist(),
        "class_names": list(class_names),
        "support": int(len(y_true)),
    }


def _evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: tuple[str, ...],
    latency_ms: float | None = None,
) -> dict[str, Any]:
    r2l = class_names.index("R2L")
    u2r = class_names.index("U2R")
    report = classification_report(
        y_true, y_pred, output_dict=True, zero_division=0
    )
    detail = _class_detail(y_true, y_pred, class_names)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "recall_r2l": float(report[str(r2l)]["recall"]),
        "recall_u2r": float(report[str(u2r)]["recall"]),
        "fpr": calculate_fpr(y_true, y_pred),
    }
    if latency_ms is not None:
        metrics["latency_ms"] = float(latency_ms)
    return {"metrics": metrics, **detail}


def _saved_metrics_for(summary: dict[str, Any], model_key: str) -> dict[str, float]:
    values: dict[str, float] = {}
    for label, key in METRIC_KEYS.items():
        values[key] = float(summary[label][model_key])
    if "Latency (ms)" in summary:
        values["latency_ms"] = float(summary["Latency (ms)"][model_key])
    return values


def _verify_against_saved(
    computed: dict[str, float], saved: dict[str, float], model_name: str, tolerance: float = 1e-9
) -> float:
    differences = {
        key: abs(computed[key] - saved[key])
        for key in METRIC_KEYS.values()
    }
    worst_key = max(differences, key=differences.get)
    worst = differences[worst_key]
    if worst > tolerance:
        raise ArtifactError(
            f"Reproductibilité rompue pour {model_name} : {worst_key} diffère de "
            f"{worst:.3e} (recalculé={computed[worst_key]:.6f}, sauvegardé={saved[worst_key]:.6f})."
        )
    return worst


def _load_keras_models(models_dir: Path) -> dict[str, Any] | None:
    try:
        from tensorflow import keras  # type: ignore
    except Exception:  # noqa: BLE001 - optional dependency
        return None
    models = {}
    for key, filename in (("MLP", "mlp_model.keras"), ("1D-CNN", "cnn1d_model.keras")):
        path = models_dir / filename
        if path.is_file():
            models[key] = keras.models.load_model(path)
    return models or None


def build_evaluation_report(
    raw_dir: str | Path,
    models_dir: str | Path,
    include_neural: bool = True,
) -> dict[str, Any]:
    """Recompute and verify the frozen evaluation, then return a JSON-ready dict."""

    models_dir = Path(models_dir)
    split: DatasetSplit = reproduce_split(raw_dir)
    with (models_dir / config.FEATURE_CONFIG_PATH).open(encoding="utf-8") as handle:
        feature_config = json.load(handle)
    import joblib

    preprocessor = joblib.load(models_dir / config.PREPROCESSOR_PATH)
    label_encoder = joblib.load(models_dir / config.LABEL_ENCODER_PATH)
    rf = joblib.load(models_dir / config.RF_MODEL_PATH)
    with (models_dir / config.METRICS_SUMMARY_PATH).open(encoding="utf-8") as handle:
        summary = json.load(handle)

    class_names = tuple(str(name) for name in label_encoder.classes_)
    X_val_proc = preprocessor.transform(split.X_val)
    X_test_proc = preprocessor.transform(split.X_test)
    y_val = label_encoder.transform(split.y_val)
    y_test = label_encoder.transform(split.y_test)

    # Random Forest: recompute validation and test details.
    started = time.perf_counter()
    rf_test_pred = rf.predict(X_test_proc)
    rf_latency = (time.perf_counter() - started) / len(split.X_test) * 1000
    rf_val_pred = rf.predict(X_val_proc)
    rf_test = _evaluate_predictions(y_test, rf_test_pred, class_names, rf_latency)
    rf_val = _evaluate_predictions(y_val, rf_val_pred, class_names)

    saved_test = _saved_metrics_for(summary, "Random Forest (Test)")
    saved_val_note = "Les métriques de validation RF sont recalculées (non présentes dans metrics_summary.json)."
    worst_test = _verify_against_saved(rf_test["metrics"], saved_test, "Random Forest")
    # The comparison table must not mix latencies measured on different
    # machines: keep the Week 3 reference value for all three models.
    rf_test["metrics"]["latency_ms"] = saved_test["latency_ms"]
    # Validation reference from the notebook output (cell 17).
    validation_reference = {
        "accuracy": 0.989944,
        "macro_f1": 0.933994,
        "weighted_f1": 0.989897,
        "recall_r2l": 0.976293,
        "recall_u2r": 0.583333,
        "fpr": 0.002945,
    }
    worst_val = _verify_against_saved(
        rf_val["metrics"],
        validation_reference,
        "Random Forest (validation)",
        # The notebook printed validation values with six decimals.
        tolerance=1e-5,
    )

    models: dict[str, Any] = {
        "Random Forest": {
            **rf_test,
            "artifact": "rf_model.joblib",
            "detail_available": True,
            "source": "Test set — recomputed from frozen artifacts and verified against metrics_summary.json",
        },
        "MLP": {
            "metrics": _saved_metrics_for(summary, "MLP (Test)"),
            "detail_available": False,
            "artifact": "mlp_model.keras",
            "source": "Test set — metrics_summary.json (Week 3 export)",
        },
        "1D-CNN": {
            "metrics": _saved_metrics_for(summary, "1D-CNN (Test)"),
            "detail_available": False,
            "artifact": "cnn1d_model.keras",
            "source": "Test set — metrics_summary.json (Week 3 export)",
        },
    }

    neural_notes: list[str] = []
    if include_neural:
        keras_models = _load_keras_models(models_dir)
        if keras_models:
            for key, model in keras_models.items():
                try:
                    if key == "1D-CNN":
                        inputs = X_test_proc.reshape((X_test_proc.shape[0], X_test_proc.shape[1], 1))
                    else:
                        inputs = X_test_proc
                    started = time.perf_counter()
                    probabilities = model.predict(inputs, verbose=0)
                    latency = (time.perf_counter() - started) / len(split.X_test) * 1000
                    prediction = np.argmax(probabilities, axis=1)
                    evaluated = _evaluate_predictions(y_test, prediction, class_names, latency)
                    saved = _saved_metrics_for(summary, f"{key} (Test)")
                    # Same rationale as the Random Forest: compare Week 3 latencies.
                    evaluated["metrics"]["latency_ms"] = saved["latency_ms"]
                    try:
                        difference = _verify_against_saved(evaluated["metrics"], saved, key)
                        verification = {
                            "verified": True,
                            "max_abs_difference": difference,
                        }
                    except ArtifactError as exc:
                        verification = {"verified": False, "message": str(exc)}
                    models[key] = {
                        **evaluated,
                        "artifact": "mlp_model.keras" if key == "MLP" else "cnn1d_model.keras",
                        "detail_available": True,
                        "source": "Test set — recomputed from frozen Keras artifact",
                        "verification": verification,
                    }
                    neural_notes.append(
                        f"{key} : détail par classe recalculé, écart maximal vs métriques sauvegardées "
                        f"= {difference:.2e}."
                    )
                except Exception as exc:  # noqa: BLE001 - optional reproducibility step
                    neural_notes.append(f"{key} : détail non recalculé ({type(exc).__name__}: {exc}).")
        else:
            neural_notes.append(
                "TensorFlow indisponible : les détails par classe du MLP et du 1D-CNN ne sont pas recalculés."
            )

    # Validation detail for RF only (Week 3 selected model).
    validation = {
        "size": int(len(split.X_val)),
        "class_support": {
            str(name): int(count)
            for name, count in split.y_val.value_counts().reindex(config.CLASS_ORDER).items()
        },
        "models": {
            "Random Forest": {
                **rf_val,
                "detail_available": True,
                "source": "Validation set — recomputed and compared with notebook output",
            }
        },
        "note": saved_val_note,
    }

    report = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "metadata": {
            "project": "NetGuard AI — Multi-class NIDS on NSL-KDD",
            "selected_model": "Random Forest",
            "dataset": "NSL-KDD (KDDTrain+_20Percent + KDDTest+)",
            "split": "70% train / 15% validation / 15% test, stratified, random_state=42",
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scripts": "scripts/generate_evaluation_report.py",
        },
        "dataset": {
            "raw_train_rows": split.raw_train_rows,
            "raw_test_rows": split.raw_test_rows,
            "duplicates_removed": split.duplicates_removed,
            "train_size": int(len(split.X_train)),
            "validation_size": int(len(split.X_val)),
            "test_size": int(len(split.X_test)),
            "feature_count_raw": int(len(feature_config["columns"])),
            "feature_count_encoded": int(X_test_proc.shape[1]),
            "classes": list(class_names),
        },
        "reproduction": {
            "test_verified": True,
            "validation_verified": True,
            "random_forest_max_abs_difference_test": worst_test,
            "random_forest_max_abs_difference_validation": worst_val,
            "neural_notes": neural_notes,
        },
        "test_set": {
            "size": int(len(split.X_test)),
            "class_support": {
                str(name): int(count)
                for name, count in split.y_test.value_counts().reindex(config.CLASS_ORDER).items()
            },
            "models": models,
        },
        "validation_set": validation,
        "limitations": [
            "Les métriques portent sur le découpage 70/15/15 reconstruit à partir du notebook de la Semaine 3.",
            "Le jeu de test NSL-KDD contient des attaques absentes du jeu d'entraînement ; les scores ne représentent pas un trafic réseau réel.",
            "La latence dépend du matériel et n'est pas comparable entre machines.",
            "Les probabilités du Random Forest sont des fréquences de votes, pas des probabilités calibrées.",
        ],
    }
    return report


def save_evaluation_report(report: dict[str, Any], reports_dir: str | Path) -> Path:
    directory = Path(reports_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / config.MODEL_EVALUATION_PATH
    with path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    return path
