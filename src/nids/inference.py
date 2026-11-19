"""Frozen-model inference.

The training-time preprocessing is reused verbatim through the exported
``ColumnTransformer``: no scaler is re-fitted, no SMOTE is applied, and the
model is never retrained at prediction time.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from nids.artifacts import Artifacts
from nids.errors import PredictionError
from nids.validation import ValidationOutcome, validate_flow_frame


@dataclass(frozen=True)
class PredictionResult:
    """Predictions and class probabilities for one analysis batch."""

    results: pd.DataFrame
    probabilities: pd.DataFrame
    warnings: tuple[str, ...] = ()

    def __len__(self) -> int:
        return len(self.results)


def _predict_clean(frame: pd.DataFrame, artifacts: Artifacts) -> tuple[np.ndarray, np.ndarray]:
    try:
        transformed = artifacts.preprocessor.transform(frame)
    except Exception as exc:  # noqa: BLE001 - surfaced as a domain error
        raise PredictionError(f"Échec du préprocesseur : {exc}") from exc

    if not hasattr(artifacts.model, "predict_proba"):
        raise PredictionError("Le modèle sélectionné ne fournit pas de probabilités.")

    try:
        probabilities = np.asarray(artifacts.model.predict_proba(transformed), dtype=float)
    except Exception as exc:  # noqa: BLE001
        raise PredictionError(f"Échec de la prédiction : {exc}") from exc

    expected_shape = (len(frame), len(artifacts.class_names))
    if probabilities.shape != expected_shape:
        raise PredictionError(
            f"Forme de probabilités inattendue : {probabilities.shape}, attendu {expected_shape}."
        )
    if not np.isfinite(probabilities).all():
        raise PredictionError("Le modèle a produit des probabilités non finies.")
    row_sums = probabilities.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-3):
        raise PredictionError("Les probabilités retournées ne somment pas à 1.")

    predicted = np.asarray(artifacts.model.predict(transformed))
    if predicted.shape != (len(frame),):
        raise PredictionError(f"Forme de prédictions inattendue : {predicted.shape}.")
    return predicted, probabilities


def _build_result(
    frame: pd.DataFrame, predicted: np.ndarray, probabilities: np.ndarray, artifacts: Artifacts
) -> tuple[pd.DataFrame, pd.DataFrame]:
    class_names = np.asarray(artifacts.class_names)
    # The model's classes_ order is authoritative for the probability columns.
    position = {int(code): index for index, code in enumerate(np.asarray(artifacts.model.classes_))}
    labels = np.asarray([class_names[position[int(code)]] for code in predicted])

    results = pd.DataFrame(
        {
            "flow_id": np.arange(1, len(frame) + 1),
            "predicted_class": labels,
            "confidence": probabilities.max(axis=1),
            "status": np.where(labels == "Normal", "Normal", "Alerte"),
        }
    )
    probability_frame = pd.DataFrame(probabilities, columns=list(artifacts.class_names))
    return results, probability_frame


def predict_flows(frame: pd.DataFrame, artifacts: Artifacts) -> PredictionResult:
    """Classify one or many raw NSL-KDD flows.

    Parameters
    ----------
    frame:
        Raw table containing the 41 NSL-KDD features (any column order).
    artifacts:
        Frozen artifacts returned by :func:`nids.artifacts.load_artifacts`.

    Raises
    ------
    InputValidationError
        If the table does not satisfy the documented input contract.
    PredictionError
        If the model output is inconsistent.
    """

    outcome: ValidationOutcome = validate_flow_frame(frame, artifacts)
    predicted, probabilities = _predict_clean(outcome.frame, artifacts)
    results, probability_frame = _build_result(
        outcome.frame, predicted, probabilities, artifacts
    )
    return PredictionResult(
        results=results,
        probabilities=probability_frame,
        warnings=outcome.warnings,
    )


def predict_single_flow(frame: pd.DataFrame, artifacts: Artifacts) -> dict[str, object]:
    """Convenience wrapper for exactly one flow."""

    if len(frame) != 1:
        raise PredictionError(
            f"Cette fonction attend une seule ligne, {len(frame)} reçue(s)."
        )
    result = predict_flows(frame, artifacts)
    row = result.results.iloc[0]
    probabilities = result.probabilities.iloc[0].to_dict()
    return {
        "predicted_class": str(row["predicted_class"]),
        "confidence": float(row["confidence"]),
        "probabilities": {str(key): float(value) for key, value in probabilities.items()},
        "warnings": result.warnings,
    }


def predict_from_transformed(
    transformed: np.ndarray, artifacts: Artifacts
) -> np.ndarray:
    """Return raw class probabilities for already preprocessed features.

    Intended for automated tests and reproducibility checks.  The input must be
    the output of the frozen preprocessor, never a manually built matrix.
    """

    if not hasattr(artifacts.model, "predict_proba"):
        raise PredictionError("Le modèle sélectionné ne fournit pas de probabilités.")
    probabilities = np.asarray(artifacts.model.predict_proba(transformed), dtype=float)
    if probabilities.ndim != 2 or not np.isfinite(probabilities).all():
        raise PredictionError("Sortie de probabilités invalide.")
    return probabilities
