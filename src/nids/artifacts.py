"""Loading and integrity checks for the exported Week 3 artifacts."""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from nids import config
from nids.errors import ArtifactError

# The artifacts were produced with scikit-learn 1.6.1.  Unpickling a
# ColumnTransformer with another version can fail outright (1.9 removed an
# internal class) or silently change preprocessing.  Reproducibility therefore
# requires the pinned version declared in requirements.txt.
SUPPORTED_SKLEARN_VERSION = "1.6.1"


@dataclass(frozen=True)
class Artifacts:
    """Everything required to reproduce the training-time preprocessing."""

    model: Any
    preprocessor: Any
    label_encoder: Any
    feature_config: dict[str, Any]
    class_names: tuple[str, ...]
    known_categories: dict[str, tuple[str, ...]]
    metrics_summary: dict[str, dict[str, float]]

    @property
    def feature_columns(self) -> list[str]:
        return list(self.feature_config["columns"])

    @property
    def numerical_columns(self) -> list[str]:
        return list(self.feature_config["numerical"])

    @property
    def categorical_columns(self) -> list[str]:
        return list(self.feature_config["categorical"])


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ArtifactError(f"Fichier d'artefact introuvable : {path}")
    try:
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"Impossible de lire {path.name} : {exc}") from exc


def _load_joblib(path: Path) -> Any:
    if not path.is_file():
        raise ArtifactError(f"Fichier d'artefact introuvable : {path}")
    try:
        with warnings.catch_warnings():
            # A version mismatch is handled explicitly below.
            warnings.simplefilter("ignore")
            return joblib.load(path)
    except Exception as exc:  # noqa: BLE001 - surfaced as a domain error
        raise ArtifactError(
            f"Impossible de charger {path.name}. Le modèle a probablement été "
            f"exporté avec une autre version de scikit-learn. Détail : {exc}"
        ) from exc


def _verify_consistency(
    model: Any,
    preprocessor: Any,
    label_encoder: Any,
    feature_config: dict[str, Any],
) -> tuple[tuple[str, ...], dict[str, tuple[str, ...]]]:
    required = ("columns", "categorical", "numerical")
    missing = [key for key in required if key not in feature_config]
    if missing:
        raise ArtifactError(f"feature_columns.json incomplet : clés absentes {missing}")

    columns = list(feature_config["columns"])
    categorical = list(feature_config["categorical"])
    numerical = list(feature_config["numerical"])

    if len(columns) != 41:
        raise ArtifactError(
            f"Le modèle attend 41 caractéristiques NSL-KDD, le fichier en déclare {len(columns)}."
        )
    if sorted(categorical + numerical) != sorted(columns):
        raise ArtifactError(
            "feature_columns.json est incohérent : les listes 'numerical' et "
            "'categorical' ne couvrent pas exactement 'columns'."
        )

    preprocessor_columns = getattr(preprocessor, "feature_names_in_", None)
    if preprocessor_columns is not None and list(preprocessor_columns) != columns:
        raise ArtifactError(
            "L'ordre des colonnes du préprocesseur ne correspond pas à feature_columns.json."
        )

    try:
        transformed_width = len(preprocessor.get_feature_names_out())
    except Exception as exc:  # noqa: BLE001
        raise ArtifactError(f"Préprocesseur invalide : {exc}") from exc

    if getattr(model, "n_features_in_", transformed_width) != transformed_width:
        raise ArtifactError(
            "Incohérence de dimensions : le préprocesseur produit "
            f"{transformed_width} colonnes mais le modèle en attend "
            f"{model.n_features_in_}."
        )

    model_classes = np.asarray(getattr(model, "classes_", []))
    if model_classes.size == 0:
        raise ArtifactError("Le modèle ne déclare aucune classe.")
    try:
        class_names = tuple(str(name) for name in label_encoder.inverse_transform(model_classes))
    except Exception as exc:  # noqa: BLE001
        raise ArtifactError(f"LabelEncoder incompatible avec le modèle : {exc}") from exc

    if len(set(class_names)) != len(class_names):
        raise ArtifactError("Le LabelEncoder produit des noms de classes dupliqués.")

    transformer = preprocessor.named_transformers_.get("cat")
    if transformer is None or not hasattr(transformer, "categories_"):
        raise ArtifactError("Le préprocesseur ne contient pas l'encodeur catégoriel attendu.")
    if len(transformer.categories_) != len(categorical):
        raise ArtifactError("Nombre de colonnes catégorielles incohérent dans le préprocesseur.")

    known_categories = {
        column: tuple(str(value) for value in values)
        for column, values in zip(categorical, transformer.categories_, strict=True)
    }
    return class_names, known_categories


@lru_cache(maxsize=4)
def _load_artifacts_cached(models_dir: str) -> Artifacts:
    import sklearn

    if sklearn.__version__ != SUPPORTED_SKLEARN_VERSION:
        raise ArtifactError(
            "Version de scikit-learn incompatible : "
            f"{sklearn.__version__} détectée, {SUPPORTED_SKLEARN_VERSION} requise. "
            "Installez les dépendances avec `pip install -r requirements.txt`."
        )

    directory = Path(models_dir)
    feature_config = _load_json(directory / config.FEATURE_CONFIG_PATH)
    preprocessor = _load_joblib(directory / config.PREPROCESSOR_PATH)
    label_encoder = _load_joblib(directory / config.LABEL_ENCODER_PATH)
    model = _load_joblib(directory / config.RF_MODEL_PATH)

    class_names, known_categories = _verify_consistency(
        model, preprocessor, label_encoder, feature_config
    )
    metrics_summary = _load_json(directory / config.METRICS_SUMMARY_PATH)

    return Artifacts(
        model=model,
        preprocessor=preprocessor,
        label_encoder=label_encoder,
        feature_config=feature_config,
        class_names=class_names,
        known_categories=known_categories,
        metrics_summary=metrics_summary,
    )


def load_artifacts(models_dir: str | Path | None = None) -> Artifacts:
    """Load and validate the frozen Random Forest inference artifacts.

    Parameters
    ----------
    models_dir:
        Directory containing the exported artifacts.  Defaults to ``models/``
        at the project root (or ``NIDS_MODELS_DIR``).
    """

    directory = Path(models_dir) if models_dir is not None else config.MODELS_DIR
    return _load_artifacts_cached(str(directory.resolve()))


def load_metrics_summary(models_dir: str | Path | None = None) -> dict[str, dict[str, float]]:
    """Read the Week 3 test-set metrics without loading a model."""

    directory = Path(models_dir) if models_dir is not None else config.MODELS_DIR
    return _load_json(directory / config.METRICS_SUMMARY_PATH)
