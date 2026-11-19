"""Validation of raw NSL-KDD flow tables before inference.

The policy is intentionally strict and explicit:

* the 41 training columns are required, whatever their order;
* extra business columns are ignored with a warning;
* label/metadata columns are ignored with a warning;
* missing values are rejected (no imputation was learned during training);
* non-numeric values in numerical columns are rejected;
* unseen categories are accepted by the One-Hot encoder (``handle_unknown``
  was set to ``ignore`` during training) but reported as a warning.
"""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import BinaryIO, Iterable

import numpy as np
import pandas as pd

from nids import config
from nids.artifacts import Artifacts
from nids.errors import InputValidationError


@dataclass(frozen=True)
class ValidationOutcome:
    frame: pd.DataFrame
    warnings: tuple[str, ...]


def _format_examples(values: Iterable[object], limit: int = 3) -> str:
    examples = [repr(value) for value in list(values)[:limit]]
    suffix = ", …" if len(examples) == limit else ""
    return ", ".join(examples) + suffix


def read_csv_bytes(content: bytes, filename: str = "fichier.csv") -> pd.DataFrame:
    """Read a CSV upload with explicit size and parsing errors."""

    if not filename.lower().endswith(".csv"):
        raise InputValidationError(
            f"Format non pris en charge : {filename}. Importez un fichier .csv."
        )
    if not content:
        raise InputValidationError("Le fichier est vide.")
    if len(content) > config.MAX_UPLOAD_BYTES:
        raise InputValidationError(
            "Fichier trop volumineux : "
            f"{len(content) / (1024 * 1024):.1f} Mo (limite "
            f"{config.MAX_UPLOAD_BYTES / (1024 * 1024):.0f} Mo)."
        )
    try:
        frame = pd.read_csv(BytesIO(content), encoding="utf-8-sig")
    except UnicodeDecodeError as exc:
        raise InputValidationError(
            "Encodage non reconnu. Enregistrez le fichier en UTF-8 puis réessayez."
        ) from exc
    except pd.errors.EmptyDataError as exc:
        raise InputValidationError("Le fichier ne contient aucune donnée.") from exc
    except pd.errors.ParserError as exc:
        raise InputValidationError(f"CSV illisible : {exc}") from exc
    return frame


def read_csv_path(path: str | Path) -> pd.DataFrame:
    """Read a CSV from disk using the same checks as an upload."""

    file_path = Path(path)
    if not file_path.is_file():
        raise InputValidationError(f"Fichier introuvable : {file_path}")
    return read_csv_bytes(file_path.read_bytes(), file_path.name)


def read_csv_stream(stream: BinaryIO, filename: str = "fichier.csv") -> pd.DataFrame:
    """Read a binary file-like object (e.g. a Streamlit upload)."""

    return read_csv_bytes(stream.read(), filename)


def validate_flow_frame(frame: pd.DataFrame, artifacts: Artifacts) -> ValidationOutcome:
    """Validate and normalise a raw flow table.

    Returns a copy containing exactly the 41 expected columns, with numerical
    columns as ``float64`` and categorical columns as trimmed strings.  The
    original row order and index are preserved.
    """

    if not isinstance(frame, pd.DataFrame):
        raise InputValidationError("Les données doivent être fournies sous forme de tableau.")
    if frame.empty or len(frame) == 0:
        raise InputValidationError("Le fichier ne contient aucune ligne de flux à analyser.")
    if len(frame) > config.MAX_ROWS:
        raise InputValidationError(
            f"Trop de lignes : {len(frame)} (limite {config.MAX_ROWS})."
        )

    warnings: list[str] = []

    # Normalise column labels without silently renaming real features.
    stripped = [str(column).strip() for column in frame.columns]
    if len(set(stripped)) != len(stripped):
        raise InputValidationError(
            "Le fichier contient des noms de colonnes dupliqués après nettoyage des espaces."
        )
    frame = frame.copy()
    frame.columns = stripped

    required = artifacts.feature_columns
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise InputValidationError(
            "Colonnes NSL-KDD manquantes : " + ", ".join(missing),
            [f"Colonne requise absente : {column}" for column in missing],
        )

    ignored_labels = [
        column for column in config.IGNORED_LABEL_COLUMNS if column in frame.columns
    ]
    if ignored_labels:
        warnings.append(
            "Colonnes d'étiquettes ignorées (elles ne servent pas à la prédiction) : "
            + ", ".join(ignored_labels)
            + "."
        )

    extra = [
        column
        for column in frame.columns
        if column not in required and column not in config.IGNORED_LABEL_COLUMNS
    ]
    if extra:
        warnings.append(
            "Colonnes supplémentaires ignorées : " + ", ".join(extra) + "."
        )

    selected = frame[required].copy()

    # Missing values: nulls and blank strings are both rejected.
    missing_report: list[str] = []
    for column in required:
        series = selected[column]
        null_count = int(series.isna().sum())
        if series.dtype == object:
            blank_mask = series.astype("string").str.strip().eq("")
            null_count += int((blank_mask & series.notna()).sum())
        if null_count:
            missing_report.append(f"{column} ({null_count} valeur(s) manquante(s))")
    if missing_report:
        raise InputValidationError(
            "Valeurs manquantes détectées. Aucune imputation n'a été apprise "
            "pendant l'entraînement ; corrigez ou supprimez ces lignes.",
            missing_report,
        )

    # Numerical columns.
    for column in artifacts.numerical_columns:
        original = selected[column]
        converted = pd.to_numeric(original, errors="coerce")
        invalid = original.notna() & converted.isna()
        if invalid.any():
            bad_values = original[invalid].unique()
            raise InputValidationError(
                f"Valeurs non numériques dans la colonne « {column} » : "
                f"{_format_examples(bad_values)}."
            )
        if np.isinf(converted.to_numpy(dtype=float)).any():
            count = int(np.isinf(converted.to_numpy(dtype=float)).sum())
            raise InputValidationError(
                f"Valeurs infinies dans la colonne « {column} » ({count} ligne(s))."
            )
        selected[column] = converted.astype(float)

    # Categorical columns: keep the exact vocabulary seen during training.
    unknown_report: list[str] = []
    for column in artifacts.categorical_columns:
        values = selected[column].astype("string").str.strip()
        known = set(artifacts.known_categories[column])
        unknown_mask = ~values.isin(known)
        if unknown_mask.any():
            examples = values[unknown_mask].dropna().unique()
            unknown_report.append(
                f"{column} : {int(unknown_mask.sum())} valeur(s) inconnue(s), "
                f"ex. {_format_examples(examples)}"
            )
        selected[column] = values.astype(object)
    if unknown_report:
        warnings.append(
            "Catégories absentes du jeu d'entraînement : "
            + " ; ".join(unknown_report)
            + ". Elles sont ignorées par l'encodage One-Hot (comme pendant l'entraînement)."
        )

    return ValidationOutcome(frame=selected, warnings=tuple(warnings))
