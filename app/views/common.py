"""Shared data and session helpers for the Streamlit views.

Presentation helpers live in :mod:`theme`; this module only wires the frozen
artifacts and the evaluation report to the UI. It never trains or recomputes
experimental metrics from uploaded files.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import streamlit as st

from nids import config
from nids.artifacts import Artifacts, load_artifacts
from nids.errors import ArtifactError
from nids.evaluation import load_model_evaluation

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@st.cache_resource(show_spinner="Chargement des artefacts du modèle…")
def get_artifacts() -> Artifacts:
    return load_artifacts(config.MODELS_DIR)


@st.cache_data(show_spinner=False)
def get_evaluation() -> dict:
    return load_model_evaluation(config.REPORTS_DIR)


@st.cache_data(show_spinner=False)
def get_sample_csv() -> bytes:
    """Charge le CSV d'exemple une seule fois par worker."""
    return (PROJECT_ROOT / "data" / "samples" / "nsl_kdd_sample.csv").read_bytes()


def store_analysis(
    source_name: str,
    content_hash: str,
    results: pd.DataFrame,
    probabilities: pd.DataFrame,
    warnings: tuple[str, ...],
) -> None:
    st.session_state["analysis"] = {
        "source_name": source_name,
        "hash": content_hash,
        "results": results,
        "probabilities": probabilities,
        "warnings": warnings,
    }


def get_analysis() -> dict | None:
    return st.session_state.get("analysis")


def clear_analysis() -> None:
    st.session_state.pop("analysis", None)


def content_digest(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def format_percent(value: float, digits: int = 2) -> str:
    return f"{value * 100:.{digits}f} %"


def results_display(results: pd.DataFrame) -> pd.DataFrame:
    display = results.copy()
    display["confidence"] = (display["confidence"] * 100).round(2)
    display = display.rename(
        columns={
            "flow_id": "Flux",
            "predicted_class": "Classe prédite",
            "confidence": "Confiance (%)",
            "status": "Statut",
        }
    )
    return display


def evaluation_available() -> bool:
    try:
        get_evaluation()
    except ArtifactError:
        return False
    return True
