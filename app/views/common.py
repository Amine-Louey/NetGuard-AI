"""Shared Streamlit helpers and visual identity."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import streamlit as st

from nids import config
from nids.artifacts import Artifacts, load_artifacts
from nids.errors import ArtifactError
from nids.evaluation import load_model_evaluation

CLASS_COLORS = {
    "Normal": "#16a34a",
    "DoS": "#dc2626",
    "Probe": "#ea580c",
    "R2L": "#9333ea",
    "U2R": "#0f766e",
}

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: #f5f7fb; }
[data-testid="stSidebar"] { background: #0b1220; }
[data-testid="stSidebar"] * { color: #e5edf8; }
[data-testid="stSidebar"] .stRadio label { padding: 0.35rem 0.25rem; }
[data-testid="stSidebar"] hr { border-color: #1e293b; }

.block-container { padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1280px; }

.hero {
    background: linear-gradient(125deg, #0b1220 0%, #10233f 55%, #0f3b46 100%);
    border-radius: 22px; padding: 2.5rem 2.7rem; color: #f8fafc;
    border: 1px solid #1e3a5f; box-shadow: 0 18px 45px rgba(15, 23, 42, .22);
    margin-bottom: 1.6rem;
}
.hero .eyebrow { color: #5eead4; font-weight: 700; letter-spacing: .14em;
    text-transform: uppercase; font-size: .74rem; margin-bottom: .75rem; }
.hero h1 { font-size: 2.45rem; line-height: 1.13; margin: 0 0 .9rem 0;
    font-weight: 800; letter-spacing: -.02em; color: #fff; }
.hero p { color: #cbd5e1; font-size: 1.08rem; max-width: 820px; margin: 0; }
.hero .badges { margin-top: 1.35rem; display: flex; flex-wrap: wrap; gap: .55rem; }
.badge { background: rgba(94, 234, 212, .12); border: 1px solid rgba(94, 234, 212, .35);
    color: #99f6e4; border-radius: 999px; padding: .32rem .8rem;
    font-size: .78rem; font-weight: 600; }

.page-title { margin-bottom: .35rem; }
.page-title h1 { font-size: 1.85rem; color: #0f172a; margin: 0; letter-spacing: -.02em; }
.page-title p { color: #52637a; margin: .4rem 0 1.4rem 0; font-size: 1rem; }

.card { background: #fff; border: 1px solid #e2e8f0; border-radius: 16px;
    padding: 1.25rem 1.35rem; box-shadow: 0 4px 18px rgba(15, 23, 42, .04); height: 100%; }
.card h3 { color: #0f172a; font-size: 1.02rem; margin: 0 0 .45rem 0; }
.card p { color: #5b6b82; font-size: .9rem; line-height: 1.55; margin: 0; }
.card .icon { font-size: 1.5rem; margin-bottom: .55rem; }

.class-chip { display:inline-block; border-radius:999px; padding:.25rem .7rem;
    font-size:.8rem; font-weight:700; margin:.2rem .3rem .2rem 0; }
.section-label { color:#64748b; font-size:.76rem; font-weight:700;
    letter-spacing:.12em; text-transform:uppercase; margin:1.4rem 0 .6rem 0; }
.note { background:#eff6ff; border-left:4px solid #2563eb; color:#1e3a8a;
    padding:.85rem 1rem; border-radius:8px; font-size:.9rem; }
.warn-note { background:#fff7ed; border-left:4px solid #ea580c; color:#7c2d12;
    padding:.85rem 1rem; border-radius:8px; font-size:.9rem; }
.small-muted { color:#64748b; font-size:.82rem; }

div[data-testid="stMetric"] { background:#fff; border:1px solid #e2e8f0;
    border-radius:14px; padding:.85rem 1rem; box-shadow:0 4px 16px rgba(15,23,42,.04); }
div[data-testid="stMetricLabel"] p { color:#64748b; font-size:.8rem; font-weight:600; }
div[data-testid="stMetricValue"] { color:#0f172a; font-weight:700; }

.stTabs [data-baseweb="tab-list"] { gap: .4rem; }
.stTabs [data-baseweb="tab"] { background:#fff; border:1px solid #e2e8f0;
    border-radius:10px; padding:.4rem 1rem; }
.stTabs [aria-selected="true"] { background:#0b1220 !important; color:#fff !important; }

footer { visibility: hidden; }
</style>
"""


@st.cache_resource(show_spinner="Chargement des artefacts du modèle…")
def get_artifacts() -> Artifacts:
    return load_artifacts(config.MODELS_DIR)


@st.cache_data(show_spinner=False)
def get_evaluation() -> dict:
    return load_model_evaluation(config.REPORTS_DIR)


def inject_style() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def page_header(title: str, subtitle: str) -> None:
    st.markdown(
        f'<div class="page-title"><h1>{title}</h1><p>{subtitle}</p></div>',
        unsafe_allow_html=True,
    )


def section_label(text: str) -> None:
    st.markdown(f'<div class="section-label">{text}</div>', unsafe_allow_html=True)


def note(text: str, warning: bool = False) -> None:
    css_class = "warn-note" if warning else "note"
    st.markdown(f'<div class="{css_class}">{text}</div>', unsafe_allow_html=True)


def class_chips() -> str:
    return "".join(
        f'<span class="class-chip" style="background:{CLASS_COLORS[name]}1a;color:{CLASS_COLORS[name]}">'
        f"{name}</span>"
        for name in config.CLASS_ORDER
    )


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
