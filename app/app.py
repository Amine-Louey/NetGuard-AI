"""NetGuard AI — Streamlit entry point.

Run locally:
    streamlit run app/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(APP_DIR))

import streamlit as st  # noqa: E402

from views import analytics, analysis, evaluation, home  # noqa: E402
from views.common import get_artifacts, inject_style  # noqa: E402

st.set_page_config(
    page_title="NetGuard AI — NIDS NSL-KDD",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_style()

PAGES = {
    "Accueil": home.render,
    "Analyse": analysis.render,
    "Attack Analytics": analytics.render,
    "Model Evaluation": evaluation.render,
}

with st.sidebar:
    st.markdown(
        """
        <div style="padding:.4rem 0 1rem 0;">
          <div style="font-size:1.45rem;font-weight:800;letter-spacing:-.02em;color:#fff;">
            🛡️ NetGuard AI
          </div>
          <div style="font-size:.78rem;color:#94a3b8;margin-top:.25rem;">
            Multi-class NIDS · NSL-KDD
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    page = st.radio(
        "Navigation",
        list(PAGES),
        label_visibility="collapsed",
    )
    st.divider()
    try:
        artifacts = get_artifacts()
        st.markdown(
            f"""
            <div style="font-size:.78rem;line-height:1.6;color:#cbd5e1;">
              <div style="color:#5eead4;font-weight:700;margin-bottom:.3rem;">● Modèle chargé</div>
              Random Forest · 118 features encodées<br/>
              {len(artifacts.class_names)} classes · artefacts figés
            </div>
            """,
            unsafe_allow_html=True,
        )
    except Exception as exc:  # noqa: BLE001 - visible diagnostic in the sidebar
        st.error(f"Artefacts indisponibles : {exc}")
    st.divider()
    st.caption("Projet universitaire — Module Machine Learning")
    st.caption("Les résultats expérimentaux ne sont jamais recalculés sur les fichiers importés.")

PAGES[page]()
