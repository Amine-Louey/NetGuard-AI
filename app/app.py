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

from theme import inject_style  # noqa: E402
from views import analytics, analysis, evaluation, home  # noqa: E402
from views.common import get_artifacts  # noqa: E402

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
        <div class="ng-brand">
          <div class="ng-brand-mark">NG</div>
          <div>
            <div class="ng-brand-name">NetGuard AI</div>
            <div class="ng-brand-sub">Multi-class NIDS · NSL-KDD</div>
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
            <div class="ng-side-status">
              <div class="head"><span class="dot"></span>Modèle chargé</div>
              <div class="line">Random Forest · 118 features encodées</div>
              <div class="line">{len(artifacts.class_names)} classes · artefacts figés</div>
              <div class="line" style="margin-top:.35rem;">
                <code>rf_model.joblib</code> · <code>preprocessor.joblib</code>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    except Exception as exc:  # noqa: BLE001 - visible diagnostic in the sidebar
        st.error(f"Artefacts indisponibles : {exc}")

    st.markdown(
        """
        <div class="ng-side-scope" style="margin-top:.8rem;">
          <strong>Périmètre</strong> — analyse de flux fournis en CSV.
          Aucune capture réseau en direct.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()
    st.caption("Projet universitaire — Module Machine Learning")
    st.caption("Les métriques expérimentales ne sont jamais recalculées sur les imports.")

PAGES[page]()
