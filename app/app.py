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

from theme import current_theme, inject_style  # noqa: E402
from views import analytics, analysis, evaluation, home  # noqa: E402
from views.common import get_artifacts  # noqa: E402

st.set_page_config(
    page_title="NetGuard AI — NIDS NSL-KDD",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="auto",
)

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
            <div class="ng-brand-name">NetGuard <span>AI</span></div>
            <div class="ng-brand-sub">Network Defense Console</div>
          </div>
        </div>
        <div class="ng-nav-label">Navigation</div>
        """,
        unsafe_allow_html=True,
    )

    page = st.radio(
        "Navigation",
        list(PAGES),
        label_visibility="collapsed",
    )
    st.selectbox(
        "Thème",
        options=("dark", "light"),
        format_func=lambda theme: "Mode sombre" if theme == "dark" else "Mode clair",
        index=0 if current_theme() == "dark" else 1,
        key="ng_theme_mode",
    )

    st.divider()
    try:
        artifacts = get_artifacts()
        st.markdown(
            f"""
            <div class="ng-side-status">
              <div class="ng-status-head"><span class="ng-pulse"></span>Système opérationnel</div>
              <div class="ng-status-row"><span>Modèle</span><b>Random Forest</b></div>
              <div class="ng-status-row"><span>Features</span><b>118</b></div>
              <div class="ng-status-row"><span>Classes</span><b>{len(artifacts.class_names)}</b></div>
              <div class="ng-status-file">rf_model.joblib<br/>preprocessor.joblib</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    except Exception as exc:  # noqa: BLE001 - visible diagnostic in the sidebar
        st.error(f"Artefacts indisponibles : {exc}")

    st.markdown(
        """
        <div class="ng-side-scope" style="margin-top:.8rem;">
          <strong>Périmètre d'analyse</strong>
          Flux fournis en CSV · aucune capture réseau en direct.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()
    st.markdown(
        '<div class="ng-side-foot">Métriques issues du rapport d’évaluation vérifié — '
        "jamais recalculées sur les imports.</div>",
        unsafe_allow_html=True,
    )

inject_style()
PAGES[page]()
