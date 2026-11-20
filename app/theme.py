"""NetGuard AI — charte visuelle claire et composants d'interface.

Ce module ne contient que de la présentation : aucune règle métier, aucun calcul
de prédiction ou de métrique. Il centralise la palette, le CSS Streamlit et les
petits composants HTML réutilisés par les vues.
"""

from __future__ import annotations

import html
from typing import Iterable, Sequence

import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Palette — mode clair, contraste WCAG AA
# ---------------------------------------------------------------------------

SURFACE = "#FFFFFF"
CANVAS = "#F4F6FA"
BORDER = "#E2E8F0"
TEXT = "#0F172A"
MUTED = "#52637A"
FAINT = "#64748B"

INDIGO = "#2563EB"
INDIGO_DARK = "#1D4ED8"
INDIGO_SOFT = "#EEF4FF"
EMERALD = "#059669"
EMERALD_SOFT = "#ECFDF5"
AMBER = "#D97706"
AMBER_SOFT = "#FFFBEB"
RED = "#DC2626"
RED_SOFT = "#FEF2F2"
VIOLET = "#7C3AED"

CLASS_COLORS = {
    "Normal": EMERALD,
    "DoS": RED,
    "Probe": AMBER,
    "R2L": "#DB2777",
    "U2R": VIOLET,
}

TONE_COLORS = {
    "indigo": INDIGO,
    "emerald": EMERALD,
    "amber": AMBER,
    "red": RED,
    "violet": VIOLET,
    "slate": "#475569",
}

PLOTLY_CONFIG = {"displayModeBar": False, "responsive": True}

# ---------------------------------------------------------------------------
# Feuille de style
# ---------------------------------------------------------------------------

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;600;700&display=swap');

:root {
    --ng-canvas: #F4F6FA;
    --ng-surface: #FFFFFF;
    --ng-border: #E2E8F0;
    --ng-text: #0F172A;
    --ng-muted: #52637A;
    --ng-indigo: #2563EB;
    --ng-indigo-dark: #1D4ED8;
    --ng-emerald: #059669;
    --ng-amber: #D97706;
    --ng-red: #DC2626;
    --ng-radius: 12px;
    --ng-shadow: 0 1px 3px rgba(15, 23, 42, .05);
}

html, body, [data-testid="stAppViewContainer"], [data-testid="stSidebar"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}
.stApp { background: var(--ng-canvas); color: var(--ng-text); }
[data-testid="stHeader"] {
    background: rgba(244, 246, 250, .88);
    backdrop-filter: blur(8px);
    border-bottom: 1px solid transparent;
}
.block-container { padding-top: 1.5rem; padding-bottom: 3rem; max-width: 1320px; }

/* ------------------------------------------------------------------ sidebar */
[data-testid="stSidebar"] {
    background: #FFFFFF;
    border-right: 1px solid var(--ng-border);
}
[data-testid="stSidebar"] [data-testid="stSidebarContent"] { padding-top: .4rem; }
[data-testid="stSidebar"] * { color: #334155; }
[data-testid="stSidebar"] hr { border-color: var(--ng-border); margin: 1rem 0; }

.ng-brand { display: flex; align-items: center; gap: .75rem; padding: .35rem 0 1.1rem 0; }
.ng-brand-mark {
    width: 40px; height: 40px; border-radius: 11px; flex: 0 0 40px;
    background: linear-gradient(135deg, #1D4ED8, #2563EB 60%, #0EA5E9);
    color: #fff; display: flex; align-items: center; justify-content: center;
    font-weight: 800; font-size: 1rem; letter-spacing: -.03em;
    box-shadow: 0 4px 10px rgba(37, 99, 235, .25);
}
.ng-brand-name { font-weight: 800; font-size: 1.12rem; color: #0F172A; letter-spacing: -.02em; line-height: 1.1; }
.ng-brand-sub { font-size: .72rem; color: #64748B; margin-top: .18rem; font-weight: 500; }

[data-testid="stSidebar"] [role="radiogroup"] { gap: .3rem; }
[data-testid="stSidebar"] [role="radiogroup"] > label {
    display: flex; align-items: center; width: 100%;
    padding: .58rem .7rem; margin: 0; border-radius: 10px;
    border: 1px solid transparent; transition: background .15s ease, border-color .15s ease;
    cursor: pointer;
}
[data-testid="stSidebar"] [role="radiogroup"] > label:hover { background: #F8FAFC; }
[data-testid="stSidebar"] [role="radiogroup"] > label:has(input:checked) {
    background: var(--ng-indigo-soft);
    border-color: #DBEAFE;
    box-shadow: inset 3px 0 0 var(--ng-indigo);
}
[data-testid="stSidebar"] [role="radiogroup"] > label p {
    font-size: .88rem; font-weight: 600; color: #475569;
}
[data-testid="stSidebar"] [role="radiogroup"] > label:has(input:checked) p {
    color: var(--ng-indigo-dark); font-weight: 700;
}

.ng-side-status {
    background: #F8FAFC; border: 1px solid var(--ng-border);
    border-radius: 11px; padding: .8rem .9rem; font-size: .76rem; line-height: 1.55;
}
.ng-side-status .head { display: flex; align-items: center; gap: .45rem; font-weight: 700; color: #0F172A; margin-bottom: .25rem; }
.ng-side-status .dot { width: 8px; height: 8px; border-radius: 50%; background: var(--ng-emerald); box-shadow: 0 0 0 3px rgba(5, 150, 105, .12); }
.ng-side-status .line { color: #52637A; }
.ng-side-status code { font-family: 'JetBrains Mono', monospace; font-size: .7rem; background: #fff; border: 1px solid var(--ng-border); border-radius: 4px; padding: 0 .25rem; }
.ng-side-scope {
    background: #FFFBEB; border: 1px solid #FDE68A; color: #92400E;
    border-radius: 10px; padding: .7rem .8rem; font-size: .72rem; line-height: 1.5;
}

/* ------------------------------------------------------------------ headers */
.ng-page-head {
    display: flex; align-items: flex-end; justify-content: space-between; gap: 1rem;
    flex-wrap: wrap; border-bottom: 1px solid var(--ng-border);
    padding-bottom: 1rem; margin: .1rem 0 1.25rem 0;
}
.ng-page-head h1 {
    font-size: 1.72rem; line-height: 1.15; margin: 0 0 .35rem 0;
    color: var(--ng-text); font-weight: 800; letter-spacing: -.025em;
}
.ng-page-head p { margin: 0; color: var(--ng-muted); font-size: .94rem; max-width: 760px; }
.ng-meta { display: flex; flex-wrap: wrap; gap: .4rem; justify-content: flex-end; }
.ng-meta-badge {
    display: inline-flex; align-items: center; gap: .35rem;
    background: #fff; border: 1px solid var(--ng-border); color: #475569;
    border-radius: 999px; padding: .3rem .65rem; font-size: .72rem; font-weight: 600;
    white-space: nowrap;
}
.ng-meta-badge .dot { width: 6px; height: 6px; border-radius: 50%; background: var(--ng-indigo); }

/* --------------------------------------------------------------------- hero */
.ng-hero {
    position: relative; overflow: hidden;
    background: linear-gradient(120deg, #FFFFFF 0%, #F2F6FF 58%, #ECFEFF 100%);
    border: 1px solid #DBEAFE; border-radius: 18px;
    padding: 2.3rem 2.5rem; margin-bottom: 1.4rem;
    box-shadow: 0 10px 30px rgba(37, 99, 235, .06);
}
.ng-hero:after {
    content: ""; position: absolute; right: -90px; top: -110px; width: 330px; height: 330px;
    background: radial-gradient(circle, rgba(37, 99, 235, .12), rgba(37, 99, 235, 0) 66%);
    pointer-events: none;
}
.ng-hero .eyebrow {
    color: var(--ng-indigo-dark); font-weight: 700; letter-spacing: .13em;
    text-transform: uppercase; font-size: .72rem; margin-bottom: .7rem;
}
.ng-hero h1 {
    font-size: 2.35rem; line-height: 1.12; margin: 0 0 .8rem 0;
    font-weight: 800; letter-spacing: -.035em; color: #0B1220;
}
.ng-hero p { color: #475569; font-size: 1rem; max-width: 840px; margin: 0; line-height: 1.6; }
.ng-hero-badges { margin-top: 1.25rem; display: flex; flex-wrap: wrap; gap: .45rem; position: relative; z-index: 1; }
.ng-pill {
    background: #fff; border: 1px solid #DBEAFE; color: #1E40AF;
    border-radius: 999px; padding: .32rem .72rem; font-size: .74rem; font-weight: 600;
}
.ng-pill.soft { background: #EFF6FF; }
.ng-pill.emerald { background: #ECFDF5; border-color: #A7F3D0; color: #047857; }
.ng-pill.amber { background: #FFFBEB; border-color: #FDE68A; color: #92400E; }

/* --------------------------------------------------------------------- KPIs */
.ng-kpi {
    position: relative; overflow: hidden; height: 100%;
    background: var(--ng-surface); border: 1px solid var(--ng-border);
    border-radius: var(--ng-radius); padding: 1rem 1.05rem 1.05rem 1.15rem;
    box-shadow: var(--ng-shadow); min-height: 116px;
}
.ng-kpi:before {
    content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 3px;
    background: var(--accent, var(--ng-indigo));
}
.ng-kpi .label {
    color: #64748B; font-size: .69rem; font-weight: 700; letter-spacing: .08em;
    text-transform: uppercase; margin-bottom: .45rem;
}
.ng-kpi .value {
    color: #0F172A; font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 1.78rem; font-weight: 700; line-height: 1.1;
    font-variant-numeric: tabular-nums; letter-spacing: -.03em;
}
.ng-kpi .value.small { font-size: 1.42rem; }
.ng-kpi .sub { color: #64748B; font-size: .75rem; line-height: 1.45; margin-top: .4rem; }
.ng-kpi .sub .up { color: var(--ng-emerald); font-weight: 700; }
.ng-kpi .sub .down { color: var(--ng-red); font-weight: 700; }

/* ------------------------------------------------------------------- panels */
.ng-panel {
    background: var(--ng-surface); border: 1px solid var(--ng-border);
    border-radius: var(--ng-radius); padding: 1.15rem 1.25rem;
    box-shadow: var(--ng-shadow); height: 100%;
}
.ng-panel h3 { color: var(--ng-text); font-size: .98rem; margin: 0 0 .4rem 0; font-weight: 700; }
.ng-panel p { color: var(--ng-muted); font-size: .86rem; line-height: 1.6; margin: 0; }
.ng-panel .icon {
    display: inline-flex; align-items: center; justify-content: center;
    width: 32px; height: 32px; border-radius: 9px; background: var(--ng-indigo-soft);
    color: var(--ng-indigo-dark); font-size: .95rem; margin-bottom: .65rem;
}
.ng-panel .kicker { color: #64748B; font-size: .68rem; font-weight: 700; letter-spacing: .1em; text-transform: uppercase; }
.ng-panel .big { color: #0F172A; font-family: 'JetBrains Mono', monospace; font-size: 1.35rem; font-weight: 700; }

.ng-step {
    background: #fff; border: 1px solid var(--ng-border); border-radius: var(--ng-radius);
    padding: 1rem 1.05rem; height: 100%; box-shadow: var(--ng-shadow);
}
.ng-step .num {
    display: inline-flex; align-items: center; justify-content: center;
    width: 26px; height: 26px; border-radius: 7px;
    background: var(--ng-indigo-soft); color: var(--ng-indigo-dark);
    font-family: 'JetBrains Mono', monospace; font-size: .75rem; font-weight: 700;
    margin-bottom: .55rem;
}
.ng-step h3 { font-size: .9rem; color: var(--ng-text); margin: 0 0 .3rem 0; }
.ng-step p { font-size: .82rem; color: var(--ng-muted); margin: 0; line-height: 1.5; }

.ng-class {
    background: #fff; border: 1px solid var(--ng-border); border-top: 3px solid var(--class-color);
    border-radius: var(--ng-radius); padding: .9rem 1rem; height: 100%;
    box-shadow: var(--ng-shadow);
}
.ng-class .name { display: flex; align-items: center; gap: .4rem; font-size: .9rem; font-weight: 700; color: #0F172A; margin-bottom: .3rem; }
.ng-class .dot { width: 8px; height: 8px; border-radius: 50%; background: var(--class-color); }
.ng-class p { margin: 0; color: var(--ng-muted); font-size: .78rem; line-height: 1.5; }

/* ------------------------------------------------------------------ sections */
.ng-section { display: flex; align-items: center; gap: .55rem; margin: 1.6rem 0 .7rem 0; }
.ng-section .bar { width: 4px; height: 17px; border-radius: 2px; background: var(--ng-indigo); }
.ng-section h2 { margin: 0; font-size: 1rem; font-weight: 700; color: #0F172A; letter-spacing: -.01em; }
.ng-section .hint { color: #94A3B8; font-size: .76rem; margin-left: auto; }

/* --------------------------------------------------------------------- notes */
.ng-note, .ng-warn, .ng-success {
    border-radius: 10px; padding: .8rem .95rem; font-size: .84rem; line-height: 1.58;
    margin: .2rem 0 .8rem 0;
}
.ng-note { background: #EFF6FF; border: 1px solid #DBEAFE; border-left: 3px solid var(--ng-indigo); color: #1E3A8A; }
.ng-warn { background: #FFFBEB; border: 1px solid #FDE68A; border-left: 3px solid var(--ng-amber); color: #92400E; }
.ng-success { background: #ECFDF5; border: 1px solid #A7F3D0; border-left: 3px solid var(--ng-emerald); color: #065F46; }
.ng-note code, .ng-warn code, .ng-success code {
    font-family: 'JetBrains Mono', monospace; font-size: .76rem;
    background: rgba(255, 255, 255, .7); border: 1px solid rgba(15, 23, 42, .08);
    border-radius: 4px; padding: 0 .22rem;
}
.ng-caption { color: #64748B; font-size: .76rem; line-height: 1.5; }

/* ------------------------------------------------------- Streamlit controls */
div[data-testid="stFileUploader"] {
    background: #fff; border: 1px dashed #BFDBFE; border-radius: 12px;
    padding: .6rem .8rem;
}
div[data-testid="stFileUploader"] section { background: #F8FAFF; border-radius: 8px; }
div[data-testid="stFileUploader"] small { color: #64748B; }

.stButton > button, .stDownloadButton > button {
    border-radius: 8px; font-weight: 600; font-size: .85rem;
    border: 1px solid var(--ng-border); transition: all .15s ease;
}
.stButton > button:hover, .stDownloadButton > button:hover {
    border-color: #BFDBFE; color: var(--ng-indigo-dark); background: #F8FAFF;
}
.stButton > button[kind="primary"] {
    background: var(--ng-indigo); border-color: var(--ng-indigo); color: #fff;
    box-shadow: 0 3px 8px rgba(37, 99, 235, .2);
}
.stButton > button[kind="primary"]:hover {
    background: var(--ng-indigo-dark); border-color: var(--ng-indigo-dark); color: #fff;
}

div[data-baseweb="select"] > div, div[data-baseweb="input"] > div {
    border-radius: 8px; border-color: var(--ng-border); background: #fff;
}
div[data-testid="stExpander"] {
    background: #fff; border: 1px solid var(--ng-border); border-radius: 10px;
    box-shadow: none;
}
div[data-testid="stExpander"] summary { font-weight: 600; font-size: .85rem; color: #334155; }

.stTabs [data-baseweb="tab-list"] { gap: .35rem; border-bottom: 1px solid var(--ng-border); }
.stTabs [data-baseweb="tab"] {
    background: transparent; border: 1px solid transparent; border-radius: 8px 8px 0 0;
    padding: .45rem .9rem; color: #64748B; font-weight: 600; font-size: .85rem;
}
.stTabs [aria-selected="true"] {
    background: #fff !important; color: var(--ng-indigo-dark) !important;
    border-color: var(--ng-border) !important; border-bottom-color: #fff !important;
}

[data-testid="stDataFrame"] {
    background: #fff; border: 1px solid var(--ng-border);
    border-radius: 10px; overflow: hidden;
}
[data-testid="stAlert"] { border-radius: 10px; font-size: .84rem; }
[data-testid="stPlotlyChart"] {
    background: #fff; border: 1px solid var(--ng-border);
    border-radius: 12px; padding: .35rem; box-shadow: var(--ng-shadow);
}
[data-testid="stMetric"] {
    background: #fff; border: 1px solid var(--ng-border); border-radius: 12px;
    padding: .8rem 1rem; box-shadow: var(--ng-shadow);
}
[data-testid="stMetricLabel"] p { color: #64748B; font-size: .76rem; font-weight: 600; }
[data-testid="stMetricValue"] { color: #0F172A; font-family: 'JetBrains Mono', monospace; }

code, pre, .mono { font-family: 'JetBrains Mono', ui-monospace, monospace !important; }
footer { visibility: hidden; }

@media (max-width: 900px) {
    .block-container { padding-left: 1rem; padding-right: 1rem; }
    .ng-hero { padding: 1.5rem 1.3rem; }
    .ng-hero h1 { font-size: 1.8rem; }
    .ng-page-head h1 { font-size: 1.45rem; }
    .ng-meta { justify-content: flex-start; }
    .ng-kpi .value { font-size: 1.5rem; }
}
</style>
"""


# ---------------------------------------------------------------------------
# Composants
# ---------------------------------------------------------------------------


def inject_style() -> None:
    """Injecte la feuille de style une seule fois par rendu."""
    st.markdown(CSS, unsafe_allow_html=True)


def esc(value: object) -> str:
    return html.escape(str(value))


def page_header(
    title: str,
    subtitle: str,
    badges: Sequence[str] | None = None,
) -> None:
    meta = ""
    if badges:
        meta = '<div class="ng-meta">' + "".join(
            f'<span class="ng-meta-badge"><span class="dot"></span>{esc(badge)}</span>'
            for badge in badges
        ) + "</div>"
    st.markdown(
        f"""
        <div class="ng-page-head">
          <div>
            <h1>{esc(title)}</h1>
            <p>{subtitle}</p>
          </div>
          {meta}
        </div>
        """,
        unsafe_allow_html=True,
    )


def hero(eyebrow: str, title: str, text: str, badges: Iterable[str]) -> None:
    chips = "".join(f'<span class="ng-pill">{esc(badge)}</span>' for badge in badges)
    st.markdown(
        f"""
        <div class="ng-hero">
          <div class="eyebrow">{esc(eyebrow)}</div>
          <h1>{title}</h1>
          <p>{text}</p>
          <div class="ng-hero-badges">{chips}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_title(text: str, hint: str | None = None) -> None:
    hint_html = f'<span class="hint">{esc(hint)}</span>' if hint else ""
    st.markdown(
        f'<div class="ng-section"><span class="bar"></span><h2>{esc(text)}</h2>{hint_html}</div>',
        unsafe_allow_html=True,
    )


def note(text: str, warning: bool = False, success: bool = False) -> None:
    css_class = "ng-success" if success else ("ng-warn" if warning else "ng-note")
    st.markdown(f'<div class="{css_class}">{text}</div>', unsafe_allow_html=True)


def kpi_cards(items: Sequence[dict]) -> None:
    """Affiche des cartes KPI dans une grille `st.columns` responsive.

    Chaque élément accepte : label, value, sub (HTML autorisé), tone, small.
    """
    columns = st.columns(len(items), gap="small")
    for column, item in zip(columns, items):
        tone = TONE_COLORS.get(item.get("tone", "indigo"), INDIGO)
        value_class = "value small" if item.get("small") else "value"
        sub = f'<div class="sub">{item["sub"]}</div>' if item.get("sub") else ""
        with column:
            st.markdown(
                f"""
                <div class="ng-kpi" style="--accent:{tone}">
                  <div class="label">{esc(item['label'])}</div>
                  <div class="{value_class}">{item['value']}</div>
                  {sub}
                </div>
                """,
                unsafe_allow_html=True,
            )


def panel(icon: str, title: str, body: str, kicker: str | None = None) -> None:
    kicker_html = f'<div class="kicker">{esc(kicker)}</div>' if kicker else ""
    st.markdown(
        f"""
        <div class="ng-panel">
          <div class="icon">{icon}</div>
          {kicker_html}
          <h3>{esc(title)}</h3>
          <p>{body}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def class_chips() -> str:
    return "".join(
        f'<span class="ng-pill" style="background:{CLASS_COLORS[name]}14;'
        f'border-color:{CLASS_COLORS[name]}33;color:{CLASS_COLORS[name]}">{name}</span>'
        for name in ("Normal", "DoS", "Probe", "R2L", "U2R")
    )


def style_figure(
    figure: go.Figure,
    height: int = 360,
    legend: bool = True,
    margin_top: int = 48,
) -> go.Figure:
    """Applique le thème clair commun aux figures Plotly."""
    figure.update_layout(
        template="plotly_white",
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", size=12, color="#334155"),
        margin=dict(l=12, r=12, t=margin_top, b=12),
        showlegend=legend,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            font=dict(size=11),
            bgcolor="rgba(0,0,0,0)",
        ),
        hoverlabel=dict(bgcolor="#FFFFFF", bordercolor="#E2E8F0", font_color="#0F172A"),
    )
    figure.update_xaxes(
        showgrid=False,
        linecolor="#E2E8F0",
        tickfont=dict(size=11, color="#64748B"),
        title_font=dict(size=11, color="#64748B"),
    )
    figure.update_yaxes(
        gridcolor="#F1F5F9",
        zeroline=False,
        linecolor="rgba(0,0,0,0)",
        tickfont=dict(size=11, color="#64748B"),
        title_font=dict(size=11, color="#64748B"),
    )
    return figure


def render_figure(figure: go.Figure, height: int = 360, legend: bool = True) -> None:
    st.plotly_chart(
        style_figure(figure, height=height, legend=legend),
        use_container_width=True,
        config=PLOTLY_CONFIG,
    )
