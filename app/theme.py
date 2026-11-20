"""NetGuard AI — identité visuelle « security operations console ».

Ce module ne contient que de la présentation : aucune règle métier, aucun
calcul de prédiction ou de métrique. La palette suit automatiquement le thème
général clair / sombre du système (Streamlit « Use system setting ») :

* thème sombre : console NIDS bleu nuit, accents bleu électrique et violet ;
* thème clair : dashboard cyber professionnel, mêmes accents, surfaces claires.

Les composants HTML sont réutilisés par les quatre vues pour garantir une
identité cohérente.
"""

from __future__ import annotations

import html
from typing import Iterable, Sequence

import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Palette sémantique (identique dans les deux thèmes)
# ---------------------------------------------------------------------------

INDIGO = "#3B82F6"
VIOLET = "#8B5CF6"
CYAN = "#22D3EE"
EMERALD = "#10B981"
AMBER = "#F59E0B"
RED = "#F43F5E"

CLASS_COLORS = {
    "Normal": EMERALD,
    "DoS": RED,
    "Probe": AMBER,
    "R2L": "#EC4899",
    "U2R": VIOLET,
}

TONE_COLORS = {
    "indigo": INDIGO,
    "emerald": EMERALD,
    "amber": AMBER,
    "red": RED,
    "violet": VIOLET,
    "slate": "#64748B",
}

PLOTLY_CONFIG = {"displayModeBar": False, "responsive": True}


# ---------------------------------------------------------------------------
# Thème actif
# ---------------------------------------------------------------------------


def current_theme() -> str:
    """Retourne ``"dark"`` ou ``"light"`` selon le thème général actif.

    Le choix de la console prime sur le thème du navigateur. Avant le premier
    choix, Streamlit fournit le thème inféré du navigateur.
    """
    selected_theme = st.session_state.get("ng_theme_mode")
    if selected_theme in {"light", "dark"}:
        return selected_theme

    try:
        theme_type = st.context.theme.type
    except Exception:  # noqa: BLE001 - contexte absent (tests, prévisualisation)
        theme_type = None
    return "light" if theme_type == "light" else "dark"


def chart_palette() -> dict[str, str]:
    """Couleurs Plotly cohérentes avec le thème général actif."""
    if current_theme() == "light":
        return {
            "surface": "#FFFFFF",
            "text": "#0B1220",
            "muted": "#52637A",
            "grid": "rgba(100, 116, 139, .16)",
            "zero": "rgba(100, 116, 139, .25)",
            "heat_scale": ["#F4F7FD", "#DBEAFE", "#BFDBFE", "#93C5FD"],
            "heat_text": "#0B1220",
            "annotation": "#334155",
        }
    return {
        "surface": "#0B1222",
        "text": "#E9F0FB",
        "muted": "#93A6C4",
        "grid": "rgba(148, 163, 184, .14)",
        "zero": "rgba(148, 163, 184, .22)",
        "heat_scale": ["#0B1222", "#12294A", "#173B6B", "#1D4ED8"],
        "heat_text": "#F8FBFF",
        "annotation": "#C7D6EA",
    }


# ---------------------------------------------------------------------------
# Feuille de style
# ---------------------------------------------------------------------------

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Rajdhani:wght@500;600;700&family=JetBrains+Mono:wght@500;600;700&display=swap');

:root {
    color-scheme: light;
    --ng-canvas: #F2F5FB;
    --ng-surface: #FFFFFF;
    --ng-surface-2: #F8FAFF;
    --ng-elevated: #FFFFFF;
    --ng-border: #DDE5F2;
    --ng-border-strong: #C7D5E8;
    --ng-text: #0B1220;
    --ng-muted: #52637A;
    --ng-faint: #78899F;
    --ng-accent: #2563EB;
    --ng-accent-2: #7C3AED;
    --ng-cyan: #0891B2;
    --ng-ok: #059669;
    --ng-warn: #B45309;
    --ng-danger: #DC2626;
    --ng-accent-soft: rgba(37, 99, 235, .08);
    --ng-accent-line: rgba(37, 99, 235, .35);
    --ng-sidebar: #FFFFFF;
    --ng-sidebar-glow: rgba(37, 99, 235, .05);
    --ng-nav-text: #475569;
    --ng-nav-active: #1D4ED8;
    --ng-shadow: 0 1px 2px rgba(15, 23, 42, .05);
    --ng-shadow-lg: 0 14px 34px rgba(15, 23, 42, .07);
    --ng-halo-1: rgba(37, 99, 235, .07);
    --ng-halo-2: rgba(124, 58, 237, .05);
    --ng-hero-bg: linear-gradient(125deg, rgba(37, 99, 235, .07), rgba(124, 58, 237, .045) 62%, rgba(8, 145, 178, .05));
    --ng-hero-title: #0B1220;
    --ng-code-bg: #F4F7FD;
    --ng-ok-soft: rgba(5, 150, 105, .08);
    --ng-warn-soft: rgba(180, 83, 9, .08);
    --ng-danger-soft: rgba(220, 38, 38, .07);
    --ng-radius: 12px;
    --ng-radius-lg: 16px;
}

@media __NG_DARK_THEME_QUERY__ {
    :root {
        color-scheme: dark;
        --ng-canvas: #050914;
        --ng-surface: #0B1222;
        --ng-surface-2: #0E1729;
        --ng-elevated: #111C31;
        --ng-border: #1B2A44;
        --ng-border-strong: #26395C;
        --ng-text: #E9F0FB;
        --ng-muted: #93A6C4;
        --ng-faint: #6C7F9C;
        --ng-accent: #3B82F6;
        --ng-accent-2: #8B5CF6;
        --ng-cyan: #22D3EE;
        --ng-ok: #34D399;
        --ng-warn: #FBBF24;
        --ng-danger: #FB7185;
        --ng-accent-soft: rgba(59, 130, 246, .12);
        --ng-accent-line: rgba(56, 189, 248, .42);
        --ng-sidebar: #070D1B;
        --ng-sidebar-glow: rgba(56, 189, 248, .07);
        --ng-nav-text: #9DB0CC;
        --ng-nav-active: #BFDBFE;
        --ng-shadow: 0 1px 0 rgba(255, 255, 255, .025);
        --ng-shadow-lg: 0 20px 46px rgba(0, 0, 0, .45);
        --ng-halo-1: rgba(59, 130, 246, .14);
        --ng-halo-2: rgba(139, 92, 246, .10);
        --ng-hero-bg: linear-gradient(125deg, rgba(59, 130, 246, .14), rgba(139, 92, 246, .09) 58%, rgba(34, 211, 238, .07));
        --ng-hero-title: #F4F8FF;
        --ng-code-bg: #0A1222;
        --ng-ok-soft: rgba(52, 211, 153, .10);
        --ng-warn-soft: rgba(251, 191, 36, .10);
        --ng-danger-soft: rgba(251, 113, 133, .10);
    }
}

html, body, [data-testid="stAppViewContainer"], [data-testid="stSidebar"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    font-size: 16px;
}

body, .stApp {
    background: var(--ng-canvas);
    color: var(--ng-text);
}
.stApp {
    background-image:
        radial-gradient(1100px 520px at 12% -12%, var(--ng-halo-1), transparent 62%),
        radial-gradient(900px 480px at 92% -6%, var(--ng-halo-2), transparent 58%);
}
[data-testid="stHeader"] {
    background: transparent;
    border-bottom: 0;
}
[data-testid="stToolbar"] { right: .6rem; }
.block-container {
    padding-top: 1.9rem;
    padding-bottom: 3.5rem;
    max-width: 1360px;
}
p, li, .stMarkdown p { font-size: .98rem; line-height: 1.62; }
.stCaption, [data-testid="stCaptionContainer"] p {
    font-size: .84rem !important;
    color: var(--ng-muted);
}
[data-testid="stWidgetLabel"] p {
    font-size: .9rem;
    font-weight: 600;
    color: var(--ng-muted) !important;
}
[data-testid="stMarkdownContainer"] :is(h2, h3, h4, h5, h6) {
    color: var(--ng-text) !important;
}

/* ------------------------------------------------------------------ sidebar */
[data-testid="stSidebar"] {
    background: var(--ng-sidebar);
    border-right: 1px solid var(--ng-border);
    box-shadow: inset -1px 0 0 var(--ng-sidebar-glow);
}
[data-testid="stSidebar"] [data-testid="stSidebarContent"] {
    padding: .9rem 1.05rem 1.4rem 1.05rem;
}
[data-testid="stSidebar"] * { color: var(--ng-text); }
[data-testid="stSidebar"] hr {
    border: 0;
    border-top: 1px solid var(--ng-border);
    margin: 1.15rem 0;
}

.ng-brand {
    display: flex;
    align-items: center;
    gap: .85rem;
    padding: .3rem .1rem 1.15rem .1rem;
}
.ng-brand-mark {
    position: relative;
    width: 46px;
    height: 46px;
    flex: 0 0 46px;
    border-radius: 12px;
    background: linear-gradient(140deg, #2563EB 0%, #4F46E5 52%, #8B5CF6 100%);
    color: #FFFFFF;
    display: flex;
    align-items: center;
    justify-content: center;
    font-family: 'Rajdhani', sans-serif;
    font-weight: 700;
    font-size: 1.12rem;
    letter-spacing: .01em;
    box-shadow: 0 0 0 1px rgba(139, 92, 246, .35), 0 10px 24px rgba(37, 99, 235, .28);
}
.ng-brand-mark:after {
    content: "";
    position: absolute;
    inset: 5px;
    border-radius: 8px;
    border: 1px solid rgba(255, 255, 255, .22);
    pointer-events: none;
}
.ng-brand-name {
    font-family: 'Rajdhani', sans-serif;
    font-weight: 700;
    font-size: 1.48rem;
    line-height: 1;
    letter-spacing: .015em;
    color: var(--ng-text);
}
.ng-brand-name span { color: var(--ng-accent); }
.ng-brand-sub {
    font-size: .59rem;
    letter-spacing: .13em;
    text-transform: uppercase;
    color: var(--ng-faint);
    margin-top: .42rem;
    font-weight: 600;
    white-space: nowrap;
}

.ng-nav-label {
    font-size: .68rem;
    letter-spacing: .18em;
    text-transform: uppercase;
    color: var(--ng-faint);
    font-weight: 700;
    padding: .1rem .15rem .45rem .15rem;
}

[data-testid="stSidebar"] [role="radiogroup"] { gap: .28rem; }
[data-testid="stSidebar"] [role="radiogroup"] > label {
    display: flex;
    align-items: center;
    width: 100%;
    margin: 0;
    padding: .64rem .75rem;
    border-radius: 10px;
    border: 1px solid transparent;
    background: transparent;
    transition: background .16s ease, border-color .16s ease, box-shadow .16s ease;
    cursor: pointer;
}
[data-testid="stSidebar"] [role="radiogroup"] > label:hover {
    background: var(--ng-accent-soft);
    border-color: var(--ng-border);
}
[data-testid="stSidebar"] [role="radiogroup"] > label:has(input:checked) {
    background: linear-gradient(90deg, var(--ng-accent-soft), rgba(139, 92, 246, .07));
    border-color: var(--ng-accent-line);
    box-shadow: inset 3px 0 0 var(--ng-accent), 0 0 22px rgba(59, 130, 246, .10);
}
[data-testid="stSidebar"] [role="radiogroup"] > label p {
    font-size: .93rem;
    font-weight: 600;
    color: var(--ng-nav-text);
    letter-spacing: .005em;
}
[data-testid="stSidebar"] [role="radiogroup"] > label:has(input:checked) p {
    color: var(--ng-nav-active);
    font-weight: 700;
}
[data-testid="stSidebar"] [role="radiogroup"] > label > div:first-child {
    display: none;
}

.ng-side-status {
    background: var(--ng-surface-2);
    border: 1px solid var(--ng-border);
    border-radius: 12px;
    padding: .85rem .95rem;
    font-size: .82rem;
    line-height: 1.5;
}
.ng-status-head {
    display: flex;
    align-items: center;
    gap: .5rem;
    font-size: .66rem;
    letter-spacing: .16em;
    text-transform: uppercase;
    font-weight: 700;
    color: var(--ng-ok);
    margin-bottom: .65rem;
}
.ng-pulse {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--ng-ok);
    box-shadow: 0 0 0 4px rgba(16, 185, 129, .12);
    animation: ng-pulse 2.6s ease-in-out infinite;
}
@keyframes ng-pulse {
    0%, 100% { box-shadow: 0 0 0 3px rgba(16, 185, 129, .10); }
    50% { box-shadow: 0 0 0 6px rgba(16, 185, 129, .04); }
}
.ng-status-row {
    display: flex;
    justify-content: space-between;
    gap: .75rem;
    padding: .2rem 0;
    color: var(--ng-muted);
    font-size: .8rem;
}
.ng-status-row b {
    color: var(--ng-text);
    font-weight: 600;
    font-family: 'JetBrains Mono', monospace;
    font-size: .74rem;
}
.ng-status-file {
    margin-top: .6rem;
    padding-top: .55rem;
    border-top: 1px solid var(--ng-border);
    font-family: 'JetBrains Mono', monospace;
    font-size: .68rem;
    color: var(--ng-faint);
    line-height: 1.55;
}

.ng-side-scope {
    background: var(--ng-warn-soft);
    border: 1px solid rgba(245, 158, 11, .30);
    border-left: 3px solid var(--ng-warn);
    border-radius: 10px;
    padding: .75rem .85rem;
    font-size: .79rem;
    line-height: 1.55;
    color: var(--ng-text);
}
.ng-side-scope strong {
    display: block;
    font-size: .66rem;
    letter-spacing: .14em;
    text-transform: uppercase;
    color: var(--ng-warn);
    margin-bottom: .25rem;
}
.ng-side-foot {
    font-size: .74rem;
    color: var(--ng-faint);
    line-height: 1.5;
}

/* ------------------------------------------------------------------ headers */
.ng-page-head {
    display: flex;
    align-items: flex-end;
    justify-content: space-between;
    gap: 1rem;
    flex-wrap: wrap;
    border-bottom: 1px solid var(--ng-border);
    padding-bottom: 1.15rem;
    margin: .1rem 0 1.5rem 0;
}
.ng-page-kicker {
    font-size: .68rem;
    letter-spacing: .2em;
    text-transform: uppercase;
    color: var(--ng-accent);
    font-weight: 700;
    margin-bottom: .4rem;
}
.ng-page-head h1 {
    font-family: 'Rajdhani', sans-serif;
    font-size: 2.2rem;
    line-height: 1.05;
    margin: 0 0 .45rem 0;
    color: var(--ng-text);
    font-weight: 700;
    letter-spacing: .005em;
}
.ng-page-head p {
    margin: 0;
    color: var(--ng-muted);
    font-size: .98rem;
    max-width: 780px;
}
.ng-meta { display: flex; flex-wrap: wrap; gap: .4rem; justify-content: flex-end; }
.ng-meta-badge {
    display: inline-flex;
    align-items: center;
    gap: .4rem;
    background: var(--ng-surface);
    border: 1px solid var(--ng-border);
    color: var(--ng-muted);
    border-radius: 999px;
    padding: .32rem .72rem;
    font-size: .74rem;
    font-weight: 600;
    white-space: nowrap;
}
.ng-meta-badge .dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: var(--ng-accent);
    box-shadow: 0 0 8px var(--ng-accent);
}

/* --------------------------------------------------------------------- hero */
.ng-hero {
    position: relative;
    overflow: hidden;
    background: var(--ng-hero-bg), var(--ng-surface);
    border: 1px solid var(--ng-border-strong);
    border-radius: 18px;
    padding: 2.4rem 2.6rem;
    margin-bottom: 1.5rem;
    box-shadow: var(--ng-shadow-lg);
}
.ng-hero:before {
    content: "";
    position: absolute;
    inset: 0;
    background-image:
        linear-gradient(rgba(148, 163, 184, .06) 1px, transparent 1px),
        linear-gradient(90deg, rgba(148, 163, 184, .06) 1px, transparent 1px);
    background-size: 44px 44px;
    mask-image: linear-gradient(120deg, rgba(0, 0, 0, .8), transparent 62%);
    pointer-events: none;
}
.ng-hero:after {
    content: "";
    position: absolute;
    right: -120px;
    top: -150px;
    width: 420px;
    height: 420px;
    background: radial-gradient(circle, rgba(59, 130, 246, .20), rgba(59, 130, 246, 0) 64%);
    pointer-events: none;
}
.ng-hero .eyebrow {
    position: relative;
    color: var(--ng-accent);
    font-weight: 700;
    letter-spacing: .18em;
    text-transform: uppercase;
    font-size: .7rem;
    margin-bottom: .75rem;
}
.ng-hero h1 {
    position: relative;
    font-family: 'Rajdhani', sans-serif;
    font-size: 2.8rem;
    line-height: 1.04;
    margin: 0 0 .85rem 0;
    font-weight: 700;
    letter-spacing: .005em;
    color: var(--ng-hero-title);
}
.ng-hero p {
    position: relative;
    color: var(--ng-muted);
    font-size: 1.02rem;
    max-width: 860px;
    margin: 0;
    line-height: 1.62;
}
.ng-hero-badges {
    position: relative;
    margin-top: 1.35rem;
    display: flex;
    flex-wrap: wrap;
    gap: .45rem;
    z-index: 1;
}
.ng-pill {
    background: var(--ng-surface);
    border: 1px solid var(--ng-border-strong);
    color: var(--ng-text);
    border-radius: 999px;
    padding: .34rem .76rem;
    font-size: .77rem;
    font-weight: 600;
}
.ng-pill.accent {
    background: var(--ng-accent-soft);
    border-color: var(--ng-accent-line);
    color: var(--ng-accent);
}
.ng-pill.emerald {
    background: var(--ng-ok-soft);
    border-color: rgba(16, 185, 129, .35);
    color: var(--ng-ok);
}
.ng-pill.amber {
    background: var(--ng-warn-soft);
    border-color: rgba(245, 158, 11, .35);
    color: var(--ng-warn);
}

/* --------------------------------------------------------------------- KPIs */
.ng-kpi {
    position: relative;
    overflow: hidden;
    height: 100%;
    background: var(--ng-surface);
    border: 1px solid var(--ng-border);
    border-radius: var(--ng-radius);
    padding: 1.1rem 1.1rem 1.15rem 1.2rem;
    box-shadow: var(--ng-shadow);
    min-height: 128px;
    transition: border-color .16s ease, transform .16s ease, box-shadow .16s ease;
}
.ng-kpi:hover {
    border-color: var(--ng-border-strong);
    transform: translateY(-1px);
    box-shadow: var(--ng-shadow-lg);
}
.ng-kpi:before {
    content: "";
    position: absolute;
    left: 0;
    top: 0;
    bottom: 0;
    width: 3px;
    background: linear-gradient(180deg, var(--accent, var(--ng-accent)), transparent 92%);
}
.ng-kpi:after {
    content: "";
    position: absolute;
    right: -40px;
    top: -50px;
    width: 120px;
    height: 120px;
    border-radius: 50%;
    background: radial-gradient(circle, var(--accent, var(--ng-accent)), transparent 68%);
    opacity: .07;
    pointer-events: none;
}
.ng-kpi .label {
    color: var(--ng-faint);
    font-size: .7rem;
    font-weight: 700;
    letter-spacing: .12em;
    text-transform: uppercase;
    margin-bottom: .55rem;
}
.ng-kpi .value {
    color: var(--ng-text);
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 1.95rem;
    font-weight: 700;
    line-height: 1.05;
    font-variant-numeric: tabular-nums;
    letter-spacing: -.04em;
}
.ng-kpi .value.small { font-size: 1.5rem; }
.ng-kpi .sub {
    color: var(--ng-muted);
    font-size: .8rem;
    line-height: 1.48;
    margin-top: .48rem;
}
.ng-kpi .sub .up { color: var(--ng-ok); font-weight: 700; }
.ng-kpi .sub .down { color: var(--ng-danger); font-weight: 700; }

/* ------------------------------------------------------------------- panels */
.ng-panel {
    background: linear-gradient(180deg, var(--ng-surface) 0%, var(--ng-surface-2) 100%);
    border: 1px solid var(--ng-border);
    border-radius: var(--ng-radius);
    padding: 1.2rem 1.3rem;
    box-shadow: var(--ng-shadow);
    height: 100%;
}
.ng-panel h3 {
    color: var(--ng-text);
    font-family: 'Rajdhani', sans-serif;
    font-size: 1.12rem;
    margin: 0 0 .45rem 0;
    font-weight: 700;
    letter-spacing: .01em;
}
.ng-panel p {
    color: var(--ng-muted);
    font-size: .9rem;
    line-height: 1.62;
    margin: 0;
}
.ng-panel .icon {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 34px;
    height: 34px;
    border-radius: 9px;
    background: var(--ng-accent-soft);
    border: 1px solid var(--ng-accent-line);
    color: var(--ng-accent);
    font-size: 1rem;
    margin-bottom: .7rem;
}
.ng-panel .kicker,
.ng-kicker {
    color: var(--ng-faint);
    font-size: .66rem;
    font-weight: 700;
    letter-spacing: .16em;
    text-transform: uppercase;
}
.ng-panel .big {
    color: var(--ng-text);
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.42rem;
    font-weight: 700;
}

.ng-step {
    background: var(--ng-surface);
    border: 1px solid var(--ng-border);
    border-radius: var(--ng-radius);
    padding: 1.05rem 1.1rem;
    height: 100%;
    box-shadow: var(--ng-shadow);
}
.ng-step .num {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 28px;
    height: 28px;
    border-radius: 8px;
    background: var(--ng-accent-soft);
    border: 1px solid var(--ng-accent-line);
    color: var(--ng-accent);
    font-family: 'JetBrains Mono', monospace;
    font-size: .76rem;
    font-weight: 700;
    margin-bottom: .6rem;
}
.ng-step h3 {
    font-family: 'Rajdhani', sans-serif;
    font-size: 1.05rem;
    color: var(--ng-text);
    margin: 0 0 .3rem 0;
    font-weight: 700;
}
.ng-step p { font-size: .86rem; color: var(--ng-muted); margin: 0; line-height: 1.55; }

.ng-class {
    background: var(--ng-surface);
    border: 1px solid var(--ng-border);
    border-top: 3px solid var(--class-color);
    border-radius: var(--ng-radius);
    padding: .95rem 1rem;
    height: 100%;
    box-shadow: var(--ng-shadow);
}
.ng-class .name {
    display: flex;
    align-items: center;
    gap: .45rem;
    font-size: .95rem;
    font-weight: 700;
    color: var(--ng-text);
    margin-bottom: .3rem;
}
.ng-class .dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--class-color);
    box-shadow: 0 0 10px var(--class-color);
}
.ng-class p { margin: 0; color: var(--ng-muted); font-size: .82rem; line-height: 1.55; }

/* ------------------------------------------------------------------ sections */
.ng-section {
    display: flex;
    align-items: center;
    gap: .6rem;
    margin: 1.85rem 0 .85rem 0;
}
.ng-section .bar {
    width: 4px;
    height: 19px;
    border-radius: 2px;
    background: linear-gradient(180deg, var(--ng-accent), var(--ng-accent-2));
    box-shadow: 0 0 12px var(--ng-accent-line);
}
.ng-section h2 {
    margin: 0;
    font-family: 'Rajdhani', sans-serif;
    font-size: 1.16rem;
    font-weight: 700;
    color: var(--ng-text);
    letter-spacing: .045em;
    text-transform: uppercase;
}
.ng-section .hint { color: var(--ng-faint); font-size: .8rem; margin-left: auto; }

/* --------------------------------------------------------------------- notes */
.ng-note, .ng-warn, .ng-success {
    border-radius: 10px;
    padding: .9rem 1rem;
    font-size: .9rem;
    line-height: 1.6;
    margin: .2rem 0 .9rem 0;
    border: 1px solid var(--ng-border);
    border-left: 3px solid var(--ng-accent);
    background: var(--ng-accent-soft);
    color: var(--ng-text);
}
.ng-warn {
    background: var(--ng-warn-soft);
    border-color: rgba(245, 158, 11, .28);
    border-left-color: var(--ng-warn);
}
.ng-success {
    background: var(--ng-ok-soft);
    border-color: rgba(16, 185, 129, .30);
    border-left-color: var(--ng-ok);
}
.ng-note code, .ng-warn code, .ng-success code {
    font-family: 'JetBrains Mono', monospace;
    font-size: .8rem;
    background: var(--ng-code-bg);
    border: 1px solid var(--ng-border);
    border-radius: 4px;
    padding: 0 .25rem;
}
.ng-caption { color: var(--ng-muted); font-size: .82rem; line-height: 1.55; }

/* ------------------------------------------------------- Streamlit controls */
div[data-testid="stFileUploader"] {
    background: var(--ng-surface);
    border: 1px dashed var(--ng-border-strong);
    border-radius: 12px;
    padding: .65rem .85rem;
}
div[data-testid="stFileUploader"] section {
    background: var(--ng-surface-2);
    border-radius: 8px;
}
div[data-testid="stFileUploader"] small { color: var(--ng-muted); }

.stButton > button, .stDownloadButton > button {
    border-radius: 8px;
    font-weight: 600;
    font-size: .88rem;
    border: 1px solid var(--ng-border-strong);
    background: var(--ng-surface);
    color: var(--ng-text);
    transition: all .16s ease;
}
.stButton > button:hover, .stDownloadButton > button:hover {
    border-color: var(--ng-accent-line);
    color: var(--ng-accent);
    background: var(--ng-accent-soft);
}
.stButton > button[kind="primary"] {
    background: linear-gradient(120deg, #2563EB, #4F46E5 62%, #7C3AED);
    border-color: rgba(99, 102, 241, .75);
    color: #FFFFFF;
    box-shadow: 0 6px 20px rgba(37, 99, 235, .24);
}
.stButton > button[kind="primary"]:hover {
    background: linear-gradient(120deg, #1D4ED8, #4338CA 62%, #6D28D9);
    border-color: rgba(99, 102, 241, .9);
    color: #FFFFFF;
}
[data-testid="stButton"] button:not([kind="primary"]),
[data-testid="stDownloadButton"] button,
[data-testid="stFileUploader"] button {
    color: var(--ng-text) !important;
    background-color: var(--ng-surface) !important;
    border-color: var(--ng-border-strong) !important;
}
[data-testid="stButton"] button:not([kind="primary"]):hover,
[data-testid="stDownloadButton"] button:hover,
[data-testid="stFileUploader"] button:hover {
    color: var(--ng-accent) !important;
    background-color: var(--ng-accent-soft) !important;
    border-color: var(--ng-accent-line) !important;
}

div[data-baseweb="select"] > div, div[data-baseweb="input"] > div {
    border-radius: 8px;
    border-color: var(--ng-border-strong);
    background: var(--ng-surface);
    color: var(--ng-text);
}
div[data-baseweb="popover"] div[role="listbox"] {
    background: var(--ng-elevated);
    border: 1px solid var(--ng-border);
}
div[data-baseweb="popover"] [role="option"],
div[data-baseweb="popover"] [data-baseweb="option"] {
    background: var(--ng-elevated) !important;
    color: var(--ng-text) !important;
}
div[data-baseweb="popover"] [role="option"]:hover,
div[data-baseweb="popover"] [data-baseweb="option"]:hover,
div[data-baseweb="popover"] [role="option"][aria-selected="true"] {
    background: var(--ng-accent-soft) !important;
    color: var(--ng-text) !important;
}
div[data-testid="stExpander"] {
    background: var(--ng-surface);
    border: 1px solid var(--ng-border);
    border-radius: 10px;
    box-shadow: none;
}
div[data-testid="stExpander"] summary {
    font-weight: 600;
    font-size: .9rem;
    color: var(--ng-text);
    background: var(--ng-surface) !important;
}
div[data-testid="stExpander"] [data-testid="stExpanderDetails"] {
    background: var(--ng-surface) !important;
}
[data-testid="stCode"] pre,
[data-testid="stCode"] code {
    color: var(--ng-text) !important;
}
[data-testid="stJson"] .react-json-view {
    background: var(--ng-surface) !important;
    color: var(--ng-text) !important;
}

.stTabs [data-baseweb="tab-list"] {
    gap: .4rem;
    border-bottom: 1px solid var(--ng-border);
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    border: 1px solid transparent;
    border-radius: 8px 8px 0 0;
    padding: .5rem .95rem;
    color: var(--ng-muted);
    font-weight: 600;
    font-size: .9rem;
}
.stTabs [aria-selected="true"] {
    background: var(--ng-surface) !important;
    color: var(--ng-accent) !important;
    border-color: var(--ng-border) !important;
    border-bottom-color: var(--ng-surface) !important;
    box-shadow: inset 0 -2px 0 var(--ng-accent);
}

/* Neutralise les accents natifs (rouge) au profit de l'identité NetGuard. */
[data-baseweb="tab-highlight"] { background: var(--ng-accent) !important; }
label[data-baseweb="radio"] > div:first-of-type {
    background: transparent !important;
    border: 1px solid var(--ng-border-strong) !important;
    box-shadow: none !important;
}
label[data-baseweb="radio"],
label[data-baseweb="radio"] p,
label[data-baseweb="radio"] span {
    color: var(--ng-text) !important;
}
label[data-baseweb="radio"]:has(input:checked) > div:first-of-type {
    background: var(--ng-accent) !important;
    border-color: var(--ng-accent) !important;
    box-shadow: 0 0 10px var(--ng-accent-line) !important;
}
label[data-baseweb="radio"]:has(input:checked) > div:first-of-type > div {
    background: #FFFFFF !important;
}
a, a:visited { color: var(--ng-accent); }
::selection { background: rgba(59, 130, 246, .30); }
div[data-baseweb="select"] > div:focus-within,
div[data-baseweb="input"] > div:focus-within,
div[data-testid="stFileUploader"]:focus-within {
    border-color: var(--ng-accent) !important;
    box-shadow: 0 0 0 1px var(--ng-accent-line) !important;
}

[data-testid="stDataFrame"] {
    background: var(--ng-surface);
    border: 1px solid var(--ng-border);
    border-radius: 10px;
    overflow: hidden;
    color: var(--ng-text);
}
[data-testid="stDataFrame"],
[data-testid="stDataFrame"] [style*="--gdg-bg-cell"] {
    --gdg-accent-color: var(--ng-accent) !important;
    --gdg-accent-fg: #FFFFFF !important;
    --gdg-accent-light: var(--ng-accent-soft) !important;
    --gdg-text-dark: var(--ng-text) !important;
    --gdg-text-medium: var(--ng-muted) !important;
    --gdg-text-light: var(--ng-faint) !important;
    --gdg-text-bubble: var(--ng-muted) !important;
    --gdg-bg-icon-header: var(--ng-faint) !important;
    --gdg-fg-icon-header: var(--ng-text) !important;
    --gdg-text-header: var(--ng-muted) !important;
    --gdg-text-group-header: var(--ng-muted) !important;
    --gdg-bg-group-header: var(--ng-surface-2) !important;
    --gdg-bg-group-header-hovered: var(--ng-accent-soft) !important;
    --gdg-text-header-selected: #FFFFFF !important;
    --gdg-bg-cell: var(--ng-surface) !important;
    --gdg-bg-cell-medium: var(--ng-surface-2) !important;
    --gdg-bg-header: var(--ng-surface-2) !important;
    --gdg-bg-header-has-focus: var(--ng-accent-soft) !important;
    --gdg-bg-header-hovered: var(--ng-accent-soft) !important;
    --gdg-bg-bubble: var(--ng-elevated) !important;
    --gdg-bg-bubble-selected: var(--ng-accent-soft) !important;
    --gdg-border-color: var(--ng-border) !important;
    --gdg-horizontal-border-color: var(--ng-border) !important;
    --gdg-drilldown-border: var(--ng-border-strong) !important;
    --gdg-link-color: var(--ng-accent) !important;
    --gdg-bg-search-result: var(--ng-accent-soft) !important;
    --gdg-resize-indicator-color: var(--ng-accent) !important;
}
[data-testid="stAlert"] { border-radius: 10px; font-size: .88rem; }
[data-testid="stPlotlyChart"] {
    background: var(--ng-surface);
    border: 1px solid var(--ng-border);
    border-radius: 12px;
    padding: .4rem;
    box-shadow: var(--ng-shadow);
}
[data-testid="stMetric"] {
    background: var(--ng-surface);
    border: 1px solid var(--ng-border);
    border-radius: 12px;
    padding: .85rem 1rem;
    box-shadow: var(--ng-shadow);
}
[data-testid="stMetricLabel"] p { color: var(--ng-muted); font-size: .8rem; font-weight: 600; }
[data-testid="stMetricValue"] { color: var(--ng-text); font-family: 'JetBrains Mono', monospace; }

code, pre, .mono { font-family: 'JetBrains Mono', ui-monospace, monospace !important; }
pre { background: var(--ng-code-bg) !important; border: 1px solid var(--ng-border); }
footer { visibility: hidden; }

@media (max-width: 1100px) {
    .ng-hero { padding: 2rem 1.8rem; }
    .ng-hero h1 { font-size: 2.35rem; }
    .ng-page-head h1 { font-size: 1.95rem; }
}
@media (max-width: 900px) {
    .block-container { padding-left: 1rem; padding-right: 1rem; }
    .ng-hero { padding: 1.55rem 1.25rem; }
    .ng-hero h1 { font-size: 1.95rem; }
    .ng-page-head h1 { font-size: 1.65rem; }
    .ng-meta { justify-content: flex-start; }
    .ng-kpi .value { font-size: 1.62rem; }
}
@media (max-width: 640px) {
    .ng-hero h1 { font-size: 1.7rem; }
    .ng-page-head { align-items: flex-start; }
    .ng-section .hint { display: none; }
    .ng-kpi { min-height: 112px; }
    .stTabs [data-baseweb="tab-list"] { flex-wrap: wrap; }
    .stTabs [data-baseweb="tab"] {
        flex: 1 1 9rem;
        min-width: 0;
        white-space: normal;
    }
}
</style>
"""


# ---------------------------------------------------------------------------
# Composants
# ---------------------------------------------------------------------------


def inject_style() -> None:
    """Injecte la feuille de style une seule fois par rendu."""
    dark_theme_query = "all" if current_theme() == "dark" else "not all"
    css = CSS.replace("__NG_DARK_THEME_QUERY__", dark_theme_query)
    st.markdown(css, unsafe_allow_html=True)


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
            <div class="ng-page-kicker">NetGuard AI · Security Operations</div>
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
        f'border-color:{CLASS_COLORS[name]}55;color:{CLASS_COLORS[name]}">{name}</span>'
        for name in ("Normal", "DoS", "Probe", "R2L", "U2R")
    )


def style_figure(
    figure: go.Figure,
    height: int = 360,
    legend: bool = True,
    margin_top: int = 48,
) -> go.Figure:
    """Applique le thème actif commun aux figures Plotly."""
    colors = chart_palette()
    figure.update_layout(
        template="plotly_white" if current_theme() == "light" else "plotly_dark",
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", size=12, color=colors["muted"]),
        title=dict(
            x=0,
            xanchor="left",
            y=0.98,
            yanchor="top",
            font=dict(family="Rajdhani, Inter, sans-serif", size=15, color=colors["text"]),
        ),
        margin=dict(l=12, r=12, t=margin_top, b=12),
        showlegend=legend,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=11, color=colors["muted"]),
            bgcolor="rgba(0,0,0,0)",
        ),
        hoverlabel=dict(
            bgcolor=colors["surface"],
            bordercolor=colors["grid"],
            font=dict(color=colors["text"], family="Inter, sans-serif"),
        ),
    )
    figure.update_xaxes(
        showgrid=False,
        linecolor=colors["grid"],
        tickfont=dict(size=11, color=colors["muted"]),
        title_font=dict(size=11, color=colors["muted"]),
    )
    figure.update_yaxes(
        gridcolor=colors["grid"],
        zeroline=False,
        linecolor="rgba(0,0,0,0)",
        tickfont=dict(size=11, color=colors["muted"]),
        title_font=dict(size=11, color=colors["muted"]),
    )
    return figure


def render_figure(figure: go.Figure, height: int = 360, legend: bool = True) -> None:
    st.plotly_chart(
        style_figure(figure, height=height, legend=legend),
        use_container_width=True,
        config=PLOTLY_CONFIG,
    )
