"""Attack Analytics: statistics derived from the current predictions."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from nids.analytics import build_attack_statistics
from nids.errors import InputValidationError
from theme import (
    AMBER,
    CLASS_COLORS,
    EMERALD,
    RED,
    chart_palette,
    esc,
    kpi_cards,
    note,
    page_header,
    render_figure,
    section_title,
)
from views.common import get_analysis, results_display

_CONFIDENCE_BINS = [-0.01, 40, 50, 60, 70, 80, 90, 100.01]
_CONFIDENCE_LABELS = ["< 40", "40–50", "50–60", "60–70", "70–80", "80–90", "90–100"]


def _empty_state() -> None:
    page_header(
        "Attack Analytics",
        "Analyse statistique des flux classifiés pendant la session.",
        ["Aucun lot actif", "Prédictions de session"],
    )
    note(
        "Aucune analyse disponible. Importez d'abord un fichier CSV dans la page "
        "<strong>Analyse</strong> pour alimenter ce tableau de bord.",
        warning=True,
    )


def _distribution_chart(distribution: pd.DataFrame) -> None:
    attacks = distribution[distribution["category"] != "Normal"]
    attacks = attacks[attacks["count"] > 0]
    if attacks.empty:
        st.info("Aucune attaque détectée dans ce lot : le graphique de répartition est vide.")
        return
    order = [name for name in ("DoS", "Probe", "R2L", "U2R") if name in set(attacks["category"])]
    figure = px.bar(
        attacks,
        x="count",
        y="category",
        orientation="h",
        color="category",
        color_discrete_map=CLASS_COLORS,
        text="count",
        category_orders={"category": order},
        labels={"category": "", "count": "Nombre de flux"},
        title="Catégories d'attaques détectées",
    )
    figure.update_traces(textposition="outside", cliponaxis=False)
    figure.update_layout(showlegend=False, margin=dict(l=8, r=42, t=48, b=8))
    render_figure(figure, height=330, legend=False)


def _normal_vs_attack_chart(normal: int, malicious: int) -> None:
    total = normal + malicious
    share = (malicious / total * 100) if total else 0.0
    colors = chart_palette()
    figure = go.Figure(
        data=[
            go.Pie(
                labels=["Normal", "Malveillant"],
                values=[normal, malicious],
                hole=0.62,
                marker_colors=[EMERALD, RED],
                marker_line=dict(color=colors["surface"], width=2),
                textinfo="label+percent",
                textfont=dict(size=12),
                hovertemplate="%{label} : %{value} flux<extra></extra>",
            )
        ]
    )
    figure.add_annotation(
        text=f"<b>{share:.1f} %</b><br><span style='font-size:11px;color:{colors['muted']}'>malveillant</span>",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(size=17, color=colors["text"], family="JetBrains Mono, monospace"),
    )
    figure.update_layout(title="Normal / Malveillant", margin=dict(t=48, b=8, l=8, r=8))
    render_figure(figure, height=330, legend=False)


def _confidence_chart(results: pd.DataFrame) -> None:
    frame = results.copy()
    frame["Confiance (%)"] = frame["confidence"] * 100
    frame["Tranche"] = pd.cut(
        frame["Confiance (%)"],
        bins=_CONFIDENCE_BINS,
        labels=_CONFIDENCE_LABELS,
        include_lowest=True,
    )
    grouped = (
        frame.groupby(["Tranche", "status"], observed=False)
        .size()
        .reset_index(name="Flux")
    )
    figure = px.bar(
        grouped,
        x="Tranche",
        y="Flux",
        color="status",
        barmode="group",
        category_orders={"Tranche": _CONFIDENCE_LABELS, "status": ["Normal", "Alerte"]},
        color_discrete_map={"Normal": EMERALD, "Alerte": RED},
        labels={"Tranche": "Confiance (%)", "status": "Statut"},
        title="Distribution des confidences par tranche",
    )
    for trace in figure.data:
        trace.text = [str(value) if value else "" for value in trace.y]
    figure.update_traces(textposition="outside", cliponaxis=False, textfont_size=10)
    figure.update_layout(bargap=0.18, bargroupgap=0.05)
    render_figure(figure, height=350)


def _distribution_table(distribution: pd.DataFrame) -> None:
    display = distribution.copy()
    display["share"] = (display["share"] * 100).round(2)
    display["Statut"] = display["category"].map(
        lambda value: "Normal" if value == "Normal" else "Alerte"
    )
    display = display.rename(
        columns={
            "category": "Catégorie",
            "count": "Nombre de flux",
            "share": "Part du lot (%)",
        }
    )
    st.dataframe(
        display[["Catégorie", "Statut", "Nombre de flux", "Part du lot (%)"]],
        width="stretch",
        hide_index=True,
        column_config={
            "Part du lot (%)": st.column_config.NumberColumn("Part du lot (%)", format="%.2f"),
        },
    )


def render() -> None:
    analysis = get_analysis()
    if analysis is None:
        _empty_state()
        return

    results = analysis["results"]
    try:
        stats = build_attack_statistics(results)
    except InputValidationError as exc:
        st.error(str(exc))
        return

    page_header(
        "Attack Analytics",
        f"Analyse des {stats.total:,} flux prédits pour « {esc(analysis['source_name'])} ».",
        ["Prédictions de session", f"{stats.total:,} flux", "CSV statique"],
    )

    kpi_cards(
        [
            {
                "label": "Flux analysés",
                "value": f"{stats.total:,}",
                "sub": "Lot importé dans la session",
                "tone": "indigo",
            },
            {
                "label": "Flux normaux",
                "value": f"{stats.normal:,}",
                "sub": f"{stats.normal / stats.total * 100:.1f} % du lot",
                "tone": "emerald",
            },
            {
                "label": "Flux malveillants",
                "value": f"{stats.malicious:,}",
                "sub": f"{stats.malicious_rate * 100:.1f} % du lot",
                "tone": "red",
            },
            {
                "label": "Confiance moyenne des alertes",
                "value": (
                    f"{stats.mean_alert_confidence * 100:.1f} %"
                    if stats.mean_alert_confidence is not None
                    else "—"
                ),
                "sub": "Uniquement les flux classés attaque",
                "tone": "amber",
                "small": stats.mean_alert_confidence is None,
            },
        ]
    )

    section_title("Répartition des prédictions", "Sur le lot importé")
    chart_left, chart_right = st.columns([1.25, 1], gap="small")
    with chart_left:
        _distribution_chart(stats.distribution)
    with chart_right:
        _normal_vs_attack_chart(stats.normal, stats.malicious)

    with st.container(border=True):
        st.markdown("##### Détail par catégorie")
        _distribution_table(stats.distribution)

    section_title("Confiance des prédictions", "Probabilité maximale retournée")
    _confidence_chart(results)
    st.caption(
        "La confiance est la probabilité maximale retournée par le Random Forest. "
        "Elle n'est pas une garantie de justesse et n'est pas calibrée."
    )

    section_title("Alertes détectées", "Flux classés DoS, Probe, R2L ou U2R")
    if stats.alerts.empty:
        st.success("Aucune alerte : tous les flux de ce lot ont été classés « Normal ».")
    else:
        alert_display = results_display(stats.alerts)
        st.dataframe(
            alert_display,
            width="stretch",
            hide_index=True,
            column_config={
                "Confiance (%)": st.column_config.ProgressColumn(
                    "Confiance (%)", min_value=0, max_value=100, format="%.2f%%"
                )
            },
        )
        st.download_button(
            "Télécharger la liste des alertes (CSV)",
            data=stats.alerts.to_csv(index=False).encode("utf-8"),
            file_name="netguard_alerts.csv",
            mime="text/csv",
        )

    note(
        "Ces statistiques portent uniquement sur les <strong>prédictions</strong> du modèle "
        "pour le fichier importé. Elles ne mesurent pas la performance expérimentale : "
        "celle-ci est présentée dans <strong>Model Evaluation</strong>."
    )
