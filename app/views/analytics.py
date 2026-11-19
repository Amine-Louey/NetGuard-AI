"""Attack Analytics: statistics derived from the current predictions."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from nids import config
from nids.analytics import build_attack_statistics
from nids.errors import InputValidationError
from views.common import (
    CLASS_COLORS,
    get_analysis,
    note,
    page_header,
    results_display,
    section_label,
)


def _empty_state() -> None:
    page_header(
        "Attack Analytics",
        "Analyse statistique des flux classifiés pendant la session.",
    )
    note(
        "Aucune analyse disponible. Importez d'abord un fichier CSV dans la page **Analyse**.",
        warning=True,
    )


def _distribution_chart(distribution: pd.DataFrame) -> None:
    attacks = distribution[distribution["category"] != "Normal"]
    attacks = attacks[attacks["count"] > 0]
    if attacks.empty:
        st.info("Aucune attaque détectée dans ce lot : le graphique de répartition est vide.")
        return
    figure = px.bar(
        attacks,
        x="category",
        y="count",
        color="category",
        color_discrete_map=CLASS_COLORS,
        text="count",
        labels={"category": "Catégorie", "count": "Nombre de flux"},
        title="Répartition des catégories d'attaques détectées",
    )
    figure.update_layout(showlegend=False, height=380, margin=dict(l=10, r=10, t=55, b=10))
    figure.update_traces(textposition="outside")
    st.plotly_chart(figure, width="stretch")


def _normal_vs_attack_chart(normal: int, malicious: int) -> None:
    figure = go.Figure(
        data=[
            go.Pie(
                labels=["Normal", "Malveillant"],
                values=[normal, malicious],
                hole=0.58,
                marker_colors=["#16a34a", "#dc2626"],
                textinfo="label+percent",
            )
        ]
    )
    figure.update_layout(
        title="Répartition Normal / Malveillant",
        height=380,
        margin=dict(l=10, r=10, t=55, b=10),
        showlegend=False,
    )
    st.plotly_chart(figure, width="stretch")


def _confidence_chart(results: pd.DataFrame) -> None:
    frame = results.copy()
    frame["Confiance (%)"] = frame["confidence"] * 100
    figure = px.histogram(
        frame,
        x="Confiance (%)",
        color="status",
        nbins=20,
        color_discrete_map={"Normal": "#16a34a", "Alerte": "#dc2626"},
        title="Distribution des confidences",
        labels={"status": "Statut"},
    )
    figure.update_layout(height=340, margin=dict(l=10, r=10, t=55, b=10), bargap=0.06)
    st.plotly_chart(figure, width="stretch")


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
        f"Analyse des {stats.total:,} flux prédits pour `{analysis['source_name']}`.",
    )

    columns = st.columns(4)
    columns[0].metric("Flux analysés", f"{stats.total:,}")
    columns[1].metric("Flux normaux", f"{stats.normal:,}", f"{stats.normal / stats.total * 100:.1f} %")
    columns[2].metric(
        "Flux malveillants",
        f"{stats.malicious:,}",
        f"{stats.malicious_rate * 100:.1f} %",
        delta_color="inverse",
    )
    alert_confidence = (
        f"{stats.mean_alert_confidence * 100:.1f} %"
        if stats.mean_alert_confidence is not None
        else "—"
    )
    columns[3].metric("Confiance moyenne des alertes", alert_confidence)

    section_label("Répartition des prédictions")
    chart_left, chart_right = st.columns([1.25, 1])
    with chart_left:
        _distribution_chart(stats.distribution)
    with chart_right:
        _normal_vs_attack_chart(stats.normal, stats.malicious)

    distribution_display = stats.distribution.copy()
    distribution_display["share"] = (distribution_display["share"] * 100).round(2)
    distribution_display = distribution_display.rename(
        columns={
            "category": "Catégorie",
            "count": "Nombre de flux",
            "share": "Part du lot (%)",
        }
    )
    st.dataframe(distribution_display, width="stretch", hide_index=True)

    section_label("Confiance des prédictions")
    _confidence_chart(results)
    st.caption(
        "La confiance est la probabilité maximale retournée par le Random Forest. "
        "Elle n'est pas une garantie de justesse."
    )

    section_label("Alertes détectées")
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
