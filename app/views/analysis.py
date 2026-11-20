"""CSV import, validation, prediction and result export."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from nids import config
from nids.analytics import build_attack_statistics
from nids.errors import InputValidationError, NIDSError
from nids.inference import predict_flows
from nids.validation import read_csv_bytes
from theme import (
    esc,
    kpi_cards,
    note,
    page_header,
    section_title,
)
from views.common import (
    clear_analysis,
    content_digest,
    get_analysis,
    get_artifacts,
    get_sample_csv,
    results_display,
    store_analysis,
)


def _format_help(artifacts) -> None:
    with st.expander("Format attendu et règles de validation"):
        st.markdown(
            f"""
Le fichier doit être un **CSV séparé par des virgules**, encodé en **UTF-8**, contenant les
**{len(artifacts.feature_columns)} caractéristiques NSL-KDD**. L'ordre des colonnes est libre ;
les noms doivent correspondre exactement.

- **Colonnes manquantes** : analyse refusée avec la liste des colonnes absentes.
- **Colonnes supplémentaires** : ignorées avec un avertissement (sauf étiquettes, également ignorées).
- **Valeurs manquantes** : analyse refusée. Aucune imputation n'a été apprise à l'entraînement.
- **Valeurs non numériques** dans une colonne numérique : analyse refusée avec la colonne concernée.
- **Catégories inconnues** (`protocol_type`, `service`, `flag`) : acceptées, ignorées par le
  One-Hot Encoding comme pendant l'entraînement, et signalées.
- **Fichier vide** ou **format non CSV** : message d'erreur explicite.
- **Limites** : 25 Mo et {config.MAX_ROWS:,} lignes maximum.
"""
        )
        st.code(
            "duration,protocol_type,service,flag,src_bytes,dst_bytes,...,dst_host_srv_rerror_rate",
            language="text",
        )
        st.caption(
            "Une ligne unique est acceptée : le pipeline gère 1 ou N flux. "
            "Les 41 noms exacts sont listés ci-dessous."
        )
        st.write(artifacts.feature_columns)


def _import_panel() -> None:
    with st.container(border=True):
        st.markdown("#### Importer un fichier CSV")
        uploaded = st.file_uploader(
            "Glissez-déposez votre fichier ou parcourez vos dossiers",
            type=["csv"],
            help="Une ou plusieurs lignes, 41 colonnes NSL-KDD, UTF-8, 25 Mo maximum.",
        )
        st.caption(
            "CSV séparé par des virgules · encodage UTF-8 · "
            f"{config.MAX_ROWS:,} lignes maximum"
        )
    return uploaded


def _side_panel(artifacts) -> None:
    with st.container(border=True):
        st.markdown("#### Fichier d'exemple")
        st.markdown(
            "<div class='ng-caption'>Extrait du jeu de test NSL-KDD : "
            "<strong>250 flux</strong>, 41 colonnes brutes, <strong>sans étiquette</strong>.</div>",
            unsafe_allow_html=True,
        )
        try:
            sample = get_sample_csv()
        except OSError:
            st.warning("Fichier d'exemple indisponible.")
        else:
            st.download_button(
                "Télécharger l'échantillon CSV",
                data=sample,
                file_name="nsl_kdd_sample.csv",
                mime="text/csv",
                width="stretch",
                help="Extrait du jeu de test NSL-KDD, sans la classe réelle.",
            )
        st.divider()
        st.markdown(
            f"""
            <div class="ng-caption">
              <strong>{len(artifacts.feature_columns)} features</strong> attendues ·
              <strong>{len(artifacts.class_names)} classes</strong><br/>
              Modèle : Random Forest gelé<br/>
              Aucun entraînement, aucune imputation.
            </div>
            """,
            unsafe_allow_html=True,
        )


def _render_results(analysis: dict, artifacts) -> None:
    results = analysis["results"]
    probabilities = analysis["probabilities"]
    stats = build_attack_statistics(results)

    section_title("Résultats de l'analyse", "Prédictions du modèle gelé")
    kpi_cards(
        [
            {
                "label": "Flux analysés",
                "value": f"{stats.total:,}",
                "sub": f"Fichier <code>{esc(analysis['source_name'])}</code>",
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
                "label": "Confiance moyenne",
                "value": f"{stats.mean_confidence * 100:.1f} %",
                "sub": "Probabilité maximale moyenne",
                "tone": "violet",
            },
        ]
    )

    st.markdown(
        f"<div class='ng-caption' style='margin:.6rem 0 .9rem 0;'>"
        f"Modèle : <strong>Random Forest</strong> · {len(results):,} ligne(s) · "
        f"chaque probabilité est produite par le modèle gelé, sans réentraînement.</div>",
        unsafe_allow_html=True,
    )

    display = results_display(results)
    tab_all, tab_alerts = st.tabs(["Tous les flux", "Alertes uniquement"])
    with tab_all:
        st.dataframe(
            display,
            width="stretch",
            hide_index=True,
            column_config={
                "Confiance (%)": st.column_config.ProgressColumn(
                    "Confiance (%)", min_value=0, max_value=100, format="%.2f%%"
                ),
                "Statut": st.column_config.TextColumn("Statut"),
            },
        )
    with tab_alerts:
        alerts = stats.alerts
        if alerts.empty:
            st.info("Aucune alerte : tous les flux ont été classés « Normal ».")
        else:
            alert_display = results_display(alerts)
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

    export = pd.concat(
        [results.reset_index(drop=True), probabilities.reset_index(drop=True).add_prefix("proba_")],
        axis=1,
    )
    st.download_button(
        "Télécharger les prédictions (CSV)",
        data=export.to_csv(index=False).encode("utf-8"),
        file_name="netguard_predictions.csv",
        mime="text/csv",
    )

    st.caption(
        "Les statistiques décrivent les prédictions du modèle sur ce fichier, et non la "
        "vérité terrain (sauf si un fichier étiqueté est fourni séparément). "
        "Poursuivez dans Attack Analytics pour l'analyse détaillée."
    )


def render() -> None:
    page_header(
        "Analyse de flux réseau",
        "Importez un CSV NSL-KDD, vérifiez sa conformité puis lancez la classification.",
        ["CSV statique", "41 features", "UTF-8", "25 Mo max"],
    )

    try:
        artifacts = get_artifacts()
    except NIDSError as exc:
        st.error(str(exc))
        st.stop()

    section_title("Importer", "Un fichier, 1 à N flux")
    left, right = st.columns([1.65, 1], gap="small")
    with left:
        uploaded = _import_panel()
    with right:
        _side_panel(artifacts)

    _format_help(artifacts)

    if uploaded is None:
        note(
            "Aucun fichier sélectionné. Téléchargez l'échantillon de 250 flux pour tester "
            "immédiatement la chaîne complète, ou déposez votre propre CSV NSL-KDD."
        )
        return

    content = uploaded.getvalue()
    digest = content_digest(content)
    analysis = get_analysis()
    if analysis and analysis["hash"] != digest:
        clear_analysis()
        analysis = None

    try:
        frame = read_csv_bytes(content, uploaded.name)
    except InputValidationError as exc:
        clear_analysis()
        st.error(str(exc))
        for detail in exc.details:
            st.write(f"- {detail}")
        return

    section_title("Vérification", "Aperçu avant classification")
    st.markdown(
        f"""
        <div class="ng-panel" style="padding:.85rem 1rem;">
          <div style="display:flex;flex-wrap:wrap;gap:1.6rem;align-items:center;">
            <div><div class="kicker">Fichier</div>
              <div class="mono" style="font-size:.85rem;color:#0F172A;">{esc(uploaded.name)}</div></div>
            <div><div class="kicker">Lignes</div>
              <div class="big" style="font-size:1rem;">{len(frame):,}</div></div>
            <div><div class="kicker">Colonnes</div>
              <div class="big" style="font-size:1rem;">{len(frame.columns)}</div></div>
            <div><div class="kicker">Modèle</div>
              <div class="big" style="font-size:1rem;">Random Forest</div></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    with st.expander("Aperçu des données importées", expanded=False):
        st.dataframe(frame.head(20), width="stretch", hide_index=True)

    if st.button("Lancer l'analyse", type="primary", width="stretch"):
        try:
            with st.spinner("Validation et classification en cours…"):
                prediction = predict_flows(frame, artifacts)
            store_analysis(
                uploaded.name,
                digest,
                prediction.results,
                prediction.probabilities,
                prediction.warnings,
            )
            analysis = get_analysis()
            st.success(f"Analyse terminée : {len(prediction)} flux classifié(s).")
        except InputValidationError as exc:
            clear_analysis()
            st.error(str(exc))
            for detail in exc.details:
                st.write(f"- {detail}")
            return
        except NIDSError as exc:
            clear_analysis()
            st.error(f"Analyse impossible : {exc}")
            return
        except Exception as exc:  # noqa: BLE001 - last-resort guard for the UI
            clear_analysis()
            st.error(f"Erreur inattendue pendant l'analyse : {exc}")
            return

    if analysis is None:
        st.info("Le fichier est prêt. Cliquez sur « Lancer l'analyse » pour obtenir les prédictions.")
        return

    for warning in analysis["warnings"]:
        st.warning(warning)
    _render_results(analysis, artifacts)
