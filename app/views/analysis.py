"""CSV import, validation, prediction and result export."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from nids import config
from nids.analytics import build_attack_statistics
from nids.errors import InputValidationError, NIDSError
from nids.inference import predict_flows
from nids.validation import read_csv_bytes
from views.common import (
    clear_analysis,
    content_digest,
    get_analysis,
    get_artifacts,
    note,
    page_header,
    results_display,
    section_label,
    store_analysis,
)

_SAMPLE_PATH = Path(__file__).resolve().parents[2] / "data" / "samples" / "nsl_kdd_sample.csv"


def _sample_download() -> None:
    if not _SAMPLE_PATH.is_file():
        return
    st.download_button(
        "Télécharger un fichier d'exemple (250 flux, sans étiquette)",
        data=_SAMPLE_PATH.read_bytes(),
        file_name="nsl_kdd_sample.csv",
        mime="text/csv",
        help="Extrait du jeu de test NSL-KDD. Les 41 colonnes brutes sont fournies, sans la classe réelle.",
    )


def _format_help(artifacts) -> None:
    with st.expander("Format attendu et règles de validation"):
        st.markdown(
            f"""
Le fichier doit être un **CSV séparé par des virgules** contenant les
**{len(artifacts.feature_columns)} caractéristiques NSL-KDD**. L'ordre des colonnes est libre ;
les noms doivent correspondre exactement.

- **Colonnes manquantes** : analyse refusée avec la liste des colonnes absentes.
- **Colonnes supplémentaires** : ignorées avec un avertissement (sauf étiquettes, également ignorées).
- **Valeurs manquantes** : analyse refusée. Aucune imputation n'a été apprise à l'entraînement.
- **Valeurs non numériques** dans une colonne numérique : analyse refusée avec la colonne concernée.
- **Catégories inconnues** (`protocol_type`, `service`, `flag`) : acceptées, ignorées par le
  One-Hot Encoding comme pendant l'entraînement, et signalées.
- **Fichier vide** ou **format non CSV** : message d'erreur explicite.
"""
        )
        st.code(
            "duration,protocol_type,service,flag,src_bytes,dst_bytes,...,dst_host_srv_rerror_rate",
            language="text",
        )
        st.caption(
            "Un aperçu des colonnes exactes est disponible ci-dessous. "
            "Une ligne unique est acceptée : le pipeline gère 1 ou N flux."
        )
        st.write(artifacts.feature_columns)
        _sample_download()


def _render_results(analysis: dict, artifacts) -> None:
    results = analysis["results"]
    probabilities = analysis["probabilities"]
    stats = build_attack_statistics(results)

    section_label("Résultats de l'analyse")
    columns = st.columns(4)
    columns[0].metric("Flux analysés", f"{stats.total:,}")
    columns[1].metric("Flux normaux", f"{stats.normal:,}")
    columns[2].metric("Flux malveillants", f"{stats.malicious:,}")
    columns[3].metric("Confiance moyenne", f"{stats.mean_confidence * 100:.1f} %")

    st.markdown(
        f"Modèle utilisé : **Random Forest** · Fichier : `{analysis['source_name']}` · "
        f"{len(results):,} ligne(s). Chaque probabilité est produite par le modèle gelé."
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
        "Les statistiques décrivent les prédictions du modèle sur ce fichier, "
        "et non la vérité terrain (sauf si un fichier étiqueté est fourni séparément). "
        "Poursuivez dans **Attack Analytics** pour l'analyse détaillée."
    )


def render() -> None:
    page_header(
        "Analyse de flux réseau",
        "Importez un CSV NSL-KDD, vérifiez sa conformité puis lancez la classification.",
    )

    try:
        artifacts = get_artifacts()
    except NIDSError as exc:
        st.error(str(exc))
        st.stop()

    _format_help(artifacts)

    uploaded = st.file_uploader(
        "Importer un fichier CSV",
        type=["csv"],
        help="Une ou plusieurs lignes, 41 colonnes NSL-KDD.",
    )
    if uploaded is None:
        note(
            "Aucun fichier sélectionné. Utilisez le bouton d'exemple ci-dessus pour tester "
            "immédiatement la chaîne complète."
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

    st.caption(
        f"Fichier `{uploaded.name}` — {len(frame):,} ligne(s), {len(frame.columns)} colonne(s)."
    )
    with st.expander("Aperçu des données importées", expanded=False):
        st.dataframe(frame.head(20), width="stretch", hide_index=True)

    if st.button("Lancer l'analyse", type="primary", width="stretch"):
        try:
            with st.spinner("Validation et classification en cours…"):
                prediction = predict_flows(frame, artifacts)
            store_analysis(
                uploaded.name, digest, prediction.results, prediction.probabilities, prediction.warnings
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
