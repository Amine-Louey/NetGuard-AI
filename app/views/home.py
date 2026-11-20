"""Landing page: project, system and scope."""

from __future__ import annotations

import streamlit as st

from nids import config
from theme import (
    AMBER,
    EMERALD,
    INDIGO,
    RED,
    VIOLET,
    class_chips,
    hero,
    kpi_cards,
    note,
    panel,
    section_title,
)
from views.common import evaluation_available, get_evaluation

_CLASS_DESCRIPTIONS = {
    "Normal": "Trafic légitime conforme aux protocoles standards (HTTP, FTP, DNS…).",
    "DoS": "Épuisement de ressources : SYN flood, Neptune, Smurf, Pod…",
    "Probe": "Reconnaissance active : balayage de ports, Nmap, Satan, Ipsweep…",
    "R2L": "Accès distant non autorisé : Guess_password, FTP_write, Phf…",
    "U2R": "Élévation de privilèges locale : Buffer Overflow, Rootkit, Perl…",
}

_CLASS_ACCENTS = {
    "Normal": EMERALD,
    "DoS": RED,
    "Probe": AMBER,
    "R2L": "#DB2777",
    "U2R": VIOLET,
}

_STEPS = [
    (
        "01",
        "Importer",
        "Un fichier CSV contenant les 41 caractéristiques brutes d'un ou plusieurs flux.",
    ),
    (
        "02",
        "Valider",
        "Colonnes, types, valeurs manquantes et catégories sont contrôlés avant tout calcul.",
    ),
    (
        "03",
        "Prétraiter",
        "Le préprocesseur sauvegardé applique exactement l'encodage et la standardisation de l'entraînement.",
    ),
    (
        "04",
        "Classifier",
        "Le Random Forest gelé retourne une classe et cinq probabilités, sans réentraînement.",
    ),
]


def _metrics_block(report: dict) -> None:
    metrics = report["test_set"]["models"]["Random Forest"]["metrics"]
    support = report["test_set"]["class_support"]
    section_title(
        "Performance de référence",
        "Random Forest · jeu de test · évaluation unique",
    )
    kpi_cards(
        [
            {
                "label": "Accuracy (test)",
                "value": f"{metrics['accuracy'] * 100:.2f} %",
                "sub": "Vue globale, dominée par les classes majoritaires.",
                "tone": "indigo",
            },
            {
                "label": "Macro F1 (test)",
                "value": f"{metrics['macro_f1'] * 100:.2f} %",
                "sub": "Moyenne équitable des cinq classes.",
                "tone": "emerald",
            },
            {
                "label": "Recall R2L",
                "value": f"{metrics['recall_r2l'] * 100:.2f} %",
                "sub": "464 flux R2L dans le test.",
                "tone": "violet",
            },
            {
                "label": "Recall U2R",
                "value": f"{metrics['recall_u2r'] * 100:.2f} %",
                "sub": f"{support['U2R']} flux seulement : prudence.",
                "tone": "amber",
            },
        ]
    )
    note(
        "Ces valeurs proviennent de l'évaluation finale du Random Forest sur le jeu de test "
        f"retenu ({report['test_set']['size']:,} flux, dont {support['U2R']} U2R). "
        "L'Accuracy est élevée car les classes majoritaires dominent ; la Macro F1, plus "
        "sévère, expose les difficultés sur les attaques rares. Le protocole exact et sa "
        "limite principale sont détaillés dans <strong>Model Evaluation</strong>."
    )


def _comparison_block(report: dict) -> None:
    models = report["test_set"]["models"]
    section_title("Comparaison des modèles", "Sélection sur validation, mesure sur test")
    columns = st.columns(3, gap="small")
    descriptions = {
        "Random Forest": ("Machine Learning", "Ensemble de 200 arbres · modèle déployé", "indigo"),
        "MLP": ("Deep Learning", "Perceptron multicouche · comparaison", "slate"),
        "1D-CNN": ("Deep Learning", "Réseau convolutif 1D · comparaison", "slate"),
    }
    for column, (name, model) in zip(columns, models.items()):
        family, detail, tone = descriptions.get(name, ("Modèle", "", "slate"))
        metrics = model["metrics"]
        with column:
            st.markdown(
                f"""
                <div class="ng-panel">
                  <div class="kicker">{family}</div>
                  <h3 style="margin-top:.25rem;">{name}</h3>
                  <div class="big">{metrics['macro_f1'] * 100:.2f} %</div>
                  <p style="margin:.1rem 0 .55rem 0;">Macro F1 sur le test</p>
                  <p>{detail}</p>
                  <p style="margin-top:.5rem;color:#64748B;">
                    Accuracy <strong>{metrics['accuracy'] * 100:.2f} %</strong> ·
                    Weighted F1 <strong>{metrics['weighted_f1'] * 100:.2f} %</strong>
                  </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
    st.caption(
        "Le Random Forest est retenu pour le déploiement. Le MLP et le 1D-CNN restent des "
        "comparaisons de recherche ; ils ne sont pas exécutés dans l'application."
    )


def render() -> None:
    hero(
        "Machine Learning · Cybersécurité · NSL-KDD",
        "NetGuard AI<br/>Détection multiclasse d'intrusions réseau",
        "Un système qui classe des flux réseau en cinq catégories — Normal, DoS, Probe, "
        "R2L et U2R — à partir de 41 caractéristiques NSL-KDD, avec un Random Forest "
        "sélectionné sur le jeu de validation et évalué une seule fois sur un jeu de test "
        "mis de côté.",
        [
            "41 caractéristiques",
            "5 classes",
            "Random Forest",
            "Split 70 / 15 / 15",
            "Sans fuite de données",
        ],
    )

    if evaluation_available():
        report = get_evaluation()
        _metrics_block(report)
    else:
        report = None
        note(
            "Le rapport d'évaluation expérimental est absent : aucune métrique n'est affichée "
            "plutôt que d'afficher une valeur inventée.",
            warning=True,
        )

    section_title("Fonctionnement", "De l'import CSV à la prédiction")
    columns = st.columns(4, gap="small")
    for column, (number, title, text) in zip(columns, _STEPS):
        with column:
            st.markdown(
                f"""
                <div class="ng-step">
                  <div class="num">{number}</div>
                  <h3>{title}</h3>
                  <p>{text}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    section_title("Les cinq catégories détectées", "Taxonomie NSL-KDD")
    st.markdown(class_chips(), unsafe_allow_html=True)
    st.markdown("<div style='height:.55rem'></div>", unsafe_allow_html=True)
    columns = st.columns(5, gap="small")
    for column, class_name in zip(columns, config.CLASS_ORDER):
        with column:
            st.markdown(
                f"""
                <div class="ng-class" style="--class-color:{_CLASS_ACCENTS[class_name]}">
                  <div class="name"><span class="dot"></span>{class_name}</div>
                  <p>{_CLASS_DESCRIPTIONS[class_name]}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    if report is not None:
        _comparison_block(report)

    section_title("Périmètre et méthode", "Ce que l'application fait — et ne fait pas")
    left, right = st.columns(2, gap="small")
    with left:
        panel(
            "🎯",
            "Périmètre honnête",
            "L'application analyse des <strong>flux fournis dans un fichier CSV</strong> au "
            "format NSL-KDD. Elle ne capture pas le trafic réseau en direct, ne remplace pas "
            "un IDS de production et ne prétend pas détecter des attaques en temps réel. "
            "Les probabilités du Random Forest sont des fréquences de votes, non calibrées.",
            kicker="Limites",
        )
    with right:
        panel(
            "🔬",
            "Reproductibilité",
            "Découpage stratifié reconstruit avec la graine <code>42</code> (70 / 15 / 15). "
            "Modèle déployé : <code>rf_model.joblib</code> (200 arbres, profondeur 20, "
            "<code>class_weight='balanced'</code>). Prétraitement ajusté sur le seul jeu "
            "d'entraînement. Les métriques sont lues depuis le rapport vérifié, jamais "
            "recalculées sur les imports.",
            kicker="Méthode",
        )

    section_title("Démarrer", "Parcours de démonstration")
    note(
        "Utilisez le menu latéral : <strong>Analyse</strong> pour importer un CSV, "
        "<strong>Attack Analytics</strong> pour explorer les prédictions de la session, "
        "<strong>Model Evaluation</strong> pour consulter les résultats expérimentaux "
        "sauvegardés. Un fichier d'exemple de 250 flux est téléchargeable depuis la page Analyse."
    )
