"""Landing page: project, system and scope."""

from __future__ import annotations

import streamlit as st

from nids import config
from views.common import (
    class_chips,
    evaluation_available,
    get_evaluation,
    note,
    page_header,
    section_label,
)

_CLASS_DESCRIPTIONS = {
    "Normal": "Trafic légitime conforme aux protocoles standards (HTTP, FTP, DNS…).",
    "DoS": "Épuisement de ressources : SYN flood, Neptune, Smurf, Pod…",
    "Probe": "Reconnaissance active : balayage de ports, Nmap, Satan, Ipsweep…",
    "R2L": "Accès distant non autorisé : Guess_password, FTP_write, Phf…",
    "U2R": "Élévation de privilèges locale : Buffer Overflow, Rootkit, Perl…",
}


def _metric_block(report: dict) -> None:
    metrics = report["test_set"]["models"]["Random Forest"]["metrics"]
    support = report["test_set"]["class_support"]
    columns = st.columns(4)
    columns[0].metric("Accuracy (test)", f"{metrics['accuracy'] * 100:.2f} %")
    columns[1].metric("Macro F1 (test)", f"{metrics['macro_f1'] * 100:.2f} %")
    columns[2].metric("Recall R2L", f"{metrics['recall_r2l'] * 100:.2f} %")
    columns[3].metric("Recall U2R", f"{metrics['recall_u2r'] * 100:.2f} %")
    note(
        "Ces valeurs proviennent de l'évaluation finale du Random Forest sur le jeu de test "
        f"retenu ({report['test_set']['size']:,} flux, dont {support['U2R']} U2R). "
        "L'Accuracy est élevée car les classes majoritaires dominent ; la Macro F1, plus "
        "sévère, expose les difficultés sur les attaques rares. Le protocole exact et sa "
        "limite principale sont détaillés dans <strong>Model Evaluation</strong>."
    )


def render() -> None:
    st.markdown(
        """
        <div class="hero">
          <div class="eyebrow">Machine Learning · Cybersécurité · NSL-KDD</div>
          <h1>NetGuard AI<br/>Détection multiclasse d'intrusions réseau</h1>
          <p>
            Un système qui classe des flux réseau en cinq catégories — Normal, DoS, Probe,
            R2L et U2R — à partir de 41 caractéristiques NSL-KDD, avec un Random Forest
            sélectionné sur le jeu de validation et évalué une seule fois sur un jeu de test
            mis de côté.
          </p>
          <div class="badges">
            <span class="badge">41 caractéristiques</span>
            <span class="badge">5 classes</span>
            <span class="badge">Random Forest</span>
            <span class="badge">Split stratifié 70 / 15 / 15</span>
            <span class="badge">Sans fuite de données</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if evaluation_available():
        _metric_block(get_evaluation())
    else:
        note(
            "Le rapport d'évaluation expérimental est absent : aucune métrique n'est affichée "
            "plutôt que d'afficher une valeur inventée.",
            warning=True,
        )

    section_label("Ce que fait le système")
    columns = st.columns(4)
    cards = [
        ("📥", "1. Importer", "Un fichier CSV contenant les 41 caractéristiques brutes d'un ou plusieurs flux."),
        ("🧹", "2. Valider", "Colonnes, types, valeurs manquantes et catégories sont contrôlés avant tout calcul."),
        ("⚙️", "3. Prétraiter", "Le préprocesseur sauvegardé applique exactement l'encodage et la standardisation de l'entraînement."),
        ("🛡️", "4. Classifier", "Le Random Forest gelé retourne une classe et cinq probabilités, sans réentraînement."),
    ]
    for column, (icon, title, text) in zip(columns, cards):
        column.markdown(
            f'<div class="card"><div class="icon">{icon}</div><h3>{title}</h3><p>{text}</p></div>',
            unsafe_allow_html=True,
        )

    section_label("Les cinq catégories détectées")
    st.markdown(class_chips(), unsafe_allow_html=True)
    columns = st.columns(5)
    for column, class_name in zip(columns, config.CLASS_ORDER):
        column.markdown(
            f'<div class="card"><h3>{class_name}</h3>'
            f'<p>{_CLASS_DESCRIPTIONS[class_name]}</p></div>',
            unsafe_allow_html=True,
        )

    section_label("Parcours de modélisation")
    left, right = st.columns([1.15, 1])
    with left:
        st.markdown(
            """
            <div class="card">
              <h3>Une comparaison méthodologique</h3>
              <p>
                Trois familles de modèles ont été entraînées sur les mêmes données :
                un <strong>Random Forest</strong> (Machine Learning classique, modèle de référence),
                un <strong>MLP</strong> et un <strong>1D-CNN</strong> (Deep Learning).
                La sélection s'est faite uniquement sur le jeu de validation, avec la Macro F1
                comme critère principal. Le test set est resté indépendant jusqu'à l'évaluation finale.
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        st.markdown(
            """
            <div class="card">
              <h3>Périmètre honnête</h3>
              <p>
                Cette application analyse des <strong>flux fournis en entrée</strong> au format
                NSL-KDD. Elle ne capture pas le trafic réseau en direct et ne remplace pas un
                IDS de production. Les probabilités du Random Forest sont des fréquences de
                votes, pas des probabilités calibrées.
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    section_label("Démarrer")
    st.markdown(
        "Utilisez le menu latéral : **Analyse** pour importer un CSV, **Attack Analytics** "
        "pour explorer les prédictions de la session, **Model Evaluation** pour consulter les "
        "résultats expérimentaux sauvegardés."
    )
    section_label("Reproductibilité")
    st.caption(
        "Les artefacts, la graine aléatoire et la version de scikit-learn sont figés dans le projet."
    )
    st.markdown(
        "- Découpage reconstruit : graine `42`, 70 % entraînement / 15 % validation / 15 % test stratifiés.\n"
        "- Modèle déployé : `rf_model.joblib` (200 arbres, profondeur maximale 20, `class_weight='balanced'`).\n"
        "- Prétraitement : `preprocessor.joblib` ajusté sur le seul jeu d'entraînement.\n"
        "- Les métriques affichées sont lues depuis `reports/model_evaluation.json`, généré par un script de vérification."
    )
