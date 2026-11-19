"""Model Evaluation: saved experimental results, not upload-derived scores."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from nids.errors import ArtifactError
from nids.evaluation import (
    comparison_table,
    confusion_matrix_table,
    normalized_confusion_matrix_table,
    per_class_table,
)
from views.common import (
    get_evaluation,
    note,
    page_header,
    section_label,
)

_MODEL_TYPES = {
    "Random Forest": "Machine Learning — ensemble d'arbres (pas un réseau de neurones)",
    "MLP": "Deep Learning — perceptron multicouche",
    "1D-CNN": "Deep Learning — réseau convolutif 1D",
}

_CONFUSION_INTERPRETATION = (
    "Lecture : les lignes sont les classes réelles et les colonnes les classes prédites. "
    "La diagonale contient les bonnes prédictions. Les 12 flux U2R du test set sont "
    "suffisants pour illustrer une tendance, pas pour conclure statistiquement sur cette classe."
)


def _percent(value: float) -> str:
    return f"{value * 100:.2f} %"


def _metric_cards(model: dict) -> None:
    metrics = model["metrics"]
    columns = st.columns(4)
    columns[0].metric("Accuracy", _percent(metrics["accuracy"]))
    columns[1].metric("Macro F1", _percent(metrics["macro_f1"]))
    if "macro_precision" in model:
        columns[2].metric("Precision macro", _percent(model["macro_precision"]))
        columns[3].metric("Recall macro", _percent(model["macro_recall"]))
    else:
        columns[2].metric("Weighted F1", _percent(metrics["weighted_f1"]))
        columns[3].metric("FPR", f"{metrics['fpr'] * 100:.2f} %")


def _comparison_frame(evaluation: dict) -> pd.DataFrame:
    frame = comparison_table(evaluation)
    format_map = {
        column: "{:.2%}"
        for column in ["Accuracy", "Macro F1", "Weighted F1", "Recall R2L", "Recall U2R", "FPR"]
    }
    format_map["Latence (ms)"] = "{:.4f}"
    styled = frame.style.format(format_map, na_rep="—").apply(
        lambda row: [
            "background-color: #ecfdf5; font-weight: 600;" if row["Modèle"] == "Random Forest" else ""
            for _ in row
        ],
        axis=1,
    )
    return styled


def _comparison_chart(evaluation: dict) -> None:
    frame = comparison_table(evaluation)
    long = frame.melt(
        id_vars=["Modèle"],
        value_vars=["Accuracy", "Macro F1", "Weighted F1"],
        var_name="Métrique",
        value_name="Score",
    )
    figure = px.bar(
        long,
        x="Modèle",
        y="Score",
        color="Métrique",
        barmode="group",
        text_auto=".2%",
        title="Comparaison sur le jeu de test indépendant",
        color_discrete_sequence=["#0f766e", "#f59e0b", "#64748b"],
    )
    figure.update_layout(
        yaxis_range=[0, 1.05],
        yaxis_tickformat=".0%",
        height=390,
        margin=dict(l=10, r=10, t=55, b=10),
        legend_title_text="",
    )
    st.plotly_chart(figure, width="stretch")


def _detail_section(evaluation: dict, split: str, model_name: str) -> None:
    model = evaluation[split]["models"].get(model_name)
    if model is None or not model.get("detail_available", False):
        note(
            f"Le détail par classe de <strong>{model_name}</strong> n'est pas disponible "
            "dans les résultats sauvegardés. Aucune valeur n'est reconstituée ici.",
            warning=True,
        )
        return

    _metric_cards(model)
    st.caption(
        f"Source : {model.get('source', 'résultats sauvegardés')} · "
        f"{model.get('support', evaluation[split]['size']):,} flux."
    )

    per_class = per_class_table(evaluation, split, model_name)
    if per_class is not None:
        section_label("Performance par classe")
        st.dataframe(
            per_class.style.format(
                {"Precision": "{:.4f}", "Recall": "{:.4f}", "F1-score": "{:.4f}"}
            ),
            width="stretch",
            hide_index=True,
        )

    matrix = confusion_matrix_table(evaluation, split, model_name)
    normalised = normalized_confusion_matrix_table(evaluation, split, model_name)
    if matrix is not None and normalised is not None:
        section_label("Matrice de confusion")
        view = st.radio(
            "Affichage de la matrice",
            ["Effectifs", "Normalisée par ligne"],
            horizontal=True,
            key=f"cm-view-{split}-{model_name}",
        )
        selected = matrix if view == "Effectifs" else normalised
        text_format = "d" if view == "Effectifs" else ".2f"
        figure = px.imshow(
            selected,
            text_auto=text_format,
            color_continuous_scale="Blues",
            labels=dict(x="Classe prédite", y="Classe réelle", color="Flux"),
            x=selected.columns,
            y=selected.index,
            aspect="auto",
        )
        figure.update_layout(height=460, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(figure, width="stretch")
        st.caption(_CONFUSION_INTERPRETATION)


def _model_facts() -> None:
    with st.expander("Configuration des modèles et protocole expérimental"):
        st.markdown(
            """
**Random Forest (modèle déployé)**
- 200 arbres, `max_depth=20`, `class_weight='balanced'`, `random_state=42`.
- Sélectionné sur la validation (Macro F1) entre deux profondeurs ; c'est un modèle de
  Machine Learning classique, **pas un réseau de neurones**.
- Aucun réentraînement n'est effectué dans l'application.

**MLP**
- Entrée 118 dimensions → Dense 128 + ReLU → Dropout 0,3 → Dense 64 + ReLU →
  Dropout 0,2 → Softmax 5 classes.
- Adam, learning rate 1e-3, 30 époques maximum, Early Stopping (patience 5),
  `class_weight` équilibrés.

**1D-CNN**
- Entrée (118, 1) → Conv1D 64 (kernel 3) → MaxPooling → Conv1D 128 (kernel 3) →
  GlobalMaxPooling → Dense 64 + ReLU → Dropout 0,3 → Softmax.
- Les 118 features encodées sont traitées comme une séquence de colonnes, sans prétendre
  à une structure spatiale naturelle.

**Protocole**
- Découpage 70 / 15 / 15 stratifié, graine 42, sans doublons.
- StandardScaler et One-Hot Encoding ajustés sur le seul entraînement.
- Sélection des modèles sur la validation uniquement.
- Jeu de test utilisé une seule fois pour l'évaluation finale.
"""
        )


def render() -> None:
    page_header(
        "Model Evaluation",
        "Résultats expérimentaux sauvegardés du Random Forest, du MLP et du 1D-CNN.",
    )
    try:
        evaluation = get_evaluation()
    except ArtifactError as exc:
        st.error(str(exc))
        return

    metadata = evaluation["metadata"]
    reproduction = evaluation["reproduction"]
    note(
        f"Modèle sélectionné : <strong>{metadata['selected_model']}</strong> · "
        f"Jeu de test : {evaluation['test_set']['size']:,} flux · "
        f"Validation : {evaluation['validation_set']['size']:,} flux. "
        "Ces métriques proviennent des expériences de la Semaine 3 ; elles ne sont pas "
        "recalculées à partir des fichiers importés par l'utilisateur."
    )
    note(
        "Point méthodologique important : le découpage 70/15/15 a été construit après "
        "fusion et dédoublonnage de <code>KDDTrain+_20Percent</code> et <code>KDDTest+</code>. "
        "Le jeu de test est donc indépendant de l'entraînement, mais il ne reproduit pas le "
        "protocole officiel NSL-KDD où le fichier de test contient des attaques inédites. "
        "Les scores sont par conséquent optimistes par rapport à une évaluation "
        "entraînement NSL-KDD → test NSL-KDD officiel. Cette limite est discutée dans le rapport.",
        warning=True,
    )

    test_tab, validation_tab = st.tabs(
        ["Jeu de test (évaluation finale)", "Jeu de validation (sélection)"]
    )

    with test_tab:
        model_names = list(evaluation["test_set"]["models"].keys())
        selected_model = st.selectbox(
            "Modèle à détailler",
            model_names,
            index=0,
            help="Le Random Forest possède le détail par classe vérifié ; les réseaux peuvent aussi en disposer.",
        )
        _detail_section(evaluation, "test_set", selected_model)

        section_label("Tableau comparatif")
        st.dataframe(_comparison_frame(evaluation), width="stretch", hide_index=True)
        _comparison_chart(evaluation)

        st.caption(
            "FPR : taux moyen de faux positifs un-contre-tous, calculé à partir de la matrice "
            "de confusion. La latence dépend du matériel de mesure."
        )
        if reproduction.get("test_verified"):
            st.success(
                "Reproductibilité vérifiée : les métriques du Random Forest ont été recalculées "
                "à partir des artefacts et du découpage sauvegardé, avec un écart maximal de "
                f"{reproduction['random_forest_max_abs_difference_test']:.1e}."
            )
        for neural_note in reproduction.get("neural_notes", []):
            st.caption(neural_note)

    with validation_tab:
        note(
            "Le jeu de validation a servi à comparer les configurations. Il ne constitue pas "
            "une mesure finale de généralisation ; seul le jeu de test joue ce rôle."
        )
        _detail_section(evaluation, "validation_set", "Random Forest")

        rf_test = evaluation["test_set"]["models"]["Random Forest"]["metrics"]
        rf_val = evaluation["validation_set"]["models"]["Random Forest"]["metrics"]
        comparison = pd.DataFrame(
            [
                {
                    "Jeu": "Validation",
                    "Taille": evaluation["validation_set"]["size"],
                    "Accuracy": rf_val["accuracy"],
                    "Macro F1": rf_val["macro_f1"],
                    "Recall R2L": rf_val["recall_r2l"],
                    "Recall U2R": rf_val["recall_u2r"],
                },
                {
                    "Jeu": "Test",
                    "Taille": evaluation["test_set"]["size"],
                    "Accuracy": rf_test["accuracy"],
                    "Macro F1": rf_test["macro_f1"],
                    "Recall R2L": rf_test["recall_r2l"],
                    "Recall U2R": rf_test["recall_u2r"],
                },
            ]
        )
        section_label("Validation vs test — Random Forest")
        st.dataframe(
            comparison.style.format(
                {
                    "Accuracy": "{:.4f}",
                    "Macro F1": "{:.4f}",
                    "Recall R2L": "{:.4f}",
                    "Recall U2R": "{:.4f}",
                }
            ),
            width="stretch",
            hide_index=True,
        )

    section_label("Lecture critique")
    left, right = st.columns(2)
    matrix = confusion_matrix_table(evaluation, "test_set", "Random Forest")
    u2r_detected = int(matrix.loc["U2R", "U2R"]) if matrix is not None else 0
    u2r_support = int(matrix.loc["U2R"].sum()) if matrix is not None else 0
    u2r_missed = u2r_support - u2r_detected
    r2l_table = per_class_table(evaluation, "test_set", "Random Forest")
    r2l_recall = (
        float(r2l_table.loc[r2l_table["Classe"] == "R2L", "Recall"].iloc[0])
        if r2l_table is not None
        else 0.0
    )
    with left:
        st.markdown(
            """
            <div class="card">
              <h3>Ne pas confondre Accuracy et Macro F1</h3>
              <p>
                Le Random Forest atteint 98,83 % d'Accuracy mais 93,56 % de Macro F1.
                Les classes Normal, DoS et Probe, très majoritaires, expliquent l'Accuracy.
                La Macro F1 donne le même poids aux cinq classes et révèle la difficulté
                sur U2R.
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        st.markdown(
            """
            <div class="card">
              <h3>Classes minoritaires</h3>
              <p>
                U2R ne compte que {u2r_support} flux dans le test : {u2r_detected} sont
                détectés et {u2r_missed} sont classés Normal. Un Recall U2R de
                {_percent(evaluation['test_set']['models']['Random Forest']['metrics']['recall_u2r'])}
                sur un aussi petit effectif doit être interprété avec prudence.
                R2L, plus représentée, atteint {_percent(r2l_recall)} de Recall.
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    _model_facts()
