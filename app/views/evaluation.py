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
from theme import (
    AMBER,
    EMERALD,
    INDIGO,
    RED,
    esc,
    kpi_cards,
    note,
    page_header,
    render_figure,
    section_title,
)
from views.common import get_evaluation

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
    if "macro_precision" in model and "macro_recall" in model:
        kpi_cards(
            [
                {
                    "label": "Accuracy",
                    "value": _percent(metrics["accuracy"]),
                    "sub": "Exactitude globale",
                    "tone": "indigo",
                },
                {
                    "label": "Macro F1",
                    "value": _percent(metrics["macro_f1"]),
                    "sub": "Moyenne équitable des 5 classes",
                    "tone": "emerald",
                },
                {
                    "label": "Precision macro",
                    "value": _percent(model["macro_precision"]),
                    "sub": "Fiabilité des alertes",
                    "tone": "violet",
                },
                {
                    "label": "Recall macro",
                    "value": _percent(model["macro_recall"]),
                    "sub": "Capacité de détection",
                    "tone": "amber",
                },
            ]
        )
    else:
        kpi_cards(
            [
                {
                    "label": "Accuracy",
                    "value": _percent(metrics["accuracy"]),
                    "sub": "Exactitude globale",
                    "tone": "indigo",
                },
                {
                    "label": "Macro F1",
                    "value": _percent(metrics["macro_f1"]),
                    "sub": "Moyenne équitable des 5 classes",
                    "tone": "emerald",
                },
                {
                    "label": "Weighted F1",
                    "value": _percent(metrics["weighted_f1"]),
                    "sub": "Pondérée par le support",
                    "tone": "violet",
                },
                {
                    "label": "FPR",
                    "value": _percent(metrics["fpr"]),
                    "sub": "Faux positifs un-contre-tous",
                    "tone": "red",
                },
            ]
        )


def _comparison_frame(evaluation: dict) -> pd.DataFrame:
    frame = comparison_table(evaluation)
    format_map = {
        column: "{:.2%}"
        for column in ["Accuracy", "Macro F1", "Weighted F1", "Recall R2L", "Recall U2R", "FPR"]
    }
    format_map["Latence (ms)"] = "{:.4f}"
    return frame.style.format(format_map, na_rep="—").apply(
        lambda row: [
            "background-color: #ECFDF5; font-weight: 600;"
            if row["Modèle"] == "Random Forest"
            else ""
            for _ in row
        ],
        axis=1,
    )


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
        color_discrete_sequence=[INDIGO, EMERALD, AMBER],
    )
    figure.update_traces(textposition="outside", cliponaxis=False, textfont_size=10)
    figure.update_layout(
        yaxis_range=[0, 1.12],
        yaxis_tickformat=".0%",
        margin=dict(t=48, b=8),
    )
    render_figure(figure, height=370)


def _confusion_chart(evaluation: dict, split: str, model_name: str) -> None:
    matrix = confusion_matrix_table(evaluation, split, model_name)
    normalised = normalized_confusion_matrix_table(evaluation, split, model_name)
    if matrix is None or normalised is None:
        return

    section_title("Matrice de confusion", "Effectifs ou taux par ligne")
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
        color_continuous_scale=[
            [0.0, "#F8FAFC"],
            [0.35, "#DBEAFE"],
            [0.7, "#93C5FD"],
            [1.0, "#3B82F6"],
        ],
        x=selected.columns,
        y=selected.index,
        aspect="auto",
        title="",
    )
    figure.update_traces(textfont=dict(size=12, color="#0F172A"))
    figure.update_xaxes(title_text="Classe prédite")
    figure.update_yaxes(title_text="Classe réelle")
    figure.update_layout(
        height=440,
        margin=dict(l=8, r=8, t=20, b=8),
        coloraxis_colorbar=dict(title="Flux", thickness=10, len=0.7),
    )
    render_figure(figure, height=440, legend=False)
    st.caption(_CONFUSION_INTERPRETATION)


def _detail_section(evaluation: dict, split: str, model_name: str) -> None:
    model = evaluation[split]["models"].get(model_name)
    if model is None:
        note("Modèle absent des résultats sauvegardés.", warning=True)
        return

    _metric_cards(model)
    st.markdown(
        f"<div class='ng-caption' style='margin:.6rem 0 .9rem 0;'>"
        f"Source : {esc(model.get('source', 'résultats sauvegardés'))} · "
        f"{model.get('support', evaluation[split]['size']):,} flux.</div>",
        unsafe_allow_html=True,
    )

    if not model.get("detail_available", False):
        note(
            f"Le détail par classe de <strong>{model_name}</strong> n'est pas disponible "
            "dans les résultats sauvegardés. Les métriques globales ci-dessus proviennent "
            "de l'export de la Semaine 3 ; aucune valeur n'est reconstituée ici.",
            warning=True,
        )
        return

    per_class = per_class_table(evaluation, split, model_name)
    if per_class is not None:
        section_title("Performance par classe", "Precision · Recall · F1 · support")
        st.dataframe(
            per_class.style.format(
                {"Precision": "{:.4f}", "Recall": "{:.4f}", "F1-score": "{:.4f}"}
            ).apply(
                lambda row: [
                    "background-color: #EEF4FF; font-weight: 600;" if row["Classe"] == "U2R" else ""
                    for _ in row
                ],
                axis=1,
            ),
            width="stretch",
            hide_index=True,
        )

    _confusion_chart(evaluation, split, model_name)


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


def _critical_cards(evaluation: dict) -> None:
    section_title("Lecture critique", "Ce que les chiffres ne disent pas")
    rf_test = evaluation["test_set"]["models"]["Random Forest"]["metrics"]
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

    left, right = st.columns(2, gap="small")
    with left:
        st.markdown(
            f"""
            <div class="ng-panel">
              <div class="icon">⚖️</div>
              <h3>Ne pas confondre Accuracy et Macro F1</h3>
              <p>
                Le Random Forest atteint <strong>{_percent(rf_test['accuracy'])}</strong>
                d'Accuracy mais <strong>{_percent(rf_test['macro_f1'])}</strong> de Macro F1.
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
            f"""
            <div class="ng-panel">
              <div class="icon">🔎</div>
              <h3>Classes minoritaires</h3>
              <p>
                U2R ne compte que <strong>{u2r_support} flux</strong> dans le test :
                <strong>{u2r_detected}</strong> sont détectés et
                <strong>{u2r_missed}</strong> sont classés Normal. Un Recall U2R de
                <strong>{_percent(rf_test['recall_u2r'])}</strong> sur un aussi petit
                effectif doit être interprété avec prudence. R2L, plus représentée,
                atteint <strong>{_percent(r2l_recall)}</strong> de Recall.
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render() -> None:
    page_header(
        "Model Evaluation",
        "Résultats expérimentaux sauvegardés du Random Forest, du MLP et du 1D-CNN.",
        ["Semaine 3", "Test 7 161 flux", "Non recalculé sur les imports"],
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
        section_title("Détail du modèle", "Résultats sauvegardés")
        model_names = list(evaluation["test_set"]["models"].keys())
        selected_model = st.selectbox(
            "Modèle à détailler",
            model_names,
            index=0,
            help="Le Random Forest possède le détail par classe vérifié ; les réseaux "
            "disposent de leurs métriques globales sauvegardées.",
        )
        st.caption(_MODEL_TYPES.get(selected_model, ""))
        _detail_section(evaluation, "test_set", selected_model)

        section_title("Tableau comparatif", "Random Forest · MLP · 1D-CNN")
        st.dataframe(_comparison_frame(evaluation), width="stretch", hide_index=True)
        _comparison_chart(evaluation)

        st.caption(
            "FPR : taux moyen de faux positifs un-contre-tous, calculé à partir de la matrice "
            "de confusion. La latence dépend du matériel de mesure ; les trois valeurs "
            "proviennent du même export de la Semaine 3."
        )
        if reproduction.get("test_verified"):
            note(
                "<strong>Reproductibilité vérifiée.</strong> Les métriques du Random Forest "
                "ont été recalculées à partir des artefacts et du découpage sauvegardé, avec "
                f"un écart maximal de <code>{reproduction['random_forest_max_abs_difference_test']:.1e}</code>.",
                success=True,
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
        section_title("Validation vs test — Random Forest", "Stabilité du modèle")
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

    _critical_cards(evaluation)
    _model_facts()
