"""Generate the figures used by the scientific report.

All values come from the verified evaluation report or from the reconstructed
dataset split.  Training curves and the SHAP summary are genuine outputs
extracted from the Week 3 notebook (see ``--extract-notebook``).
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nids import config  # noqa: E402
from nids.dataset import reproduce_split  # noqa: E402
from nids.evaluation import (  # noqa: E402
    confusion_matrix_table,
    load_model_evaluation,
    normalized_confusion_matrix_table,
    per_class_table,
)

FIGURES = PROJECT_ROOT / "reports" / "figures"
COLORS = ["#0f766e", "#f59e0b", "#64748b", "#dc2626", "#9333ea"]


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 160,
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
        }
    )


def class_distribution() -> None:
    split = reproduce_split(PROJECT_ROOT / "data" / "raw")
    counts = pd.DataFrame(
        {
            "Entraînement": split.y_train.value_counts(),
            "Validation": split.y_val.value_counts(),
            "Test": split.y_test.value_counts(),
        }
    ).reindex(list(config.CLASS_ORDER))
    axis = counts.plot(kind="bar", figsize=(8.2, 4.2), color=["#0f766e", "#f59e0b", "#64748b"])
    axis.set_title("Distribution des cinq classes après découpage stratifié")
    axis.set_xlabel("")
    axis.set_ylabel("Nombre de flux")
    axis.tick_params(axis="x", rotation=0)
    axis.legend(frameon=False)
    for container in axis.containers:
        axis.bar_label(container, fmt="%d", fontsize=7, padding=2, rotation=90)
    axis.set_ylim(0, counts.to_numpy().max() * 1.18)
    plt.tight_layout()
    plt.savefig(FIGURES / "fig_class_distribution.png")
    plt.close()


def model_comparison(evaluation: dict) -> None:
    metrics = ["accuracy", "macro_f1", "weighted_f1"]
    labels = ["Accuracy", "Macro F1", "Weighted F1"]
    models = list(evaluation["test_set"]["models"].keys())
    values = np.array(
        [[evaluation["test_set"]["models"][model]["metrics"][key] for key in metrics] for model in models]
    )
    x = np.arange(len(models))
    width = 0.24
    figure, axis = plt.subplots(figsize=(7.6, 4.1))
    for index, label in enumerate(labels):
        bars = axis.bar(x + (index - 1) * width, values[:, index], width, label=label, color=COLORS[index])
        axis.bar_label(bars, fmt="%.3f", fontsize=7, padding=2, rotation=90)
    axis.set_xticks(x, models)
    axis.set_ylim(0, 1.15)
    axis.set_ylabel("Score")
    axis.set_title("Comparaison des modèles sur le jeu de test (7 161 flux)")
    axis.legend(frameon=False, ncols=3, loc="upper center")
    plt.tight_layout()
    plt.savefig(FIGURES / "fig_model_comparison.png")
    plt.close()


def rf_confusion(evaluation: dict) -> None:
    counts = confusion_matrix_table(evaluation)
    normalised = normalized_confusion_matrix_table(evaluation)
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
    for axis, frame, title, fmt in (
        (axes[0], counts, "Effectifs", "d"),
        (axes[1], normalised, "Normalisée par ligne", ".2f"),
    ):
        image = axis.imshow(frame.to_numpy(), cmap="Blues", vmin=0)
        axis.set_xticks(range(len(frame.columns)), frame.columns, rotation=45, ha="right")
        axis.set_yticks(range(len(frame.index)), frame.index)
        axis.set_xlabel("Classe prédite")
        axis.set_ylabel("Classe réelle")
        axis.set_title(title)
        axis.grid(False)
        threshold = frame.to_numpy().max() / 2
        for row in range(frame.shape[0]):
            for column in range(frame.shape[1]):
                value = frame.iloc[row, column]
                axis.text(
                    column,
                    row,
                    format(value, fmt),
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="white" if value > threshold else "#0f172a",
                )
    figure.suptitle("Random Forest — matrice de confusion sur le jeu de test", fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGURES / "fig_confusion_rf_test.png")
    plt.close()


def per_class_f1(evaluation: dict) -> None:
    table = per_class_table(evaluation).set_index("Classe").reindex(list(config.CLASS_ORDER))
    figure, axis = plt.subplots(figsize=(7.2, 3.8))
    bars = axis.bar(table.index, table["F1-score"], color=["#16a34a", "#dc2626", "#ea580c", "#9333ea", "#0f766e"])
    axis.bar_label(bars, fmt="%.3f", padding=3)
    axis.set_ylim(0, 1.12)
    axis.set_ylabel("F1-score")
    axis.set_title("Random Forest — F1-score par classe (test)")
    for index, (name, row) in enumerate(table.iterrows()):
        axis.text(index, 0.03, f"n={int(row['Support'])}", ha="center", color="white", fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURES / "fig_per_class_f1.png")
    plt.close()


def feature_importance() -> None:
    preprocessor = joblib.load(config.MODELS_DIR / config.PREPROCESSOR_PATH)
    model = joblib.load(config.MODELS_DIR / config.RF_MODEL_PATH)
    names = preprocessor.get_feature_names_out()

    def readable(name: str) -> str:
        if name.startswith("num__"):
            return name.removeprefix("num__")
        if name.startswith("cat__"):
            body = name.removeprefix("cat__")
            for column in ("protocol_type", "service", "flag"):
                prefix = f"{column}_"
                if body.startswith(prefix):
                    return f"{column}={body[len(prefix):]}"
            return body
        return name

    importances = pd.Series(model.feature_importances_, index=[readable(name) for name in names])
    top = importances.sort_values(ascending=False).head(15).iloc[::-1]
    figure, axis = plt.subplots(figsize=(7.6, 5.2))
    top.plot(kind="barh", ax=axis, color="#0f766e")
    axis.set_title("Random Forest — 15 variables les plus importantes")
    axis.set_xlabel("Importance moyenne (réduction d'impureté)")
    plt.tight_layout()
    plt.savefig(FIGURES / "fig_feature_importance.png")
    plt.close()
    with (PROJECT_ROOT / "reports" / "feature_importance.json").open("w", encoding="utf-8") as handle:
        json.dump(
            [{"feature": name, "importance": float(value)} for name, value in top.iloc[::-1].items()],
            handle,
            indent=2,
            ensure_ascii=False,
        )


def extract_notebook(notebook: Path) -> None:
    """Save genuine training/SHAP figures embedded in the notebook outputs."""

    wanted = {
        (13, 6): "fig_training_mlp.png",
        (15, 7): "fig_training_cnn.png",
        (17, 1): "fig_confusion_validation.png",
        (18, 2): "fig_shap_summary.png",
        (5, 2): "fig_class_distribution_notebook.png",
    }
    with notebook.open(encoding="utf-8") as handle:
        data = json.load(handle)
    for (cell_index, output_index), filename in wanted.items():
        try:
            payload = data["cells"][cell_index]["outputs"][output_index]["data"]["image/png"]
        except (KeyError, IndexError):
            continue
        if isinstance(payload, list):
            payload = "".join(payload)
        (FIGURES / filename).write_bytes(base64.b64decode(payload))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--extract-notebook",
        action="store_true",
        help="Extraire les figures déjà calculées dans le notebook de modélisation.",
    )
    args = parser.parse_args()
    FIGURES.mkdir(parents=True, exist_ok=True)
    _style()
    evaluation = load_model_evaluation(PROJECT_ROOT / "reports")
    class_distribution()
    model_comparison(evaluation)
    rf_confusion(evaluation)
    per_class_f1(evaluation)
    feature_importance()
    if args.extract_notebook:
        extract_notebook(PROJECT_ROOT / "notebooks" / "NetGuard_AI_Modeling.ipynb")
    print(f"Figures écrites dans {FIGURES}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
