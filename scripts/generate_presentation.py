"""Generate the oral-defence presentation (PowerPoint).

Usage:
    python scripts/generate_presentation.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Cm, Pt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nids.evaluation import load_model_evaluation  # noqa: E402

DOCS = PROJECT_ROOT / "docs"
FIGURES = PROJECT_ROOT / "reports" / "figures"

NAVY = RGBColor(0x0B, 0x12, 0x20)
TEAL = RGBColor(0x0F, 0x76, 0x6E)
MINT = RGBColor(0x5E, 0xEA, 0xD4)
SLATE = RGBColor(0x54, 0x66, 0x7E)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
LIGHT = RGBColor(0xF5, 0xF7, 0xFB)

W, H = Cm(33.867), Cm(19.05)


def pct(value: float, digits: int = 2) -> str:
    return f"{value * 100:.{digits}f} %".replace(".", ",")


def fr(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}".replace(".", ",")


class Deck:
    def __init__(self) -> None:
        self.presentation = Presentation()
        self.presentation.slide_width = W
        self.presentation.slide_height = H
        self.blank = self.presentation.slide_layouts[6]
        self.index = 0

    def _background(self, slide, color) -> None:
        fill = slide.background.fill
        fill.solid()
        fill.fore_color.rgb = color

    def _text(self, slide, left, top, width, height, text, size=18, bold=False,
              color=NAVY, align=PP_ALIGN.LEFT, italic=False):
        box = slide.shapes.add_textbox(left, top, width, height)
        frame = box.text_frame
        frame.word_wrap = True
        paragraph = frame.paragraphs[0]
        paragraph.alignment = align
        run = paragraph.add_run()
        run.text = text
        run.font.size = Pt(size)
        run.font.bold = bold
        run.font.italic = italic
        run.font.color.rgb = color
        run.font.name = "Calibri"
        return box

    def _bars(self, slide, color=TEAL) -> None:
        slide.shapes.add_shape(1, 0, 0, W, Cm(0.25)).fill.solid()
        slide.shapes[-1].fill.fore_color.rgb = color
        slide.shapes[-1].line.fill.background()

    def slide(self, title: str, subtitle: str | None = None) -> object:
        slide = self.presentation.slides.add_slide(self.blank)
        self._background(slide, WHITE)
        self._bars(slide)
        self.index += 1
        self._text(slide, Cm(2.2), Cm(1.2), Cm(28), Cm(1.5), title, 30, True, NAVY)
        if subtitle:
            self._text(slide, Cm(2.2), Cm(2.8), Cm(28), Cm(1.1), subtitle, 14, False, SLATE)
        self._text(slide, Cm(2.2), Cm(17.9), Cm(12), Cm(0.8),
                   "NetGuard AI — NIDS multiclasse NSL-KDD", 9, False, SLATE)
        self._text(slide, Cm(30.0), Cm(17.9), Cm(2), Cm(0.8), str(self.index), 10, True, SLATE,
                   PP_ALIGN.RIGHT)
        return slide

    def bullets(self, slide, items, left=Cm(2.4), top=Cm(4.0), width=Cm(17.5),
                size=16, spacing=1.35) -> None:
        box = slide.shapes.add_textbox(left, top, width, Cm(12))
        frame = box.text_frame
        frame.word_wrap = True
        for index, item in enumerate(items):
            paragraph = frame.paragraphs[0] if index == 0 else frame.add_paragraph()
            paragraph.space_after = Pt(9)
            paragraph.line_spacing = spacing
            run = paragraph.add_run()
            run.text = "•  " + item
            run.font.size = Pt(size)
            run.font.color.rgb = NAVY
            run.font.name = "Calibri"

    def note(self, slide, text, top=Cm(15.6), color=TEAL) -> None:
        bar = slide.shapes.add_shape(1, Cm(2.2), top, Cm(0.12), Cm(1.15))
        bar.fill.solid()
        bar.fill.fore_color.rgb = color
        bar.line.fill.background()
        self._text(slide, Cm(2.6), top + Cm(0.05), Cm(27.5), Cm(1.2), text, 12, False, SLATE,
                   italic=True)

    def picture(self, slide, filename, left, top, width) -> None:
        path = FIGURES / filename
        if path.is_file():
            slide.shapes.add_picture(str(path), left, top, width=width)

    def metric_cards(self, slide, values) -> None:
        count = len(values)
        gap = Cm(0.5)
        total = Cm(29.4)
        card_width = int((total - gap * (count - 1)) / count)
        left = Cm(2.2)
        for label, value in values:
            shape = slide.shapes.add_shape(1, left, Cm(4.1), card_width, Cm(2.5))
            shape.fill.solid()
            shape.fill.fore_color.rgb = LIGHT
            shape.line.color.rgb = RGBColor(0xE2, 0xE8, 0xF0)
            self._text(slide, left + Cm(0.5), Cm(4.35), card_width - Cm(1), Cm(0.8),
                       label.upper(), 10, True, SLATE)
            self._text(slide, left + Cm(0.5), Cm(5.15), card_width - Cm(1), Cm(1.2),
                       value, 24, True, TEAL)
            left += card_width + gap

    def table(self, slide, headers, rows, left=Cm(2.2), top=Cm(4.6), width=Cm(29.4),
              font=12, row_height=Cm(1.05)) -> None:
        shape = slide.shapes.add_table(len(rows) + 1, len(headers), left, top, width,
                                       row_height * (len(rows) + 1))
        table = shape.table
        for index, header in enumerate(headers):
            cell = table.cell(0, index)
            cell.text = header
            cell.fill.solid()
            cell.fill.fore_color.rgb = TEAL
            paragraph = cell.text_frame.paragraphs[0]
            paragraph.alignment = PP_ALIGN.CENTER
            run = paragraph.runs[0]
            run.font.size = Pt(font)
            run.font.bold = True
            run.font.color.rgb = WHITE
        for row_index, row in enumerate(rows, start=1):
            for column_index, value in enumerate(row):
                cell = table.cell(row_index, column_index)
                cell.text = str(value)
                cell.fill.solid()
                cell.fill.fore_color.rgb = WHITE if row_index % 2 else LIGHT
                paragraph = cell.text_frame.paragraphs[0]
                paragraph.alignment = PP_ALIGN.CENTER
                run = paragraph.runs[0]
                run.font.size = Pt(font)
                run.font.color.rgb = NAVY

    def save(self, path: Path) -> None:
        self.presentation.save(path)


def build(evaluation: dict) -> Deck:
    deck = Deck()
    rf = evaluation["test_set"]["models"]["Random Forest"]
    mlp = evaluation["test_set"]["models"]["MLP"]
    cnn = evaluation["test_set"]["models"]["1D-CNN"]
    val = evaluation["validation_set"]["models"]["Random Forest"]["metrics"]

    # 1 — title
    slide = deck.presentation.slides.add_slide(deck.blank)
    deck._background(slide, NAVY)
    deck.index += 1
    accent = slide.shapes.add_shape(1, Cm(2.4), Cm(5.0), Cm(0.25), Cm(7.5))
    accent.fill.solid()
    accent.fill.fore_color.rgb = MINT
    accent.line.fill.background()
    deck._text(slide, Cm(3.2), Cm(4.5), Cm(25), Cm(1.0),
               "MACHINE LEARNING · CYBERSÉCURITÉ · NSL-KDD", 13, True, MINT)
    deck._text(slide, Cm(3.2), Cm(5.8), Cm(27), Cm(3.4), "NetGuard AI", 54, True, WHITE)
    deck._text(slide, Cm(3.2), Cm(9.2), Cm(26), Cm(2.2),
               "Détection multiclasse d'intrusions réseau\npar Machine Learning et Deep Learning",
               26, True, WHITE)
    deck._text(slide, Cm(3.2), Cm(13.2), Cm(26), Cm(1.2),
               "Random Forest · MLP · 1D-CNN  |  Split 70 / 15 / 15  |  5 classes",
               14, False, RGBColor(0xCB, 0xD5, 0xE1))
    deck._text(slide, Cm(3.2), Cm(15.6), Cm(26), Cm(1.0),
               "Mini-projet Machine Learning — 4ᵉ année ingénieur réseau", 12, False,
               RGBColor(0x94, 0xA3, 0xB8))

    # 2 — problem
    slide = deck.slide("Problématique", "Pourquoi un NIDS par apprentissage automatique ?")
    deck.bullets(slide, [
        "Les IDS à signatures reconnaissent les attaques connues mais manquent les variantes.",
        "Le trafic chiffré et les comportements lents limitent l'inspection par motifs.",
        "Objectif : classer un flux NSL-KDD parmi 5 catégories — Normal, DoS, Probe, R2L, U2R.",
        "Contrainte : mesurer honnêtement la performance sur des classes très déséquilibrées.",
        "Livrable : pipeline reproductible + application Streamlit + rapport scientifique.",
    ], width=Cm(19))
    deck.picture(slide, "fig_class_distribution.png", Cm(22.4), Cm(4.0), Cm(9.6))
    deck._text(slide, Cm(22.4), Cm(10.8), Cm(9.5), Cm(1.0),
               "U2R = 0,16 % du train", 12, True, SLATE)

    # 3 — data / pipeline
    slide = deck.slide("Données et pipeline", "NSL-KDD · 41 caractéristiques · 5 classes")
    steps = [
        ("1", "Fusion + dédoublonnage", "25 192 + 22 544 lignes brutes, 0 doublon"),
        ("2", "Split stratifié 70/15/15", "33 415 / 7 160 / 7 161 · graine 42"),
        ("3", "Prétraitement ajusté sur train", "38 num. standardisées + 3 cat. One-Hot → 118"),
        ("4", "Modélisation", "Random Forest · MLP · 1D-CNN, class_weight"),
        ("5", "Évaluation unique sur test", "Macro F1, Recall par classe, matrice de confusion"),
    ]
    top = Cm(4.2)
    for number, title, detail in steps:
        bubble = slide.shapes.add_shape(9, Cm(2.4), top, Cm(1.5), Cm(1.5))
        bubble.fill.solid()
        bubble.fill.fore_color.rgb = TEAL
        bubble.line.fill.background()
        deck._text(slide, Cm(2.4), top + Cm(0.25), Cm(1.5), Cm(1.0), number, 14, True, WHITE,
                   PP_ALIGN.CENTER)
        deck._text(slide, Cm(4.4), top + Cm(0.1), Cm(11), Cm(0.9), title, 16, True, NAVY)
        deck._text(slide, Cm(15.6), top + Cm(0.18), Cm(15.5), Cm(1.2), detail, 12, False, SLATE)
        top += Cm(2.25)

    # 4 — models
    slide = deck.slide("Trois modèles comparés", "Un modèle classique contre deux réseaux de neurones")
    deck.table(
        slide,
        ["Élément", "Random Forest", "MLP", "1D-CNN"],
        [
            ["Famille", "ML classique", "Deep Learning", "Deep Learning"],
            ["Entrée", "118 features", "118 features", "118 × 1"],
            ["Architecture", "200 arbres, depth=20", "128 → 64 → Softmax", "Conv 64 → Conv 128 → Dense"],
            ["Régularisation", "class_weight", "Dropout 0,3 / 0,2", "Dropout 0,3"],
            ["Sélection", "Macro F1 (validation)", "Early Stopping", "Early Stopping"],
        ],
        top=Cm(4.4),
        font=13,
        row_height=Cm(1.5),
    )
    deck.note(slide, "Le Random Forest est le modèle déployé ; les réseaux restent une comparaison de recherche "
                     "et ne sont pas exécutés dans l'application.", top=Cm(15.5))

    # 5 — results
    slide = deck.slide("Résultats sur le jeu de test", "7 161 flux · évaluation finale unique")
    deck.table(
        slide,
        ["Modèle", "Accuracy", "Macro F1", "Weighted F1", "Recall R2L", "Recall U2R"],
        [
            ["Random Forest", pct(rf["metrics"]["accuracy"]), pct(rf["metrics"]["macro_f1"]),
             pct(rf["metrics"]["weighted_f1"]), pct(rf["metrics"]["recall_r2l"]),
             pct(rf["metrics"]["recall_u2r"])],
            ["MLP", pct(mlp["metrics"]["accuracy"]), pct(mlp["metrics"]["macro_f1"]),
             pct(mlp["metrics"]["weighted_f1"]), pct(mlp["metrics"]["recall_r2l"]),
             pct(mlp["metrics"]["recall_u2r"])],
            ["1D-CNN", pct(cnn["metrics"]["accuracy"]), pct(cnn["metrics"]["macro_f1"]),
             pct(cnn["metrics"]["weighted_f1"]), pct(cnn["metrics"]["recall_r2l"]),
             pct(cnn["metrics"]["recall_u2r"])],
        ],
        top=Cm(3.9), font=12, row_height=Cm(1.25),
    )
    deck.picture(slide, "fig_model_comparison.png", Cm(2.2), Cm(8.4), Cm(19.6))

    # 6 — per class
    slide = deck.slide("Random Forest : performance par classe", "L'Accuracy cache les classes rares")
    deck.picture(slide, "fig_per_class_f1.png", Cm(2.2), Cm(4.2), Cm(15.5))
    deck.bullets(slide, [
        "Normal, DoS, Probe : F1 ≥ 98,8 %.",
        "R2L : 93,97 % de F1 sur 464 flux.",
        "U2R : 76,19 % de F1 sur 12 flux seulement.",
        "4 U2R classés Normal, 1 classé R2L.",
    ], left=Cm(19.4), top=Cm(4.4), width=Cm(12.2), size=14)
    deck.note(slide, "Recall U2R = 66,67 % ; intervalle de Wilson à 95 % ≈ [39,1 % ; 86,2 %]. "
                     "Effectif trop faible pour conclure.", top=Cm(15.4))

    # 7 — confusion
    slide = deck.slide("Matrice de confusion", "Où se concentrent les erreurs ?")
    deck.picture(slide, "fig_confusion_rf_test.png", Cm(2.2), Cm(4.2), Cm(20))
    deck.bullets(slide, [
        "Erreurs R2L → Normal : 19 flux.",
        "Normal → R2L : 37 flux (fausses alertes).",
        "U2R → Normal : 4 flux, jamais signalés.",
        "Aucune confusion entre DoS et Probe.",
    ], left=Cm(23.2), top=Cm(4.6), width=Cm(8.8), size=13)

    # 8 — validation vs test
    slide = deck.slide("Validation vs test", "Pas de surapprentissage manifeste sur ce découpage")
    deck.metric_cards(slide, [
        ("Accuracy validation", pct(val["accuracy"])),
        ("Accuracy test", pct(rf["metrics"]["accuracy"])),
        ("Macro F1 validation", pct(val["macro_f1"])),
        ("Macro F1 test", pct(rf["metrics"]["macro_f1"])),
    ])
    deck.bullets(slide, [
        "Sélection des hyperparamètres uniquement sur la validation.",
        "Test utilisé une seule fois, après gel du modèle.",
        "Écart validation/test inférieur à 0,5 point.",
        "Script de vérification : écart maximal de 0,0e+00 sur le test.",
    ], top=Cm(7.4), width=Cm(26), size=15)
    deck.note(slide, "Stabilité interne confirmée, mais elle ne corrige pas la limite de validité externe "
                     "liée à la fusion des fichiers.", top=Cm(15.0))

    # 9 — critical limitation
    slide = deck.slide("Limite méthodologique majeure", "Un test indépendant, mais pas le protocole officiel")
    deck.bullets(slide, [
        "KDDTrain+_20Percent et KDDTest+ ont été fusionnés avant le split.",
        "Des attaques du test officiel (mscan, apache2, processtable…) sont donc vues à l'entraînement.",
        "Aucune fuite de prétraitement : scaler et encodeur ajustés sur le train seul.",
        "Mais les scores ne mesurent pas la généralisation à des attaques inédites.",
        "Conséquence : 98,83 % d'Accuracy n'est pas comparable au benchmark NSL-KDD standard.",
    ], width=Cm(27), size=16)
    deck.note(slide, "Perspective n°1 : réentraîner sans fusion et évaluer sur KDDTest+ officiel.", top=Cm(15.2),
              color=RGBColor(0xDC, 0x26, 0x26))

    # 10 — application
    slide = deck.slide("Application Streamlit", "Quatre pages, une seule logique métier testée")
    deck.table(
        slide,
        ["Page", "Contenu"],
        [
            ["Accueil", "Contexte, 5 catégories, résultats de test, périmètre"],
            ["Analyse", "Import CSV, validation, prédictions, confiance, export"],
            ["Attack Analytics", "Répartition, alertes, histogramme de confiance"],
            ["Model Evaluation", "Métriques sauvegardées, comparaison, matrices"],
        ],
        top=Cm(4.2), width=Cm(28.5), font=14, row_height=Cm(1.55),
    )
    deck.note(slide, "L'interface n'entraîne rien et ne recalcule aucune métrique expérimentale. "
                     "Les résultats affichés sont lus depuis un rapport vérifié.", top=Cm(13.6))

    # 11 — tests & reproducibility
    slide = deck.slide("Tests et reproductibilité", "Ce qui est vérifié automatiquement")
    deck.metric_cards(slide, [
        ("Tests pytest", "51"),
        ("Prédictions figées", "92 flux"),
        ("Pages testées", "4"),
        ("Écart RF test", "0,0e+00"),
    ])
    deck.bullets(slide, [
        "CSV valide, colonnes manquantes ou en trop, valeurs manquantes, fichier vide, format incorrect.",
        "Une ligne ou plusieurs milliers, cohérence du prétraitement, ordre des colonnes.",
        "Statistiques d'attaque et intégrité du rapport d'évaluation.",
        "Démarrage réel des quatre pages via streamlit.testing.AppTest.",
    ], top=Cm(7.6), width=Cm(28), size=14)
    deck.note(slide, "Toute régression de l'ordre des features, du préprocesseur ou du modèle fait échouer la suite.",
              top=Cm(14.8))

    # 12 — conclusion
    slide = deck.slide("Conclusion", "Rigueur du pipeline avant le score")
    deck.bullets(slide, [
        "Chaîne complète : données → modèles → évaluation → application → documentation.",
        "Random Forest retenu : 98,83 % d'Accuracy et 93,56 % de Macro F1.",
        "Résultat solide sur Normal, DoS, Probe et R2L ; fragile sur U2R (12 flux).",
        "Preuve de reproductibilité indépendante et 51 tests automatisés.",
        "Limite assumée : test non canonique, scores optimistes, non comparables à l'état de l'art.",
    ], width=Cm(28), size=16)
    deck.note(slide, "Prochaine étape scientifique : évaluation sur KDDTest+ officiel sans fusion.", top=Cm(15.2))

    # 13 — thank you
    slide = deck.presentation.slides.add_slide(deck.blank)
    deck._background(slide, NAVY)
    deck.index += 1
    deck._text(slide, Cm(3.0), Cm(7.0), Cm(27), Cm(2.0), "Merci de votre attention", 40, True, WHITE)
    deck._text(slide, Cm(3.0), Cm(9.4), Cm(27), Cm(1.2), "Questions & démonstration", 20, False, MINT)
    deck._text(slide, Cm(3.0), Cm(14.0), Cm(27), Cm(1.0),
               "NetGuard AI — Mini-projet Machine Learning", 12, False, RGBColor(0x94, 0xA3, 0xB8))

    return deck


def main() -> int:
    DOCS.mkdir(parents=True, exist_ok=True)
    evaluation = load_model_evaluation(PROJECT_ROOT / "reports")
    deck = build(evaluation)
    path = DOCS / "NetGuard_AI_Soutenance.pptx"
    deck.save(path)
    print(f"Présentation écrite : {path} ({len(deck.presentation.slides)} diapositives)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
