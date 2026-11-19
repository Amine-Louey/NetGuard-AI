"""Generate the final academic report (DOCX + PDF).

Usage:
    python scripts/generate_report.py
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Iterable

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nids.dataset import reproduce_split  # noqa: E402
from nids.evaluation import load_model_evaluation  # noqa: E402

DOCS = PROJECT_ROOT / "docs"
FIGURES = PROJECT_ROOT / "reports" / "figures"
TEAL = RGBColor(0x0F, 0x76, 0x6E)
NAVY = RGBColor(0x0B, 0x12, 0x20)
GREY = RGBColor(0x5B, 0x6B, 0x82)


def fr(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}".replace(".", ",")


def pct(value: float, digits: int = 2) -> str:
    return f"{fr(value * 100, digits)} %"


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    proportion = k / n
    denominator = 1 + z * z / n
    centre = (proportion + z * z / (2 * n)) / denominator
    half = z * math.sqrt(proportion * (1 - proportion) / n + z * z / (4 * n * n)) / denominator
    return centre - half, centre + half


class Report:
    def __init__(self) -> None:
        self.document = Document()
        self._setup_page()
        self._setup_styles()

    def _setup_page(self) -> None:
        section = self.document.sections[0]
        section.page_height = Cm(29.7)
        section.page_width = Cm(21.0)
        section.top_margin = Cm(2.0)
        section.bottom_margin = Cm(1.9)
        section.left_margin = Cm(2.2)
        section.right_margin = Cm(2.2)
        section.header_distance = Cm(1.0)
        section.footer_distance = Cm(0.9)

    def _setup_styles(self) -> None:
        normal = self.document.styles["Normal"]
        normal.font.name = "Calibri"
        normal.font.size = Pt(10.5)
        normal.paragraph_format.space_after = Pt(4)
        normal.paragraph_format.line_spacing = 1.12
        for level, size in ((1, 15), (2, 12.5), (3, 11)):
            style = self.document.styles[f"Heading {level}"]
            style.font.name = "Calibri"
            style.font.size = Pt(size)
            style.font.color.rgb = TEAL if level == 1 else NAVY
            style.font.bold = True
            style.paragraph_format.space_before = Pt(12 if level == 1 else 8)
            style.paragraph_format.space_after = Pt(4)
            style.paragraph_format.keep_with_next = True

    def h1(self, text: str) -> None:
        self.document.add_heading(text, level=1)

    def h2(self, text: str) -> None:
        self.document.add_heading(text, level=2)

    def h3(self, text: str) -> None:
        self.document.add_heading(text, level=3)

    def p(self, text: str, bold: bool = False, italic: bool = False, size: float | None = None) -> None:
        paragraph = self.document.add_paragraph()
        run = paragraph.add_run(text)
        run.bold = bold
        run.italic = italic
        if size:
            run.font.size = Pt(size)
        paragraph.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

    def bullets(self, items: Iterable[str]) -> None:
        for item in items:
            paragraph = self.document.add_paragraph(item, style="List Bullet")
            paragraph.paragraph_format.space_after = Pt(2)
            paragraph.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

    def numbered(self, items: Iterable[str]) -> None:
        for item in items:
            paragraph = self.document.add_paragraph(item, style="List Number")
            paragraph.paragraph_format.space_after = Pt(2)

    def code(self, text: str) -> None:
        paragraph = self.document.add_paragraph()
        run = paragraph.add_run(text)
        run.font.name = "Consolas"
        run.font.size = Pt(8.5)
        paragraph.paragraph_format.left_indent = Cm(0.6)
        paragraph.paragraph_format.space_before = Pt(2)
        paragraph.paragraph_format.space_after = Pt(6)

    def caption(self, text: str, table: bool = False) -> None:
        paragraph = self.document.add_paragraph()
        run = paragraph.add_run(("Tableau " if table else "Figure ") + text)
        run.italic = True
        run.font.size = Pt(8.5)
        run.font.color.rgb = GREY
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        paragraph.paragraph_format.space_after = Pt(8)

    def figure(self, filename: str, caption: str, width: float = 15.5) -> None:
        path = FIGURES / filename
        if not path.is_file():
            self.p(f"[Figure absente : {filename}]", italic=True)
            return
        self.document.add_picture(str(path), width=Cm(width))
        self.document.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
        self.caption(caption)

    def _shade(self, cell, color: str) -> None:
        shading = OxmlElement("w:shd")
        shading.set(qn("w:fill"), color)
        cell._tc.get_or_add_tcPr().append(shading)

    def table(
        self,
        headers: list[str],
        rows: list[list[str]],
        caption: str,
        widths: list[float] | None = None,
        font_size: float = 8.8,
    ) -> None:
        table = self.document.add_table(rows=1, cols=len(headers))
        table.style = "Table Grid"
        table.alignment = WD_TABLE_ALIGNMENT.CENTER
        table.autofit = True
        header_cells = table.rows[0].cells
        for index, header in enumerate(headers):
            header_cells[index].text = ""
            run = header_cells[index].paragraphs[0].add_run(header)
            run.bold = True
            run.font.size = Pt(font_size)
            run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
            header_cells[index].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
            self._shade(header_cells[index], "0F766E")
        for row in rows:
            cells = table.add_row().cells
            for index, value in enumerate(row):
                cells[index].text = ""
                run = cells[index].paragraphs[0].add_run(str(value))
                run.font.size = Pt(font_size)
                cells[index].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
        if widths:
            for row in table.rows:
                for index, width in enumerate(widths):
                    row.cells[index].width = Cm(width)
        self.caption(caption, table=True)

    def page_break(self) -> None:
        self.document.add_paragraph().add_run().add_break(WD_BREAK.PAGE)

    def footer(self) -> None:
        for section in self.document.sections:
            footer = section.footer
            paragraph = footer.paragraphs[0]
            paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
            run = paragraph.add_run("NetGuard AI — Rapport final · ")
            run.font.size = Pt(8)
            run.font.color.rgb = GREY
            field_begin = OxmlElement("w:fldChar")
            field_begin.set(qn("w:fldCharType"), "begin")
            instruction = OxmlElement("w:instrText")
            instruction.set(qn("xml:space"), "preserve")
            instruction.text = "PAGE"
            field_end = OxmlElement("w:fldChar")
            field_end.set(qn("w:fldCharType"), "end")
            run._r.append(field_begin)
            run._r.append(instruction)
            run._r.append(field_end)


def cover(report: Report, evaluation: dict) -> None:
    document = report.document
    document.add_paragraph()
    p = document.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run("UNIVERSITÉ DJILALI LIABES DE SIDI BEL ABBÈS")
    run.bold = True
    run.font.size = Pt(12)
    run.font.color.rgb = NAVY
    p2 = document.add_paragraph()
    p2.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p2.add_run("Faculté des sciences exactes · Département d'informatique\n"
                     "Module : Machine Learning · 4ᵉ année ingénieur réseau")
    run.font.size = Pt(10)
    run.font.color.rgb = GREY

    document.add_paragraph()
    document.add_paragraph()
    title = document.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = title.add_run("NetGuard AI")
    run.bold = True
    run.font.size = Pt(34)
    run.font.color.rgb = TEAL
    subtitle = document.add_paragraph()
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = subtitle.add_run(
        "Conception et implémentation d'un système multiclasse de détection "
        "d'intrusions réseau par Machine Learning et Deep Learning"
    )
    run.font.size = Pt(16)
    run.bold = True
    run.font.color.rgb = NAVY
    line = document.add_paragraph()
    line.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = line.add_run("Dataset NSL-KDD · 5 classes · Random Forest, MLP et 1D-CNN")
    run.italic = True
    run.font.size = Pt(11)
    run.font.color.rgb = GREY

    document.add_paragraph()
    document.add_paragraph()
    report.table(
        ["Élément", "Valeur"],
        [
            ["Projet", "Mini-projet de Machine Learning — 4 semaines"],
            ["Dataset", "NSL-KDD (KDDTrain+_20Percent, KDDTest+)"],
            ["Tâche", "Classification multiclasse supervisée (5 classes)"],
            ["Modèle déployé", evaluation["metadata"]["selected_model"]],
            ["Application", "Streamlit — analyse de flux fournis en CSV"],
            ["Année académique", "2026 – 2027"],
        ],
        "Identification du projet.",
        widths=[4.5, 10.5],
        font_size=10,
    )

    document.add_paragraph()
    note = document.add_paragraph()
    note.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = note.add_run(
        "Rapport scientifique final — les résultats présentés proviennent des expériences "
        "réellement exécutées et des artefacts sauvegardés."
    )
    run.italic = True
    run.font.size = Pt(9)
    run.font.color.rgb = GREY
    report.page_break()


def summary(report: Report, evaluation: dict) -> None:
    rf = evaluation["test_set"]["models"]["Random Forest"]["metrics"]
    mlp = evaluation["test_set"]["models"]["MLP"]["metrics"]
    cnn = evaluation["test_set"]["models"]["1D-CNN"]["metrics"]
    report.h1("Résumé")
    report.p(
        "Ce rapport présente la conception, l'entraînement, l'évaluation et la mise en "
        "application d'un système de détection d'intrusions réseau (NIDS) multiclasse fondé "
        "sur le jeu de données NSL-KDD. Cinq catégories sont distinguées : Normal, DoS, "
        "Probe, R2L et U2R. Le pipeline nettoie les données, encode les variables "
        "catégorielles par One-Hot Encoding, standardise les variables numériques, puis "
        "découpe les flux en 70 % d'entraînement, 15 % de validation et 15 % de test de "
        "manière stratifiée. Le déséquilibre est traité par pondération des classes ; une "
        "variante SMOTE a été construite mais n'a pas été retenue pour les modèles finaux."
    )
    report.p(
        f"Trois modèles sont comparés : un Random Forest (Machine Learning classique), un "
        f"perceptron multicouche (MLP) et un réseau convolutif 1D (1D-CNN). Sur le jeu de "
        f"test retenu, le Random Forest obtient une Accuracy de {pct(rf['accuracy'])} et une "
        f"Macro F1 de {pct(rf['macro_f1'])} ; le MLP atteint {pct(mlp['accuracy'])} et "
        f"{pct(mlp['macro_f1'])} ; le 1D-CNN {pct(cnn['accuracy'])} et {pct(cnn['macro_f1'])}. "
        "Le Random Forest est donc sélectionné et intégré à une application Streamlit qui "
        "valide les fichiers CSV, applique le prétraitement gelé et affiche les prédictions, "
        "les statistiques d'attaques et les résultats expérimentaux."
    )
    report.p(
        "Le rapport insiste sur une limite méthodologique majeure : le découpage a été "
        "réalisé après fusion des fichiers d'entraînement et de test NSL-KDD. Le jeu de test "
        "est indépendant de l'entraînement, mais il ne reproduit pas le protocole officiel "
        "où le fichier de test contient des attaques inédites. Les scores sont donc "
        "optimistes par rapport aux benchmarks NSL-KDD canoniques et ne doivent pas être "
        "présentés comme un état de l'art."
    )
    report.p(
        "Mots-clés : détection d'intrusions, NSL-KDD, classification multiclasse, Random "
        "Forest, MLP, 1D-CNN, déséquilibre de classes, Streamlit, reproductibilité.",
        italic=True,
        size=9.5,
    )

    report.h1("Sommaire")
    for item in [
        "1. Introduction et problématique",
        "2. État de l'art",
        "3. Dataset NSL-KDD",
        "4. Préparation des données",
        "5. Architectures et hyperparamètres",
        "6. Protocole d'évaluation et métriques",
        "7. Résultats expérimentaux",
        "8. Analyse critique et menaces de validité",
        "9. Application développée",
        "10. Limites et perspectives",
        "11. Conclusion",
        "Références",
    ]:
        paragraph = report.document.add_paragraph(item)
        paragraph.paragraph_format.space_after = Pt(1)
    report.page_break()


def introduction(report: Report) -> None:
    report.h1("1. Introduction et problématique")
    report.p(
        "L'expansion des infrastructures interconnectées et la sophistication croissante "
        "des menaces rendent insuffisants les mécanismes de sécurité statiques fondés sur "
        "des signatures. Un pare-feu ou un IDS à signatures reconnaît les attaques connues, "
        "mais détecte mal les variantes, le trafic chiffré ou les comportements lents. "
        "L'apprentissage automatique offre une réponse complémentaire : apprendre, à partir "
        "d'exemples étiquetés, des motifs discriminants dans les caractéristiques "
        "statistiques des flux réseau."
    )
    report.p(
        "Le projet répond à une question précise : peut-on classer automatiquement un flux "
        "NSL-KDD parmi cinq catégories — Normal, DoS, Probe, R2L, U2R — avec un taux de "
        "détection élevé et un taux de fausses alertes maîtrisé, tout en conservant une "
        "démarche scientifique reproductible ? La difficulté centrale n'est pas seulement "
        "d'entraîner un modèle, mais de mesurer honnêtement sa performance sur des classes "
        "très déséquilibrées et de l'intégrer dans une application démontrable."
    )
    report.h2("1.1 Objectifs")
    report.bullets(
        [
            "Construire un pipeline de prétraitement sans fuite de données, ajusté sur le seul jeu d'entraînement.",
            "Entraîner et comparer trois familles de modèles : Random Forest, MLP et 1D-CNN.",
            "Gérer le déséquilibre des classes et analyser spécifiquement R2L et U2R.",
            "Évaluer avec Accuracy, Precision, Recall, F1, Macro F1, Weighted F1 et FPR.",
            "Développer une application Streamlit claire, testée et déployable publiquement.",
            "Documenter les limites et ne jamais substituer une valeur inventée à une mesure absente.",
        ]
    )
    report.h2("1.2 Contributions")
    report.p(
        "Le travail produit quatre contributions : (i) un pipeline reproductible et vérifié "
        "de reconstruction du découpage et des métriques ; (ii) une comparaison honnête "
        "entre un modèle d'arbres et deux réseaux de neurones sur des données tabulaires "
        "déséquilibrées ; (iii) un module d'inférence séparé de l'interface, avec validation "
        "stricte du contrat d'entrée ; (iv) une application qui distingue explicitement les "
        "prédictions d'un fichier utilisateur des performances expérimentales sauvegardées."
    )
    report.h2("1.3 Périmètre")
    report.p(
        "L'application analyse des flux fournis dans un fichier CSV au format NSL-KDD. Elle "
        "ne capture pas le trafic en direct, ne constitue pas un IDS de production et ne "
        "prétend pas remplacer un système de détection opérationnel. Les probabilités du "
        "Random Forest sont des fréquences de votes et non des probabilités calibrées."
    )


def state_of_the_art(report: Report) -> None:
    report.h1("2. État de l'art")
    report.h2("2.1 Détection d'intrusions : signatures et anomalies")
    report.p(
        "Un NIDS surveille le trafic pour identifier des activités malveillantes. Les "
        "approches par signatures comparent les paquets ou flux à une base de motifs connus : "
        "elles sont précises mais aveugles aux attaques nouvelles. Les approches par anomalies "
        "modélisent le comportement normal et signalent les écarts ; elles détectent davantage "
        "d'attaques inédites, au prix de fausses alertes. Le Machine Learning supervisé se "
        "situe entre les deux : il apprend des frontières de décision à partir d'exemples "
        "étiquetés, à condition que les classes rares soient correctement représentées."
    )
    report.h2("2.2 Modèles classiques et profonds")
    report.p(
        "Le Random Forest (Breiman, 2001) combine de nombreux arbres de décision entraînés "
        "sur des sous-échantillons et des sous-espaces de variables. Il est robuste, peu "
        "sensible à l'échelle des variables et fournit une importance globale des "
        "caractéristiques. Sur les données tabulaires, il constitue une référence difficile "
        "à battre. Le perceptron multicouche (MLP) empile des couches denses et des "
        "activations non linéaires ; il peut approximer des frontières complexes mais exige "
        "une standardisation, un réglage du taux d'apprentissage et une régularisation. Le "
        "réseau convolutif 1D applique des filtres locaux à une séquence ; il est naturel "
        "pour les signaux temporels, mais son usage sur des colonnes tabulaires est "
        "conventionnel : l'ordre des 118 features encodées n'a pas de signification spatiale."
    )
    report.p(
        "Les LSTM, parfois utilisés pour des séquences temporelles, n'ont pas été développés "
        "ici : les lignes NSL-KDD sont des flux indépendants, sans ordre temporel naturel. "
        "Cette décision évite d'ajouter une architecture non justifiée méthodologiquement."
    )
    report.h2("2.3 Déséquilibre et évaluation")
    report.p(
        "Les attaques rares (R2L, U2R) sont sous-représentées. Deux stratégies dominent : "
        "la pondération des classes, qui augmente le coût des erreurs sur les classes rares, "
        "et SMOTE (Chawla et al., 2002), qui synthétise des exemples par interpolation entre "
        "voisins. SMOTE doit rester strictement dans l'ensemble d'entraînement. L'Accuracy "
        "seule est trompeuse : un modèle qui prédit toujours « Normal » obtient un score "
        "élevé tout en manquant toutes les attaques. La Macro F1, moyenne non pondérée des "
        "F1 par classe, et le Recall par classe sont donc indispensables."
    )
    report.h2("2.4 Positionnement du projet")
    report.p(
        "Le projet ne cherche pas à proposer une nouvelle architecture. Il applique une "
        "démarche d'ingénierie rigoureuse : comparaison contrôlée, sélection sur validation, "
        "évaluation unique sur test, analyse des classes minoritaires, explicabilité et "
        "industrialisation légère. Sa valeur scientifique réside dans la transparence du "
        "protocole et dans l'analyse critique des résultats, y compris de leurs limites."
    )


def dataset(report: Report, evaluation: dict, split) -> None:
    report.h1("3. Dataset NSL-KDD")
    report.h2("3.1 Présentation")
    report.p(
        "NSL-KDD est une version corrigée du dataset KDD Cup 1999. Chaque ligne décrit une "
        "connexion réseau par 41 caractéristiques : durée, protocole, service, drapeaux, "
        "volumes d'octets, taux d'erreurs, nombres de connexions et statistiques d'hôte. "
        "La colonne attack_type contient le nom précis de l'attaque ; la colonne "
        "difficulty_level, retirée du modèle, indique la difficulté de classification."
    )
    report.p(
        "Les 39 types d'attaques sont regroupés en quatre familles : DoS (déni de service), "
        "Probe (reconnaissance), R2L (accès distant non autorisé) et U2R (élévation de "
        "privilèges), auxquelles s'ajoute la classe Normal. Le mapping a été vérifié : "
        "aucun type d'attaque des fichiers utilisés n'est resté sans correspondance."
    )
    report.h2("3.2 Fichiers et volumes")
    report.p(
        f"Deux fichiers publics ont été utilisés : KDDTrain+_20Percent "
        f"({evaluation['dataset']['raw_train_rows']:,} lignes) et KDDTest+ "
        f"({evaluation['dataset']['raw_test_rows']:,} lignes). Après dédoublonnage et retrait "
        f"des colonnes d'identification, {evaluation['dataset']['train_size'] + evaluation['dataset']['validation_size'] + evaluation['dataset']['test_size']:,} "
        f"flux ont été répartis en {evaluation['dataset']['train_size']:,} pour l'entraînement, "
        f"{evaluation['dataset']['validation_size']:,} pour la validation et "
        f"{evaluation['dataset']['test_size']:,} pour le test. Le nombre de doublons exacts "
        f"supprimés est de {evaluation['dataset']['duplicates_removed']}."
    )
    support = evaluation["test_set"]["class_support"]
    rows = []
    train_counts = split.y_train.value_counts()
    val_counts = split.y_val.value_counts()
    for class_name in ["Normal", "DoS", "Probe", "R2L", "U2R"]:
        rows.append(
            [
                class_name,
                f"{int(train_counts[class_name]):,}",
                pct(train_counts[class_name] / len(split.y_train)),
                f"{int(val_counts[class_name]):,}",
                f"{support[class_name]:,}",
            ]
        )
    report.table(
        ["Classe", "Train", "% train", "Validation", "Test"],
        rows,
        "Distribution des classes après découpage stratifié.",
        widths=[3.4, 3.2, 2.6, 3.2, 2.6],
    )
    report.p(
        "Le déséquilibre est extrême : U2R représente 0,16 % de l'entraînement et 12 flux "
        "seulement dans le test. Toute métrique calculée sur U2R est donc affectée d'une "
        "incertitude importante."
    )
    report.figure(
        "fig_class_distribution.png",
        "distribution des cinq classes dans les ensembles d'entraînement, de validation et de test.",
        15.0,
    )
    report.h2("3.3 Limites intrinsèques du protocole retenu")
    report.p(
        "Le notebook de la Semaine 3 fusionne KDDTrain+_20Percent et KDDTest+ avant de "
        "reconstruire un découpage 70/15/15. Ce choix garantit que le scaler et l'encodeur "
        "ne voient jamais le test, mais il modifie la nature du test : les attaques présentes "
        "uniquement dans KDDTest+ sont désormais représentées dans l'ensemble d'entraînement. "
        "Le test cesse donc d'être un test externe avec attaques inédites. C'est la principale "
        "menace à la validité externe de ce travail, analysée en section 8."
    )
    report.p(
        "De plus, KDDTrain+_20Percent ne contient que 20 % du fichier d'entraînement complet. "
        "Le modèle dispose donc de moins de données que dans un protocole NSL-KDD standard. "
        "Enfin, NSL-KDD reste un jeu ancien, agrégé et non représentatif du trafic chiffré "
        "moderne."
    )


def preprocessing(report: Report) -> None:
    report.h1("4. Préparation des données")
    report.h2("4.1 Nettoyage et contrôles")
    report.p(
        "L'audit a mesuré zéro valeur manquante et zéro doublon exact dans les fichiers "
        "utilisés. Aucune imputation n'a donc été nécessaire. Les valeurs extrêmes n'ont pas "
        "été supprimées automatiquement : en sécurité, un volume d'octets ou un taux d'erreur "
        "exceptionnel peut précisément caractériser une attaque. Les colonnes attack_type et "
        "difficulty_level ont été retirées pour éviter toute fuite d'étiquette."
    )
    report.h2("4.2 Encodage et standardisation")
    report.p(
        "Les 38 variables numériques sont standardisées par StandardScaler (moyenne 0, "
        "écart-type 1). Les trois variables catégorielles protocol_type, service et flag sont "
        "encodées par One-Hot Encoding avec handle_unknown='ignore'. L'encodage produit "
        "80 colonnes binaires ; les 118 features finales combinent 38 numériques et 80 "
        "binaires. Le ColumnTransformer complet est ajusté une seule fois sur X_train, puis "
        "appliqué à la validation et au test."
    )
    report.code(
        "preprocessor = ColumnTransformer([\n"
        "    ('num', StandardScaler(), numerical_cols),\n"
        "    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols),\n"
        "])\n"
        "X_train_proc = preprocessor.fit_transform(X_train)   # fit unique sur le train\n"
        "X_val_proc   = preprocessor.transform(X_val)\n"
        "X_test_proc  = preprocessor.transform(X_test)"
    )
    report.h2("4.3 Découpage et absence de fuite")
    report.p(
        "Le découpage est stratifié avec random_state=42 : 70 % d'entraînement, 15 % de "
        "validation, 15 % de test. L'ordre des opérations est essentiel : le split précède "
        "tout ajustement. Le scaler, l'encodeur et l'encodeur de labels ne sont jamais "
        "réajustés sur la validation ou le test. Le pipeline d'inférence réutilise "
        "exactement l'objet sauvegardé, ce qui garantit la cohérence entre entraînement et "
        "prédiction."
    )
    report.h2("4.4 Déséquilibre des classes")
    report.p(
        "Les poids de classe équilibrés sont calculés sur y_train : environ 0,41 pour Normal, "
        "0,57 pour DoS, 2,03 pour Probe, 3,09 pour R2L et 123,76 pour U2R. Ces poids "
        "pénalisent fortement les erreurs sur U2R. Une variante SMOTE a été générée sur le "
        "train uniquement (81 060 exemples synthétiques), mais elle n'a pas été utilisée pour "
        "entraîner les modèles finaux. Ce choix évite d'introduire des exemples artificiels "
        "dans un espace de features encodées et de revendiquer une amélioration non mesurée."
    )


def models(report: Report) -> None:
    report.h1("5. Architectures et hyperparamètres")
    report.h2("5.1 Random Forest")
    report.p(
        "Le Random Forest est le modèle de référence. Deux profondeurs ont été comparées sur "
        "la validation : aucune limite (Macro F1 = 0,9192) et max_depth = 20 "
        "(Macro F1 = 0,9340). La seconde configuration est retenue. Le modèle final compte "
        "200 arbres, utilise class_weight='balanced', random_state=42 et n_jobs=-1. "
        "C'est un modèle de Machine Learning classique, pas un réseau de neurones."
    )
    report.table(
        ["Paramètre", "Valeur"],
        [
            ["n_estimators", "200"],
            ["max_depth", "20 (sélectionné sur validation)"],
            ["class_weight", "balanced"],
            ["random_state", "42"],
            ["Critère de sélection", "Macro F1 de validation"],
        ],
        "Hyperparamètres du Random Forest.",
        widths=[6.0, 8.5],
    )
    report.h2("5.2 MLP")
    report.p(
        "Le MLP suit l'architecture demandée : entrée 118 dimensions, Dense 128 + ReLU, "
        "Dropout 0,3, Dense 64 + ReLU, Dropout 0,2, sortie Softmax à 5 neurones. Il compte "
        "23 813 paramètres. L'optimiseur est Adam avec un learning rate de 1e-3, la perte est "
        "l'entropie croisée catégorielle creuse, le batch size est 128 et l'entraînement est "
        "limité à 30 époques avec Early Stopping (patience 5, restauration des meilleurs poids)."
    )
    report.h2("5.3 1D-CNN")
    report.p(
        "Le 1D-CNN traite les 118 features comme une séquence de longueur 118 et un canal. "
        "Il enchaîne Conv1D 64 (kernel 3, padding same), MaxPooling1D, Conv1D 128 (kernel 3), "
        "GlobalMaxPooling1D, Dense 64 + ReLU, Dropout 0,3 et Softmax. Il compte 33 541 "
        "paramètres et utilise la même procédure d'entraînement que le MLP. L'ordre des "
        "colonnes est conventionnel : le modèle ne doit pas être interprété comme capturant "
        "une structure spatiale naturelle."
    )
    report.table(
        ["Élément", "Random Forest", "MLP", "1D-CNN"],
        [
            ["Famille", "ML classique", "Deep Learning", "Deep Learning"],
            ["Entrée", "118 features", "118 features", "118 × 1"],
            ["Paramètres appris", "200 arbres", "23 813", "33 541"],
            ["Régularisation", "class_weight", "Dropout 0,3 / 0,2", "Dropout 0,3"],
            ["Optimiseur", "—", "Adam 1e-3", "Adam 1e-3"],
            ["Arrêt", "—", "Early Stopping (5)", "Early Stopping (5)"],
        ],
        "Comparaison des configurations.",
        widths=[4.0, 4.0, 3.6, 3.6],
    )
    report.p(
        "La recherche d'hyperparamètres est volontairement limitée : deux profondeurs pour "
        "le Random Forest et une configuration par réseau, avec Early Stopping. Il ne s'agit "
        "pas d'une optimisation exhaustive ; les réseaux pourraient bénéficier de réglages "
        "supplémentaires, ce qui interdit de conclure à une supériorité générale du Random "
        "Forest sur le Deep Learning."
    )


def protocol(report: Report, evaluation: dict) -> None:
    report.h1("6. Protocole d'évaluation et métriques")
    report.p(
        "La sélection des modèles utilise uniquement la validation. Le test set est évalué "
        "une seule fois, après gel des modèles. Les métriques sont calculées par classe puis "
        "agrégées :"
    )
    report.table(
        ["Métrique", "Définition", "Usage"],
        [
            ["Accuracy", "(TP+TN) / total", "Vue globale, trompeuse si déséquilibre"],
            ["Precision", "TP / (TP+FP)", "Fiabilité des alertes"],
            ["Recall", "TP / (TP+FN)", "Capacité à détecter une attaque"],
            ["F1", "2PR / (P+R)", "Compromis Precision/Recall par classe"],
            ["Macro F1", "Moyenne simple des F1", "Critère principal, classes à égalité"],
            ["Weighted F1", "Moyenne pondérée par support", "Reflète la distribution"],
            ["FPR", "FP / (FP+TN)", "Charge de fausses alertes"],
        ],
        "Métriques utilisées.",
        widths=[3.2, 5.4, 6.2],
    )
    report.p(
        "La Macro F1 est le critère principal car elle donne le même poids aux cinq classes, "
        "y compris U2R. Le FPR est calculé en un-contre-tous puis moyenné, comme dans le "
        "notebook. La latence est mesurée par flux sur le même matériel ; elle dépend du "
        "processeur et ne constitue pas une mesure absolue."
    )
    report.h2("6.1 Reproductibilité")
    report.p(
        f"Les artefacts ont été exportés avec scikit-learn 1.6.1. Le script "
        f"scripts/generate_evaluation_report.py reconstruit le découpage, recharge le modèle "
        f"et recalcule les métriques. L'écart maximal observé sur le test du Random Forest "
        f"est de {evaluation['reproduction']['random_forest_max_abs_difference_test']:.1e}, "
        f"et de {evaluation['reproduction']['random_forest_max_abs_difference_validation']:.1e} "
        f"sur la validation (référence arrondie à six décimales dans le notebook). "
        "Les tests automatisés figent également les prédictions attendues de 92 flux."
    )


def results(report: Report, evaluation: dict) -> None:
    rf = evaluation["test_set"]["models"]["Random Forest"]
    mlp = evaluation["test_set"]["models"]["MLP"]
    cnn = evaluation["test_set"]["models"]["1D-CNN"]
    report.h1("7. Résultats expérimentaux")
    report.h2("7.1 Comparaison sur le jeu de test")
    rows = []
    for name, model in [
        ("Random Forest", rf),
        ("MLP", mlp),
        ("1D-CNN", cnn),
    ]:
        metrics = model["metrics"]
        rows.append(
            [
                name,
                pct(metrics["accuracy"]),
                pct(metrics["macro_f1"]),
                pct(metrics["weighted_f1"]),
                pct(metrics["recall_r2l"]),
                pct(metrics["recall_u2r"]),
                f"{fr(metrics['fpr'] * 100)} %",
            ]
        )
    report.table(
        ["Modèle", "Accuracy", "Macro F1", "Weighted F1", "Recall R2L", "Recall U2R", "FPR"],
        rows,
        "Résultats sur le jeu de test (7 161 flux).",
        widths=[3.1, 2.2, 2.2, 2.4, 2.2, 2.2, 2.0],
        font_size=8.4,
    )
    report.figure(
        "fig_model_comparison.png",
        "comparaison de l'Accuracy, de la Macro F1 et de la Weighted F1 sur le test.",
        14.8,
    )
    report.p(
        f"Le Random Forest domine la comparaison avec {pct(rf['metrics']['accuracy'])} "
        f"d'Accuracy et {pct(rf['metrics']['macro_f1'])} de Macro F1. Le MLP obtient une "
        f"Accuracy correcte ({pct(mlp['metrics']['accuracy'])}) mais une Macro F1 de "
        f"{pct(mlp['metrics']['macro_f1'])}, ce qui révèle de faibles performances sur "
        f"certaines classes malgré un Recall U2R de {pct(mlp['metrics']['recall_u2r'])}. "
        f"Le 1D-CNN est le plus faible ({pct(cnn['metrics']['macro_f1'])} de Macro F1), "
        "ce qui confirme que traiter des colonnes tabulaires comme une séquence n'apporte "
        "pas d'avantage sur ce problème."
    )
    report.h2("7.2 Performance par classe du Random Forest")
    per_class = rf["per_class"]
    rows = [
        [
            name,
            pct(values["precision"]),
            pct(values["recall"]),
            pct(values["f1"]),
            str(values["support"]),
        ]
        for name, values in per_class.items()
    ]
    report.table(
        ["Classe", "Precision", "Recall", "F1", "Support"],
        rows,
        "Performance par classe du Random Forest sur le test.",
        widths=[3.6, 3.0, 3.0, 3.0, 2.5],
    )
    report.figure(
        "fig_per_class_f1.png",
        "F1-score par classe du Random Forest sur le jeu de test.",
        13.8,
    )
    report.figure(
        "fig_confusion_rf_test.png",
        "matrice de confusion du Random Forest sur le test : effectifs et normalisation par ligne.",
        16.0,
    )
    matrix = rf["confusion_matrix"]
    names = rf["class_names"]
    index = {name: i for i, name in enumerate(names)}
    normal_to_r2l = matrix[index["Normal"]][index["R2L"]]
    r2l_to_normal = matrix[index["R2L"]][index["Normal"]]
    u2r_to_normal = matrix[index["U2R"]][index["Normal"]]
    u2r_detected = matrix[index["U2R"]][index["U2R"]]
    report.p(
        f"Les erreurs sont concentrées sur les classes rares. Sur 464 flux R2L, "
        f"{r2l_to_normal} sont classés Normal ; sur 3 474 flux normaux, {normal_to_r2l} sont "
        f"classés R2L. U2R ne compte que 12 flux : {u2r_detected} sont correctement détectés "
        f"et {u2r_to_normal} sont classés Normal. Ces confusions sont plus coûteuses qu'une "
        "confusion entre deux familles d'attaques, car une attaque classée Normal n'est pas "
        "signalée."
    )
    low, high = wilson(u2r_detected, sum(matrix[index["U2R"]]))
    report.p(
        f"Avec seulement 12 flux U2R, le Recall observé de {pct(per_class['U2R']['recall'])} "
        f"a un intervalle de confiance de Wilson à 95 % d'environ [{pct(low, 1)}, {pct(high, 1)}]. "
        "Cette incertitude interdit de conclure à une performance stable sur U2R. À l'inverse, "
        f"le Recall R2L de {pct(per_class['R2L']['recall'])} repose sur 464 flux et est plus fiable."
    )
    report.h2("7.3 Validation et stabilité")
    val = evaluation["validation_set"]["models"]["Random Forest"]["metrics"]
    report.table(
        ["Jeu", "Taille", "Accuracy", "Macro F1", "Recall R2L", "Recall U2R"],
        [
            [
                "Validation",
                f"{evaluation['validation_set']['size']:,}",
                pct(val["accuracy"]),
                pct(val["macro_f1"]),
                pct(val["recall_r2l"]),
                pct(val["recall_u2r"]),
            ],
            [
                "Test",
                f"{evaluation['test_set']['size']:,}",
                pct(rf["metrics"]["accuracy"]),
                pct(rf["metrics"]["macro_f1"]),
                pct(rf["metrics"]["recall_r2l"]),
                pct(rf["metrics"]["recall_u2r"]),
            ],
        ],
        "Random Forest : validation vs test.",
        widths=[3.2, 2.5, 2.8, 2.8, 2.8, 2.8],
    )
    report.p(
        "Les performances de validation et de test sont proches, ce qui indique l'absence de "
        "surapprentissage manifeste sur ce découpage. Cette stabilité ne corrige toutefois pas "
        "la limite de validité externe liée à la fusion des fichiers."
    )
    report.h2("7.4 Interprétabilité")
    report.figure(
        "fig_feature_importance.png",
        "les 15 variables les plus importantes du Random Forest (réduction d'impureté).",
        14.5,
    )
    report.figure(
        "fig_shap_summary.png",
        "résumé SHAP des 15 variables les plus influentes pour la classe Normal (sortie du notebook).",
        14.0,
    )
    report.p(
        "Les variables dominantes sont src_bytes, dst_bytes, dst_host_srv_count, count, "
        "srv_count, duration et service=http. Elles décrivent le volume échangé et le "
        "contexte de connexion, ce qui est cohérent avec des comportements réseau connus : "
        "un scan ou un déni de service modifie les volumes et les compteurs de connexions. "
        "Le service HTTP joue un rôle particulier car de nombreuses attaques R2L l'utilisent. "
        "Ces importances décrivent le modèle, pas une causalité physique."
    )
    report.p(
        "SHAP confirme l'importance de src_bytes, service_http, dst_bytes, "
        "dst_host_srv_count et srv_count. L'analyse est limitée au Random Forest : les "
        "explications SHAP des réseaux n'ont pas été produites, et une importance élevée "
        "n'implique pas que la variable soit la cause de l'attaque."
    )


def critical_analysis(report: Report, evaluation: dict) -> None:
    report.h1("8. Analyse critique et menaces de validité")
    report.h2("8.1 Le protocole de test est optimiste")
    report.p(
        "La faiblesse principale est méthodologique. En fusionnant KDDTrain+_20Percent et "
        "KDDTest+ avant le split, le protocole mélange les distributions et fait apparaître "
        "dans l'entraînement des types d'attaques que le test officiel était censé garder "
        "inédits (mscan, apache2, processtable, snmpguess, etc.). Le test reste valide pour "
        "vérifier que le pipeline n'a pas de fuite au moment du fit, mais il ne mesure pas la "
        "généralisation à des attaques nouvelles. Les scores de 98,83 % d'Accuracy ne sont "
        "donc pas comparables aux résultats publiés sur le protocole NSL-KDD standard."
    )
    report.p(
        "Cette limite doit être présentée explicitement en soutenance. La corriger exigerait "
        "de réentraîner sur KDDTrain+ (ou KDDTrain+_20Percent) et d'évaluer sur KDDTest+ "
        "sans fusion. Ce travail n'a pas été réalisé ici pour ne pas modifier les résultats "
        "expérimentaux de la Semaine 3 ; il constitue la première perspective."
    )
    report.h2("8.2 Accuracy contre Macro F1")
    report.p(
        "Le Random Forest affiche 98,83 % d'Accuracy mais 93,56 % de Macro F1. L'écart "
        "s'explique par les classes majoritaires : Normal, DoS et Probe totalisent plus de "
        "93 % du test. Une erreur sur U2R coûte très peu à l'Accuracy. La Macro F1 et le "
        "Recall par classe sont donc les indicateurs à privilégier pour un NIDS."
    )
    report.h2("8.3 Classes rares et puissance statistique")
    report.p(
        "U2R ne contient que 12 flux de test. Un Recall de 66,67 % correspond à 8 détections "
        "et 4 erreurs ; l'intervalle de confiance est large. Le MLP obtient 91,67 % de Recall "
        "U2R, mais avec une Macro F1 globale de 76,57 %, ce qui suggère un compromis différent "
        "et probablement davantage de fausses alertes. Sans le détail par classe du MLP, il "
        "serait imprudent d'affirmer qu'il est meilleur sur U2R de manière robuste. Le nombre "
        "d'exemples est trop faible pour une conclusion forte."
    )
    report.h2("8.4 Réseaux de neurones")
    report.p(
        "Le MLP et le 1D-CNN sont désavantagés par une recherche d'hyperparamètres limitée, "
        "un jeu d'entraînement de 33 415 flux seulement et un déséquilibre sévère. Leur "
        "contre-performance ne prouve pas que le Deep Learning est inadapté aux NIDS ; elle "
        "montre que, dans ce protocole et avec ce budget d'optimisation, le Random Forest est "
        "le meilleur compromis. Le 1D-CNN souffre en outre d'une hypothèse de séquence non "
        "naturelle sur des features tabulaires."
    )
    report.h2("8.5 Autres menaces de validité")
    report.table(
        ["Menace", "Description", "Impact"],
        [
            ["Validité externe", "Fusion train/test avant split", "Scores optimistes, non comparables au benchmark"],
            ["Validité interne", "Scaler et encodeur ajustés sur le train uniquement", "Pas de fuite de prétraitement"],
            ["Conclusion statistique", "12 flux U2R dans le test", "Intervalles larges, prudence"],
            ["Construction", "Mapping des 39 attaques vérifié", "Aucun type non mappé"],
            ["Reproductibilité", "Graine 42, versions figées, script de vérification", "Métriques RF reproduites exactement"],
            ["Calibration", "Probabilités RF = fréquences de votes", "Confiance non calibrée"],
        ],
        "Synthèse des menaces de validité.",
        widths=[3.3, 6.2, 5.3],
        font_size=8.5,
    )
    report.h2("8.6 Ce qui reste solide")
    report.p(
        "Malgré ces limites, plusieurs éléments sont robustes : le scaler et l'encodeur ne "
        "voient jamais la validation ni le test ; les doublons ont été contrôlés ; la "
        "sélection s'est faite sur la validation ; le test n'a servi qu'une fois ; les "
        "métriques du Random Forest sont reproduites à l'identique par un script indépendant ; "
        "et l'application ne recalcule jamais les performances expérimentales à partir des "
        "fichiers importés."
    )


def application(report: Report) -> None:
    report.h1("9. Application développée")
    report.h2("9.1 Architecture")
    report.p(
        "L'application Streamlit est séparée en quatre vues : Accueil, Analyse, Attack "
        "Analytics et Model Evaluation. La logique de Machine Learning réside dans le paquet "
        "src/nids : chargement et vérification des artefacts, validation du CSV, "
        "prétraitement gelé, prédiction, statistiques et lecture du rapport d'évaluation. "
        "L'interface ne contient aucune formule de modèle."
    )
    report.code(
        "CSV → validation (colonnes, types, manquants, catégories)\n"
        "    → preprocessor.transform (ajusté une seule fois sur le train)\n"
        "    → rf_model.predict_proba\n"
        "    → classes + probabilités + statistiques"
    )
    report.h2("9.2 Fonctionnalités")
    report.bullets(
        [
            "Accueil : objectif, cinq catégories, résultats de test et limites.",
            "Analyse : import CSV, aperçu, validation, bouton d'analyse, tableau des prédictions, confiance, export CSV, alertes.",
            "Attack Analytics : total, normaux, malveillants, répartition, histogramme de confiance, liste des alertes, export.",
            "Model Evaluation : métriques de test et de validation, comparaison des trois modèles, performance par classe, matrice de confusion.",
        ]
    )
    report.p(
        "Le contrat d'entrée est strict : les 41 colonnes sont requises, les colonnes "
        "supplémentaires et les étiquettes sont ignorées avec avertissement, les valeurs "
        "manquantes et non numériques sont refusées. Les catégories inconnues sont acceptées "
        "car l'encodeur d'entraînement utilise handle_unknown='ignore'. Une ligne unique ou "
        "un lot de plusieurs milliers de lignes sont acceptés."
    )
    report.h2("9.3 Tests et validation")
    report.p(
        "La suite pytest couvre 51 tests : chargement des artefacts, CSV valide, colonnes "
        "manquantes ou supplémentaires, valeurs manquantes, valeurs non numériques, fichier "
        "vide, format incorrect, catégories inconnues, une ou plusieurs lignes, cohérence "
        "du prétraitement, prédictions, statistiques, intégrité du rapport et démarrage des "
        "pages Streamlit. Les prédictions attendues de 92 flux sont figées dans une fixture : "
        "toute régression de l'ordre des features ou du préprocesseur fait échouer les tests."
    )
    report.h2("9.4 Déploiement")
    report.p(
        "L'application est préparée pour Streamlit Community Cloud : requirements épinglés, "
        "Python 3.12, point d'entrée app/app.py, aucun secret requis. Le modèle de 31 Mo est "
        "versionné dans models/. Le déploiement public et sa vérification de bout en bout "
        "sont décrits dans le README ; l'URL publique est ajoutée à la livraison finale."
    )


def limits(report: Report) -> None:
    report.h1("10. Limites et perspectives")
    report.h2("10.1 Limites")
    report.bullets(
        [
            "Le test n'est pas le test officiel NSL-KDD : la fusion préalable rend les scores optimistes.",
            "Le jeu d'entraînement est le sous-ensemble 20 %, plus petit que KDDTrain+ complet.",
            "U2R est trop rare pour une estimation stable ; R2L reste la classe minoritaire la plus exploitable.",
            "La recherche d'hyperparamètres des réseaux est limitée ; leurs résultats ne sont pas une borne supérieure.",
            "Les probabilités du Random Forest ne sont pas calibrées.",
            "L'application n'analyse pas le trafic en direct et ne remplace pas un IDS opérationnel.",
            "NSL-KDD est ancien et ne représente pas le trafic chiffré moderne.",
        ]
    )
    report.h2("10.2 Perspectives")
    report.bullets(
        [
            "Réentraîner sans fusion et évaluer sur KDDTest+ officiel pour mesurer la généralisation aux attaques inédites.",
            "Utiliser KDDTrain+ complet et répéter l'expérience avec plusieurs graines pour estimer la variance.",
            "Ajouter une calibration des probabilités (Platt, isotonique) et un seuil d'alerte ajustable.",
            "Tester des architectures spécialisées pour le tabulaire (TabNet, transformeurs) et un réglage plus systématique.",
            "Comparer class weights et SMOTE dans un protocole identique, avec mesures de FPR et de Recall.",
            "Étendre à UNSW-NB15 ou CICIDS2017 pour des scénarios modernes.",
            "Connecter l'inférence à un collecteur de flux (Zeek, Scapy) dans un travail futur, sans modifier le périmètre actuel.",
        ]
    )


def conclusion(report: Report, evaluation: dict) -> None:
    rf = evaluation["test_set"]["models"]["Random Forest"]["metrics"]
    report.h1("11. Conclusion")
    report.p(
        "Le projet a construit une chaîne complète, de la préparation NSL-KDD à une "
        "application Streamlit testée. Le Random Forest sélectionné atteint "
        f"{pct(rf['accuracy'])} d'Accuracy et {pct(rf['macro_f1'])} de Macro F1 sur le jeu "
        "de test retenu, devant le MLP et le 1D-CNN. L'analyse par classe montre cependant "
        "que les performances sur U2R restent fragiles et que l'Accuracy globale masque les "
        "difficultés sur les classes rares."
    )
    report.p(
        "La contribution principale n'est pas un score isolé, mais la rigueur du pipeline : "
        "absence de fuite de prétraitement, sélection sur validation, évaluation unique sur "
        "test, vérification indépendante des métriques, application qui distingue prédictions "
        "et performances expérimentales. La limite du protocole de test est clairement "
        "identifiée : les résultats ne doivent pas être présentés comme un état de l'art "
        "NSL-KDD. Cette honnêteté scientifique, associée à une démonstration fonctionnelle, "
        "constitue la base la plus solide pour la soutenance et pour des travaux futurs."
    )


def references(report: Report) -> None:
    report.h1("Références")
    items = [
        "Tavallaee, M., Bagheri, E., Lu, W., Ghorbani, A. A. (2009). A detailed analysis of the KDD CUP 99 data set. IEEE Symposium on Computational Intelligence for Security and Defense Applications.",
        "Breiman, L. (2001). Random Forests. Machine Learning, 45(1), 5–32.",
        "Chawla, N. V., Bowyer, K. W., Hall, L. O., Kegelmeyer, W. P. (2002). SMOTE: Synthetic Minority Over-sampling Technique. Journal of Artificial Intelligence Research, 16, 321–357.",
        "Kingma, D. P., Ba, J. (2015). Adam: A Method for Stochastic Optimization. ICLR.",
        "Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., Salakhutdinov, R. (2014). Dropout: A Simple Way to Prevent Neural Networks from Overfitting. JMLR, 15, 1929–1958.",
        "Lundberg, S. M., Lee, S.-I. (2017). A Unified Approach to Interpreting Model Predictions. NeurIPS.",
        "Goodfellow, I., Bengio, Y., Courville, A. (2016). Deep Learning. MIT Press.",
        "Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. JMLR, 12, 2825–2830.",
        "Streamlit Documentation. https://docs.streamlit.io (consultée en 2026).",
        "NSL-KDD Dataset, Canadian Institute for Cybersecurity, University of New Brunswick. https://www.unb.ca/cic/datasets/nsl.html",
    ]
    for index, item in enumerate(items, start=1):
        paragraph = report.document.add_paragraph(f"[{index}] {item}")
        paragraph.paragraph_format.space_after = Pt(2)
        paragraph.paragraph_format.left_indent = Cm(0.6)
        paragraph.paragraph_format.first_line_indent = Cm(-0.6)
        for run in paragraph.runs:
            run.font.size = Pt(9)


def main() -> int:
    DOCS.mkdir(parents=True, exist_ok=True)
    evaluation = load_model_evaluation(PROJECT_ROOT / "reports")
    split = reproduce_split(PROJECT_ROOT / "data" / "raw")

    report = Report()
    cover(report, evaluation)
    summary(report, evaluation)
    introduction(report)
    state_of_the_art(report)
    dataset(report, evaluation, split)
    preprocessing(report)
    models(report)
    protocol(report, evaluation)
    results(report, evaluation)
    critical_analysis(report, evaluation)
    application(report)
    limits(report)
    conclusion(report, evaluation)
    references(report)
    report.footer()

    docx_path = DOCS / "NetGuard_AI_Rapport.docx"
    report.document.save(docx_path)
    print(f"DOCX écrit : {docx_path}")

    pdf_path = DOCS / "NetGuard_AI_Rapport.pdf"
    try:
        subprocess.run(
            [
                "libreoffice",
                "--headless",
                "--convert-to",
                "pdf",
                "--outdir",
                str(DOCS),
                str(docx_path),
            ],
            check=True,
            capture_output=True,
            timeout=180,
        )
        if pdf_path.is_file():
            print(f"PDF écrit : {pdf_path}")
    except Exception as exc:  # noqa: BLE001
        print(f"Conversion PDF impossible : {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
