# Guide de reproduction — NetGuard AI

Ce document décrit les étapes exactes permettant de reproduire l'installation, les
tests, l'évaluation expérimentale et la génération des livrables. Il ne remplace pas le
README : il détaille les commandes et les résultats attendus.

## 1. Environnement

| Élément | Valeur |
|---|---|
| Python | 3.12.16 |
| scikit-learn | **1.6.1** (obligatoire pour désérialiser `preprocessor.joblib`) |
| numpy | 2.2.6 |
| pandas | 2.2.3 |
| scipy | 1.15.3 |
| joblib | 1.5.1 |
| streamlit | 1.50.0 |
| plotly | 6.3.0 |
| pytest | 8.4.2 |
| TensorFlow | optionnel, uniquement pour recalculer le détail MLP / 1D-CNN |

> **Attention** : charger `preprocessor.joblib` avec scikit-learn ≥ 1.7 échoue
> (`_RemainderColsList`). Utilisez impérativement la version 1.6.1.

```bash
cd "NetGuard AI"
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
```

## 2. Contenu du dépôt

| Chemin | Rôle |
|---|---|
| `data/raw/KDDTrain+_20Percent.txt` | Fichier d'entraînement NSL-KDD (20 %) |
| `data/raw/KDDTest+.txt` | Fichier de test NSL-KDD |
| `models/rf_model.joblib` | Random Forest gelé (modèle déployé) |
| `models/preprocessor.joblib` | ColumnTransformer ajusté sur le train uniquement |
| `models/label_encoder.joblib` | LabelEncoder des 5 classes |
| `models/metrics_summary.json` | Métriques de test exportées en Semaine 3 |
| `models/*.keras` | MLP et 1D-CNN (comparaison de recherche) |
| `reports/model_evaluation.json` | Rapport d'évaluation vérifié, lu par l'application |
| `tests/fixtures/expected_predictions.csv` | Prédictions figées de 92 flux |

## 3. Vérification de l'évaluation

```bash
python scripts/generate_evaluation_report.py --no-neural
```

Résultat attendu :

- `Random Forest — écart max test : 0.00e+00`
- `validation : 4.83e-07` (la référence du notebook est arrondie à 6 décimales)

Le script :

1. recharge les fichiers bruts et reconstruit le split 70/15/15 (graine 42) ;
2. recharge le préprocesseur et le Random Forest sans réentraînement ;
3. recalcule Accuracy, Macro F1, Weighted F1, Recall R2L/U2R et FPR ;
4. **échoue** si un écart dépasse `1e-9` par rapport à `metrics_summary.json` ;
5. réécrit `reports/model_evaluation.json`.

Les latences de comparaison proviennent toutes de `metrics_summary.json` (mesures de la
Semaine 3) afin de ne pas mélanger des mesures effectuées sur des machines différentes.

Pour recalculer aussi les détails par classe du MLP et du 1D-CNN (nécessite
TensorFlow) :

```bash
pip install tensorflow-cpu==2.21.0
python scripts/generate_evaluation_report.py
```

## 4. Tests automatisés

```bash
pytest
```

Résultat attendu : **51 tests passés**. La suite couvre :

- chargement et intégrité des artefacts ;
- CSV valide, colonnes manquantes ou supplémentaires, étiquettes ignorées ;
- valeurs manquantes, non numériques, infinies, fichier vide, format incorrect ;
- une ligne, plusieurs lignes, ordre des colonnes, doublons ;
- cohérence stricte entre le pipeline et le préprocesseur gelé ;
- statistiques Attack Analytics ;
- intégrité du rapport d'évaluation et cohérence matrice / Recall ;
- démarrage réel des quatre pages Streamlit (`streamlit.testing.v1.AppTest`).

Les prédictions de `tests/fixtures/expected_predictions.csv` sont figées : toute
modification de l'ordre des features, du préprocesseur ou du modèle fait échouer les
tests.

## 5. Données de démonstration

```bash
python scripts/make_demo_sample.py
```

- `data/samples/nsl_kdd_sample.csv` : 250 flux du jeu de test, **sans étiquette** ;
- `tests/fixtures/expected_predictions.csv` : 92 flux avec prédictions attendues.

Ces fichiers sont déjà présents dans le dépôt ; la commande documente leur provenance.

## 6. Application

```bash
streamlit run app/app.py
```

Vérifications manuelles recommandées :

1. Accueil : les métriques affichées correspondent à `reports/model_evaluation.json`.
2. Analyse : télécharger l'échantillon, l'importer, lancer l'analyse (250 prédictions).
3. Attack Analytics : totaux cohérents avec la page Analyse.
4. Model Evaluation : onglets test / validation et matrice de confusion.

## 7. Régénération des livrables

```bash
python scripts/generate_report_figures.py --extract-notebook
python scripts/generate_report.py
python scripts/generate_presentation.py
```

- rapport : `docs/NetGuard_AI_Rapport.docx` et `.pdf` (15 pages A4) ;
- présentation : `docs/NetGuard_AI_Soutenance.pptx` et `.pdf` (13 diapositives).

La conversion PDF utilise LibreOffice :

```bash
libreoffice --headless --convert-to pdf --outdir docs docs/NetGuard_AI_Rapport.docx
```

## 8. Résultats de référence

Jeu de test : 7 161 flux (Normal 3 474, DoS 2 504, Probe 707, R2L 464, U2R 12).

| Modèle | Accuracy | Macro F1 | Weighted F1 | Recall R2L | Recall U2R | FPR |
|---|---:|---:|---:|---:|---:|---:|
| Random Forest | 0,988270 | 0,935584 | 0,988281 | 0,956897 | 0,666667 | 0,003420 |
| MLP | 0,943583 | 0,765706 | 0,953414 | 0,883621 | 0,916667 | 0,012692 |
| 1D-CNN | 0,873761 | 0,686542 | 0,894279 | 0,877155 | 0,750000 | 0,028541 |

Matrice de confusion Random Forest (ordre DoS, Normal, Probe, R2L, U2R) :

```
[[2499,    5,   0,   0,  0],
 [   4, 3426,   7,  37,  0],
 [   0,    7, 700,   0,  0],
 [   0,   19,   0, 444,  1],
 [   0,    4,   0,   0,  8]]
```

## 9. Limite méthodologique à rappeler

Le split a été construit après fusion de `KDDTrain+_20Percent` et `KDDTest+`. Le test est
indépendant de l'entraînement (aucune fuite de prétraitement), mais il contient des types
d'attaques vus à l'entraînement. Les scores sont donc optimistes et **non comparables**
aux résultats publiés sur le protocole NSL-KDD officiel. La correction (réentraînement
sans fusion, évaluation sur `KDDTest+`) est identifiée comme perspective prioritaire.
