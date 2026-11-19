# NetGuard AI — NIDS multiclasse sur NSL-KDD

Application académique de détection d'intrusions réseau par Machine Learning et Deep
Learning. Elle classe des flux NSL-KDD en cinq catégories — **Normal, DoS, Probe, R2L,
U2R** — à partir de 41 caractéristiques, avec un **Random Forest** sélectionné sur le jeu
de validation et évalué une seule fois sur un jeu de test indépendant.

Le projet couvre la chaîne complète : préparation des données, comparaison de trois
modèles (Random Forest, MLP, 1D-CNN), évaluation scientifique, explicabilité, pipeline
d'inférence, application Streamlit, tests et documentation.

> **Périmètre** : l'application analyse des flux **fournis dans un fichier CSV**. Elle ne
> capture pas le trafic réseau en direct et ne remplace pas un IDS de production.

---

## 1. Résultats expérimentaux

Évaluation finale sur le jeu de test (7 161 flux), telle qu'exportée pendant la Semaine 3
et vérifiée par le script de reproductibilité :

| Modèle | Type | Accuracy | Macro F1 | Weighted F1 | Recall R2L | Recall U2R |
|---|---|---:|---:|---:|---:|---:|
| **Random Forest** | ML (arbres) | **98,83 %** | **93,56 %** | 98,83 % | 95,69 % | 66,67 % |
| MLP | Deep Learning | 94,36 % | 76,57 % | 95,34 % | 88,36 % | 91,67 % |
| 1D-CNN | Deep Learning | 87,38 % | 68,65 % | 89,43 % | 87,72 % | 75,00 % |

Le Random Forest est le modèle déployé. **L'Accuracy élevée ne doit pas masquer la
Macro F1** : les classes Normal, DoS et Probe dominent le jeu de test, tandis que U2R ne
compte que 12 flux. Le MLP obtient un meilleur Recall U2R mais une Macro F1 bien plus
faible. Ces résultats sont discutés dans le rapport.

---

## 2. Structure du projet

```
NetGuard AI/
├── app/
│   ├── app.py                    # point d'entrée Streamlit
│   └── views/                    # Accueil, Analyse, Attack Analytics, Model Evaluation
├── src/nids/
│   ├── artifacts.py              # chargement + vérification des artefacts
│   ├── validation.py             # contrat d'entrée CSV
│   ├── inference.py              # prétraitement gelé + prédictions
│   ├── analytics.py              # statistiques des prédictions
│   ├── evaluation.py             # lecture des résultats expérimentaux
│   ├── dataset.py                # reconstruction reproductible du split
│   └── reproducibility.py        # vérification des métriques sauvegardées
├── models/                       # artefacts gelés (rf, preprocessor, encoder, métriques)
├── reports/model_evaluation.json # évaluation vérifiée, consommée par l'application
├── data/
│   ├── raw/                      # KDDTrain+_20Percent.txt, KDDTest+.txt
│   └── samples/                  # fichier de démonstration (41 features, sans étiquette)
├── notebooks/                    # notebook de modélisation des Semaines 1 à 3
├── scripts/                      # génération de l'évaluation et des fixtures
├── tests/                        # pytest : validation, inférence, analytics, évaluation, UI
├── docs/                         # rapport scientifique et soutenance
├── requirements.txt
└── README.md
```

L'interface ne contient aucune logique de Machine Learning : elle appelle les modules de
`src/nids`. Les résultats expérimentaux sont produits hors ligne et **lus** par la page
Model Evaluation ; ils ne sont jamais recalculés à partir des fichiers importés.

---

## 3. Installation

Prérequis : Python **3.12** (les artefacts ont été exportés avec scikit-learn 1.6.1).

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Pour les tests et la génération des rapports :

```bash
pip install -r requirements-dev.txt
```

### Lancer l'application

```bash
streamlit run app/app.py
```

Ouvrir <http://localhost:8501>. Le chargement du Random Forest (31 Mo) prend quelques
secondes au premier démarrage.

---

## 4. Utilisation

1. **Accueil** : contexte, catégories, résultats de test.
2. **Analyse** : importer un CSV, lancer l'analyse, consulter et télécharger les prédictions.
3. **Attack Analytics** : totaux, répartition, confiances, liste des alertes du lot courant.
4. **Model Evaluation** : métriques de test et de validation, comparaison, matrice de
   confusion et performance par classe.

Un fichier d'exemple est téléchargeable depuis la page Analyse
(`data/samples/nsl_kdd_sample.csv`, 250 flux issus du jeu de test, sans étiquette).

### Format d'entrée

- CSV séparé par des virgules, encodage UTF-8.
- Les **41 colonnes NSL-KDD** sont requises, dans n'importe quel ordre.
- Les colonnes supplémentaires et les étiquettes (`attack_type`, `attack_category`,
  `difficulty_level`, `label`…) sont ignorées avec un avertissement.
- Valeurs manquantes et valeurs non numériques dans une colonne numérique : analyse refusée
  avec un message explicite. Aucune imputation n'a été apprise à l'entraînement.
- Catégories inconnues : acceptées, ignorées par le One-Hot Encoding (comme à
  l'entraînement), et signalées.
- Une ligne ou plusieurs milliers de lignes sont acceptées (limite : 100 000 lignes, 25 Mo).

---

## 5. Tests

```bash
pytest
```

La suite couvre : chargement des artefacts, CSV valide, colonnes manquantes ou
supplémentaires, valeurs manquantes, valeurs non numériques, fichier vide, format
incorrect, catégories inconnues, une ou plusieurs lignes, cohérence du prétraitement,
prédictions, statistiques, intégrité du rapport d'évaluation et démarrage des pages
Streamlit.

Les prédictions attendues sont figées dans `tests/fixtures/expected_predictions.csv` :
toute modification de l'ordre des features, du préprocesseur ou du modèle fait échouer les
tests.

### Vérifier la reproductibilité de l'évaluation

```bash
python scripts/generate_evaluation_report.py --no-neural
```

Le script reconstruit le découpage 70/15/15 (graine 42) à partir des fichiers bruts,
recharge le modèle, recalcule les métriques et **échoue** si elles diffèrent de
`models/metrics_summary.json`. Il régénère `reports/model_evaluation.json`.

---

## 6. Choix techniques

- **Aucune fuite de données** : doublons supprimés, split stratifié avant tout `fit`,
  `StandardScaler` et `One-HotEncoder` ajustés sur le seul entraînement.
- **SMOTE** comparé sur l'entraînement uniquement ; les modèles finaux utilisent des
  `class_weight` équilibrés. Aucun rééchantillonnage à l'inférence.
- **Sélection sur validation** : Macro F1 ; le test set n'est utilisé qu'une fois.
- **Modèle déployé** : Random Forest, 200 arbres, `max_depth=20`,
  `class_weight='balanced'`, `random_state=42`.
- **Reproductibilité** : versions épinglées, graine fixée, rapport de vérification généré.

---

## 7. Déploiement

L'application est prévue pour **Streamlit Community Cloud** :

1. pousser le dépôt sur GitHub (sans secrets ni données personnelles) ;
2. créer une application Streamlit pointant vers `app/app.py` ;
3. utiliser Python 3.12 (`runtime.txt`) et `requirements.txt` ;
4. vérifier après déploiement le chargement du modèle, l'import CSV, les prédictions et
   les pages Attack Analytics / Model Evaluation.

Aucun secret n'est nécessaire : le modèle et les artefacts sont des fichiers du dépôt.

---

## 8. Limites

- Les flux NSL-KDD sont des enregistrements statiques : l'application ne capture pas de
  trafic réel et n'offre pas de détection temps réel.
- Le jeu de test contient des attaques absentes de l'entraînement ; les scores ne se
  transposent pas directement à un réseau opérationnel.
- U2R est très rare (12 flux de test) : les métriques de cette classe ont une forte
  incertitude.
- Les probabilités du Random Forest sont des fréquences de votes, non calibrées.
- La latence dépend du matériel et n'est pas comparable entre machines.

---

## 9. Livrables et reproduction

| Livrable | Fichier |
|---|---|
| Rapport scientifique (15 pages) | `docs/NetGuard_AI_Rapport.pdf` / `.docx` |
| Présentation de soutenance (13 diapositives) | `docs/NetGuard_AI_Soutenance.pdf` / `.pptx` |
| Guide de reproduction détaillé | `docs/REPRODUCTION.md` |
| Évaluation vérifiée (source de l'UI) | `reports/model_evaluation.json` |
| Figures du rapport | `reports/figures/` |

Pour vérifier l'ensemble en une commande :

```bash
pytest && python scripts/generate_evaluation_report.py --no-neural
```

## 10. Contexte académique

Mini-projet du module **Machine Learning**, 4ᵉ année ingénieur réseau, Université Djilali
Liabes de Sidi Bel Abbès. Dataset : **NSL-KDD**.
