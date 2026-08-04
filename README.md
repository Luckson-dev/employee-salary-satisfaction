# 📊 Employee Salary Satisfaction Prediction

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Classification-green.svg)
![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange.svg)

Ce projet a été réalisé dans le cadre de mon **apprentissage du Machine Learning** afin de mettre en pratique l'analyse de données et les algorithmes de classification. Il vise à explorer et prédire la **satisfaction salariale des employés** en fonction de divers facteurs socio-professionnels (expérience, poste, niveau d'études, charge de travail, etc.).

À travers ce cas pratique de **classification binaire**, l'objectif principal est de maîtriser l'ensemble du pipeline data science : de l'exploration des données (EDA) au prétraitement, jusqu'à l'entraînement et l'évaluation de plusieurs modèles prédictifs.
## Objectifs

* **Analyser les facteurs clés** qui influencent la satisfaction salariale des employés.
* **Construire et comparer plusieurs modèles de classification** pour identifier le plus performant.
* **Proposer un outil prédictif** utilisable pour des simulations RH.

## Structure du projet

```text
├── data/               # Jeux de données (raw et processed)
├── notebooks/          # Notebooks Jupyter (EDA, Preprocessing, Modélisation)
├── src/                # Scripts Python réutilisables (clean_data.py, train.py)
├── models/             # Modèles entraînés sauvegardés (.pkl / .joblib)
├── .gitignore          # Fichiers à ignorer par Git
├── README.md           # Description du projet
└── requirements.txt    # Dépendances Python

```

## Démarche méthodologique

1. **Exploration des données (EDA) :** Analyse de la distribution des salaires, corrélations et traitement des valeurs manquantes.
2. **Prétraitement :** Encodage des variables catégorielles, normalisation/standardisation des variables numériques.
3. **Modélisation :** Entraînement de plusieurs algorithmes de classification :
* Régression Logistique
* Arbres de Décision (Decision Trees)
* Forêts Aléatoires (Random Forest)
* XGBoost / LightGBM *(optionnel)*


4. **Évaluation :** Comparaison des modèles avec des métriques adaptées (**Accuracy, Précision, Rappel, F1-Score, ROC-AUC**).

## Installation et Utilisation

### Préréquis

Assure-toi d'avoir **Python 3.8+** installé sur ta machine.

### 1. Cloner le dépôt

```bash
git clone https://github.com/Luckson-dev/employee-salary-satisfaction.git

cd employee-salary-satisfaction
```

### 2. Créer un environnement virtuel

```bash
python -m venv venv

# Activation sur Windows :
venv\Scripts\activate

# Activation sur Mac/Linux :
source venv/bin/activate
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

## Résultats obtenus

| Modèle | Accuracy | F1-Score | ROC-AUC |
| --- | --- | --- | --- |
| Régression Logistique | 0.XX | 0.XX | 0.XX |
| Random Forest | **0.XX** | **0.XX** | **0.XX** |
| XGBoost | 0.XX | 0.XX | 0.XX |

*(Le modèle **Random Forest** a donné les meilleurs résultats globaux sur le jeu de test.)*

## Contribution

Les contributions, suggestions et retours sont les bienvenus ! N'hésite pas à ouvrir une *Issue* ou à soumettre une *Pull Request*.

## Licence

Ce projet est sous licence MIT - voir le fichier [LICENSE](https://www.google.com/search?q=LICENSE) pour plus de détails.