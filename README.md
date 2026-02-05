# 🏥 Aide à la Décision - Urgences Pitié-Salpêtrière

Ce repo contient le prototype de tableau de bord intéractif pour le suivi et la prédiction de la charge aux urgences de la Pitié-Salpêtrière.

## 🚀 État du Projet : Branche SARIMAX

> **Note importante :** Cette branche spécifique (`Laura`) contient uniquement l'intégration dans Streamlit du modèle de prédiction temporelle du temps moyen passé par un patient par heure.

### 🧠 Intégration du Modèle Prédictif
Nous avons développé et intégré dans cette branche un modèle **SARIMAX (Seasonal AutoRegressive Integrated Moving Average with eXogenous regressors)** capable de prédire le **Length of Stay (LOS)** à un horizon de 24 heures (H+24).

Le modèle prend en compte des variables exogènes critiques :
* Effectifs soignants présents.
* Disponibilité des lits en aval.
* Flux de patients entrants.
* Saisonnalité (jour de la semaine).

## ⚠️ Statut de l'Intégration au MVP

Bien que le pipeline de prédiction soit fonctionnel dans le Jupyter Notebook associé et porté dans l'interface Streamlit, **cette fonctionnalité n'a pas été retenue pour le MVP (Minimum Viable Product) final** en raison de contrainte de temps.

## 🛠️ Structure Technique (sur cette branche)

* `model_sarimax.pkl` (disponible sur Edsquare uniquement) : Modèle entraîné "allégé".
* `last_y.csv` / `last_exog.csv` : Données de contexte nécessaires à l'initialisation du modèle.
* `dashboard.py` : Interface Streamlit intégrant le formulaire de simulation (mode démo).
* `urgences_data.csv` : Jeu de données utilisé pour l'entraînement du modèle.
* `Projet_data.ipynb` : Jupyter Notebook contenant le code de développement du modèle.

## 💻 Installation & Test

Pour explorer les travaux de prédiction :

1. Basculer sur cette branche :
   ```bash
   git checkout Laura
   ```

2. Installer les dépendances spécifiques :
    ```bash
    pip install statsmodels joblib pandas streamlit
    ```

3. Lancer l'interface :
    ```bash
    streamlit run app.py
    ```