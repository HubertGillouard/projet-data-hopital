# 🏥 Aide à la Décision - Urgences Pitié-Salpêtrière

Ce repo contient le prototype de tableau de bord intéractif pour le suivi et la prédiction de la charge aux urgences de la Pitié-Salpêtrière.

## 🛠️ Structure Technique

* `dashboard/app.py` : Interface Streamlit du MVP.
* `urgences_data.csv` : Jeu de données utilisé pour l'entraînement de nos modèles.
* `EDA_Pitie_Salpetriere.ipynb` : Jupyter Notebook contenant notre Analyse Exploratoire des Données.
* `Modelisation_Predictive_PSL.ipynb` : Jupyter Notebook contenant le code de notre modèle XGBoost.
* `model_sarimax.pkl`: Voir branche ```Laura```.

## 💻 Installation & Test

1. Basculer sur cette branche :
   ```bash
   git checkout main
   ```

2. Installer les dépendances spécifiques :
    ```bash
    pip install statsmodels joblib pandas streamlit
    ```

3. Lancer l'interface :
    ```bash
    cd dashboard
    streamlit run app.py
    ```