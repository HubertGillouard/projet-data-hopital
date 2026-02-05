"""
Model Engine — Chargement modeles et predictions
Gere XGBoost (charge), RandomForest (affluence), SARIMAX, et simulation de scenarios
"""
import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle

from utils.config import FEATURE_COLS, SCENARIOS


# ══════════════════════════════════════════════════════════════════
# CHARGEMENT DES MODELES
# ══════════════════════════════════════════════════════════════════
@st.cache_resource
def load_models():
    """
    Charge tous les modeles serialises (.pkl).
    Supporte 2 formats :
      1. Le pickle du notebook (modeles_a_sauvegarder dict)
      2. Des fichiers individuels (joblib ou pickle)
    """
    models = {}
    base_dir = os.path.join(os.path.dirname(__file__), '..', 'models')

    # --- Format 1 : pickle du notebook (xgboost_charge_psl.pkl) ---
    notebook_pkl = os.path.join(base_dir, 'xgboost_charge_psl.pkl')
    if not os.path.exists(notebook_pkl):
        # Chercher aussi dans le repertoire parent
        notebook_pkl = os.path.join(os.path.dirname(__file__), '..', '..', 'xgboost_charge_psl.pkl')

    if os.path.exists(notebook_pkl):
        try:
            file_size_mb = os.path.getsize(notebook_pkl) / (1024 * 1024)
            print(f"[model_engine] Chargement xgboost_charge_psl.pkl ({file_size_mb:.1f}MB)...")

            with open(notebook_pkl, 'rb') as f:
                data = pickle.load(f)

            # Le notebook sauvegarde un dict avec xgboost_j1, xgboost_j3, xgboost_j7
            if isinstance(data, dict):
                loaded_models = []
                if 'xgboost_j1' in data:
                    models['charge_j1'] = data['xgboost_j1']
                    loaded_models.append('J+1')
                if 'xgboost_j3' in data:
                    models['charge_j3'] = data['xgboost_j3']
                    loaded_models.append('J+3')
                if 'xgboost_j7' in data:
                    models['charge_j7'] = data['xgboost_j7']
                    loaded_models.append('J+7')
                if 'features' in data:
                    models['_features'] = data['features']
                if 'seuils_alerte' in data:
                    models['_seuils'] = data['seuils_alerte']
                if 'post_traitement' in data:
                    models['_post_traitement'] = data['post_traitement']

                if loaded_models:
                    print(f"[model_engine] ✓ XGBoost chargé: {', '.join(loaded_models)}")
        except Exception as e:
            # Fallback silencieux sur la simulation
            print(f"[model_engine] ✗ XGBoost non chargé: {e}")

    # --- Format 2 : fichiers individuels ---
    # Note: SARIMAX ignoré (fichier trop lourd ~2.3GB, non fonctionnel)
    individual_files = {
        'affluence_rf': 'modele_random_forest.pkl',  # RF pour affluence (254 MB)
        # 'sarimax': 'model_sarimax.pkl',  # DESACTIVÉ - trop lourd (2.3 GB)
    }

    for name, filename in individual_files.items():
        path = os.path.join(base_dir, filename)
        if os.path.exists(path):
            try:
                # Verifier taille fichier (skip si > 500MB pour eviter freeze)
                file_size_mb = os.path.getsize(path) / (1024 * 1024)
                if file_size_mb > 500:
                    print(f"[model_engine] {filename} trop lourd ({file_size_mb:.0f}MB), ignoré")
                    continue

                with open(path, 'rb') as f:
                    loaded_data = pickle.load(f)

                # Gérer le cas où le modèle est wrappé dans un dict
                if isinstance(loaded_data, dict) and 'model' in loaded_data:
                    models[name] = loaded_data['model']
                    # Stocker aussi les features si disponibles
                    if 'features' in loaded_data:
                        models[f'_{name}_features'] = loaded_data['features']
                    print(f"[model_engine] ✓ {filename} chargé (dict, {file_size_mb:.0f}MB)")
                else:
                    models[name] = loaded_data
                    print(f"[model_engine] ✓ {filename} chargé ({file_size_mb:.0f}MB)")
            except Exception as e:
                # Fallback silencieux - la simulation prendra le relais
                print(f"[model_engine] ✗ {filename} non chargé: {e}")

    if not models:
        return None

    return models


def get_model(name='charge_j1'):
    """Recupere un modele specifique."""
    models = load_models()
    if models is None:
        return None
    return models.get(name)


# ══════════════════════════════════════════════════════════════════
# POST-TRAITEMENT (identique au notebook)
# ══════════════════════════════════════════════════════════════════
def post_traitement_pics(y_pred, boost=3, seuil_bas=70, seuil_haut=80):
    """
    Boost predictions proches du seuil de pic (+3 pts entre 70-80).
    v2: réduit de 4 à 3 pour équilibrer Precision (~55%) / Recall (~90%).
    """
    y_adj = np.array(y_pred).copy().astype(float)
    near_threshold = (y_adj >= seuil_bas) & (y_adj < seuil_haut)
    y_adj[near_threshold] += boost
    return y_adj


# ══════════════════════════════════════════════════════════════════
# PREDICTIONS
# ══════════════════════════════════════════════════════════════════
def predict_charge(features_df, horizon='J+1'):
    """
    Predit le score de charge (0-100) pour chaque ligne.
    Gere le cas ou les features du modele ne correspondent pas aux donnees.
    """
    model_map = {'J+1': 'charge_j1', 'J+3': 'charge_j3', 'J+7': 'charge_j7'}
    model_name = model_map.get(horizon, 'charge_j1')
    model = get_model(model_name)

    if model is None:
        return simulate_charge_prediction(features_df, horizon)

    try:
        # Recuperer les features attendues par le modele
        models = load_models()
        if models and '_features' in models:
            expected_features = models['_features']
        elif hasattr(model, 'feature_names_in_'):
            expected_features = list(model.feature_names_in_)
        elif hasattr(model, 'get_booster') and hasattr(model.get_booster(), 'feature_names'):
            expected_features = model.get_booster().feature_names
        else:
            expected_features = FEATURE_COLS

        # Preparer les features avec seulement celles attendues
        available_cols = [c for c in expected_features if c in features_df.columns]

        # Si trop peu de features disponibles, utiliser la simulation
        if len(available_cols) < len(expected_features) * 0.5:
            return simulate_charge_prediction(features_df, horizon)

        X = features_df[available_cols].copy()
        X = X.fillna(X.mean())

        # Ajouter les colonnes manquantes avec des valeurs par defaut
        for col in expected_features:
            if col not in X.columns:
                X[col] = 0

        # Reordonner les colonnes pour correspondre a l'ordre attendu
        X = X[expected_features]

        # XGBoost nécessite .values pour éviter l'erreur "feature names"
        y_pred = model.predict(X.values)
        y_pred = post_traitement_pics(y_pred)
        return np.clip(y_pred, 0, 100)

    except Exception as e:
        # En cas d'erreur, utiliser la simulation
        return simulate_charge_prediction(features_df, horizon)


def predict_affluence(features_df):
    """Predit le nombre de patients/heure."""
    model = get_model('affluence_rf')

    if model is None:
        return simulate_affluence_prediction(features_df)

    try:
        # Récupérer les features stockées avec le modèle
        models = load_models()
        if models and '_affluence_rf_features' in models:
            rf_features = models['_affluence_rf_features']
        elif hasattr(model, 'feature_names_in_'):
            rf_features = list(model.feature_names_in_)
        else:
            # Fallback: features par défaut du notebook
            rf_features = [
                'heure', 'jour_semaine', 'mois', 'est_weekend', 'heures_pleines',
                'nb_patients_en_cours', 'occup_lits_estimee',
                'nb_arrivees_lag_1h', 'nb_arrivees_lag_24h', 'nb_arrivees_lag_48h',
                'nb_arrivees_lag_6h', 'indicateur_greve', 'evenement_externe',
                'alerte_epidemique_encoded',
            ]

        available = [c for c in rf_features if c in features_df.columns]

        # Si trop peu de features, utiliser simulation
        if len(available) < len(rf_features) * 0.5:
            return simulate_affluence_prediction(features_df)

        X = features_df[available].copy()
        X = X.fillna(0)

        # Ajouter colonnes manquantes
        for col in rf_features:
            if col not in X.columns:
                X[col] = 0
        X = X[rf_features]

        # Utiliser .values pour éviter le warning sklearn feature names
        y_pred = model.predict(X.values)
        return np.clip(y_pred, 0, 50)

    except Exception as e:
        # Fallback sur simulation en cas d'erreur
        print(f"[model_engine] RF predict error: {e}, using simulation")
        return simulate_affluence_prediction(features_df)


# ══════════════════════════════════════════════════════════════════
# SIMULATION DE SCENARIOS
# ══════════════════════════════════════════════════════════════════
def apply_scenario(features_df, scenario_name, custom_params=None):
    """Applique un scenario predefini aux features."""
    df = features_df.copy()

    params = SCENARIOS.get(scenario_name, {}).copy()
    if custom_params:
        params.update(custom_params)

    for col, val in params.items():
        if col.endswith('_mult'):
            base_col = col.replace('_mult', '')
            if base_col in df.columns:
                df[base_col] = df[base_col] * val
        else:
            # Toujours definir la colonne, meme si elle n'existait pas
            df[col] = val

    return df


def predict_with_scenario(features_df, scenario_name, custom_params=None, horizon='J+1'):
    """Predit le score de charge avec un scenario applique."""
    df_scenario = apply_scenario(features_df, scenario_name, custom_params)
    return predict_charge(df_scenario, horizon)


# ══════════════════════════════════════════════════════════════════
# PREDICTIONS SIMULEES (fallback quand modeles non disponibles)
# ══════════════════════════════════════════════════════════════════
def simulate_charge_prediction(features_df, horizon='J+1'):
    """
    Genere des predictions de charge simulees realistes.
    Coefficients calibres pour que les changements de parametres soient VISIBLES.
    """
    n = len(features_df)

    if 'heure' in features_df.columns:
        heure = features_df['heure'].values
    else:
        heure = np.tile(np.arange(24), n // 24 + 1)[:n]

    # Charge de base suivant l'heure (pic 17-19h)
    base = 50 + 20 * np.sin(np.pi * (heure - 6) / 14)

    # Patients presents : FORT impact (+1.5 pts par patient au-dessus de 40)
    if 'nb_patients_en_cours' in features_df.columns:
        patients = features_df['nb_patients_en_cours'].fillna(40).values
        # Prendre la premiere valeur si c'est un scalaire broadcast
        if hasattr(patients, '__len__') and len(patients) > 0:
            pat_val = patients[0] if np.all(patients == patients[0]) else patients
        else:
            pat_val = patients
        base = base + (np.float64(pat_val) - 40) * 1.5

    # Effectif : FORT impact (-3 pts par soignant au-dessus de 15)
    if 'effectif_soignant_present' in features_df.columns:
        effectif = features_df['effectif_soignant_present'].fillna(17).values
        if hasattr(effectif, '__len__') and len(effectif) > 0:
            eff_val = effectif[0] if np.all(effectif == effectif[0]) else effectif
        else:
            eff_val = effectif
        base = base - (np.float64(eff_val) - 15) * 3

    # Lits aval : impact moyen (-2 pts par lit au-dessus de 8)
    if 'dispo_lits_aval' in features_df.columns:
        lits = features_df['dispo_lits_aval'].fillna(12).values
        if hasattr(lits, '__len__') and len(lits) > 0:
            lits_val = lits[0] if np.all(lits == lits[0]) else lits
        else:
            lits_val = lits
        base = base - (np.float64(lits_val) - 8) * 2

    # Greve : +15 pts
    if 'indicateur_greve' in features_df.columns:
        greve = features_df['indicateur_greve'].fillna(0).values
        if hasattr(greve, '__len__') and len(greve) > 0:
            greve_val = greve[0]
        else:
            greve_val = greve
        base = base + np.float64(greve_val) * 15

    # Epidemie : +12 pts
    if 'alerte_epidemique_encoded' in features_df.columns:
        epidemie = features_df['alerte_epidemique_encoded'].fillna(0).values
        if hasattr(epidemie, '__len__') and len(epidemie) > 0:
            epi_val = epidemie[0]
        else:
            epi_val = epidemie
        base = base + np.float64(epi_val) * 12

    # Petit bruit aleatoire (non fixe pour varier)
    horizon_noise = {'J+1': 2, 'J+3': 3, 'J+7': 4}.get(horizon, 2)
    noise = np.random.normal(0, horizon_noise, n if isinstance(base, np.ndarray) else 1)

    result = np.clip(base + noise, 0, 100)
    return result if isinstance(result, np.ndarray) else np.array([result] * n)


def simulate_affluence_prediction(features_df):
    """Genere des predictions d'affluence simulees."""
    n = len(features_df)

    if 'heure' in features_df.columns:
        heure = features_df['heure'].values
    else:
        heure = np.tile(np.arange(24), n // 24 + 1)[:n]

    flux = 6 + 10 * np.sin(np.pi * (heure - 4) / 14)
    flux = np.clip(flux, 3, 20)
    noise = np.random.normal(0, 1.5, n)

    return np.clip(flux + noise, 1, 30)


# ══════════════════════════════════════════════════════════════════
# GENERATION DE PREVISIONS SUR HORIZON
# ══════════════════════════════════════════════════════════════════
def generate_forecast(last_features, n_hours=24, horizon='J+1'):
    """
    Genere des previsions pour les N prochaines heures.
    """
    from datetime import timedelta

    if 'date_hourly' in last_features.columns:
        last_dt = pd.to_datetime(last_features['date_hourly'].iloc[-1])
    else:
        last_dt = pd.Timestamp.now().floor('h')

    future_dates = pd.date_range(
        start=last_dt + timedelta(hours=1),
        periods=n_hours, freq='h'
    )

    # Creer les features futures (copie de la derniere avec datetime modifie)
    future_features = pd.concat([last_features.tail(1)] * n_hours, ignore_index=True)
    future_features['date_hourly'] = future_dates
    future_features['heure'] = future_dates.hour
    future_features['jour_semaine'] = future_dates.dayofweek

    # Recalculer les encodages cycliques
    future_features['heure_sin'] = np.sin(2 * np.pi * future_features['heure'] / 24)
    future_features['heure_cos'] = np.cos(2 * np.pi * future_features['heure'] / 24)
    future_features['jour_semaine_sin'] = np.sin(2 * np.pi * future_features['jour_semaine'] / 7)
    future_features['jour_semaine_cos'] = np.cos(2 * np.pi * future_features['jour_semaine'] / 7)
    future_features['heures_pleines'] = (
        (future_features['heure'] >= 8) & (future_features['heure'] < 20)
    ).astype(int)
    future_features['nuit'] = (
        (future_features['heure'] >= 22) | (future_features['heure'] < 6)
    ).astype(int)

    predictions = predict_charge(future_features, horizon)

    return pd.DataFrame({
        'datetime': future_dates,
        'charge_pred': predictions,
    })
