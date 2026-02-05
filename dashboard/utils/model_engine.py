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
            with open(notebook_pkl, 'rb') as f:
                data = pickle.load(f)
            # Le notebook sauvegarde un dict avec xgboost_j1, xgboost_j3, xgboost_j7
            if isinstance(data, dict):
                if 'xgboost_j1' in data:
                    models['charge_j1'] = data['xgboost_j1']
                if 'xgboost_j3' in data:
                    models['charge_j3'] = data['xgboost_j3']
                if 'xgboost_j7' in data:
                    models['charge_j7'] = data['xgboost_j7']
                if 'features' in data:
                    models['_features'] = data['features']
                if 'seuils_alerte' in data:
                    models['_seuils'] = data['seuils_alerte']
                if 'post_traitement' in data:
                    models['_post_traitement'] = data['post_traitement']
        except Exception as e:
            st.warning(f"Erreur chargement {notebook_pkl}: {e}")

    # --- Format 2 : fichiers individuels ---
    individual_files = {
        'affluence_rf': 'rf_affluence.pkl',
        'sarimax': 'sarimax.pkl',
    }

    for name, filename in individual_files.items():
        path = os.path.join(base_dir, filename)
        if os.path.exists(path):
            try:
                with open(path, 'rb') as f:
                    models[name] = pickle.load(f)
            except Exception as e:
                st.warning(f"Erreur chargement {filename}: {e}")

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
def post_traitement_pics(y_pred, boost=4, seuil_bas=70, seuil_haut=80):
    """
    Boost predictions proches du seuil de pic (+4 pts entre 70-80).
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
    """
    model_map = {'J+1': 'charge_j1', 'J+3': 'charge_j3', 'J+7': 'charge_j7'}
    model_name = model_map.get(horizon, 'charge_j1')
    model = get_model(model_name)

    if model is None:
        return simulate_charge_prediction(features_df, horizon)

    # Preparer les features
    available_cols = [c for c in FEATURE_COLS if c in features_df.columns]
    X = features_df[available_cols].copy()
    X = X.fillna(X.mean())

    y_pred = model.predict(X)
    y_pred = post_traitement_pics(y_pred)
    return np.clip(y_pred, 0, 100)


def predict_affluence(features_df):
    """Predit le nombre de patients/heure."""
    model = get_model('affluence_rf')

    if model is None:
        return simulate_affluence_prediction(features_df)

    rf_features = [
        'heure_sin', 'heure_cos', 'jour_semaine_sin', 'jour_semaine_cos',
        'heures_pleines', 'est_weekend', 'nb_patients_en_cours',
        'patient_count_lag_1h', 'patient_count_lag_24h',
        'temperature_max', 'indicateur_greve', 'alerte_epidemique_encoded',
    ]
    available = [c for c in rf_features if c in features_df.columns]
    X = features_df[available].fillna(0)

    y_pred = model.predict(X)
    return np.clip(y_pred, 0, 50)


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
        elif col in df.columns:
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
    """Genere des predictions de charge simulees realistes."""
    n = len(features_df)
    np.random.seed(42)

    if 'heure' in features_df.columns:
        heure = features_df['heure'].values
    else:
        heure = np.tile(np.arange(24), n // 24 + 1)[:n]

    # Charge de base suivant l'heure (pic 17-19h)
    base = 45 + 25 * np.sin(np.pi * (heure - 6) / 14)

    if 'nb_patients_en_cours' in features_df.columns:
        patients = features_df['nb_patients_en_cours'].fillna(40).values
        base += (patients - 40) * 0.5

    if 'effectif_soignant_present' in features_df.columns:
        effectif = features_df['effectif_soignant_present'].fillna(17).values
        base -= (effectif - 17) * 2

    if 'dispo_lits_aval' in features_df.columns:
        lits = features_df['dispo_lits_aval'].fillna(12).values
        base -= (lits - 12) * 0.8

    if 'indicateur_greve' in features_df.columns:
        greve = features_df['indicateur_greve'].fillna(0).values
        base += greve * 10

    if 'alerte_epidemique_encoded' in features_df.columns:
        epidemie = features_df['alerte_epidemique_encoded'].fillna(0).values
        base += epidemie * 8

    horizon_noise = {'J+1': 3, 'J+3': 5, 'J+7': 8}.get(horizon, 3)
    noise = np.random.normal(0, horizon_noise, n)

    return np.clip(base + noise, 0, 100)


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
