"""
Data Loader — Chargement, cache et feature engineering
Charge les donnees depuis CSV et les modeles depuis .pkl
"""
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

from utils.config import FEATURE_COLS, COEF_PATHOLOGIE, SEUIL_SATURATION, CAPACITE_URGENCES


# ══════════════════════════════════════════════════════════════════
# RESOLUTION DU CHEMIN CSV
# ══════════════════════════════════════════════════════════════════
def _resolve_csv_path():
    """Cherche urgences_data.csv dans data/ puis dans le repertoire parent."""
    candidates = [
        os.path.join(os.path.dirname(__file__), '..', 'data', 'urgences_data.csv'),
        os.path.join(os.path.dirname(__file__), '..', '..', 'urgences_data.csv'),
        'data/urgences_data.csv',
        '../urgences_data.csv',
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


# ══════════════════════════════════════════════════════════════════
# CHARGEMENT DONNEES BRUTES
# ══════════════════════════════════════════════════════════════════
@st.cache_data(ttl=3600)
def load_raw_data():
    """
    Charge le dataset brut (570K lignes).
    Encodage Latin-1 (comme dans les notebooks).
    """
    filepath = _resolve_csv_path()
    if filepath is None:
        st.warning("Fichier urgences_data.csv non trouve. Donnees simulees utilisees.")
        return None

    df = pd.read_csv(filepath, encoding='latin1')

    # Parse datetime
    if 'date_heure_arrivee' in df.columns:
        df['date_heure_arrivee'] = pd.to_datetime(
            df['date_heure_arrivee'],
            format='%d/%m/%Y %H:%M',
            errors='coerce'
        )

    return df


@st.cache_data(ttl=3600)
def load_hourly_data():
    """
    Charge ou construit les donnees agregees horaires.
    """
    df_raw = load_raw_data()
    if df_raw is not None:
        return aggregate_hourly(df_raw)

    return generate_simulated_hourly()


# ══════════════════════════════════════════════════════════════════
# AGREGATION HORAIRE
# ══════════════════════════════════════════════════════════════════
def aggregate_hourly(df):
    """
    Agrege les donnees patient-level vers le niveau horaire.
    Reproduit la logique du notebook.
    """
    df = df.copy()

    # S'assurer que la colonne datetime est bien parsee
    if not pd.api.types.is_datetime64_any_dtype(df['date_heure_arrivee']):
        df['date_heure_arrivee'] = pd.to_datetime(
            df['date_heure_arrivee'], format='%d/%m/%Y %H:%M', errors='coerce'
        )

    # Creer colonne date_hourly
    df['date_hourly'] = df['date_heure_arrivee'].dt.floor('h')

    # Colonnes d'aggregation — uniquement celles presentes dans le CSV
    agg_dict = {}
    col_map = {
        'id_passage': ('count', 'patient_count'),
        'temps_passage_total': ('mean', 'avg_passage_time'),
        'mois': ('first', 'mois'),
        'heure': ('first', 'heure'),
        'jour_semaine': ('first', 'jour_semaine'),
        'annee': ('first', 'annee'),
        'temperature_max': ('max', 'temperature_max'),
        'indicateur_greve': ('max', 'indicateur_greve'),
        'evenement_externe': ('max', 'evenement_externe'),
        'niveau_pollution': ('max', 'niveau_pollution'),
        'age_patient': ('mean', 'age_patient'),
        'score_IAO': ('mean', 'score_IAO'),
        'effectif_soignant_present': ('mean', 'effectif_soignant_present'),
        'dispo_lits_aval': ('min', 'dispo_lits_aval'),
        'consommation_O2': ('sum', 'consommation_O2'),
        'kit_traumatologie': ('sum', 'kit_traumatologie'),
        'solutes_hydratation': ('sum', 'solutes_hydratation'),
        'alerte_epidemique_encoded': ('first', 'alerte_epidemique_encoded'),
        'batiment_accueil_encoded': ('mean', 'batiment_accueil_encoded'),
        'site_accueil_encoded': ('first', 'site_accueil_encoded'),
        'filiere_pathologie_encoded': ('mean', 'filiere_pathologie_encoded'),
        'mode_transport_encoded': ('mean', 'mode_transport_encoded'),
        'besoin_imagerie_encoded': ('mean', 'besoin_imagerie_encoded'),
        'devenir_patient_encoded': ('mean', 'devenir_patient_encoded'),
    }

    for col, (func, _) in col_map.items():
        if col in df.columns:
            agg_dict[col] = func

    df_indexed = df.set_index('date_hourly')
    df_hourly = df_indexed.resample('h').agg(agg_dict)

    # Renommer
    rename_map = {}
    for col, (_, new_name) in col_map.items():
        if col in df_hourly.columns and new_name != col:
            rename_map[col] = new_name
    df_hourly = df_hourly.rename(columns=rename_map)

    df_hourly['patient_count'] = df_hourly['patient_count'].fillna(0).astype(int)
    df_hourly = df_hourly.reset_index()

    # Calculer nb_patients_en_cours (estimation : cumul glissant sur fenetre de 4h = duree moy)
    # Approximation : nb_patients_en_cours ~ somme des arrivees sur les 4 dernieres heures
    df_hourly['nb_patients_en_cours'] = (
        df_hourly['patient_count'].rolling(4, min_periods=1).sum()
    ).clip(0, SEUIL_SATURATION + 10)

    # Taux d'occupation
    df_hourly['taux_occupation_urgences'] = (
        df_hourly['nb_patients_en_cours'] / CAPACITE_URGENCES
    ).clip(0, 2.0)

    # Variables derivees
    df_hourly['est_weekend'] = (df_hourly['jour_semaine'] >= 5).astype(int)
    df_hourly['heures_pleines'] = (
        (df_hourly['heure'] >= 8) & (df_hourly['heure'] < 20)
    ).astype(int)
    df_hourly['nuit'] = (
        (df_hourly['heure'] >= 22) | (df_hourly['heure'] < 6)
    ).astype(int)

    # Calculer le score de charge
    df_hourly['charge_soin_totale'] = df_hourly.apply(calculer_charge_soin, axis=1)

    # Feature engineering complet
    df_hourly = add_features(df_hourly)

    return df_hourly


# ══════════════════════════════════════════════════════════════════
# CALCUL SCORE DE CHARGE (formule exacte du notebook)
# ══════════════════════════════════════════════════════════════════
def calculer_charge_soin(row):
    """
    Score composite 0-100 :
    - Affluence (40%) : nb_patients / seuil saturation
    - Gravite  (30%) : IAO inverse (IAO 1=vital -> score eleve)
    - Pathologie(20%) : coefficient complexite filiere
    - Ressources(10%) : inverse ratio effectif+lits / optimal
    """
    def safe_float(val, default):
        """Retourne default si val est NaN ou None."""
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return default
        return float(val)

    # 1. Affluence
    nb_patients = safe_float(row.get('nb_patients_en_cours', 40), 40)
    score_affluence = min((nb_patients / SEUIL_SATURATION) * 100, 100)

    # 2. Gravite (IAO inverse)
    score_iao = safe_float(row.get('score_IAO', 3), 3)
    gravite_inversee = 6 - score_iao
    score_gravite = (gravite_inversee / 5) * 100

    # 3. Pathologie
    filiere_raw = safe_float(row.get('filiere_pathologie_encoded', 0), 0)
    filiere = int(round(filiere_raw))
    coef = COEF_PATHOLOGIE.get(filiere, 1.0)
    score_pathologie = (coef / 1.8) * 100

    # 4. Ressources (peu = score eleve)
    effectif = safe_float(row.get('effectif_soignant_present', 17), 17)
    lits = safe_float(row.get('dispo_lits_aval', 12), 12)
    ratio_ress = (effectif / 25) * 0.5 + (lits / 40) * 0.5
    ratio_ress = max(min(ratio_ress, 1.0), 0.2)
    score_ressources = (1 - ratio_ress) * 100

    # Score final pondere
    charge = (
        score_affluence * 0.40
        + score_gravite * 0.30
        + score_pathologie * 0.20
        + score_ressources * 0.10
    )
    return min(charge, 100)


# ══════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════
def add_features(df):
    """
    Ajoute toutes les features necessaires au modele XGBoost.
    Reproduit le code du notebook (Cell 19).
    """
    df = df.copy()

    # Extraire datetime components si pas presents
    if 'date_hourly' in df.columns:
        dt = pd.to_datetime(df['date_hourly'])
        if 'heure' not in df.columns:
            df['heure'] = dt.dt.hour
        if 'jour_semaine' not in df.columns:
            df['jour_semaine'] = dt.dt.dayofweek
        if 'mois' not in df.columns:
            df['mois'] = dt.dt.month
        df['jour_du_mois'] = dt.dt.day
        df['semaine_du_mois'] = (dt.dt.day - 1) // 7 + 1

    # Lag features (charge)
    for lag in [1, 4, 6, 12, 24, 48, 72, 168]:
        col_name = f'charge_lag_{lag}h'
        if col_name not in df.columns and 'charge_soin_totale' in df.columns:
            df[col_name] = df['charge_soin_totale'].shift(lag)

    # Lag features (patient_count)
    for lag in [1, 4, 24, 168]:
        col_name = f'patient_count_lag_{lag}h'
        if col_name not in df.columns and 'patient_count' in df.columns:
            df[col_name] = df['patient_count'].shift(lag)

    # Rolling statistics
    if 'charge_soin_totale' in df.columns:
        df['charge_mean_24h'] = df['charge_soin_totale'].rolling(24, min_periods=1).mean()
        df['charge_std_24h'] = df['charge_soin_totale'].rolling(24, min_periods=1).std().fillna(0)
        df['charge_max_7d'] = df['charge_soin_totale'].rolling(168, min_periods=1).max()
        df['charge_mean_12h'] = df['charge_soin_totale'].rolling(12, min_periods=1).mean()
        df['charge_std_12h'] = df['charge_soin_totale'].rolling(12, min_periods=1).std().fillna(0)

    if 'patient_count' in df.columns:
        df['patient_count_mean_24h'] = df['patient_count'].rolling(24, min_periods=1).mean()

    # Encodages cycliques
    if 'heure' in df.columns:
        df['heure_sin'] = np.sin(2 * np.pi * df['heure'] / 24)
        df['heure_cos'] = np.cos(2 * np.pi * df['heure'] / 24)
    if 'jour_semaine' in df.columns:
        df['jour_semaine_sin'] = np.sin(2 * np.pi * df['jour_semaine'] / 7)
        df['jour_semaine_cos'] = np.cos(2 * np.pi * df['jour_semaine'] / 7)
    if 'mois' in df.columns:
        df['mois_sin'] = np.sin(2 * np.pi * df['mois'] / 12)
        df['mois_cos'] = np.cos(2 * np.pi * df['mois'] / 12)

    # Variables derivees (si pas deja calculees)
    if 'heure' in df.columns:
        if 'nuit' not in df.columns:
            df['nuit'] = ((df['heure'] >= 22) | (df['heure'] < 6)).astype(int)
        if 'heures_pleines' not in df.columns:
            df['heures_pleines'] = ((df['heure'] >= 8) & (df['heure'] < 20)).astype(int)
    if 'jour_semaine' in df.columns:
        if 'est_weekend' not in df.columns:
            df['est_weekend'] = (df['jour_semaine'] >= 5).astype(int)

    # Interactions
    if 'est_weekend' in df.columns and 'alerte_epidemique_encoded' in df.columns:
        df['weekend_epidemie'] = df['est_weekend'] * df['alerte_epidemique_encoded']
    if 'nuit' in df.columns and 'effectif_soignant_present' in df.columns:
        df['nuit_effectif'] = df['nuit'] * df['effectif_soignant_present']
    if 'indicateur_greve' in df.columns and 'jour_semaine' in df.columns:
        df['greve_jour_semaine'] = df['indicateur_greve'] * df['jour_semaine']

    return df


# ══════════════════════════════════════════════════════════════════
# SNAPSHOT TEMPS REEL
# ══════════════════════════════════════════════════════════════════
def get_current_snapshot(df):
    """Retourne la derniere heure disponible (simule 'temps reel')."""
    latest = df.iloc[-1]
    return {
        'datetime': latest.get('date_hourly', datetime.now()),
        'charge': latest.get('charge_soin_totale', 65),
        'patients': latest.get('nb_patients_en_cours', 45),
        'effectif': latest.get('effectif_soignant_present', 18),
        'lits': latest.get('dispo_lits_aval', 12),
        'flux': latest.get('patient_count', 13),
        'temps_passage': latest.get('avg_passage_time', 240),
        'score_IAO': latest.get('score_IAO', 3.2),
    }


def get_recent_hours(df, n=24):
    """Retourne les N dernieres heures."""
    return df.tail(n).copy()


def get_features_for_prediction(df):
    """
    Extrait les features necessaires pour le modele XGBoost.
    Retourne uniquement les colonnes presentes dans FEATURE_COLS.
    """
    available = [c for c in FEATURE_COLS if c in df.columns]
    return df[available].copy()


# ══════════════════════════════════════════════════════════════════
# DONNEES SIMULEES (fallback)
# ══════════════════════════════════════════════════════════════════
def generate_simulated_hourly():
    """
    Genere des donnees horaires simulees realistes.
    Utilise quand aucun fichier de donnees n'est disponible.
    """
    np.random.seed(42)
    n_hours = 24 * 90  # 90 jours

    dates = pd.date_range(
        end=datetime.now().replace(minute=0, second=0, microsecond=0),
        periods=n_hours, freq='h'
    )

    heure = dates.hour
    jour = dates.dayofweek
    mois = dates.month

    # Profil horaire realiste (pic 17-19h)
    profil_h = 8 + 6 * np.sin(np.pi * (heure - 6) / 16)
    profil_h = np.clip(profil_h, 3, 18)
    noise = np.random.normal(0, 1.5, n_hours)

    patient_count = (profil_h + noise).clip(2, 25).astype(int)

    # Patients en cours (suit le profil horaire avec inertie)
    nb_patients = (40 + 15 * np.sin(np.pi * (heure - 8) / 14)
                   + np.random.normal(0, 5, n_hours)).clip(10, 65)

    # Effectif (jour vs nuit)
    effectif = np.where(
        (heure >= 8) & (heure < 20),
        np.random.normal(20, 2, n_hours),
        np.random.normal(10, 1.5, n_hours)
    ).clip(6, 26)

    # Lits aval
    lits = (12 + np.random.normal(0, 4, n_hours)).clip(1, 23)

    # Score IAO
    score_iao = np.random.normal(3.2, 0.5, n_hours).clip(1, 5)

    # Temperature (saisonniere)
    temp_base = 15 + 10 * np.sin(2 * np.pi * (mois - 4) / 12)
    temperature = (temp_base + np.random.normal(0, 3, n_hours)).clip(-5, 40)

    # Indicateurs binaires
    indicateur_greve = np.random.binomial(1, 0.02, n_hours)
    alerte_epidemique = np.random.binomial(1, 0.05, n_hours)
    evenement_externe = np.random.binomial(1, 0.01, n_hours)

    df = pd.DataFrame({
        'date_hourly': dates,
        'heure': heure,
        'jour_semaine': jour,
        'mois': mois,
        'patient_count': patient_count,
        'nb_patients_en_cours': nb_patients,
        'effectif_soignant_present': effectif,
        'dispo_lits_aval': lits,
        'score_IAO': score_iao,
        'temperature_max': temperature,
        'indicateur_greve': indicateur_greve,
        'alerte_epidemique_encoded': alerte_epidemique,
        'evenement_externe': evenement_externe,
        'niveau_pollution': np.random.randint(1, 6, n_hours),
        'filiere_pathologie_encoded': np.random.randint(0, 7, n_hours),
        'batiment_accueil_encoded': np.random.randint(0, 5, n_hours),
        'site_accueil_encoded': 0,
        'besoin_imagerie_encoded': np.random.binomial(1, 0.69, n_hours),
        'consommation_O2': np.random.randint(0, 3, n_hours),
        'kit_traumatologie': np.random.binomial(1, 0.19, n_hours),
        'solutes_hydratation': np.random.binomial(1, 0.11, n_hours),
        'avg_passage_time': (240 + 60 * np.sin(np.pi * (heure - 10) / 12)
                             + np.random.normal(0, 30, n_hours)).clip(60, 480),
        'est_weekend': (jour >= 5).astype(int),
        'heures_pleines': ((heure >= 8) & (heure < 20)).astype(int),
        'nuit': ((heure >= 22) | (heure < 6)).astype(int),
    })

    # Calculer charge et features
    df['charge_soin_totale'] = df.apply(calculer_charge_soin, axis=1)
    df = add_features(df)

    return df
