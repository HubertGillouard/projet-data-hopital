import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import joblib

# Configuration de la page
st.set_page_config(
    page_title="Pitié-Salpêtrière - Aide à la décision",
    page_icon="🏥",
    layout="wide"
)

# CHARGEMENT DU MODELE SARIMAX

@st.cache_resource
def load_model():
    return joblib.load('model_sarimax.pkl')

model_load_state = st.text('Loading model...')
model = load_model()
model_load_state.text("Done! (using st.cache_resource)")

# CHARGEMENT DES DONNEES

@st.cache_data
def load_data():
    """
    Charge, nettoie et transforme les données d'urgences pour une analyse horaire.

    Cette fonction effectue le pipeline ETL (Extract, Transform, Load) suivant :
    1. Chargement du fichier CSV 'urgences_data.csv'.
    2. Feature Engineering : calcul du taux d'occupation en temps réel, identification 
       des weekends et des heures pleines.
    3. Estimation du flux : calcule les sorties via le temps de passage pour déduire 
       le nombre de patients présents simultanément.
    4. Rééchantillonnage (Resampling) : agrège les données individuelles par pas de 
       temps d'une heure.
    5. Nettoyage final : imputation des valeurs manquantes et formatage des colonnes encodées.

    Returns:
        pd.DataFrame: Un DataFrame indexé par 'date_hourly' (DatetimeIndex) où chaque ligne 
            représente une heure précise avec des agrégations sur les indicateurs médicaux, 
            environnementaux et de ressources.
    """

    # Charger le dataset
    df = pd.read_csv('urgences_data.csv', encoding='latin1')

    # Conversion de la colonne en format datetime
    df['date_heure_arrivee'] = pd.to_datetime(df['date_heure_arrivee'], format='%d/%m/%Y %H:%M')

    # Fix the timestamp logic
    # - Day part from original date
    # - Hour part from 'heure' column
    # - Minutes always '00'
    df['date_hourly'] = df['date_heure_arrivee'].dt.normalize() + pd.to_timedelta(df['heure'], unit='h')

    # Estimer la date et l'heure de sortie
    df["date_heure_sortie_estimee"] = (
        df["date_heure_arrivee"] + pd.to_timedelta(df["temps_passage_total"], unit="m")
    )

    # Créer deux dataframes : un pour les arrivées, un pour les départs
    arrivals = pd.DataFrame({'time': df['date_heure_arrivee'], 'change': 1})
    exits = pd.DataFrame({'time': df['date_heure_sortie_estimee'], 'change': -1})
    # Combinez-les et triez-les par ordre chronologique
    events = pd.concat([arrivals, exits]).sort_values(by='time')
    # Calculer le total cumulé (cumulative sum)
    events['occupancy'] = events['change'].cumsum()
    # Fusionner avec le dataframe d'origine pour obtenir le nombre à l'heure d'arrivée.
    # 'merge_asof' est parfait ici : il détecte l'état d'occupation au moment exact de l'arrivée.
    df = df.sort_values('date_heure_arrivee')
    df = pd.merge_asof(
        df, 
        events[['time', 'occupancy']], 
        left_on='date_heure_arrivee', 
        right_on='time', 
        direction='backward'
    )
    # Calcul final
    CAPACITE_LITS = 30
    df["nb_patients_en_cours"] = df['occupancy']
    df["occup_lits_estimee"] = df["nb_patients_en_cours"] / CAPACITE_LITS

    # Définir l'index
    df = df.set_index('date_hourly')

    # Utilisez resample('h') pour que chaque ligne corresponde à une HEURE
    # Nous utilisons .agg pour calculer simultanément la moyenne (durée) et le nombre (patients).
    df_hourly = df.resample('h').agg({
        'id_passage': 'count',          # Nombre de patients pour cette heure
        'temps_passage_total': 'mean',  # Temps passé moyen pour cette heure
        'mois': 'mean',
        'heure': 'mean',
        'jour_semaine': 'mean',
        'annee': 'mean',
        'temperature_max': 'max',
        'indicateur_greve': 'max',
        'evenement_externe': 'max',
        'niveau_pollution': 'max',
        'age_patient': 'mean',
        'score_IAO': 'max',
        'effectif_soignant_present': 'mean',
        'dispo_lits_aval': 'min',
        'consommation_O2': 'min',
        'kit_traumatologie': 'min',
        'solutes_hydratation': 'min',
        'alerte_epidemique_encoded': 'mean',
        'batiment_accueil_encoded': 'mean',
        'filiere_pathologie_encoded': 'mean',
        'mode_transport_encoded': 'mean',
        'besoin_imagerie_encoded': 'mean',
        'devenir_patient_encoded': 'mean',
        'nb_patients_en_cours': 'max',
        'occup_lits_estimee': 'max'
    }).rename(columns={
        'temps_passage_total': 'temps_passage_moyen',
        'id_passage': 'patient_count'
    })

    # Gérer les NaN (requis pour SARIMAX et avant la conversion en int)
    df_hourly['patient_count'] = df_hourly['patient_count'].fillna(0)
    # Forward fill les données pour gérer les heures sans patients
    df_hourly = df_hourly.ffill().bfill()

    # Identifiez toutes les colonnes qui se terminent par '_encoded'
    encoded_cols = [col for col in df_hourly.columns if col.endswith('_encoded')]
    # Appliquez l'arrondi et la conversion en entier à tous les éléments
    df_hourly[encoded_cols] = df_hourly[encoded_cols].round().astype(int)

    return df_hourly

data_load_state = st.text('Loading data...')
df = load_data()
data_load_state.text("Done! (using st.cache_data)")

# INTERFACE DE SIMULATION COMPLETE (FRONT-END)

st.title("Simulation & Prédiction de Charge")

# Utilisation d'un formulaire pour regrouper les paramètres
with st.container(border=True):
    st.subheader("🛠️ Paramètres de Simulation (Variables Exogènes)")
    
    col_env, col_hosp = st.columns(2)
    
    with col_env:
        st.markdown("**Environnement & Contexte**")
        target_date = st.date_input("Date à simuler", datetime.now() + timedelta(days=1))
        
        # Le jour de la semaine est calculé automatiquement à partir de la date
        jour_semaine = target_date.weekday()
        st.info(f"Jour de la semaine détecté : {target_date.strftime('%A')}")

        epidemie = st.selectbox("Alerte Épidémique", ["Aucune", "COVID", "Grippe", "Bronchiolite", "Gastro"])
        temp = st.slider("Température prévue (°C)", -5, 40, 15)
        greve = st.toggle("Indicateur de Grève active")

    with col_hosp:
        st.markdown("**Ressources Hospitalières**")
        
        # effectif_soignant_present
        staff = st.number_input("Effectif soignant présent", min_value=1, max_value=50)
        
        # dispo_lits_aval
        lits = st.number_input("Disponibilité lits aval", min_value=0, max_value=100)
        
        # patient_count
        current_patients = st.number_input("Nombre de patients déjà en attente", min_value=0)

    # Bouton de lancement
    btn_predict = st.button("🚀 CALCULER L'IMPACT", width="stretch", type="primary")


# PREPARATION DU VECTEUR EXOGENE POUR LE MODELE (BACK-END)

if btn_predict:
    # Encodage des variables textuelles comme dans notre notebook
    mapping_epi = {"Aucune": 0, "COVID": 2, "Grippe": 4, "Bronchiolite": 1, "Gastro": 3}
    
    # Création du DataFrame exogène pour la prédiction
    exog_simu = pd.DataFrame({
        'patient_count': [current_patients],
        'jour_semaine': [jour_semaine],
        'effectif_soignant_present': [staff],
        'dispo_lits_aval': [lits],
        # 'alerte_epidemique_encoded': [mapping_epi[epidemie]],
        # 'temperature_max': [temp],
        # 'indicateur_greve': [1 if greve else 0],
    })

    # Appel du modèle
    try:
        # On prédit par exemple les 24 prochaines heures
        # Note : Pour SARIMAX, si nous avons plusieurs steps, il faut répéter les exog
        prediction = model.forecast(steps=1, exog=exog_simu)
        valeur_predite = int(prediction.iloc[0])
        
        # Affichage
        st.divider()
        st.metric("Nombre d'entrées prévues", f"{valeur_predite} patients")
        
    except Exception as e:
        st.error(f"Erreur lors de la prédiction : {e}")