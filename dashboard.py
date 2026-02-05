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
    # Charger le modèle "léger"
    model_hollow = joblib.load('model_sarimax.pkl')
    
    # Charger les données de contexte
    last_y = pd.read_csv('last_y.csv', index_col=0, parse_dates=True)
    last_exog = pd.read_csv('last_exog.csv', index_col=0, parse_dates=True)
    
    # "Injecter" la mémoire dans le modèle vide via .apply()
    # Cela permet de faire des prédictions à partir de la fin de last_y
    model_ready = model_hollow.apply(endog=last_y, exog=last_exog)
    
    return model_ready

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

    # Définissez uniquement les colonnes dont vous avez besoin pour le dashboard
    keep_cols = ['date_heure_arrivee', 'heure', 'id_passage', 'temps_passage_total', 'jour_semaine', 'effectif_soignant_present', 'dispo_lits_aval']

    # Charger le dataset
    df = pd.read_csv('urgences_data.csv', encoding='latin1', usecols=keep_cols)

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
        'jour_semaine': 'mean',
        'effectif_soignant_present': 'mean',
        'dispo_lits_aval': 'min',
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
        current_patients = st.number_input("Nombre de patients déjà en attente ou en cours de prise en charge", min_value=0)

    # Bouton de lancement
    btn_predict = st.button("🚀 CALCULER L'IMPACT", width="stretch", type="primary")

# PREPARATION DU VECTEUR EXOGENE POUR LE MODELE (BACK-END)

if btn_predict:
    # Création de la séquence de dates au format datetime64 de Pandas
    base_time = pd.Timestamp(target_date).normalize()

    # Génération des 24 heures suivantes directement
    future_dates = pd.date_range(start=base_time, periods=24, freq='h')

    # Préparation des données numériques
    future_exog_data = []
    for current_dt in future_dates:
        future_exog_data.append({
            'patient_count': current_patients,
            'jour_semaine': current_dt.weekday(),
            'effectif_soignant_present': staff,
            'dispo_lits_aval': lits
        })

    # Création du DataFrame avec l'index temporel
    exog_24h = pd.DataFrame(future_exog_data, index=future_dates)

    # Nommage de l'index comme dans le notebook
    exog_24h.index.name = 'date_hourly'

    # Appel du modèle pour 24 pas de temps (H+24)
    try:
        with st.spinner("Calcul des prévisions des Length Of Stay sur 24h..."):
            # Prédiction des 24 prochaines heures avec les exogènes préparées
            prediction_24h = model.forecast(steps=24, exog=exog_24h)
            
            # Nettoyage des valeurs (pas de temps négatif)
            prediction_24h = [max(0, val) for val in prediction_24h]
            
            avg_los_24h = np.mean(prediction_24h)
            peak_los = np.max(prediction_24h)

        # Affichage des Indicateurs de Charge
        st.divider()
        st.subheader("📊 Analyse du Temps de Passage Total (LOS)")
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Moyenne sur 24h", f"{int(avg_los_24h)} min")
        m2.metric("Pic de prise en charge", f"{int(peak_los)} min")
        m3.metric("Tension de service", "Critique" if peak_los > 240 else "Fluide")

        # Graphique temporel
        st.write("**Évolution du temps de prise en charge moyen par heure**")
        chart_df = pd.DataFrame({
            'Heure': [f"H+{i}" for i in range(1, 25)],
            'Temps de passage (min)': forecast_los
        }).set_index('Heure')
        
        st.line_chart(chart_df)

        # Recommandations stratégiques
        st.subheader("💡 Étude d'impact et recommandations")
        
        if peak_los > 300: # Seuil de 5h de passage moyen
            st.error("🚨 **ALERTE ENCOMBREMENT**")
            st.markdown(f"""
            - **Constat :** Le temps de passage total atteint un pic de **{int(peak_los)} min**. Le service est en risque de saturation.
            - **Action :** Prioriser les dossiers en attente de résultats d'imagerie pour libérer les box.
            - **Stratégie :** Augmenter le ratio soignants/patients (actuellement basé sur {staff} soignants).
            """)
        else:
            st.success("✅ **FLUX OPTIMAL**")
            st.write("Le temps de passage moyen est maîtrisé. Les ressources actuelles permettent une prise en charge fluide.")

    except Exception as e:
        st.error(f"Erreur modèle : {e}")