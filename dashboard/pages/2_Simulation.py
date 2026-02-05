"""
Page 2 — Simulation & Decisions (VERSION REPENSEE)
Interface de simulation pour tester des scenarios et obtenir des recommandations.

Ameliorations majeures:
- Boutons horizon (J+1, J+3, J+7) en haut, tres visibles
- Suppression du multiplicateur obscur -> Scenarios predefinis clairs
- Parametres regroupes en 2 colonnes (externe vs interne)
- Jour auto-detecte affiche
- Alertes a 4 niveaux (50-70-85)
- Recommandations actionnables
- Design coherent avec l'identite AP-HP
- Resultats visibles par defaut
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

from utils.config import (
    BLEU_FONCE, BLEU_CLAIR, BLEU_TRES_CLAIR, GRIS_MEDICAL, GRIS_TEXTE, BLANC,
    VERT_OK, JAUNE_VIGIL, ORANGE_ALERT, ROUGE_CRIT,
    SCENARIOS, SCENARIOS_AFFLUENCE, get_charge_level,
)
from utils.charts import comparison_overlay, double_gauge
from utils.data_loader import load_hourly_data, get_recent_hours, get_features_for_prediction
from utils.model_engine import predict_charge, predict_with_scenario, generate_forecast, predict_affluence
from utils.recommendation_engine import (
    generate_recommendations, format_recommendation_html,
    count_red_hours, count_alert_hours, calculate_total_gain,
)

# ══════════════════════════════════════════════════════════════════
# APPLICATION DES RECOMMANDATIONS PENDING (AVANT widgets)
# Cette section DOIT etre AVANT l'instantiation des widgets
# ══════════════════════════════════════════════════════════════════
# Si une recommandation a ete appliquee, on prepare les nouvelles valeurs
if 'pending_effectif' in st.session_state:
    st.session_state.w_effectif = st.session_state.pending_effectif
    del st.session_state.pending_effectif

if 'pending_lits' in st.session_state:
    st.session_state.w_lits = st.session_state.pending_lits
    del st.session_state.pending_lits

if 'pending_patients' in st.session_state:
    st.session_state.w_patients = st.session_state.pending_patients
    del st.session_state.pending_patients

# ══════════════════════════════════════════════════════════════════
# CONSTANTES DE DESIGN
# ══════════════════════════════════════════════════════════════════
CARD_STYLE = f"""
    background: {BLANC};
    border-radius: 12px;
    padding: 24px;
    box-shadow: 0 2px 8px rgba(0,61,122,0.08);
    border: 1px solid rgba(0,61,122,0.08);
"""


def get_alert_level(score):
    """4 niveaux d'alerte."""
    if score < 50:
        return "NORMAL", VERT_OK, "✓", "Le service fonctionne normalement"
    elif score < 70:
        return "VIGILANCE", JAUNE_VIGIL, "⚠", "Restez attentif a l'evolution"
    elif score < 85:
        return "ATTENTION", ORANGE_ALERT, "⚠⚠", "Renforcement recommande"
    else:
        return "CRITIQUE", ROUGE_CRIT, "🚨", "Action immediate requise"


# ══════════════════════════════════════════════════════════════════
# CSS GLOBAL
# ══════════════════════════════════════════════════════════════════
st.markdown(f"""
<style>
/* Sidebar bleu fonce AP-HP */
[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, {BLEU_FONCE} 0%, #1B5E9E 100%);
}}
[data-testid="stSidebar"] * {{
    color: white !important;
}}
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stRadio label {{
    color: white !important;
    font-weight: 600 !important;
}}
/* Navigation sidebar - texte blanc visible */
section[data-testid="stSidebar"] nav a span {{
    color: white !important;
    font-weight: 500 !important;
}}
section[data-testid="stSidebar"] nav a[aria-current="page"] {{
    background: rgba(255,255,255,0.15) !important;
}}

/* Header bar */
.header-bar {{
    background: linear-gradient(135deg, {BLEU_FONCE} 0%, #1B5E9E 100%);
    padding: 28px 40px;
    border-radius: 12px;
    margin-bottom: 24px;
}}
.header-bar h1 {{
    color: white !important;
    font-size: 28px !important;
    font-weight: 700 !important;
    margin: 0 !important;
}}
.header-bar p {{
    color: rgba(255,255,255,0.9) !important;
    font-size: 15px !important;
    margin: 8px 0 0 0 !important;
}}

/* Boutons horizon custom */
div[data-testid="column"] button {{
    min-height: 70px !important;
    font-size: 16px !important;
}}

/* Bouton primary */
.stButton > button[kind="primary"] {{
    background: {BLEU_FONCE} !important;
    color: white !important;
    font-weight: 700 !important;
    font-size: 16px !important;
    padding: 16px 32px !important;
    border-radius: 10px !important;
    min-height: 56px !important;
}}
.stButton > button[kind="primary"]:hover {{
    background: #1B5E9E !important;
    box-shadow: 0 6px 16px rgba(0,61,122,0.3) !important;
}}

/* Section headers */
.section-header {{
    font-size: 20px;
    font-weight: 700;
    color: {BLEU_FONCE};
    margin: 32px 0 16px 0;
    padding-left: 12px;
    border-left: 4px solid {BLEU_FONCE};
}}

/* Recommendation cards */
.reco-card {{
    background: #F0F9FF;
    border-left: 4px solid {BLEU_FONCE};
    padding: 16px 20px;
    border-radius: 0 8px 8px 0;
    margin-bottom: 12px;
}}
.reco-title {{
    font-weight: 700;
    color: {BLEU_FONCE};
    font-size: 15px;
    margin-bottom: 6px;
}}
.reco-detail {{
    color: {GRIS_TEXTE};
    font-size: 14px;
    line-height: 1.6;
}}

/* Result block */
.result-block {{
    padding: 16px 24px;
    border-radius: 10px;
    font-weight: 600;
    font-size: 15px;
    text-align: center;
    margin-top: 16px;
}}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="header-bar">
    <h1>Simulation & Decisions</h1>
    <p>Testez des scenarios  •  Obtenez des recommandations automatiques</p>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# CHARGEMENT DONNEES
# ══════════════════════════════════════════════════════════════════
df = load_hourly_data()
df_recent = get_recent_hours(df, 48)
features = get_features_for_prediction(df_recent)

# Dernieres valeurs connues pour defaults
last_row = df.iloc[-1]
default_effectif = int(last_row.get('effectif_soignant_present', 18))
default_lits = int(last_row.get('dispo_lits_aval', 12))

# ══════════════════════════════════════════════════════════════════
# SECTION 1 : CHOIX D'HORIZON (TRES VISIBLE)
# ══════════════════════════════════════════════════════════════════
st.markdown("<div class='section-header'>1. Choisissez l'horizon de prevision</div>", unsafe_allow_html=True)

# State management pour l'horizon selectionne
if 'horizon' not in st.session_state:
    st.session_state.horizon = 'J+1'

col_j1, col_j3, col_j7 = st.columns(3)

with col_j1:
    btn_style_j1 = "primary" if st.session_state.horizon == 'J+1' else "secondary"
    if st.button("J+1  •  24 heures", key="btn_j1", use_container_width=True,
                 type=btn_style_j1):
        st.session_state.horizon = 'J+1'
        st.rerun()

with col_j3:
    btn_style_j3 = "primary" if st.session_state.horizon == 'J+3' else "secondary"
    if st.button("J+3  •  72 heures", key="btn_j3", use_container_width=True,
                 type=btn_style_j3):
        st.session_state.horizon = 'J+3'
        st.rerun()

with col_j7:
    btn_style_j7 = "primary" if st.session_state.horizon == 'J+7' else "secondary"
    if st.button("J+7  •  7 jours", key="btn_j7", use_container_width=True,
                 type=btn_style_j7):
        st.session_state.horizon = 'J+7'
        st.rerun()

horizon = st.session_state.horizon

st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# SECTION 2 : PARAMETRES DE SIMULATION (2 colonnes)
# ══════════════════════════════════════════════════════════════════
st.markdown("<div class='section-header'>2. Configurez les parametres</div>", unsafe_allow_html=True)

# Jour auto-detecte
target_date = datetime.now() + timedelta(days={'J+1': 1, 'J+3': 3, 'J+7': 7}[horizon])
jour_semaine = target_date.weekday()
jours = ["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi", "Dimanche"]

st.markdown(f"""
<div style="background:{BLEU_TRES_CLAIR};border-radius:8px;padding:12px 20px;
            margin-bottom:20px;display:flex;align-items:center;gap:16px">
    <div style="font-size:15px;color:{BLEU_FONCE}">
        <b>Horizon :</b> {horizon}
    </div>
    <div style="font-size:15px;color:{BLEU_FONCE}">
        <b>Jour cible :</b> {jours[jour_semaine]} {target_date.strftime('%d/%m/%Y')}
    </div>
</div>
""", unsafe_allow_html=True)

col_env, col_hosp = st.columns(2)

with col_env:
    st.markdown(f"""
    <div style="{CARD_STYLE}">
        <div style="font-size:16px;font-weight:700;color:{BLEU_FONCE};margin-bottom:16px">
            Contexte externe
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Scenario d'affluence (remplace le multiplicateur obscur)
    scenario_affluence = st.selectbox(
        "Scenario d'affluence",
        list(SCENARIOS_AFFLUENCE.keys()),
        index=0,
        help="Impact sur le flux de patients entrants"
    )
    affluence_mult = SCENARIOS_AFFLUENCE[scenario_affluence]

    # Alerte epidemique (select au lieu de toggle)
    epidemie_options = ["Aucune", "Grippe", "COVID-19", "Bronchiolite", "Gastro-enterite"]
    epidemie_choice = st.selectbox(
        "Alerte epidemique",
        epidemie_options,
        help="Impact sur l'affluence et la duree de prise en charge"
    )
    epidemie = epidemie_choice != "Aucune"

    # Temperature
    temp = st.slider(
        "Temperature prevue (C)",
        -5, 40, 20,
        help="Temperatures extremes augmentent l'affluence"
    )
    canicule = temp >= 35

    # Greve
    greve = st.toggle(
        "Greve des transports",
        help="Impacte l'acheminement des patients et l'arrivee du personnel"
    )

with col_hosp:
    st.markdown(f"""
    <div style="{CARD_STYLE}">
        <div style="font-size:16px;font-weight:700;color:{BLEU_FONCE};margin-bottom:16px">
            Ressources internes
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Init session state pour widgets
    if 'w_effectif' not in st.session_state:
        st.session_state.w_effectif = default_effectif
    if 'w_lits' not in st.session_state:
        st.session_state.w_lits = default_lits
    patients_data = int(last_row.get('nb_patients_en_cours', 35))
    if 'w_patients' not in st.session_state:
        st.session_state.w_patients = patients_data

    effectif = st.number_input(
        "Effectif soignant present",
        min_value=5,
        max_value=30,
        key="w_effectif",
        help=f"Actuel : {default_effectif} | Seuil critique : < 10"
    )

    lits = st.number_input(
        "Lits d'aval disponibles",
        min_value=0,
        max_value=25,
        key="w_lits",
        help=f"Actuel : {default_lits} | Seuil critique : <= 3"
    )

    patients_actuels = st.number_input(
        "Patients deja presents",
        min_value=0,
        max_value=max(80, patients_data + 10),
        key="w_patients",
        help="Nombre de patients aux urgences actuellement"
    )

st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

# Bouton de lancement (relance la simulation avec les nouveaux params)
btn_simulate = st.button(
    "ACTUALISER LA SIMULATION",
    type="primary",
    use_container_width=True
)

# ══════════════════════════════════════════════════════════════════
# SECTION 3 : RESULTATS (toujours visibles par defaut)
# ══════════════════════════════════════════════════════════════════
st.markdown("<div style='height:32px'></div>", unsafe_allow_html=True)
st.markdown("<div class='section-header'>3. Resultats de la simulation</div>", unsafe_allow_html=True)

# Afficher les parametres actuels
st.markdown(f"""
<div style="background:{BLEU_TRES_CLAIR};border-radius:8px;padding:12px 20px;margin-bottom:16px">
    <span style="color:{BLEU_FONCE};font-weight:600">Parametres actuels :</span>
    <span style="color:{GRIS_TEXTE}">
        Effectif = <b>{effectif}</b> |
        Lits = <b>{lits}</b> |
        Patients = <b>{patients_actuels}</b> |
        Affluence = <b>{scenario_affluence}</b>
    </span>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# CALCUL DIRECT DU SCORE (sans passer par les fonctions complexes)
# Ceci garantit que les changements de parametres sont VISIBLES
# ══════════════════════════════════════════════════════════════════

# Score de base selon l'heure (pic vers 17h)
import math
heure_actuelle = datetime.now().hour
base_horaire = 50 + 20 * math.sin(math.pi * (heure_actuelle - 6) / 14)

# Impact des parametres utilisateur (coefficients forts pour etre visibles)
impact_patients = (patients_actuels - 40) * 1.5      # +1.5 pts par patient > 40
impact_effectif = -(effectif - 15) * 3               # -3 pts par soignant > 15
impact_lits = -(lits - 8) * 2                        # -2 pts par lit > 8
impact_greve = 15 if greve else 0
impact_epidemie = 12 if epidemie else 0
impact_canicule = 8 if canicule else 0
impact_affluence = (affluence_mult - 1.0) * 20       # +20 pts si affluence +100%

# Score scenario = base + tous les impacts
scenario_score_raw = (base_horaire + impact_patients + impact_effectif +
                      impact_lits + impact_greve + impact_epidemie +
                      impact_canicule + impact_affluence)
scenario_score = max(0, min(100, scenario_score_raw))

# Score baseline = juste la base horaire (sans modifications utilisateur)
baseline_score = max(0, min(100, base_horaire))

# Pour le graphique : generer 24 heures de predictions
hours_list = list(range(24))
baseline_24 = []
scenario_24 = []
for h in hours_list:
    base_h = 50 + 20 * math.sin(math.pi * (h - 6) / 14)
    baseline_24.append(max(0, min(100, base_h)))
    scenario_h = (base_h + impact_patients + impact_effectif +
                  impact_lits + impact_greve + impact_epidemie +
                  impact_canicule + impact_affluence)
    scenario_24.append(max(0, min(100, scenario_h)))

baseline_24 = np.array(baseline_24)
scenario_24 = np.array(scenario_24)

baseline_mean = float(np.mean(baseline_24))
scenario_mean = float(np.mean(scenario_24))
delta = scenario_mean - baseline_mean

# Comptage des heures critiques
heures_rouges_base = sum(1 for s in baseline_24 if s >= 85)
heures_rouges_scen = sum(1 for s in scenario_24 if s >= 85)
heures_alerte_base = sum(1 for s in baseline_24 if 70 <= s < 85)
heures_alerte_scen = sum(1 for s in scenario_24 if 70 <= s < 85)

# Pour les variables utilisees plus tard
scenario_choice = "Normal"
if epidemie: scenario_choice = "Epidemie"
elif greve: scenario_choice = "Greve"
elif canicule: scenario_choice = "Canicule"
elif affluence_mult >= 1.5: scenario_choice = "Afflux massif"

# Utiliser hours_list pour le graphique
hours = hours_list

# ── Metriques principales ──
st.markdown(f"""
<div style="{CARD_STYLE}">
    <div style="font-size:16px;font-weight:700;color:{BLEU_FONCE};margin-bottom:16px">
        Indicateurs de charge prevus
    </div>
</div>
""", unsafe_allow_html=True)

col_m1, col_m2, col_m3, col_m4 = st.columns(4)

with col_m1:
    st.markdown(f"""
    <div style="background:{BLANC};border-radius:10px;padding:20px;text-align:center;
                box-shadow:0 2px 6px rgba(0,0,0,0.06)">
        <div style="font-size:36px;font-weight:800;color:{BLEU_FONCE}">{baseline_mean:.0f}</div>
        <div style="font-size:13px;color:{GRIS_TEXTE};margin-top:8px;text-transform:uppercase">
            Score baseline
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_m2:
    niveau, color, icon, _ = get_alert_level(scenario_mean)
    st.markdown(f"""
    <div style="background:{BLANC};border-radius:10px;padding:20px;text-align:center;
                box-shadow:0 2px 6px rgba(0,0,0,0.06);border-top:4px solid {color}">
        <div style="font-size:36px;font-weight:800;color:{BLEU_FONCE}">{scenario_mean:.0f}</div>
        <div style="font-size:13px;color:{GRIS_TEXTE};margin-top:8px;text-transform:uppercase">
            Score scenario
        </div>
        <div style="color:{color};font-weight:600;margin-top:8px">{icon} {niveau}</div>
    </div>
    """, unsafe_allow_html=True)

with col_m3:
    delta_color = ROUGE_CRIT if delta > 0 else VERT_OK
    delta_sign = "+" if delta > 0 else ""
    st.markdown(f"""
    <div style="background:{BLANC};border-radius:10px;padding:20px;text-align:center;
                box-shadow:0 2px 6px rgba(0,0,0,0.06)">
        <div style="font-size:36px;font-weight:800;color:{delta_color}">{delta_sign}{delta:.1f}</div>
        <div style="font-size:13px;color:{GRIS_TEXTE};margin-top:8px;text-transform:uppercase">
            Delta (pts)
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_m4:
    st.markdown(f"""
    <div style="background:{BLANC};border-radius:10px;padding:20px;text-align:center;
                box-shadow:0 2px 6px rgba(0,0,0,0.06)">
        <div style="font-size:36px;font-weight:800;color:{ROUGE_CRIT if heures_rouges_scen > 0 else BLEU_FONCE}">
            {heures_rouges_scen}h
        </div>
        <div style="font-size:13px;color:{GRIS_TEXTE};margin-top:8px;text-transform:uppercase">
            Heures critiques
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

# ── Graphique overlay ──
st.markdown(f"""
<div style="{CARD_STYLE}">
    <div style="font-size:16px;font-weight:700;color:{BLEU_FONCE};margin-bottom:8px">
        Evolution de la charge : Baseline vs Scenario
    </div>
</div>
""", unsafe_allow_html=True)

fig_comp = comparison_overlay(
    [f"{h}h" for h in hours],
    baseline_24, scenario_24
)
st.plotly_chart(fig_comp, use_container_width=True, config={'displayModeBar': False})

# ══════════════════════════════════════════════════════════════════
# SECTION 3b : PREVISION AFFLUENCE (modele RF - toujours visible)
# ══════════════════════════════════════════════════════════════════
# Affluence prevue (modele RF) - toujours calculee
affluence_pred = predict_affluence(df_recent.tail(24))
affluence_max = float(np.max(affluence_pred))
affluence_mean = float(np.mean(affluence_pred))

# Couleur selon niveau d'affluence
if affluence_max >= 12:
    aff_color = ORANGE_ALERT
    aff_status = "Elevee"
    aff_bg = "#FFF3E0"
elif affluence_max >= 8:
    aff_color = JAUNE_VIGIL
    aff_status = "Moderee"
    aff_bg = "#FFFDE7"
else:
    aff_color = VERT_OK
    aff_status = "Normale"
    aff_bg = "#E8F5E9"

st.markdown(f"""
<div style="background:{aff_bg};border-left:4px solid {aff_color};
            border-radius:0 8px 8px 0;padding:16px 20px;margin:16px 0">
    <div style="font-weight:700;color:{BLEU_FONCE};margin-bottom:8px;font-size:15px">
        Prevision affluence (modele RandomForest)
    </div>
    <div style="font-size:14px;color:{GRIS_TEXTE};line-height:1.6">
        <b>Pic prevu</b> : {affluence_max:.1f} patients/heure<br>
        <b>Moyenne</b> : {affluence_mean:.1f} patients/heure<br>
        <b>Statut</b> : <span style="color:{aff_color};font-weight:600">{aff_status}</span>
        <span style="color:{GRIS_TEXTE};font-style:italic;font-size:12px"> (seuil alerte : 12 pat/h)</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# SECTION 3c : CAUSES IDENTIFIEES (si score eleve)
# ══════════════════════════════════════════════════════════════════
if scenario_mean >= 70:
    # Temps de passage observe (donnees reelles ou fallback)
    if 'avg_passage_time' in df.columns:
        tdp_data = df.groupby('heure')['avg_passage_time'].mean()
        tdp_vals = [tdp_data.get(h, 240) for h in range(24)]
    else:
        tdp_vals = [180, 170, 160, 155, 160, 175, 200, 230, 260, 275, 280, 285,
                    290, 300, 310, 320, 330, 340, 320, 300, 270, 240, 210, 190]
    tdp_max = float(max(tdp_vals))

    causes = []
    if affluence_max >= 12:
        causes.append(
            f"<b>Affluence elevee</b> : {affluence_max:.0f} pat/h (seuil : 12)"
        )
    if tdp_max >= 240:
        causes.append(
            f"<b>Temps passage long</b> : {tdp_max:.0f} min ({tdp_max/60:.1f}h)"
        )
    if effectif < 15:
        causes.append(
            f"<b>Sous-effectif</b> : {effectif} soignants (critique &lt; 15)"
        )
    if lits < 5:
        causes.append(
            f"<b>Lits satures</b> : {lits} disponibles (critique &le; 3)"
        )

    if causes:
        st.markdown(f"""
        <div style="background:#FFEBEE;border-left:4px solid {ROUGE_CRIT};
                    border-radius:0 8px 8px 0;padding:16px 20px;margin:16px 0">
            <div style="font-weight:700;color:{BLEU_FONCE};margin-bottom:10px;font-size:15px">
                Causes identifiees du score eleve
            </div>
            <div style="font-size:14px;color:{GRIS_TEXTE};line-height:1.8">
                {"<br>".join(causes)}
            </div>
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# SECTION 4 : RECOMMANDATIONS AUTOMATIQUES
# ══════════════════════════════════════════════════════════════════
st.markdown("<div class='section-header'>4. Recommandations strategiques</div>", unsafe_allow_html=True)

# Afficher message si une reco vient d'etre appliquee
if 'last_applied' in st.session_state:
    st.success(f"Recommandation appliquee : {st.session_state.last_applied}")
    del st.session_state.last_applied

st.markdown(f"""
<div style="{CARD_STYLE}">
    <div style="font-size:14px;color:{GRIS_TEXTE};margin-bottom:16px">
        Basees sur le scenario simule et les regles metier (EDA + SHAP)
    </div>
</div>
""", unsafe_allow_html=True)

# Generer les recommandations
current_feat = df.iloc[-1].to_dict()
current_feat['effectif_soignant_present'] = effectif
current_feat['dispo_lits_aval'] = lits
if greve:
    current_feat['indicateur_greve'] = 1
if epidemie:
    current_feat['alerte_epidemique_encoded'] = 1

recos = generate_recommendations(current_feat, baseline_score=scenario_mean)

total_gain = 0
for reco in recos:
    border_color = ROUGE_CRIT if reco['niveau'] == 'critique' else ORANGE_ALERT
    icon = '🚨' if reco['niveau'] == 'critique' else '⚠'
    gain_color = VERT_OK if reco['gain_score'] < 0 else ROUGE_CRIT
    total_gain += reco['gain_score']

    col_reco, col_btn = st.columns([5, 1])

    with col_reco:
        st.markdown(f"""
        <div class="reco-card" style="border-left-color:{border_color}">
            <div class="reco-title">{icon} {reco['priorite']} — {reco['titre']}</div>
            <div class="reco-detail">
                <b>Cause :</b> {reco['cause']}<br>
                <b>Action :</b> {reco['action']}<br>
                <span style="color:{gain_color};font-weight:700">
                    Gain estime : {reco['gain_score']:+.1f} pts de score
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_btn:
        st.markdown("<div style='height:30px'></div>", unsafe_allow_html=True)
        if reco['feature'] and reco['target_value']:
            if st.button(f"Appliquer", key=f"apply_{reco['id']}", type="primary"):
                # Stocker dans pending_* pour appliquer au prochain rerun (AVANT widgets)
                if reco['feature'] == 'effectif_soignant_present':
                    old_val = st.session_state.get('w_effectif', effectif)
                    st.session_state.pending_effectif = int(reco['target_value'])
                    st.session_state.last_applied = f"Effectif: {old_val} → {reco['target_value']}"
                elif reco['feature'] == 'dispo_lits_aval':
                    old_val = st.session_state.get('w_lits', lits)
                    st.session_state.pending_lits = int(reco['target_value'])
                    st.session_state.last_applied = f"Lits: {old_val} → {reco['target_value']}"
                st.rerun()

# Bloc resultat global
score_apres_recos = scenario_mean + total_gain
score_apres_recos = max(0, min(100, score_apres_recos))
level_final = get_charge_level(score_apres_recos)
bg_result = '#ECFDF5' if score_apres_recos < 50 else '#FEF3C7' if score_apres_recos < 70 else '#FFEDD5' if score_apres_recos < 85 else '#FEE2E2'

st.markdown(f"""
<div class="result-block" style="background:{bg_result};border:2px solid {level_final['color']}">
    Avec les {len(recos)} actions : score {scenario_mean:.0f} → {score_apres_recos:.0f}
    ({level_final['icon']} {level_final['label']})
    &nbsp;•&nbsp; Gain total : {total_gain:+.1f} pts
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# SECTION 5 : EXPORT PDF
# ══════════════════════════════════════════════════════════════════
st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

col_export, _ = st.columns([1, 2])
with col_export:
    if st.button("Exporter le rapport PDF", use_container_width=True):
        try:
            from utils.pdf_export import generate_simulation_pdf

            pdf_bytes = generate_simulation_pdf(
                scenario=scenario_choice,
                horizon=horizon,
                effectif=effectif,
                lits=lits,
                baseline_mean=baseline_mean,
                scenario_mean=scenario_mean,
                delta=delta,
                heures_rouges_base=heures_rouges_base,
                heures_rouges_scen=heures_rouges_scen,
                recos=recos,
                total_gain=total_gain,
                score_final=score_apres_recos,
            )

            timestamp = datetime.now().strftime('%Y%m%d_%H%M')
            st.download_button(
                label="Telecharger le rapport PDF",
                data=pdf_bytes,
                file_name=f"rapport_simulation_{timestamp}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
            st.success("Rapport genere avec succes.")

        except ImportError:
            st.error("Module fpdf2 non installe. Executez : pip install fpdf2")
        except Exception as e:
            st.error(f"Erreur generation PDF : {e}")
