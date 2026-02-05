"""
Page 1 — Pilotage
Vue temps reel pour la direction et le cadre de garde.
Objectif : comprendre la situation en 30 secondes.
"""
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from utils.config import (
    BLEU_FONCE, BLEU_CLAIR, BLEU_TRES_CLAIR, GRIS_TEXTE, BLANC,
    VERT_OK, JAUNE_VIGIL, ORANGE_ALERT, ROUGE_CRIT,
    get_charge_level, get_kpi_status, status_color, status_icon,
)
from utils.charts import (
    gauge_charge, charge_curve_with_forecast,
    creneaux_critiques, heatmap_jour_heure,
)
from utils.data_loader import load_hourly_data, get_current_snapshot, get_recent_hours
from utils.model_engine import generate_forecast

# ══════════════════════════════════════════════════════════════════
# CSS - Header bar compacte + Sidebar bleu fonce
# ══════════════════════════════════════════════════════════════════
st.markdown(f"""
<style>
/* Sidebar bleu fonce AP-HP - tout en blanc */
[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, {BLEU_FONCE} 0%, #1B5E9E 100%);
}}
[data-testid="stSidebar"] * {{
    color: white !important;
}}
/* Navigation sidebar - texte blanc visible */
section[data-testid="stSidebar"] nav a span {{
    color: white !important;
    font-weight: 500 !important;
}}
section[data-testid="stSidebar"] nav a[aria-current="page"] {{
    background: rgba(255,255,255,0.15) !important;
}}

/* Header bar large */
.header-bar {{
    background: linear-gradient(135deg, {BLEU_FONCE} 0%, #1B5E9E 100%);
    padding: 28px 40px;
    border-radius: 12px;
    margin-bottom: 24px;
}}
.header-bar h2 {{
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

/* KPI Cards */
.kpi-card {{
    background: {BLANC};
    border-radius: 12px;
    padding: 20px;
    box-shadow: 0 2px 8px rgba(0,61,122,0.08);
    border: 1px solid rgba(0,61,122,0.08);
    min-height: 140px;
}}
.kpi-label {{
    font-size: 13px;
    color: {GRIS_TEXTE};
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 8px;
}}
.kpi-value {{
    font-size: 40px;
    font-weight: 800;
    color: {BLEU_FONCE};
    line-height: 1;
}}
.kpi-unit {{
    font-size: 13px;
    color: {GRIS_TEXTE};
    margin-top: 6px;
}}

/* Section cards */
.section-card {{
    background: {BLANC};
    border-radius: 12px;
    padding: 20px;
    box-shadow: 0 2px 8px rgba(0,61,122,0.08);
    border: 1px solid rgba(0,61,122,0.08);
    margin-bottom: 16px;
}}
.section-title {{
    font-size: 16px;
    font-weight: 700;
    color: {BLEU_FONCE};
    margin-bottom: 4px;
}}
.section-subtitle {{
    font-size: 13px;
    color: {GRIS_TEXTE};
}}

/* Bouton primary */
.stButton > button[kind="primary"] {{
    background: {BLEU_FONCE} !important;
    color: white !important;
    font-weight: 600 !important;
    padding: 12px 24px !important;
    border-radius: 8px !important;
    border: none !important;
}}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# HEADER BAR COMPACTE
# ══════════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="header-bar">
    <h2>Pilotage des Urgences</h2>
    <p>Vue temps reel  •  Derniere mise a jour : {datetime.now().strftime('%d/%m/%Y %H:%M')}</p>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# CHARGEMENT DONNEES
# ══════════════════════════════════════════════════════════════════
df = load_hourly_data()
snapshot = get_current_snapshot(df)
score_actuel = snapshot['charge']
level = get_charge_level(score_actuel)

# ══════════════════════════════════════════════════════════════════
# SECTION 1 : JAUGE + 3 KPI CARDS (4 colonnes pour eviter superposition)
# ══════════════════════════════════════════════════════════════════
col_gauge, col_k1, col_k2, col_k3 = st.columns([1.5, 1, 1, 1])

with col_gauge:
    fig_gauge = gauge_charge(score_actuel)
    st.plotly_chart(fig_gauge, use_container_width=True, config={'displayModeBar': False})
    st.markdown(f"""
    <div style="text-align:center;margin-top:-10px">
        <span style="font-size:16px;font-weight:700;color:{level['color']}">
            {level['icon']} {level['label']}
        </span>
    </div>
    """, unsafe_allow_html=True)


def kpi_card(label, value, unit, kpi_name, fmt=".0f"):
    """Carte KPI avec bordure coloree selon statut."""
    status = get_kpi_status(value, kpi_name)
    color = status_color(status)
    icon = status_icon(status)
    st.markdown(f"""
    <div class="kpi-card" style="border-left:4px solid {color}">
        <div class="kpi-label">{icon} {label}</div>
        <div class="kpi-value">{value:{fmt}}</div>
        <div class="kpi-unit">{unit}</div>
    </div>
    """, unsafe_allow_html=True)


with col_k1:
    kpi_card("Patients presents", snapshot['patients'],
             "/ 55 places", "patients")
with col_k2:
    kpi_card("Effectif soignant", snapshot['effectif'],
             "soignants", "effectif")
with col_k3:
    kpi_card("Lits aval dispo", snapshot['lits'],
             "lits libres", "lits")

st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# SECTION 2 : COURBE CHARGE 24H + PREVISION
# ══════════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="section-card">
    <div class="section-title">Charge des dernieres 24h + Prevision J+1</div>
    <div class="section-subtitle">
        Trait plein = observe  •  Pointille = prevision XGBoost  •  Zone grisee = +/- MAE (4.5 pts)
    </div>
</div>
""", unsafe_allow_html=True)

# Donnees passees (24h)
df_past = get_recent_hours(df, 24)
df_past = df_past.rename(columns={'date_hourly': 'datetime'})

# Prevision (24h suivantes)
df_recent = get_recent_hours(df, 48)
df_forecast = generate_forecast(df_recent, n_hours=24, horizon='J+1')

fig_curve = charge_curve_with_forecast(df_past, df_forecast)
st.plotly_chart(fig_curve, use_container_width=True, config={'displayModeBar': False})

st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# SECTION 3 : CRENEAUX CRITIQUES (pleine largeur)
# ══════════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="section-card">
    <div class="section-title">Creneaux critiques (prochaines 24h)</div>
    <div class="section-subtitle">Score prevu par heure — Seuils : Vigilance 50 | Attention 70 | Critique 85</div>
</div>
""", unsafe_allow_html=True)

fig_bars = creneaux_critiques(
    [f"{h}h" for h in df_forecast['datetime'].dt.hour],
    df_forecast['charge_pred'].values
)
st.plotly_chart(fig_bars, use_container_width=True, config={'displayModeBar': False})

st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# SECTION 4 : HEATMAP (pleine largeur)
# ══════════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="section-card">
    <div class="section-title">Heatmap — Flux moyen par creneau</div>
    <div class="section-subtitle">Donnees historiques (5 ans) — Le marqueur indique votre position actuelle</div>
</div>
""", unsafe_allow_html=True)

current_h = datetime.now().hour
current_d = datetime.now().weekday()
fig_heat = heatmap_jour_heure(df, current_hour=current_h, current_day=current_d)
st.plotly_chart(fig_heat, use_container_width=True, config={'displayModeBar': False})

# ══════════════════════════════════════════════════════════════════
# BOUTON SIMULATION (en bas de page)
# ══════════════════════════════════════════════════════════════════
st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)
col_spacer1, col_btn, col_spacer2 = st.columns([1, 2, 1])
with col_btn:
    if st.button("Simuler un scenario", use_container_width=True, type="primary"):
        st.switch_page("pages/2_Simulation.py")
