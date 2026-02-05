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
    VERT_OK, ORANGE_ALERT, ROUGE_CRIT,
    get_charge_level, get_kpi_status, status_color, status_icon,
)
from utils.charts import (
    gauge_charge, charge_curve_with_forecast,
    creneaux_critiques, heatmap_jour_heure,
)
from utils.data_loader import load_hourly_data, get_current_snapshot, get_recent_hours
from utils.model_engine import generate_forecast

# ══════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="header-bar">
    <div>
        <h1>Pilotage des Urgences</h1>
        <p>Vue temps reel  •  Derniere mise a jour : {datetime.now().strftime('%d/%m/%Y %H:%M')}</p>
    </div>
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
# SECTION 1 : JAUGE + 4 KPI CARDS
# ══════════════════════════════════════════════════════════════════
col_gauge, col_k1, col_k2, col_k3, col_k4 = st.columns([1.4, 1, 1, 1, 1])

with col_gauge:
    fig_gauge = gauge_charge(score_actuel)
    st.plotly_chart(fig_gauge, use_container_width=True, config={'displayModeBar': False})
    st.markdown(f"""
    <div style="text-align:center;margin-top:-10px">
        <span style="font-size:14px;font-weight:600;color:{level['color']}">
            {level['icon']} {level['label']}
        </span>
    </div>
    """, unsafe_allow_html=True)

def kpi_card(label, value, unit, kpi_name, fmt=".0f"):
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
with col_k4:
    kpi_card("Flux patients/h", snapshot['flux'],
             "patients/h", "flux")

st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# SECTION 2 : COURBE CHARGE 24H + PREVISION
# ══════════════════════════════════════════════════════════════════
st.markdown(f"""
<div style="background:{BLANC};border-radius:10px;padding:20px;
            box-shadow:0 2px 8px rgba(0,0,0,0.06);margin-bottom:20px">
    <div style="font-size:15px;font-weight:700;color:{BLEU_FONCE};margin-bottom:4px">
        Charge des dernieres 24h + Prevision J+1
    </div>
    <div style="font-size:12px;color:{GRIS_TEXTE}">
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


# ══════════════════════════════════════════════════════════════════
# SECTION 3 : CRENEAUX CRITIQUES + HEATMAP (2 colonnes)
# ══════════════════════════════════════════════════════════════════
col_left, col_right = st.columns([1, 1.3])

with col_left:
    st.markdown(f"""
    <div style="background:{BLANC};border-radius:10px;padding:20px;
                box-shadow:0 2px 8px rgba(0,0,0,0.06)">
        <div style="font-size:15px;font-weight:700;color:{BLEU_FONCE};margin-bottom:12px">
            Creneaux critiques (prochaines 24h)
        </div>
    </div>
    """, unsafe_allow_html=True)

    fig_bars = creneaux_critiques(
        [f"{h}h" for h in df_forecast['datetime'].dt.hour],
        df_forecast['charge_pred'].values
    )
    st.plotly_chart(fig_bars, use_container_width=True, config={'displayModeBar': False})

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    if st.button("Simuler un scenario", use_container_width=True, type="primary"):
        st.switch_page("pages/2_🔮_Simulation.py")

with col_right:
    st.markdown(f"""
    <div style="background:{BLANC};border-radius:10px;padding:20px;
                box-shadow:0 2px 8px rgba(0,0,0,0.06)">
        <div style="font-size:15px;font-weight:700;color:{BLEU_FONCE};margin-bottom:4px">
            Heatmap — Flux moyen par creneau
        </div>
        <div style="font-size:12px;color:{GRIS_TEXTE}">
            Donnees historiques
        </div>
    </div>
    """, unsafe_allow_html=True)

    current_h = datetime.now().hour
    current_d = datetime.now().weekday()
    fig_heat = heatmap_jour_heure(df, current_hour=current_h, current_day=current_d)
    st.plotly_chart(fig_heat, use_container_width=True, config={'displayModeBar': False})
