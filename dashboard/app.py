"""
MVP Dashboard Urgences — Pitie-Salpetriere (AP-HP)
Point d'entree Streamlit multi-pages
Lancer : streamlit run app.py
"""
import streamlit as st
import os

# ══════════════════════════════════════════════════════════════════
# CONFIG PAGE
# ══════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Urgences PSL — Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════
# CSS GLOBAL
# ══════════════════════════════════════════════════════════════════
st.markdown("""
<style>
    /* --- Fond general gris medical --- */
    .stApp { background-color: #E8ECF0; }

    /* --- Sidebar bleu AP-HP --- */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #003D7A 0%, #1B5E9E 100%);
    }
    [data-testid="stSidebar"] * { color: white !important; }
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stRadio label { font-weight: 600; }

    /* --- Masquer header Streamlit --- */
    header[data-testid="stHeader"] { display: none; }
    .block-container { padding-top: 0.5rem; max-width: 1400px; }

    /* --- Header custom --- */
    .header-bar {
        background: linear-gradient(135deg, #003D7A 0%, #1B5E9E 100%);
        color: white; padding: 18px 30px; border-radius: 0 0 12px 12px;
        margin: -0.5rem -1rem 20px -1rem; display: flex;
        align-items: center; gap: 16px;
        box-shadow: 0 4px 12px rgba(0,61,122,0.25);
    }
    .header-bar h1 { margin: 0; font-size: 22px; font-weight: 700; }
    .header-bar p  { margin: 2px 0 0; font-size: 13px; opacity: 0.8; }

    /* --- KPI Cards --- */
    .kpi-card {
        background: white; border-radius: 10px; padding: 18px 16px;
        text-align: center; box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        transition: transform 0.15s; height: 140px;
        display: flex; flex-direction: column; justify-content: center;
    }
    .kpi-card:hover { transform: translateY(-2px); box-shadow: 0 4px 12px rgba(0,0,0,0.1); }
    .kpi-label { font-size: 12px; color: #4A5568; text-transform: uppercase;
                 letter-spacing: 0.5px; margin-bottom: 6px; }
    .kpi-value { font-size: 34px; font-weight: 800; color: #003D7A; line-height: 1; }
    .kpi-unit  { font-size: 12px; color: #718096; margin-top: 4px; }

    /* --- Reco cards --- */
    .reco-card {
        background: white; border-radius: 0 10px 10px 0; padding: 16px 20px;
        margin-bottom: 12px; box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    }
    .reco-title { font-weight: 700; color: #003D7A; font-size: 15px; }
    .reco-detail { color: #4A5568; font-size: 13px; margin-top: 6px; line-height: 1.5; }

    /* --- Scenario buttons --- */
    .scenario-bar {
        background: #D6E6F2; border-radius: 10px; padding: 12px 20px;
        margin-bottom: 20px; display: flex; gap: 10px; flex-wrap: wrap;
        align-items: center;
    }

    /* --- Result block --- */
    .result-block {
        border-radius: 10px; padding: 16px 24px; text-align: center;
        font-weight: 700; font-size: 16px; margin-top: 16px;
    }

    /* --- Tabs --- */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #D6E6F2; border-radius: 8px 8px 0 0;
        color: #003D7A; font-weight: 600; padding: 8px 20px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #003D7A !important; color: white !important;
    }

    /* --- Plotly chart containers --- */
    .stPlotlyChart { border-radius: 10px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════
with st.sidebar:
    # Logo
    logo_path = os.path.join(os.path.dirname(__file__), 'assets', 'logoPSL-630x214V6.png')
    if os.path.exists(logo_path):
        st.image(logo_path, use_container_width=True)
    else:
        st.markdown("""
        <div style="text-align:center;padding:10px 0 20px">
            <div style="font-size:42px">🏥</div>
            <div style="font-size:16px;font-weight:700;margin-top:4px">
                Pitie-Salpetriere
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align:center;padding:0 0 10px">
        <div style="font-size:11px;opacity:0.7">AP-HP  •  Urgences</div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    st.markdown("""
    <div style="padding:8px 0">
        <div style="font-size:11px;opacity:0.6;text-transform:uppercase;
                    letter-spacing:1px;margin-bottom:12px">Navigation</div>
    </div>
    """, unsafe_allow_html=True)

    # Les pages sont gerees automatiquement par Streamlit multi-pages
    # (fichiers dans pages/)

    st.divider()

    with st.expander("A propos du MVP"):
        st.markdown("""
        **Modeles predictifs :**
        - Charge : XGBoost + LSTM + Ensemble
        - Affluence : RandomForest
        - Tendance : SARIMAX

        **Donnees :**
        570K passages  •  Nov 2020 — Oct 2025
        Donnees synthetiques (pas de donnees reelles)

        **Conformite :**
        Aucune donnee identifiante  •  RGPD OK
        Production : hebergement HDS requis
        """)

    st.divider()
    st.markdown("""
    <div style="font-size:10px;opacity:0.5;text-align:center;padding-top:20px">
        MVP v1.0  •  Fevrier 2026<br>
        Donnees synthetiques
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# PAGE D'ACCUEIL
# ══════════════════════════════════════════════════════════════════
st.markdown("""
<div class="header-bar">
    <div>
        <h1>Systeme de Pilotage des Urgences</h1>
        <p>Hopital Pitie-Salpetriere  •  AP-HP  •  Dashboard predictif</p>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
### Bienvenue sur le MVP de gestion hospitaliere

Ce tableau de bord permet de :

- **Piloter** — Visualiser la charge en temps reel et anticiper les prochaines 24h
- **Simuler** — Tester des scenarios (epidemie, greve, afflux) et obtenir des recommandations
- **Analyser** — Comprendre les patterns historiques et les facteurs de tension

**Selectionnez une page dans le menu lateral pour commencer.**
""")

# Quick KPIs on home page
st.markdown("---")
st.markdown("#### Dernieres metriques disponibles")

# Charger les vraies donnees pour les KPIs d'accueil
try:
    from utils.data_loader import load_hourly_data, get_current_snapshot
    df_hourly = load_hourly_data()
    snapshot = get_current_snapshot(df_hourly)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Score de charge", f"{snapshot['charge']:.0f}/100")
    with col2:
        st.metric("Patients presents", f"{snapshot['patients']:.0f}")
    with col3:
        st.metric("Effectif soignant", f"{snapshot['effectif']:.0f}")
    with col4:
        st.metric("Lits aval", f"{snapshot['lits']:.0f}")
except Exception:
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Score de charge", "—", help="Ouvrir Page Pilotage")
    with col2:
        st.metric("Patients presents", "—")
    with col3:
        st.metric("Effectif soignant", "—")
    with col4:
        st.metric("Lits aval", "—")
    st.info("Les donnees seront chargees lorsque vous ouvrirez la page Pilotage.")
