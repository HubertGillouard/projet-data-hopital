"""
Page 3 — Analyse & Patterns
Graphiques EDA selectionnes + SHAP pour le jury et les cadres.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

from utils.config import (
    BLEU_FONCE, BLEU_MOYEN, BLEU_CLAIR, BLEU_TRES_CLAIR,
    GRIS_MEDICAL, GRIS_TEXTE, BLANC,
    VERT_OK, JAUNE_VIGIL, ORANGE_ALERT, ROUGE_CRIT,
)
from utils.data_loader import load_raw_data, load_hourly_data

# ══════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="header-bar">
    <div>
        <h1>Analyse & Patterns</h1>
        <p>Tendances historiques  •  Facteurs de tension  •  Interpretabilite</p>
    </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# CHARGEMENT DONNEES REELLES
# ══════════════════════════════════════════════════════════════════
LAYOUT = dict(
    plot_bgcolor=BLANC, paper_bgcolor='rgba(0,0,0,0)',
    font=dict(family="Arial, sans-serif", color=GRIS_TEXTE),
    margin=dict(t=40, b=40, l=50, r=20),
)


@st.cache_data(ttl=3600)
def prepare_eda_data():
    """Prepare les donnees EDA depuis le CSV brut."""
    df = load_raw_data()

    if df is None:
        # Fallback : donnees simulees
        return _simulated_eda_data()

    # S'assurer que datetime est parse
    if not pd.api.types.is_datetime64_any_dtype(df['date_heure_arrivee']):
        df['date_heure_arrivee'] = pd.to_datetime(
            df['date_heure_arrivee'], format='%d/%m/%Y %H:%M', errors='coerce'
        )

    df['date'] = df['date_heure_arrivee'].dt.date

    # Admissions quotidiennes
    admissions_jour = df.groupby('date').size().reset_index(name='admissions_jour')
    admissions_jour['date'] = pd.to_datetime(admissions_jour['date'])
    admissions_jour = admissions_jour.sort_values('date')
    admissions_jour['ma_7j'] = admissions_jour['admissions_jour'].rolling(7).mean()
    admissions_jour['ma_30j'] = admissions_jour['admissions_jour'].rolling(30).mean()

    return {
        'df': df,
        'df_daily': admissions_jour,
        'n_passages': len(df),
    }


def _simulated_eda_data():
    """Fallback pour donnees simulees."""
    np.random.seed(42)
    dates = pd.date_range('2021-01-01', '2025-10-31', freq='D')
    trend = np.linspace(240, 350, len(dates))
    seasonal = 20 * np.sin(2 * np.pi * np.arange(len(dates)) / 365)
    noise = np.random.normal(0, 25, len(dates))
    admissions = trend + seasonal + noise

    df_daily = pd.DataFrame({
        'date': dates,
        'admissions_jour': admissions,
        'ma_7j': pd.Series(admissions).rolling(7).mean().values,
        'ma_30j': pd.Series(admissions).rolling(30).mean().values,
    })
    return {'df': None, 'df_daily': df_daily, 'n_passages': 570282}


data = prepare_eda_data()
df_daily = data['df_daily']
df = data['df']

# Indicateur source
source = "Donnees reelles (570K passages)" if df is not None else "Donnees simulees"
st.caption(f"Source : {source}")


# ══════════════════════════════════════════════════════════════════
# SECTION 1 : TENDANCE STRUCTURELLE (Graph 16)
# ══════════════════════════════════════════════════════════════════
st.markdown(f"""
<div style="background:{BLANC};border-radius:10px;padding:20px;
            box-shadow:0 2px 8px rgba(0,0,0,0.06);margin-bottom:20px">
    <div style="font-size:16px;font-weight:700;color:{BLEU_FONCE}">
        Tendance structurelle — Admissions quotidiennes
    </div>
    <div style="font-size:13px;color:{GRIS_TEXTE}">
        +45% en 5 ans (240 → 350 admissions/jour)  •  Nov 2020 — Oct 2025
    </div>
</div>
""", unsafe_allow_html=True)

fig_trend = go.Figure()
fig_trend.add_trace(go.Scatter(
    x=df_daily['date'], y=df_daily['admissions_jour'],
    mode='lines', name='Admissions/jour',
    line=dict(color=BLEU_TRES_CLAIR, width=0.7), opacity=0.4,
))
fig_trend.add_trace(go.Scatter(
    x=df_daily['date'], y=df_daily['ma_7j'],
    mode='lines', name='MA 7 jours',
    line=dict(color=BLEU_CLAIR, width=1.5),
))
fig_trend.add_trace(go.Scatter(
    x=df_daily['date'], y=df_daily['ma_30j'],
    mode='lines', name='MA 30 jours',
    line=dict(color=BLEU_FONCE, width=2.5),
))
fig_trend.update_layout(
    **LAYOUT, height=350,
    yaxis=dict(title="Admissions/jour", gridcolor=GRIS_MEDICAL),
    legend=dict(orientation='h', yanchor='bottom', y=1.02),
)
st.plotly_chart(fig_trend, use_container_width=True, config={'displayModeBar': False})


# ══════════════════════════════════════════════════════════════════
# SECTION 2 : 2x2 GRAPHIQUES EDA
# ══════════════════════════════════════════════════════════════════
col1, col2 = st.columns(2)

# --- Graph 6 : Desequilibre effectif / demande ---
with col1:
    st.markdown(f"""
    <div style="background:{BLANC};border-radius:10px;padding:16px;
                box-shadow:0 2px 8px rgba(0,0,0,0.06)">
        <div style="font-size:14px;font-weight:700;color:{BLEU_FONCE}">
            Desequilibre Effectif vs Demande
        </div>
        <div style="font-size:12px;color:{GRIS_TEXTE}">
            La baisse d'effectif apres-midi coincide avec le pic de demande
        </div>
    </div>
    """, unsafe_allow_html=True)

    if df is not None and 'effectif_soignant_present' in df.columns:
        # Calculer depuis les vraies donnees
        periodes_def = {'Nuit\n(0-8h)': (0, 8), 'Matin\n(8-12h)': (8, 12),
                        'Apres-midi\n(12-18h)': (12, 18), 'Soir\n(18-0h)': (18, 24)}
        effectif_data = []
        flux_data = []
        periodes_labels = []
        for label, (h_start, h_end) in periodes_def.items():
            mask = (df['heure'] >= h_start) & (df['heure'] < h_end)
            effectif_data.append(df.loc[mask, 'effectif_soignant_present'].mean())
            flux_data.append(df.loc[mask].groupby(df.loc[mask, 'date_heure_arrivee'].dt.floor('h')).size().mean())
            periodes_labels.append(label)
    else:
        periodes_labels = ['Nuit\n(0-8h)', 'Matin\n(8-12h)', 'Apres-midi\n(12-18h)', 'Soir\n(18-0h)']
        effectif_data = [10.5, 22, 19, 16]
        flux_data = [6, 12, 14, 17]

    fig_eff = go.Figure()
    fig_eff.add_trace(go.Bar(name='Effectif soignant', x=periodes_labels, y=effectif_data,
                              marker_color=BLEU_FONCE, width=0.35))
    fig_eff.add_trace(go.Bar(name='Flux patients/h', x=periodes_labels, y=flux_data,
                              marker_color=ORANGE_ALERT, width=0.35))
    fig_eff.update_layout(**LAYOUT, height=320, barmode='group',
                           legend=dict(orientation='h', yanchor='bottom', y=1.02))
    st.plotly_chart(fig_eff, use_container_width=True, config={'displayModeBar': False})


# --- Graph 23 : Impact imagerie ---
with col2:
    st.markdown(f"""
    <div style="background:{BLANC};border-radius:10px;padding:16px;
                box-shadow:0 2px 8px rgba(0,0,0,0.06)">
        <div style="font-size:14px;font-weight:700;color:{BLEU_FONCE}">
            Impact Imagerie sur la Duree
        </div>
        <div style="font-size:12px;color:{GRIS_TEXTE}">
            +40 min systematiques  •  69% des patients concernes
        </div>
    </div>
    """, unsafe_allow_html=True)

    if df is not None and 'besoin_imagerie' in df.columns and 'temps_passage_total' in df.columns:
        sans = df.loc[df['besoin_imagerie'] == 'Non', 'temps_passage_total'].sample(min(1000, len(df)), random_state=42).values
        avec = df.loc[df['besoin_imagerie'] == 'Oui', 'temps_passage_total'].sample(min(2000, len(df)), random_state=42).values
    else:
        np.random.seed(42)
        sans = np.random.normal(200, 60, 500).clip(60, 500)
        avec = np.random.normal(280, 70, 1200).clip(60, 600)

    fig_imag = go.Figure()
    fig_imag.add_trace(go.Violin(y=sans, name='Sans imagerie',
                                  box_visible=True, meanline_visible=True,
                                  fillcolor=BLEU_TRES_CLAIR, line_color=BLEU_CLAIR))
    fig_imag.add_trace(go.Violin(y=avec, name='Avec imagerie',
                                  box_visible=True, meanline_visible=True,
                                  fillcolor='#FFCCBC', line_color=ORANGE_ALERT))
    fig_imag.add_annotation(x=1, y=max(np.median(avec) + 30, 310), text="+40 min",
                             showarrow=True, arrowhead=2,
                             font=dict(size=14, color=ROUGE_CRIT))
    fig_imag.update_layout(**LAYOUT, height=320,
                            yaxis=dict(title="Temps passage (min)", gridcolor=GRIS_MEDICAL))
    st.plotly_chart(fig_imag, use_container_width=True, config={'displayModeBar': False})


# --- Row 2 ---
col3, col4 = st.columns(2)

# --- Graph 24 : Impact greves ---
with col3:
    st.markdown(f"""
    <div style="background:{BLANC};border-radius:10px;padding:16px;
                box-shadow:0 2px 8px rgba(0,0,0,0.06)">
        <div style="font-size:14px;font-weight:700;color:{BLEU_FONCE}">
            Impact Greves sur la Duree
        </div>
        <div style="font-size:12px;color:{GRIS_TEXTE}">
            +20.3% duree passage (p &lt; 0.001)  •  Effectif -13%
        </div>
    </div>
    """, unsafe_allow_html=True)

    if df is not None and 'indicateur_greve' in df.columns and 'temps_passage_total' in df.columns:
        normal_passage = df.loc[df['indicateur_greve'] == 0, 'temps_passage_total'].sample(
            min(2000, (df['indicateur_greve'] == 0).sum()), random_state=55
        ).values
        greve_df = df.loc[df['indicateur_greve'] == 1, 'temps_passage_total']
        greve_passage = greve_df.sample(min(500, len(greve_df)), random_state=55).values
    else:
        np.random.seed(55)
        normal_passage = np.random.normal(240, 65, 800).clip(60, 600)
        greve_passage = np.random.normal(289, 80, 200).clip(60, 700)

    fig_greve = go.Figure()
    fig_greve.add_trace(go.Box(y=normal_passage, name='Hors greve',
                                marker_color=BLEU_CLAIR, boxmean=True))
    fig_greve.add_trace(go.Box(y=greve_passage, name='Greve',
                                marker_color=ROUGE_CRIT, boxmean=True))

    pct_diff = (np.mean(greve_passage) / np.mean(normal_passage) - 1) * 100
    fig_greve.add_annotation(x=1, y=np.percentile(greve_passage, 75) + 30,
                              text=f"+{pct_diff:.1f}%", showarrow=False,
                              font=dict(size=16, color=ROUGE_CRIT))
    fig_greve.update_layout(**LAYOUT, height=320,
                             yaxis=dict(title="Temps passage (min)", gridcolor=GRIS_MEDICAL))
    st.plotly_chart(fig_greve, use_container_width=True, config={'displayModeBar': False})


# --- Decomposition STL (Graph 19 simplifie) ---
with col4:
    st.markdown(f"""
    <div style="background:{BLANC};border-radius:10px;padding:16px;
                box-shadow:0 2px 8px rgba(0,0,0,0.06)">
        <div style="font-size:14px;font-weight:700;color:{BLEU_FONCE}">
            Decomposition STL Saisonniere
        </div>
        <div style="font-size:12px;color:{GRIS_TEXTE}">
            Tendance haussiere + saisonnalite hebdomadaire
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Utiliser les vraies donnees quotidiennes pour STL
    if df_daily is not None and len(df_daily) > 60:
        recent = df_daily.dropna().tail(365)
        dates_stl = recent['date']
        observed = recent['admissions_jour'].values
        trend_stl = recent['ma_30j'].values
        seasonal_stl = observed - trend_stl
        seasonal_stl = np.where(np.isnan(seasonal_stl), 0, seasonal_stl)
        residual_stl = observed - trend_stl - seasonal_stl
        residual_stl = np.where(np.isnan(residual_stl), 0, residual_stl)
    else:
        n = 365
        dates_stl = pd.date_range('2025-01-01', periods=n, freq='D')
        trend_stl = np.linspace(320, 350, n)
        seasonal_stl = 15 * np.sin(2 * np.pi * np.arange(n) / 7)
        residual_stl = np.random.normal(0, 10, n)

    fig_stl = make_subplots(rows=3, cols=1, shared_xaxes=True,
                             subplot_titles=['Tendance', 'Saisonnalite', 'Residus'],
                             vertical_spacing=0.08)
    fig_stl.add_trace(go.Scatter(x=dates_stl, y=trend_stl,
                                  line=dict(color=BLEU_FONCE, width=2), showlegend=False),
                       row=1, col=1)
    fig_stl.add_trace(go.Scatter(x=dates_stl, y=seasonal_stl,
                                  line=dict(color=BLEU_CLAIR, width=1.5), showlegend=False),
                       row=2, col=1)
    fig_stl.add_trace(go.Scatter(x=dates_stl, y=residual_stl,
                                  line=dict(color=GRIS_TEXTE, width=0.8), showlegend=False),
                       row=3, col=1)
    fig_stl.update_layout(height=320, plot_bgcolor=BLANC, paper_bgcolor='rgba(0,0,0,0)',
                           margin=dict(t=30, b=20, l=40, r=10),
                           font=dict(size=10, color=GRIS_TEXTE))
    st.plotly_chart(fig_stl, use_container_width=True, config={'displayModeBar': False})


# ══════════════════════════════════════════════════════════════════
# SECTION 3 : SHAP — Top features
# ══════════════════════════════════════════════════════════════════
st.markdown(f"""
<div style="background:{BLANC};border-radius:10px;padding:20px;
            box-shadow:0 2px 8px rgba(0,0,0,0.06);margin-top:20px">
    <div style="font-size:16px;font-weight:700;color:{BLEU_FONCE}">
        Facteurs explicatifs — SHAP (XGBoost J+1)
    </div>
    <div style="font-size:13px;color:{GRIS_TEXTE}">
        Variables qui contribuent le plus au score de charge predit
    </div>
</div>
""", unsafe_allow_html=True)

# Tenter de charger les vraies valeurs SHAP depuis le modele
shap_loaded = False
try:
    from utils.model_engine import get_model
    model = get_model('charge_j1')
    if model is not None:
        import shap
        df_hourly = load_hourly_data()
        from utils.data_loader import get_features_for_prediction
        from utils.config import FEATURE_COLS

        features_df = get_features_for_prediction(df_hourly.tail(1000))
        features_df = features_df.fillna(features_df.mean())

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(features_df)

        mean_abs = np.abs(shap_values).mean(axis=0)
        top_idx = np.argsort(mean_abs)[::-1][:10]
        features_shap = [features_df.columns[i] for i in top_idx]
        shap_vals = [float(np.mean(shap_values[:, i])) for i in top_idx]
        shap_loaded = True
except Exception:
    pass

if not shap_loaded:
    # Valeurs SHAP simulees (basees sur l'analyse du notebook)
    features_shap = [
        'nb_patients_en_cours', 'charge_lag_1h', 'effectif_soignant_present',
        'heures_pleines', 'dispo_lits_aval', 'patient_count',
        'charge_mean_24h', 'temperature_max', 'score_IAO', 'indicateur_greve'
    ]
    shap_vals = [12.3, 9.1, -6.2, 4.8, -3.9, 3.5, 3.2, 2.1, 1.8, 1.5]

colors_shap = [ROUGE_CRIT if v > 0 else VERT_OK for v in shap_vals]

labels_display = [f"{f} ({v:+.1f})" for f, v in zip(features_shap, shap_vals)]

fig_shap = go.Figure(go.Bar(
    y=labels_display[::-1], x=[abs(v) for v in shap_vals[::-1]],
    orientation='h',
    marker_color=colors_shap[::-1],
    text=[f'{v:+.1f}' for v in shap_vals[::-1]],
    textposition='outside', textfont=dict(size=11),
))
fig_shap.update_layout(
    **LAYOUT, height=380,
    xaxis=dict(title="|SHAP value| (impact sur score)", gridcolor=GRIS_MEDICAL),
    yaxis=dict(title=""),
)

col_shap, col_legend = st.columns([3, 1])
with col_shap:
    st.plotly_chart(fig_shap, use_container_width=True, config={'displayModeBar': False})
with col_legend:
    shap_source = "Valeurs SHAP reelles (modele charge)" if shap_loaded else "Valeurs SHAP estimees (modele non charge)"
    st.markdown(f"""
    <div style="background:{BLANC};border-radius:8px;padding:16px;margin-top:20px">
        <div style="font-size:13px;font-weight:600;color:{BLEU_FONCE};margin-bottom:10px">
            Lecture SHAP
        </div>
        <div style="font-size:12px;color:{GRIS_TEXTE};line-height:1.8">
            <span style="color:{ROUGE_CRIT}">■ Rouge</span> : pousse le score ↑ (tension)<br>
            <span style="color:{VERT_OK}">■ Vert</span> : pousse le score ↓ (protege)<br><br>
            Le <b>nb de patients</b> et la <b>charge recente</b> sont les 2 facteurs
            les plus determinants.<br><br>
            L'<b>effectif soignant</b> est le principal facteur protecteur.<br><br>
            <i style="font-size:11px">{shap_source}</i>
        </div>
    </div>
    """, unsafe_allow_html=True)
