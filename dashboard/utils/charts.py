"""
Fonctions graphiques Plotly reutilisables — Charte AP-HP
"""
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

from utils.config import (
    BLEU_FONCE, BLEU_MOYEN, BLEU_CLAIR, BLEU_TRES_CLAIR,
    GRIS_MEDICAL, GRIS_TEXTE, BLANC,
    VERT_OK, JAUNE_VIGIL, ORANGE_ALERT, ROUGE_CRIT,
    get_charge_level,
)

LAYOUT_BASE = dict(
    plot_bgcolor=BLANC,
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(family="Arial, sans-serif", color=GRIS_TEXTE),
    margin=dict(t=40, b=40, l=50, r=20),
)


# ══════════════════════════════════════════════════════════════════
# N1 — JAUGE SCORE DE CHARGE
# ══════════════════════════════════════════════════════════════════
def gauge_charge(score, delta=None, title="Score de Charge"):
    level = get_charge_level(score)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        title={'text': title, 'font': {'size': 16, 'color': BLEU_FONCE}},
        number={'font': {'size': 52, 'color': BLEU_FONCE}, 'suffix': '/100'},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': GRIS_TEXTE,
                     'dtick': 25},
            'bar': {'color': level['color'], 'thickness': 0.3},
            'bgcolor': BLANC,
            'steps': [
                {'range': [0, 50],   'color': '#E8F5E9'},
                {'range': [50, 70],  'color': '#FFF8E1'},
                {'range': [70, 85],  'color': '#FFF3E0'},
                {'range': [85, 100], 'color': '#FFEBEE'},
            ],
            'threshold': {
                'line': {'color': ROUGE_CRIT, 'width': 3},
                'thickness': 0.8, 'value': 85,
            },
        },
    ))
    fig.update_layout(height=220, margin=dict(t=50, b=10, l=30, r=30),
                      paper_bgcolor='rgba(0,0,0,0)')
    return fig


# ══════════════════════════════════════════════════════════════════
# N2 — COURBE CHARGE 24h + PREVISION
# ══════════════════════════════════════════════════════════════════
def charge_curve_with_forecast(df_past, df_forecast):
    """
    df_past    : dernieres 24h avec colonnes [datetime, charge_soin_totale]
    df_forecast: prochaines 24h avec colonnes [datetime, charge_pred]
    """
    fig = go.Figure()

    # Reel
    fig.add_trace(go.Scatter(
        x=df_past['datetime'], y=df_past['charge_soin_totale'],
        mode='lines', name='Observe',
        line=dict(color=BLEU_FONCE, width=2.5),
    ))

    # Prevision
    fig.add_trace(go.Scatter(
        x=df_forecast['datetime'], y=df_forecast['charge_pred'],
        mode='lines', name='Prevision J+1',
        line=dict(color=BLEU_CLAIR, width=2, dash='dash'),
    ))

    # Intervalle confiance (+-MAE ~4.5 pts)
    mae = 4.5
    upper = df_forecast['charge_pred'] + mae
    lower = (df_forecast['charge_pred'] - mae).clip(0)
    fig.add_trace(go.Scatter(
        x=pd.concat([df_forecast['datetime'], df_forecast['datetime'][::-1]]),
        y=pd.concat([upper, lower[::-1]]),
        fill='toself', fillcolor='rgba(91,141,184,0.12)',
        line=dict(color='rgba(0,0,0,0)'), showlegend=False,
    ))

    # Seuils
    fig.add_hline(y=70, line_dash="dot", line_color=JAUNE_VIGIL, line_width=1,
                  annotation_text="Vigilance (70)", annotation_position="right",
                  annotation_font_size=10, annotation_font_color=JAUNE_VIGIL)
    fig.add_hline(y=85, line_dash="dot", line_color=ROUGE_CRIT, line_width=1,
                  annotation_text="Saturation (85)", annotation_position="right",
                  annotation_font_size=10, annotation_font_color=ROUGE_CRIT)

    fig.update_layout(
        **LAYOUT_BASE, height=340,
        yaxis=dict(range=[0, 105], title="Score", gridcolor=GRIS_MEDICAL),
        xaxis=dict(title="", gridcolor=GRIS_MEDICAL),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0),
    )
    return fig


# ══════════════════════════════════════════════════════════════════
# N3 — CRENEAUX CRITIQUES (barres colorees par seuil)
# ══════════════════════════════════════════════════════════════════
def creneaux_critiques(hours, scores):
    """Barres horaires colorees selon le score prevu."""
    colors = []
    for s in scores:
        level = get_charge_level(s)
        colors.append(level['color'])

    fig = go.Figure(go.Bar(
        x=hours, y=scores, marker_color=colors,
        text=[f'{s:.0f}' for s in scores],
        textposition='outside', textfont=dict(size=10),
    ))
    fig.add_hline(y=70, line_dash="dot", line_color=JAUNE_VIGIL, line_width=1)
    fig.add_hline(y=85, line_dash="dot", line_color=ROUGE_CRIT, line_width=1)
    fig.update_layout(
        **LAYOUT_BASE, height=280,
        yaxis=dict(range=[0, 105], title="Score prevu"),
        xaxis=dict(title="Heure", dtick=1),
        showlegend=False,
    )
    return fig


# ══════════════════════════════════════════════════════════════════
# N4 — OVERLAY BASELINE VS SCENARIO (Page 2)
# ══════════════════════════════════════════════════════════════════
def comparison_overlay(hours, baseline, scenario):
    fig = go.Figure()

    # Baseline (gris pointille)
    fig.add_trace(go.Scatter(
        x=hours, y=baseline,
        mode='lines', name='Baseline (normal)',
        line=dict(color=GRIS_TEXTE, width=2, dash='dash'),
    ))

    # Scenario (trait plein + marqueurs colores)
    colors = [ROUGE_CRIT if s >= 85 else ORANGE_ALERT if s >= 70
              else JAUNE_VIGIL if s >= 50 else VERT_OK for s in scenario]
    fig.add_trace(go.Scatter(
        x=hours, y=scenario,
        mode='lines+markers', name='Scenario simule',
        line=dict(color=ROUGE_CRIT, width=2.5),
        marker=dict(color=colors, size=7, line=dict(width=1, color=BLANC)),
    ))

    # Zone delta
    fig.add_trace(go.Scatter(
        x=list(hours) + list(hours)[::-1],
        y=list(scenario) + list(baseline)[::-1],
        fill='toself', fillcolor='rgba(220,53,69,0.08)',
        line=dict(color='rgba(0,0,0,0)'), showlegend=False,
    ))

    fig.add_hline(y=85, line_dash="dot", line_color=ROUGE_CRIT, line_width=1)
    fig.add_hline(y=70, line_dash="dot", line_color=ORANGE_ALERT, line_width=1)

    fig.update_layout(
        **LAYOUT_BASE, height=380,
        yaxis=dict(range=[0, 105], title="Score de charge", gridcolor=GRIS_MEDICAL),
        xaxis=dict(title="Heure", gridcolor=GRIS_MEDICAL),
        legend=dict(orientation='h', yanchor='bottom', y=1.02),
    )
    return fig


# ══════════════════════════════════════════════════════════════════
# N5 — DOUBLE JAUGE AVANT / APRES (Page 2)
# ══════════════════════════════════════════════════════════════════
def double_gauge(before, after):
    fig = make_subplots(rows=1, cols=2, specs=[[{'type': 'indicator'}, {'type': 'indicator'}]],
                        subplot_titles=['AVANT (baseline)', 'APRES (scenario)'])
    for i, (val, col) in enumerate([(before, 1), (after, 2)]):
        level = get_charge_level(val)
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=val,
            number={'font': {'size': 40, 'color': BLEU_FONCE}, 'suffix': ''},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': level['color'], 'thickness': 0.3},
                'steps': [
                    {'range': [0, 50], 'color': '#E8F5E9'},
                    {'range': [50, 70], 'color': '#FFF8E1'},
                    {'range': [70, 85], 'color': '#FFF3E0'},
                    {'range': [85, 100], 'color': '#FFEBEE'},
                ],
            },
        ), row=1, col=col)
    fig.update_layout(height=200, margin=dict(t=40, b=10, l=20, r=20),
                      paper_bgcolor='rgba(0,0,0,0)')
    return fig


# ══════════════════════════════════════════════════════════════════
# G17 — HEATMAP JOUR x HEURE (EDA Graph 17)
# ══════════════════════════════════════════════════════════════════
def heatmap_jour_heure(df_hourly, current_hour=None, current_day=None):
    """
    df_hourly doit avoir colonnes : heure (0-23), jour_semaine (0-6), patient_count
    """
    pivot = df_hourly.groupby(['jour_semaine', 'heure'])['patient_count'].mean().reset_index()
    pivot_table = pivot.pivot(index='jour_semaine', columns='heure', values='patient_count')

    jours = ['Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi', 'Dimanche']

    fig = go.Figure(go.Heatmap(
        z=pivot_table.values,
        x=list(range(24)),
        y=jours[:len(pivot_table)],
        colorscale=[[0, BLANC], [0.3, BLEU_TRES_CLAIR], [0.6, BLEU_CLAIR],
                     [0.85, BLEU_MOYEN], [1, BLEU_FONCE]],
        colorbar=dict(title='Pat./h', thickness=15),
        hovertemplate='%{y} %{x}h : %{z:.1f} pat/h<extra></extra>',
    ))

    # Marqueur "vous etes ici"
    if current_hour is not None and current_day is not None:
        fig.add_annotation(
            x=current_hour, y=jours[current_day],
            text="📍", showarrow=False, font=dict(size=20),
        )

    fig.update_layout(
        **LAYOUT_BASE, height=300,
        xaxis=dict(title="Heure", dtick=1, side='bottom'),
        yaxis=dict(title="", autorange='reversed'),
        title=dict(text="Flux moyen par creneau", font=dict(size=14, color=BLEU_FONCE)),
    )
    return fig


# ══════════════════════════════════════════════════════════════════
# G16 — SERIE TEMPORELLE ADMISSIONS (EDA Graph 16)
# ══════════════════════════════════════════════════════════════════
def serie_temporelle_admissions(df_daily):
    """df_daily : colonnes [date, admissions_jour, ma_7j, ma_30j]"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_daily['date'], y=df_daily['admissions_jour'],
        mode='lines', name='Admissions/jour',
        line=dict(color=BLEU_TRES_CLAIR, width=0.8), opacity=0.5,
    ))
    fig.add_trace(go.Scatter(
        x=df_daily['date'], y=df_daily['ma_7j'],
        mode='lines', name='Moyenne mobile 7j',
        line=dict(color=BLEU_CLAIR, width=2),
    ))
    fig.add_trace(go.Scatter(
        x=df_daily['date'], y=df_daily['ma_30j'],
        mode='lines', name='Moyenne mobile 30j',
        line=dict(color=BLEU_FONCE, width=2.5),
    ))
    fig.update_layout(
        **LAYOUT_BASE, height=350,
        title=dict(text="Admissions quotidiennes — Tendance +45% sur 5 ans",
                   font=dict(size=14, color=BLEU_FONCE)),
        yaxis=dict(title="Admissions/jour", gridcolor=GRIS_MEDICAL),
        legend=dict(orientation='h', yanchor='bottom', y=1.02),
    )
    return fig


# ══════════════════════════════════════════════════════════════════
# G6 — DESEQUILIBRE EFFECTIF / DEMANDE (EDA Graph 6)
# ══════════════════════════════════════════════════════════════════
def desequilibre_effectif(df_hourly):
    """Barres groupees : effectif moyen vs patient_count par periode."""
    periodes = {'Nuit (0-8h)': (0, 8), 'Matin (8-12h)': (8, 12),
                'Apres-midi (12-18h)': (12, 18), 'Soir (18-0h)': (18, 24)}
    data_eff, data_flux = [], []
    labels = []
    for label, (h_start, h_end) in periodes.items():
        mask = (df_hourly['heure'] >= h_start) & (df_hourly['heure'] < h_end)
        data_eff.append(df_hourly.loc[mask, 'effectif_soignant_present'].mean())
        data_flux.append(df_hourly.loc[mask, 'patient_count'].mean())
        labels.append(label)

    fig = go.Figure()
    fig.add_trace(go.Bar(name='Effectif soignant', x=labels, y=data_eff,
                         marker_color=BLEU_FONCE))
    fig.add_trace(go.Bar(name='Flux patients/h', x=labels, y=data_flux,
                         marker_color=ORANGE_ALERT))
    fig.update_layout(
        **LAYOUT_BASE, height=320, barmode='group',
        title=dict(text="Effectif vs Demande par periode",
                   font=dict(size=14, color=BLEU_FONCE)),
        legend=dict(orientation='h', yanchor='bottom', y=1.02),
    )
    return fig


# ══════════════════════════════════════════════════════════════════
# N6 — DECOMPOSITION SCORE (4 composantes)
# ══════════════════════════════════════════════════════════════════
def decomposition_score(score_affluence, score_gravite, score_patho, score_ressources):
    """Barres horizontales des 4 composantes du score."""
    composantes = ['Affluence (40%)', 'Gravite (30%)', 'Pathologie (20%)', 'Ressources (10%)']
    valeurs = [score_affluence, score_gravite, score_patho, score_ressources]
    colors = [BLEU_FONCE, BLEU_MOYEN, BLEU_CLAIR, GRIS_TEXTE]

    fig = go.Figure(go.Bar(
        y=composantes, x=valeurs, orientation='h',
        marker_color=colors,
        text=[f'{v:.0f}/100' for v in valeurs],
        textposition='inside', textfont=dict(color=BLANC, size=13),
    ))
    fig.update_layout(
        **LAYOUT_BASE, height=200,
        xaxis=dict(range=[0, 105], title="Score composante"),
        yaxis=dict(autorange='reversed'),
        title=dict(text="Decomposition du score actuel",
                   font=dict(size=14, color=BLEU_FONCE)),
    )
    return fig
