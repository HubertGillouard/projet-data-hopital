"""
Configuration globale — Couleurs AP-HP, seuils, constantes
"""

# ══════════════════════════════════════════════════════════════════
# PALETTE AP-HP
# ══════════════════════════════════════════════════════════════════
BLEU_FONCE      = '#003D7A'
BLEU_MOYEN      = '#1B5E9E'
BLEU_CLAIR      = '#5B8DB8'
BLEU_TRES_CLAIR = '#D6E6F2'
GRIS_MEDICAL    = '#E8ECF0'
GRIS_TEXTE      = '#4A5568'
BLANC           = '#FFFFFF'

# Indicateurs de severite
VERT_OK      = '#28A745'
JAUNE_VIGIL  = '#FFC107'
ORANGE_ALERT = '#FF8C42'
ROUGE_CRIT   = '#DC3545'

# ══════════════════════════════════════════════════════════════════
# SEUILS
# ══════════════════════════════════════════════════════════════════
SEUILS_CHARGE = [
    {'label': 'Normal',     'min': 0,  'max': 50, 'color': VERT_OK,     'icon': '🟢', 'bg': '#E8F5E9'},
    {'label': 'Vigilance',  'min': 50, 'max': 70, 'color': JAUNE_VIGIL, 'icon': '🟡', 'bg': '#FFF8E1'},
    {'label': 'Alerte',     'min': 70, 'max': 85, 'color': ORANGE_ALERT,'icon': '🟠', 'bg': '#FFF3E0'},
    {'label': 'Saturation', 'min': 85, 'max': 100,'color': ROUGE_CRIT,  'icon': '🔴', 'bg': '#FFEBEE'},
]

SEUILS_KPI = {
    'patients':  {'ok': 40, 'warn': 52, 'crit': 60, 'direction': 'high_bad'},
    'effectif':  {'ok': 15, 'warn': 10, 'crit': 8,  'direction': 'low_bad'},
    'lits':      {'ok': 10, 'warn': 5,  'crit': 3,  'direction': 'low_bad'},
    'flux':      {'ok': 10, 'warn': 12, 'crit': 18, 'direction': 'high_bad'},
}

# ══════════════════════════════════════════════════════════════════
# FEATURES XGBOOST (exactes du notebook)
# ══════════════════════════════════════════════════════════════════
FEATURE_COLS = [
    'heure_sin', 'heure_cos', 'jour_semaine_sin', 'jour_semaine_cos',
    'mois_sin', 'mois_cos',
    'temperature_max', 'niveau_pollution', 'indicateur_greve',
    'alerte_epidemique_encoded', 'evenement_externe',
    'score_IAO', 'patient_count', 'nb_patients_en_cours',
    'effectif_soignant_present', 'dispo_lits_aval',
    'charge_lag_1h', 'charge_lag_4h', 'charge_lag_6h',
    'charge_lag_12h', 'charge_lag_24h', 'charge_lag_48h',
    'charge_lag_72h', 'charge_lag_168h',
    'patient_count_lag_1h', 'patient_count_lag_4h',
    'patient_count_lag_24h', 'patient_count_lag_168h',
    'charge_mean_24h', 'charge_std_24h', 'charge_max_7d',
    'patient_count_mean_24h', 'charge_mean_12h', 'charge_std_12h',
    'weekend_epidemie', 'nuit_effectif', 'greve_jour_semaine',
]

# ══════════════════════════════════════════════════════════════════
# SCENARIOS PREDEFINIS
# ══════════════════════════════════════════════════════════════════
SCENARIOS = {
    '🟢 Normal': {},
    '🦠 Epidemie': {
        'alerte_epidemique_encoded': 1,
        'patient_count_mult': 1.30,
        'nb_patients_en_cours_mult': 1.25,
    },
    '✊ Greve': {
        'indicateur_greve': 1,
        'effectif_soignant_present_mult': 0.87,
    },
    '🌡️ Canicule': {
        'temperature_max': 38,
        'patient_count_mult': 1.15,
        'nb_patients_en_cours_mult': 1.10,
    },
    '💥 Afflux massif': {
        'patient_count_mult': 1.50,
        'nb_patients_en_cours_mult': 1.40,
        'evenement_externe': 1,
    },
}

# Coefficients pathologie (du notebook — 7 entrees intentionnelles)
COEF_PATHOLOGIE = {0: 1.0, 1: 1.7, 2: 1.8, 3: 1.3, 4: 1.4, 5: 0.8, 6: 1.5}

# Capacite et seuils (du notebook)
CAPACITE_URGENCES = 55
SEUIL_SATURATION = 66

# ══════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════
def get_charge_level(score):
    """Retourne le dict du niveau de charge correspondant."""
    for s in SEUILS_CHARGE:
        if s['min'] <= score < s['max']:
            return s
    return SEUILS_CHARGE[-1]

def get_kpi_status(value, kpi_name):
    """Retourne 'ok'|'warning'|'critical' selon les seuils du KPI."""
    cfg = SEUILS_KPI.get(kpi_name)
    if not cfg:
        return 'ok'
    if cfg['direction'] == 'high_bad':
        if value >= cfg['crit']:  return 'critical'
        if value >= cfg['warn']:  return 'warning'
        return 'ok'
    else:  # low_bad
        if value <= cfg['crit']:  return 'critical'
        if value <= cfg['warn']:  return 'warning'
        return 'ok'

def status_color(status):
    return {'ok': VERT_OK, 'warning': ORANGE_ALERT, 'critical': ROUGE_CRIT}.get(status, GRIS_TEXTE)

def status_icon(status):
    return {'ok': '✅', 'warning': '⚠️', 'critical': '🔴'}.get(status, '•')
