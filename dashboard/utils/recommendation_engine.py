"""
Recommendation Engine — Moteur de recommandations automatiques
Base sur regles metier (EDA) + rejeu du modele XGBoost pour estimer les gains
"""
import pandas as pd
import numpy as np

from utils.config import SEUILS_KPI, FEATURE_COLS


# ══════════════════════════════════════════════════════════════════
# REGLES DE RECOMMANDATION
# ══════════════════════════════════════════════════════════════════
RULES = [
    {
        'id': 'effectif_16_20',
        'condition': lambda f: f.get('effectif_soignant_present', 20) < 18 and f.get('heures_pleines', 0) == 1,
        'priorite': 'P1',
        'niveau': 'critique',
        'titre': 'Renforcer effectif 16h-20h',
        'cause_template': 'Effectif {effectif:.0f} soignants en heures pleines (recommande : 20-24)',
        'action': '+4 soignants sur creneau apres-midi/soir',
        'feature': 'effectif_soignant_present',
        'target_value': lambda f: min(f.get('effectif_soignant_present', 18) + 4, 26),
        'estimated_gain': -12,
    },
    {
        'id': 'lits_aval_critique',
        'condition': lambda f: f.get('dispo_lits_aval', 12) < 8,
        'priorite': 'P2',
        'niveau': lambda f: 'critique' if f.get('dispo_lits_aval', 12) <= 3 else 'warning',
        'titre': 'Anticiper sorties aval',
        'cause_template': '{lits:.0f} lits disponibles (seuil critique <= 3)',
        'action': 'Accelerer sorties programmees, liberer 5-8 lits',
        'feature': 'dispo_lits_aval',
        'target_value': lambda f: min(f.get('dispo_lits_aval', 8) + 8, 23),
        'estimated_gain': -6,
    },
    {
        'id': 'imagerie_goulot',
        'condition': lambda f: f.get('besoin_imagerie_encoded', 0) > 0.5,
        'priorite': 'P3',
        'niveau': 'warning',
        'titre': 'Creneaux imagerie dedies urgences',
        'cause_template': '69% des patients necessitent imagerie (+40 min par patient)',
        'action': 'Reserver radio 14h-20h pour urgences',
        'feature': None,
        'target_value': None,
        'estimated_gain': -5,
    },
    {
        'id': 'protocole_epidemie',
        'condition': lambda f: f.get('alerte_epidemique_encoded', 0) > 0.5,
        'priorite': 'P2',
        'niveau': 'warning',
        'titre': 'Activer protocole epidemie',
        'cause_template': 'Alerte epidemique active (+0.3 patient/h selon SARIMAX)',
        'action': 'Renforcer tri, ouvrir zone tampon, anticiper stocks',
        'feature': None,
        'target_value': None,
        'estimated_gain': -3,
    },
    {
        'id': 'effectif_nuit',
        'condition': lambda f: f.get('nuit', 0) == 1 and f.get('effectif_soignant_present', 15) < 12,
        'priorite': 'P2',
        'niveau': 'warning',
        'titre': 'Renforcer effectif de nuit',
        'cause_template': 'Effectif nuit {effectif:.0f} (seuil critique < 10)',
        'action': '+2 soignants sur creneau 00h-08h',
        'feature': 'effectif_soignant_present',
        'target_value': lambda f: min(f.get('effectif_soignant_present', 10) + 2, 14),
        'estimated_gain': -8,
    },
    {
        'id': 'greve_active',
        'condition': lambda f: f.get('indicateur_greve', 0) == 1,
        'priorite': 'P1',
        'niveau': 'critique',
        'titre': 'Plan de continuite greve',
        'cause_template': 'Greve active (effectif -13%, duree passage +20.3%)',
        'action': 'Activer protocole greve, rappeler reservistes, prioriser urgences vraies',
        'feature': None,
        'target_value': None,
        'estimated_gain': -10,
    },
]


# ══════════════════════════════════════════════════════════════════
# GENERATION DES RECOMMANDATIONS
# ══════════════════════════════════════════════════════════════════
def generate_recommendations(current_features, model=None, baseline_score=None, max_recos=3):
    """
    Genere les recommandations automatiques basees sur les regles et le contexte.
    """
    # Convertir en dict si necessaire
    if isinstance(current_features, pd.Series):
        feat = current_features.to_dict()
    elif isinstance(current_features, pd.DataFrame):
        feat = current_features.iloc[-1].to_dict()
    else:
        feat = dict(current_features)

    recos = []

    for rule in RULES:
        try:
            if not rule['condition'](feat):
                continue
        except Exception:
            continue

        niveau = rule['niveau']
        if callable(niveau):
            niveau = niveau(feat)

        cause = rule['cause_template'].format(
            effectif=feat.get('effectif_soignant_present', 0),
            lits=feat.get('dispo_lits_aval', 0),
            patients=feat.get('nb_patients_en_cours', 0),
        )

        target = rule['target_value']
        if callable(target):
            target = target(feat)

        gain = rule['estimated_gain']
        if model is not None and rule['feature'] is not None and target is not None:
            gain = estimate_gain_with_model(
                feat, rule['feature'], target, model, baseline_score
            )

        recos.append({
            'id': rule['id'],
            'priorite': rule['priorite'],
            'niveau': niveau,
            'titre': rule['titre'],
            'cause': cause,
            'action': rule['action'],
            'gain_score': gain,
            'feature': rule['feature'],
            'target_value': target,
        })

    recos.sort(key=lambda r: r['gain_score'])
    return recos[:max_recos]


def estimate_gain_with_model(features_dict, feature_name, new_value, model, baseline_score):
    """Estime le gain en points de score en rejouant le modele."""
    try:
        feat_modif = dict(features_dict)
        feat_modif[feature_name] = new_value

        available_cols = [c for c in FEATURE_COLS if c in feat_modif]
        X = pd.DataFrame([{c: feat_modif.get(c, 0) for c in available_cols}])

        new_score = model.predict(X)[0]
        new_score = np.clip(new_score, 0, 100)

        return float(new_score - baseline_score)
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════
# APPLICATION DES RECOMMANDATIONS
# ══════════════════════════════════════════════════════════════════
def apply_recommendation(current_features, reco):
    """Applique une recommandation aux features."""
    if isinstance(current_features, pd.DataFrame):
        feat = current_features.copy()
    else:
        feat = dict(current_features)

    if reco['feature'] and reco['target_value'] is not None:
        if isinstance(feat, pd.DataFrame):
            feat[reco['feature']] = reco['target_value']
        else:
            feat[reco['feature']] = reco['target_value']

    return feat


def apply_all_recommendations(current_features, recos):
    """Applique toutes les recommandations simultanement."""
    feat = dict(current_features) if not isinstance(current_features, dict) else current_features.copy()

    for reco in recos:
        if reco['feature'] and reco['target_value'] is not None:
            feat[reco['feature']] = reco['target_value']

    return feat


def calculate_total_gain(recos):
    """Calcule le gain total (avec facteur de reduction non-lineaire)."""
    total = sum(r['gain_score'] for r in recos if r['gain_score'] is not None)
    return total * 0.85


def count_red_hours(scores, threshold=85):
    """Compte le nombre d'heures ou le score depasse le seuil."""
    return sum(1 for s in scores if s >= threshold)


def count_alert_hours(scores, threshold=70):
    """Compte le nombre d'heures ou le score depasse le seuil d'alerte."""
    return sum(1 for s in scores if s >= threshold)


def format_recommendation_html(reco):
    """Formate une recommandation en HTML pour l'affichage Streamlit."""
    from utils.config import ROUGE_CRIT, ORANGE_ALERT, VERT_OK, BLEU_FONCE, GRIS_TEXTE

    border_color = ROUGE_CRIT if reco['niveau'] == 'critique' else ORANGE_ALERT
    icon = '🔴' if reco['niveau'] == 'critique' else '🟡'
    gain_color = VERT_OK if reco['gain_score'] < 0 else ROUGE_CRIT

    return f"""
    <div style="border-left:5px solid {border_color};background:white;
                padding:16px 20px;border-radius:0 8px 8px 0;margin-bottom:12px;
                box-shadow:0 1px 4px rgba(0,0,0,0.06)">
        <div style="font-weight:700;color:{BLEU_FONCE};font-size:15px">
            {icon} {reco['priorite']} — {reco['titre']}
        </div>
        <div style="color:{GRIS_TEXTE};font-size:13px;margin-top:6px;line-height:1.6">
            <b>Cause :</b> {reco['cause']}<br>
            <b>Action :</b> {reco['action']}<br>
            <span style="color:{gain_color};font-weight:700">
                Gain estime : {reco['gain_score']:+.1f} pts de score
            </span>
        </div>
    </div>
    """
