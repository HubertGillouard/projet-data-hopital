"""
PDF Export -Generation de rapports PDF pour le dashboard
Utilise fpdf2 (pip install fpdf2)
"""
from fpdf import FPDF
from datetime import datetime
import os


def _sanitize(text):
    """Remplace les caracteres non Latin-1 (emojis, tirets longs, etc.)."""
    replacements = {
        '\u2014': '-', '\u2013': '-', '\u2018': "'", '\u2019': "'",
        '\u201c': '"', '\u201d': '"', '\u2026': '...', '\u2192': '->',
        '\u2191': '^', '\u2193': 'v',
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    # Supprimer tout caractere non encodable en latin-1
    return text.encode('latin-1', errors='replace').decode('latin-1')


class RapportPDF(FPDF):
    """PDF personnalise avec en-tete et pied de page AP-HP."""

    def header(self):
        # Logo si disponible
        logo_path = os.path.join(os.path.dirname(__file__), '..', 'assets', 'logoPSL-630x214V6.png')
        if os.path.exists(logo_path):
            self.image(logo_path, 10, 6, 40)
        # Titre
        self.set_font('Helvetica', 'B', 14)
        self.set_text_color(0, 61, 122)  # BLEU_FONCE
        self.cell(0, 10, 'Rapport de Simulation -Urgences PSL', align='C', new_x='LMARGIN', new_y='NEXT')
        self.set_font('Helvetica', '', 9)
        self.set_text_color(100, 100, 100)
        self.cell(0, 5, f'Genere le {datetime.now().strftime("%d/%m/%Y a %H:%M")}', align='C', new_x='LMARGIN', new_y='NEXT')
        self.ln(5)
        # Ligne de separation
        self.set_draw_color(0, 61, 122)
        self.set_line_width(0.5)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f'Urgences Pitie-Salpetriere (AP-HP) -MVP Dashboard -Page {self.page_no()}/{{nb}}',
                  align='C')

    def section_title(self, title):
        self.set_font('Helvetica', 'B', 12)
        self.set_text_color(0, 61, 122)
        self.cell(0, 10, title, new_x='LMARGIN', new_y='NEXT')
        self.set_draw_color(0, 61, 122)
        self.set_line_width(0.3)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(3)

    def kpi_box(self, label, value, status_color=(0, 0, 0)):
        self.set_font('Helvetica', '', 10)
        self.set_text_color(80, 80, 80)
        self.cell(45, 7, label, new_x='RIGHT')
        self.set_font('Helvetica', 'B', 11)
        self.set_text_color(*status_color)
        self.cell(30, 7, str(value), new_x='LMARGIN', new_y='NEXT')

    def reco_block(self, reco):
        is_crit = reco.get('niveau') == 'critique'
        border_color = (220, 53, 69) if is_crit else (255, 140, 66)

        self.set_draw_color(*border_color)
        self.set_line_width(0.8)
        x_start = self.get_x()
        y_start = self.get_y()
        self.line(x_start, y_start, x_start, y_start + 18)

        self.set_x(x_start + 3)
        self.set_font('Helvetica', 'B', 10)
        self.set_text_color(0, 61, 122)
        icon = '[CRIT]' if is_crit else '[ALRT]'
        self.cell(0, 5, _sanitize(f"{icon} {reco['priorite']} - {reco['titre']}"), new_x='LMARGIN', new_y='NEXT')

        self.set_x(x_start + 3)
        self.set_font('Helvetica', '', 9)
        self.set_text_color(80, 80, 80)
        self.cell(0, 4, _sanitize(f"Cause : {reco['cause']}"), new_x='LMARGIN', new_y='NEXT')

        self.set_x(x_start + 3)
        self.cell(0, 4, _sanitize(f"Action : {reco['action']}"), new_x='LMARGIN', new_y='NEXT')

        self.set_x(x_start + 3)
        gain_color = (40, 167, 69) if reco['gain_score'] < 0 else (220, 53, 69)
        self.set_text_color(*gain_color)
        self.set_font('Helvetica', 'B', 9)
        self.cell(0, 4, f"Gain estime : {reco['gain_score']:+.1f} pts", new_x='LMARGIN', new_y='NEXT')
        self.ln(3)


def generate_simulation_pdf(
    scenario, horizon, effectif, lits,
    baseline_mean, scenario_mean, delta,
    heures_rouges_base, heures_rouges_scen,
    recos, total_gain, score_final,
):
    """Genere un rapport PDF de simulation et retourne les bytes."""
    pdf = RapportPDF()
    pdf.alias_nb_pages()
    pdf.add_page()

    # ── Section 1 : Parametres ──
    pdf.section_title('1. Parametres de la simulation')

    pdf.set_font('Helvetica', '', 10)
    pdf.set_text_color(60, 60, 60)

    params = [
        ('Scenario', _sanitize(scenario)),
        ('Horizon', horizon),
        ('Effectif soignant', str(effectif)),
        ('Lits aval disponibles', str(lits)),
    ]
    for label, val in params:
        pdf.set_font('Helvetica', '', 10)
        pdf.cell(55, 6, f'{label} :', new_x='RIGHT')
        pdf.set_font('Helvetica', 'B', 10)
        pdf.cell(0, 6, val, new_x='LMARGIN', new_y='NEXT')

    pdf.ln(5)

    # ── Section 2 : Resultats ──
    pdf.section_title('2. Resultats Avant / Apres')

    # Avant
    pdf.set_font('Helvetica', 'B', 11)
    pdf.set_text_color(0, 61, 122)
    pdf.cell(90, 8, 'AVANT (baseline)', align='C', new_x='RIGHT')
    pdf.cell(90, 8, 'APRES (scenario)', align='C', new_x='LMARGIN', new_y='NEXT')

    # Scores
    pdf.set_font('Helvetica', 'B', 24)
    color_before = _score_color(baseline_mean)
    color_after = _score_color(scenario_mean)

    pdf.set_text_color(*color_before)
    pdf.cell(90, 14, f'{baseline_mean:.0f}', align='C', new_x='RIGHT')
    pdf.set_text_color(*color_after)
    pdf.cell(90, 14, f'{scenario_mean:.0f}', align='C', new_x='LMARGIN', new_y='NEXT')

    # Details
    pdf.set_font('Helvetica', '', 10)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(90, 6, f'{heures_rouges_base}h rouges', align='C', new_x='RIGHT')
    pdf.cell(90, 6, f'{heures_rouges_scen}h rouges', align='C', new_x='LMARGIN', new_y='NEXT')

    # Delta
    pdf.ln(3)
    pdf.set_font('Helvetica', 'B', 12)
    delta_color = (220, 53, 69) if delta > 0 else (40, 167, 69)
    pdf.set_text_color(*delta_color)
    sign = '+' if delta > 0 else ''
    pdf.cell(0, 8, f'Delta : {sign}{delta:.1f} pts', align='C', new_x='LMARGIN', new_y='NEXT')

    pdf.ln(5)

    # ── Section 3 : Recommandations ──
    pdf.section_title('3. Recommandations automatiques')

    if not recos:
        pdf.set_font('Helvetica', 'I', 10)
        pdf.set_text_color(100, 100, 100)
        pdf.cell(0, 8, 'Aucune recommandation declenchee pour ce scenario.',
                 new_x='LMARGIN', new_y='NEXT')
    else:
        for reco in recos:
            pdf.reco_block(reco)

    # Resultat global
    pdf.ln(3)
    pdf.set_draw_color(*_score_color(score_final))
    pdf.set_line_width(0.5)
    y = pdf.get_y()
    pdf.rect(10, y, 190, 12)

    pdf.set_font('Helvetica', 'B', 11)
    pdf.set_text_color(*_score_color(score_final))
    level_label = _level_label(score_final)
    pdf.cell(0, 12,
             f'Avec {len(recos)} actions : {scenario_mean:.0f} -> {score_final:.0f} ({level_label}) | '
             f'Gain total : {total_gain:+.1f} pts',
             align='C', new_x='LMARGIN', new_y='NEXT')

    pdf.ln(8)

    # ── Section 4 : Notes ──
    pdf.section_title('4. Notes methodologiques')

    pdf.set_font('Helvetica', '', 9)
    pdf.set_text_color(80, 80, 80)

    notes = [
        "Modele principal : XGBoost optimise Optuna (30 trials, TimeSeriesSplit 3 folds)",
        "Post-traitement : +4 pts pour predictions dans [70,80) (amelioration detection pics)",
        "Score de charge = Affluence(40%) + Gravite(30%) + Pathologie(20%) + Ressources(10%)",
        "Gains estimes bases sur les coefficients SHAP et regles metier (EDA)",
        "Donnees synthetiques -570 282 passages, Nov 2020 -Oct 2025",
        "Conformite : aucune donnee identifiante, RGPD compatible",
    ]
    for note in notes:
        pdf.cell(5, 5, '-', new_x='RIGHT')
        pdf.cell(0, 5, note, new_x='LMARGIN', new_y='NEXT')

    pdf.ln(5)
    pdf.set_font('Helvetica', 'I', 8)
    pdf.set_text_color(150, 150, 150)
    pdf.cell(0, 5, 'Document genere automatiquement -MVP Dashboard Urgences PSL (AP-HP)',
             align='C', new_x='LMARGIN', new_y='NEXT')

    return pdf.output()


def _score_color(score):
    """Retourne un tuple RGB selon le score."""
    if score >= 85:
        return (220, 53, 69)    # ROUGE_CRIT
    elif score >= 70:
        return (255, 140, 66)   # ORANGE_ALERT
    elif score >= 50:
        return (255, 193, 7)    # JAUNE_VIGIL
    else:
        return (40, 167, 69)    # VERT_OK


def _level_label(score):
    """Retourne le label texte du niveau."""
    if score >= 85:
        return 'Saturation'
    elif score >= 70:
        return 'Alerte'
    elif score >= 50:
        return 'Vigilance'
    else:
        return 'Normal'
