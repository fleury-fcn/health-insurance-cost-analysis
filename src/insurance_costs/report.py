"""Generate a self-contained HTML report."""

import json
from pathlib import Path

from .config import ARTIFACT_DIR, REPORT_DIR


def generate_report(audit: dict, statistics: dict, metrics: dict) -> Path:
    best = metrics["best_model"]
    model_rows = "".join(
        f"<tr><td>{name}</td><td>{values['test_rmse']:,.0f}</td>"
        f"<td>{values['test_mae']:,.0f}</td><td>{values['test_r2']:.3f}</td></tr>"
        for name, values in metrics["models"].items()
    )
    html = f"""<!doctype html><html lang="fr"><head><meta charset="utf-8">
    <title>Analyse des coûts d'assurance maladie</title>
    <style>body{{font-family:Arial;max-width:1000px;margin:40px auto;color:#172033}}
    h1,h2{{color:#1d4ed8}} table{{border-collapse:collapse;width:100%}}
    td,th{{border:1px solid #ddd;padding:8px;text-align:left}} img{{max-width:100%}}
    .card{{background:#eff6ff;padding:16px;border-radius:8px}}</style></head><body>
    <h1>Analyse et modélisation des coûts d'assurance maladie</h1>
    <div class="card"><b>Résumé exécutif.</b> Le meilleur modèle est
    <b>{best}</b>. Les fumeurs ont une charge moyenne de
    {statistics['smoker_mean']:,.0f}, contre {statistics['non_smoker_mean']:,.0f}
    pour les non-fumeurs. La différence est statistiquement significative
    (Welch p={statistics['welch_t']['p_value']:.2e}).</div>
    <h2>1. Périmètre et qualité</h2><p>{audit['rows']} lignes, {audit['columns']} variables,
    {audit['duplicates']} doublon(s), {audit['missing_values']} valeur(s) manquante(s).
    Les valeurs physiquement impossibles sont exclues ; les valeurs extrêmes sont
    conservées car elles représentent un risque assuré réel.</p>
    <h2>2. Exploration</h2><p>Les charges sont fortement asymétriques
    (skewness={statistics['skewness']:.2f}), ce qui motive le GLM Gamma à lien log
    et l'évaluation par MAE/RMSE.</p><img src="../../artifacts/eda_distributions.png">
    <h2>3. Inférence</h2><p>Le test de Mann-Whitney confirme la différence de
    distributions entre fumeurs et non-fumeurs (p={statistics['mann_whitney']['p_value']:.2e}).
    Taille d'effet de Cohen d={statistics['cohens_d']:.2f}. L'ANOVA par région
    donne p={statistics['anova_region']['p_value']:.3f}; le chi² région-tabagisme
    donne p={statistics['chi2_region_smoker']['p_value']:.3f}.</p>
    <h2>4. Benchmark hors-échantillon</h2><table><tr><th>Modèle</th><th>RMSE</th>
    <th>MAE</th><th>R²</th></tr>{model_rows}</table>
    <h2>5. Méthode et limites</h2><p>Split 80/20 fixe, validation croisée K-fold
    (5 plis), encodage one-hot, standardisation des variables numériques et
    interactions âge/IMC avec le tabagisme. Jeu de données synthétique/éducatif,
    non représentatif d'un portefeuille réel, sans causalité ni usage médical.
    Une calibration temporelle, une validation externe et l'étude de la fréquence
    des sinistres seraient nécessaires en production.</p>
    <h2>6. Recommandations métier</h2><ul><li>Segmenter le risque en intégrant le
    tabagisme et ses interactions, sans interpréter la prédiction comme une décision
    individuelle.</li><li>Surveiller les erreurs par sous-groupe avant tout usage.</li>
    <li>Recalibrer périodiquement sur données de portefeuille réelles.</li></ul>
    </body></html>"""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    target = REPORT_DIR / "report.html"
    target.write_text(html, encoding="utf-8")
    return target
