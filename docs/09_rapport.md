# Phase 9 — Rapport automatique

- **Gabarit** : [`templates/report.html.j2`](../src/insurance_costs/templates/report.html.j2)
  (Jinja2), rendu par [`report.py`](../src/insurance_costs/report.py).
- **Source unique** : le rapport ne lit que `artifacts/*.json`, les figures et la model card.
  Il ne peut donc pas contredire la dernière exécution, et il se régénère sans réentraîner
  (`insurance-costs report`).
- **Autonome** : un seul fichier HTML (≈ 0,4 Mo). Les graphiques Plotly sont interactifs
  (plotly.js chargé depuis le CDN officiel). Mise en page responsive et feuille de style
  d'impression.
- **PDF** : `insurance-costs report --pdf` (ou `make report`) l'imprime via Chrome/Chromium
  headless s'il est installé → `reports/generated/report.pdf`.
- **Textes générés** : interprétations, conclusions de tests et alertes d'équité sont
  produites par le code à partir des chiffres (avec leur sens et leur significativité),
  jamais écrites en dur.

## Plan

1. Résumé exécutif (4 KPI + 5 messages)
2. Données et qualité (audit, nettoyage journalisé)
3. Analyse exploratoire (7 figures commentées)
4. Statistiques inférentielles (tableau des tests, IC, conclusions)
5. Feature engineering (ablation)
6. Modélisation (protocole, tableau avec IC, stabilité CV, résidus, calibration/Lorenz,
   relativités GLM)
7. Interprétation et équité (SHAP, permutation, PDP, sous-groupes, unisexe)
8. Limites
9. Recommandations métier
10. Annexe de reproductibilité (commit, empreinte, durées, model card)

## Publication GitHub Pages

Le workflow [`pages.yml`](../.github/workflows/pages.yml) publie le rapport versionné
(`reports/generated/report.html` → `index.html`, plus le PDF) à chaque push qui le modifie.

**Pourquoi publier le rapport versionné plutôt que le régénérer en CI ?** Pour que les
chiffres en ligne soient exactement ceux du README et du commit. La CI vérifie séparément
que le pipeline tourne de bout en bout.

Activation (une fois) : *Settings → Pages → Source : GitHub Actions*, ou
`gh api -X POST repos/<owner>/<repo>/pages -f build_type=workflow`.
