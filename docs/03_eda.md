# Phase 3 — Analyse exploratoire

Script : [`src/insurance_costs/eda.py`](../src/insurance_costs/eda.py). Les figures interactives
(Plotly, JSON) et statiques (PNG) sont dans `artifacts/figures/`. Les chiffres et
interprétations sont dans `artifacts/eda.json` et intégrés automatiquement au rapport.

![Vue d'ensemble](../artifacts/figures/eda_overview.png)

## Constats clés

1. **Asymétrie** : skewness 1,51 ; moyenne 13 279 $ contre médiane 9 386 $. Les 10 % d'assurés
   les plus coûteux portent 32 % de la charge. Après passage au log, la skewness est de −0,09.
   → cible log, GLM Gamma à lien log, suivi conjoint de la RMSE et de la MAE.
2. **Tabagisme** : ×3,8 sur le coût moyen (32 050 $ contre 8 441 $), corrélation r = 0,79.
3. **Âge** : effet régulier et quasi linéaire (≈ 267 $/an chez les non-fumeurs). C'est le
   premier facteur en rangs (Spearman ρ = 0,53).
4. **Interaction IMC × tabagisme** : franchir IMC 30 coûte +879 $ à un non-fumeur et
   +20 195 $ à un fumeur. L'effet de l'IMC n'est donc pas additif.
5. **Segmentation** : non-fumeurs 8 441 $, fumeurs avec IMC < 30 21 363 $, fumeurs avec
   IMC ≥ 30 41 558 $, avec un gradient d'âge dans chaque segment.
6. **Sexe et région** : écarts bruts modestes, à tester en phase 4 en contrôlant le tabagisme.
7. **Multicolinéarité** : aucune (VIF max ≈ 1,6).

## Choix de visualisation

- Une même couleur par statut tabagique dans toutes les figures (bleu pour non-fumeur,
  orange pour fumeur), toujours accompagnée d'une légende ou d'un libellé. La palette a été
  validée pour les principales formes de daltonisme.
- Corrélations sur une échelle divergente bleu ↔ rouge centrée sur un gris neutre en 0.
  Coûts en heatmap sur une échelle séquentielle à teinte unique.
- Pearson et Spearman côte à côte : les coûts étant asymétriques, la corrélation de
  rangs est plus robuste.
