# Phase 7 — Interprétation et équité

Script : [`src/insurance_costs/interpretation.py`](../src/insurance_costs/interpretation.py).
Résultats : `artifacts/interpretation.json`.

![Interprétation](../artifacts/figures/interpretation_overview.png)

## Méthodes et justification

| Outil | Ce qu'il mesure | Choix technique |
|-------|-----------------|-----------------|
| **Importance par permutation** | hausse de RMSE quand une variable est brouillée | calculée sur le **test** (pas sur le train, où elle mesurerait le sur-apprentissage), 30 répétitions |
| **SHAP** | contribution additive de chaque variable à une prédiction | explainer **exact** (6 variables → 64 coalitions), agnostique au modèle, sur les **6 variables d'origine**. Les interactions construites sont réattribuées au tabagisme, à l'IMC et à l'âge, ce qui reste lisible pour un métier. Additivité vérifiée (écart < 1e-9 $) |
| **Dépendance partielle conditionnelle** | effet moyen d'une variable, séparément pour fumeurs et non-fumeurs | la PDP globale masquerait l'interaction |
| **Erreurs par sous-groupe** | MAE, RMSE, biais (prédit − observé), ratio de calibration avec IC 95 % bootstrap | l'IC évite de sur-interpréter des groupes de 50 à 85 assurés |
| **Neutralisation du sexe** | coût en précision d'une tarification unisexe | réentraînement de la même configuration avec un sexe constant |

## Résultats

1. **Hiérarchie** : tabagisme (|SHAP| moyen 6 471 $) > âge (3 157 $) > IMC (1 745 $) >
   enfants (744 $) > région (297 $) > sexe (253 $). Lasso, XGBoost et GLM Gamma
   **s'accordent sur le top 3**, donc la conclusion ne dépend pas du modèle.
2. **Seuil d'IMC** : chez un fumeur, passer de 28 à 32 d'IMC ajoute ≈ 17 000 $ ; chez un
   non-fumeur, l'effet est nul.
3. **Âge** : effet légèrement convexe, parallèle pour fumeurs et non-fumeurs.
4. **Équité** :
   - l'erreur absolue varie peu entre sexes (×1,18) et régions (×1,04), davantage entre âges
     (×1,76, maximum chez les 40-49 ans, où se trouvent plusieurs chocs de coûts) ;
   - **point de vigilance** : les **18-29 ans sont surestimés de 17 %** (ratio 1,17, IC 95 %
     [1,11 ; 1,27]), au-delà du niveau global du test (1,05). Le modèle leur attribue une part
     du risque de choc des tranches plus âgées. Correctifs possibles : âge en spline,
     recalibrage par tranche ou terme âge × non-fumeur ;
   - les fumeurs sont mieux calibrés que la moyenne (1,00) : leurs coûts sont plus
     déterministes.
5. **Tarification unisexe** (obligatoire dans l'UE depuis l'arrêt *Test-Achats*, 2011) :
   retirer le sexe fait varier la RMSE de −16 $, dans le bruit. **La conformité ne coûte rien.**

## Limites

- SHAP et la PDP supposent implicitement des variables indépendantes. C'est raisonnable ici
  (VIF ≈ 1), mais pas en général.
- L'équité est évaluée sur 268 assurés de test : c'est une alerte, pas une preuve.
  Il faudrait refaire l'évaluation sur un portefeuille réel et suivre ces indicateurs en
  production.
- Les explications portent sur des **associations** apprises, pas sur des causes : le
  modèle ne dit pas qu'arrêter de fumer réduirait le coût d'un individu de 23 000 $.
