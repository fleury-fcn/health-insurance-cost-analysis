# Phase 5 — Feature engineering

Script : [`src/insurance_costs/features.py`](../src/insurance_costs/features.py). Ablation :
`artifacts/feature_ablation.json` (RMSE en validation croisée 5 plis, **sur le train uniquement**).

## Principes

- **Tout est dans le pipeline scikit-learn** (`InsuranceFeatures` → encodage → modèle).
  Les statistiques apprises (moyenne d'âge pour centrer `age_sq`, moyennes/écarts-types
  du scaler) sont calculées sur chaque pli d'entraînement : aucune fuite. L'app et l'API
  envoient les 6 variables brutes, le modèle sérialisé applique exactement les mêmes
  transformations.
- **Split unique 80/20 stratifié sur le tabagisme** (`features.split`) : la variable la plus
  influente n'est présente qu'à 20 %, on garantit la même proportion dans le test.

## Variables construites

| Variable | Définition | Justification (EDA/inférence) |
|----------|-----------|-------------------------------|
| `obese` | IMC ≥ 30 (seuil OMS) | rupture de coûts observée à 30 |
| `smoker_obese` | fumeur × obèse | segment à 41 558 $ ; variable la plus utile après les brutes |
| `bmi_smoker` | IMC × fumeur | pente d'IMC propre aux fumeurs |
| `age_smoker` | âge × fumeur | indispensable sur l'échelle log (voir ci-dessous) |
| `age_sq` | (âge − moyenne)² | légère courbure de l'effet âge ; centrage pour limiter la colinéarité avec `age` |

## Encodage et standardisation

| Famille de modèles | Catégorielles | Numériques | Pourquoi |
|--------------------|---------------|-----------|----------|
| Linéaires, GLM | one-hot (`drop="if_binary"`) | `StandardScaler` | la pénalisation L1/L2 dépend de l'échelle ; les coefficients deviennent comparables |
| Arbres (RF, XGBoost, LightGBM) | ordinal | inchangées | les arbres sont invariants aux transformations monotones ; le one-hot les fragmenterait inutilement |

Les catégories sont figées (`CATEGORY_LEVELS`). Une modalité inconnue en production est
ignorée (one-hot) ou codée −1 (ordinal), sans faire planter le service.

## Transformation log de la cible

`SmearedLogRegressor` ajuste le modèle sur log(coûts) puis retransforme avec le **facteur de
Duan** (moyenne des exp(résidus)). Sans lui, exp(ŷ) estime la *médiane* conditionnelle et
sous-estime la moyenne. Or une prime pure doit être une moyenne.

## Ablation (RMSE CV, Ridge)

| Configuration | RMSE CV | R² CV |
|---------------|--------:|------:|
| Variables brutes | 6 297 | 0,723 |
| **Avec interactions** | **4 607** | **0,849** |
| Sans `smoker_obese` | 4 953 | 0,826 |
| Sans `age_sq` | 4 657 | 0,846 |
| Sans `bmi_smoker` | 4 667 | 0,845 |
| Sans `obese` | 4 591 | 0,850 |
| Cible log, variables brutes | 9 745 | 0,341 |
| Cible log, interactions | 6 116 | 0,739 |
| Cible log, interactions sans `age_smoker` | 10 169 | 0,287 |

**Lectures** :
1. Les interactions réduisent l'erreur de 27 %, et `smoker_obese` y contribue le plus.
2. `obese` seule est redondante une fois `smoker_obese` présent (−16 $, dans le bruit de
   validation). Elle est conservée car elle rend le GLM lisible (effet de l'obésité chez les
   non-fumeurs, puis supplément pour les fumeurs).
3. **Leçon de méthode : la cible log n'est pas une solution miracle.** Les données montrent
   des bandes de coûts *parallèles* (même pente d'âge en dollars pour fumeurs et
   non-fumeurs), donc une structure **additive**. Un modèle sur l'échelle log est
   multiplicatif : sans `age_smoker`, il impose aux fumeurs une pente d'âge environ 4 fois
   trop forte. L'interaction divise l'erreur par 1,7, mais l'échelle naturelle reste
   meilleure ici. Le même raisonnement vaut pour le GLM Gamma à lien log (RMSE test
   7 355 → 4 422 avec `age_smoker`). Un GLM Gamma à lien identité ne converge pas dans
   scikit-learn ; il n'est pas retenu.
