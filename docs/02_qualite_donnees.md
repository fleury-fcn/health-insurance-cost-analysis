# Phase 2 — Qualité des données

Script : [`src/insurance_costs/data.py`](../src/insurance_costs/data.py). Sortie complète :
[`artifacts/data_quality.json`](../artifacts/data_quality.json), régénérée à chaque exécution.

## Audit

| Contrôle | Résultat | Méthode |
|----------|----------|---------|
| Schéma | 7 colonnes attendues, aucune en trop | comparaison au dictionnaire |
| Types | numériques : `age`, `bmi`, `children`, `charges` ; texte : `sex`, `smoker`, `region` | conversion numérique stricte |
| Doublons exacts | **1** (homme de 19 ans, IMC 30,59, nord-ouest, 1 639,56 $) | `duplicated()` |
| Valeurs manquantes | 0 | par colonne |
| Modalités inattendues | 0 | normalisation casse/espaces puis comparaison |
| Valeurs impossibles | 0 | bornes : âge 18–100, IMC 10–80, enfants 0–20, coûts ≥ 0 |
| Atypiques IMC | 9 (IQR) / 2 (z-score robuste > 3,5) | valeurs 47 à 53, plausibles (obésité morbide) |
| Atypiques coûts | 139 au-delà de Q3 + 1,5·IQR (34 489 $) | IQR et z-score robuste (MAD) |

**Profil des coûts extrêmes** : 97,8 % sont des fumeurs (20,5 % dans l'ensemble) et
96,4 % sont obèses. Ces points ne sont pas des erreurs : ils forment une sous-population
à haut risque, celle que l'assurance doit précisément tarifer.

## Décisions de nettoyage

1. **Normaliser** les libellés (minuscules, sans espaces) et forcer les types numériques,
   ce qui rend le pipeline robuste à une nouvelle extraction mal formatée.
2. **Supprimer le doublon** : faute d'identifiant, une coïncidence parfaite au centime près
   sur 7 variables est presque certainement une double saisie. 1 338 → **1 337 lignes**.
3. **Pas d'imputation** : aucune valeur manquante. Le code supprime d'éventuelles lignes
   incomplètes plutôt que d'imputer la cible ou ses déterminants.
4. **Exclure les valeurs impossibles** (aucune ici), règle conservée pour les données futures.
5. **Conserver les valeurs extrêmes** : les retirer biaiserait la prime pure vers le bas.
   Le choix de la cible log, du GLM Gamma et de métriques comme la MAE les gère
   statistiquement, sans les effacer.

Chaque étape est journalisée (`cleaning_log`) avec le nombre de lignes retirées et sa
justification. Les tests `tests/test_data.py` vérifient ces règles sur des cas pathologiques.
