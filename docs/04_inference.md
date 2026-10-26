# Phase 4 — Statistiques inférentielles

Script : [`src/insurance_costs/inference.py`](../src/insurance_costs/inference.py). Résultats
complets : `artifacts/inference.json`. Seuil α = 5 %, bootstrap de 10 000 tirages (seed 42).

## Démarche : vérifier avant de tester

| Hypothèse | Test de vérification | Résultat | Conséquence |
|-----------|---------------------|----------|-------------|
| Normalité par groupe | Shapiro-Wilk | rejetée partout (p ≈ 1e-28) | tests de rangs (Mann-Whitney, Kruskal-Wallis) + bootstrap |
| Égalité des variances | Levene (centré sur la médiane, robuste) | rejetée (p < 0,001) | Welch plutôt que Student ; Alexander-Govern plutôt qu'ANOVA |
| Effectifs attendus du χ² | min ≥ 5 | respectée | χ² valide |
| Confusion (ex. : plus de fumeurs chez les hommes) | — | — | test F partiel sur log(coûts), à âge, IMC, enfants et tabagisme égaux |
| Tests multiples (8) | — | — | correction de Holm (contrôle du FWER, moins conservatrice que Bonferroni) |

Le t de Student et l'ANOVA classique restent dans les sorties à titre de comparaison :
la région en est l'illustration, avec un écart « significatif » par ANOVA (p = 0,033) qui
disparaît avec les tests robustes (Kruskal-Wallis p = 0,20).

## Résultats

| Question | Test retenu | p | Taille d'effet | Lecture |
|----------|-------------|---|----------------|---------|
| Fumeurs vs non-fumeurs | Mann-Whitney, Welch, bootstrap | < 0,001 | g = 3,16 ; r rang-bisérial = 0,95 | +23 610 $ (IC 95 % [22 186 ; 25 039]) : effet massif |
| Hommes vs femmes (brut) | Mann-Whitney | 0,69 | g = 0,12 | pas d'écart robuste |
| Hommes vs femmes (ajusté) | F partiel, log-OLS | < 0,001 | η²p = 0,010 | −8,3 % pour un homme à profil égal (confusion par le tabagisme) |
| Région (brut) | Kruskal-Wallis / Alexander-Govern | 0,20 / 0,053 | η² = 0,007 | négligeable |
| Région (ajusté) | F partiel, log-OLS | < 0,001 | η²p = 0,019 | sud-est −13,9 %, sud-ouest −12,8 % vs nord-est |
| Enfants | Kruskal-Wallis | < 0,001 | ε² = 0,018 | effet faible |
| Tabagisme × région | χ² | 0,064 | V = 0,07 | indépendance non rejetée |
| Tabagisme × sexe | χ² | 0,006 | V = 0,07 | hommes un peu plus fumeurs |
| Âge – coûts | Pearson (IC Fisher z) | < 0,001 | r = 0,30 [0,25 ; 0,35] | modéré |
| IMC – coûts | Pearson | < 0,001 | r = 0,20 [0,15 ; 0,25] | faible globalement, fort chez les fumeurs |

## Messages clés

- **Significatif ≠ important** : avec n = 1 337, de petits effets deviennent significatifs.
  Les tailles d'effet hiérarchisent les facteurs : le tabagisme écrase tout.
- **Le paradoxe du sexe** : l'écart brut favorable aux femmes s'inverse à profil égal.
  C'est l'exemple type d'une variable de confusion. Dans l'UE, le sexe est de toute façon
  un critère tarifaire interdit (CJUE, *Test-Achats*, 2011). Son impact est évalué en phase 7.
- **Limite** : le test F ajusté suppose des résidus normaux et homoscédastiques sur l'échelle
  log, ce qui n'est qu'approximatif. Il sert à contrôler la confusion, pas à prédire.
