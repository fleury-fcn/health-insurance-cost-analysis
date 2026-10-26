# Phase 1 — Cadrage

## Problématique

> **Quels facteurs expliquent les coûts médicaux annuels facturés à un assuré, et
> comment les prédire de façon fiable, interprétable et équitable ?**

Un assureur santé a besoin de ces réponses pour trois usages : la **tarification**
(prime pure = coût attendu), le **pilotage du risque** (identifier les segments qui
concentrent la charge) et la **prévention** (cibler des programmes, par exemple le
sevrage tabagique).

## Questions d'analyse

| # | Question | Phase |
|---|----------|-------|
| Q1 | Quelle est la forme de la distribution des coûts (asymétrie, queue) ? | 3 |
| Q2 | Quelles variables sont associées aux coûts, seules et en interaction ? | 3, 4 |
| Q3 | Les écarts observés (fumeurs, régions, sexe) sont-ils significatifs et d'ampleur importante ? | 4 |
| Q4 | Quel modèle prédit le mieux hors échantillon, et à quel prix d'interprétabilité ? | 6 |
| Q5 | Comment chaque facteur contribue-t-il à une prédiction individuelle ? | 7 |
| Q6 | Le modèle se trompe-t-il davantage sur certains sous-groupes ? | 7 |

## Critères de succès

- **Statistiques** : RMSE, MAE et R² hors échantillon, comparés à une baseline naïve
  (prédire la moyenne) ; ratio de calibration (somme prédite / somme observée) proche de 1,
  exigence actuarielle pour qu'un portefeuille soit tarifé à l'équilibre.
- **Interprétabilité** : contributions SHAP par profil, effets partiels lisibles.
- **Équité** : écarts d'erreur par âge, sexe et région documentés.
- **Reproductibilité** : une seule commande régénère données, modèles, métriques et rapport
  à l'identique (seed `42`, dépendances figées, empreinte des données vérifiée).

## Données

*Medical Cost Personal Dataset* (1 338 assurés, 7 variables), voir
[`data/README.md`](../data/README.md) pour la source, la licence et le dictionnaire.

**Pourquoi ce jeu de données ?** C'est la référence publique de la tarification santé
individuelle : petit, sans données personnelles, et il contient un vrai problème actuariel
(cible positive et asymétrique, interaction forte tabagisme × obésité). **Limite assumée** :
les données sont simulées à partir de statistiques démographiques américaines. Les
conclusions illustrent une méthode, elles ne décrivent pas un portefeuille réel.

## Choix structurants (justifiés)

| Choix | Justification |
|-------|---------------|
| Pipeline Python packagé (`src/`) plutôt que notebooks comme source de vérité | Les notebooks gardent un état caché ; un package testé et appelé par une CLI se rejoue à l'identique. Le notebook sert de vitrine. |
| Split train/test 80/20 fixe + validation croisée 5 plis sur le train | Le test n'est jamais vu pendant le choix du modèle ni le réglage des hyperparamètres, donc la performance annoncée reste non biaisée. |
| GLM Gamma à lien log comme référence actuarielle | Standard de marché pour les montants de sinistres : cible strictement positive, variance qui croît avec la moyenne, effets multiplicatifs lisibles comme des coefficients tarifaires. |
| Sélection du modèle sur la RMSE de validation croisée | La RMSE pénalise les gros écarts, qui coûtent cher en assurance. La MAE et la calibration sont suivies en parallèle. |

## Arborescence

```
data/raw         données brutes téléchargées (non versionnées, empreinte vérifiée)
data/processed   données nettoyées (versionnées pour la démo)
notebooks/       notebook de restitution exécuté à partir du package
src/insurance_costs/  code du pipeline (données → EDA → stats → modèles → rapport)
app/             application Streamlit et API FastAPI
reports/         rapport HTML généré (publié sur GitHub Pages)
artifacts/       métriques JSON et figures
models/          meilleur modèle sérialisé (joblib)
tests/           tests pytest
docs/            notes méthodologiques
```
