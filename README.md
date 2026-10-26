# Analyse et modélisation des coûts d'assurance maladie

[![CI](https://github.com/fleury-fcn/health-insurance-cost-analysis/actions/workflows/ci.yml/badge.svg)](https://github.com/fleury-fcn/health-insurance-cost-analysis/actions)

## 🇫🇷 Objectif

Quels facteurs expliquent les coûts d'assurance maladie et peut-on les prédire
avec une précision utile à la segmentation actuarielle ? Ce portfolio met en
place un pipeline reproductible de la qualité des données à une API et une
démo Streamlit.

## Données et éthique

Le fichier **Medical Cost Personal** (1 338 observations) est téléchargé depuis
le miroir public [stedy/Machine-Learning-with-R-datasets](https://github.com/stedy/Machine-Learning-with-R-datasets/blob/master/insurance.csv),
lui-même couramment distribué pour l'enseignement. La licence du miroir n'est
pas explicitement déclarée : vérifier les droits avant tout usage commercial.
Les charges sont des unités monétaires non précisées. Le jeu est éducatif,
non représentatif d'un portefeuille et ne doit pas servir à prendre une
décision individuelle.

## Reproduire le projet

```bash
python -m venv .venv && source .venv/bin/activate
make install
pip install -e .
make run
pytest -q
```

Le pipeline fixe `random_state=42`, nettoie les incohérences, produit les
statistiques (`artifacts/statistics.json`), le benchmark (`artifacts/metrics.json`),
le modèle (`models/best_model.joblib`) et le rapport
`reports/generated/report.html`. Les métriques sont également suivies dans
MLflow (`artifacts/mlflow.db`).

## Méthodologie

1. **Qualité** : types, manquants, doublons, contraintes âge/IMC/charge ; les
   valeurs extrêmes sont conservées car elles portent du risque.
2. **EDA/inférence** : distributions, asymétrie, corrélations, Welch,
   Mann-Whitney, ANOVA région et chi² région-tabagisme, avec taille d'effet.
3. **Variables** : one-hot, standardisation, `bmi × smoker` et `age × smoker`.
4. **Modèles** : baseline linéaire régularisée (Ridge/Lasso), GLM Gamma à lien
   log (cohérent avec une charge positive asymétrique), Random Forest et
   Gradient Boosting. Validation croisée 5-fold puis test 20 % ; RMSE, MAE, R².
5. **Limites** : pas de causalité, pas de validation externe/temps, pas de
   fréquence-sévérité, biais de sélection et définition de licence à vérifier.

## Démo/API

```bash
streamlit run app/streamlit_app.py
uvicorn app.api:app --reload
```

Swagger est disponible sur `/docs`. Pour Streamlit Community Cloud, choisir
`app/streamlit_app.py`, Python 3.11 et `requirements.txt`; lancer `make run`
avant de publier un modèle pré-calculé ou ajouter une étape de démarrage.

## 🇬🇧 English summary

This reproducible portfolio asks which policyholder characteristics explain
medical insurance charges and how accurately they can be predicted. It
includes data-quality checks, exploratory and inferential statistics, an
actuarial Gamma-log GLM, regularized linear models, tree ensembles, cross
validation, an HTML report, Streamlit demo and FastAPI endpoint. The dataset
is educational and must not be used for individual underwriting decisions.
