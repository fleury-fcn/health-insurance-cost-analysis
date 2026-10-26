# Phase 10 — Application, API et déploiement

## Lancer en local

```bash
make app   # Streamlit → http://localhost:8501
make api   # FastAPI   → http://127.0.0.1:8000/docs (Swagger)
```

| Onglet Streamlit | Contenu |
|------------------|---------|
| 🔮 Prédire un profil | formulaire → coût estimé, comparaison avec des assurés similaires, **cascade SHAP** expliquant le montant |
| 🔎 Explorer les données | filtres (tabagisme, région, sexe, âge), KPI, nuage de points, distribution, export CSV |
| 🏁 Comparer les modèles | tableau CV/test, test d'égalité statistique, stabilité CV, calibration, résidus |
| 🧠 Interprétation & équité | importance SHAP/permutation, dépendance partielle, audit par sous-groupe |
| ℹ️ À propos | méthode, limites, liens |

L'app et l'API **réutilisent le package** (`explain_profile`, figures du pipeline) : aucune
logique dupliquée, et donc aucun écart possible avec le rapport.

### API

```bash
curl -X POST "http://127.0.0.1:8000/predict?explain=true" \
  -H "Content-Type: application/json" \
  -d '{"age": 45, "sex": "male", "bmi": 32.5, "children": 2, "smoker": "yes", "region": "southeast"}'
```

```json
{"predicted_charges": 41422.02, "model": "Lasso", "typical_error": 1991.95,
 "base_value": 13643.73,
 "contributions": {"age": 930.74, "sex": -245.27, "bmi": 4489.15, "children": 557.98,
                   "smoker": 22070.42, "region": -24.73}}
```

Validation stricte des entrées par Pydantic (bornes, modalités `Literal`) : une valeur hors
domaine renvoie une **422** au lieu d'une prédiction absurde. Endpoints `/health` et
`/model` (model card).

![Swagger](images/api_swagger.png)

## Déployer sur Streamlit Community Cloud (gratuit)

Le dépôt contient déjà tout le nécessaire : le modèle léger (`models/best_model.joblib`,
≈ 2 Ko), les données nettoyées, les artefacts, `app/requirements.txt` (dépendances minimales,
lues en priorité car placées à côté du point d'entrée) et `.streamlit/config.toml` (thème).

1. Aller sur <https://share.streamlit.io> et se connecter avec GitHub (compte `fleury-fcn`).
2. **Create app → Deploy a public app from GitHub**.
3. Renseigner :
   - *Repository* : `fleury-fcn/health-insurance-cost-analysis`
   - *Branch* : `main`
   - *Main file path* : `app/streamlit_app.py`
   - *App URL* : par exemple `health-insurance-costs` (→ `https://health-insurance-costs.streamlit.app`)
4. **Advanced settings → Python version : 3.12**. Aucun secret n'est nécessaire.
5. **Deploy**. Le premier build prend environ 3 à 5 minutes (installation de SHAP/numba).
6. Copier l'URL obtenue dans le badge et la section « Démo » du README.

Chaque push sur `main` redéploie l'app automatiquement.

**Dépannage** : en cas d'erreur `libgomp.so.1` (LightGBM), ajouter à la racine un fichier
`packages.txt` contenant `libgomp1`. En cas de manque de mémoire, rien n'est entraîné au
démarrage : seul le modèle de 2 Ko est chargé.

## Alternative : Hugging Face Spaces (Docker)

1. <https://huggingface.co/new-space> → *SDK : Docker* → *Blank*.
2. Pousser le dépôt vers le Space :
   ```bash
   git remote add space https://huggingface.co/spaces/<user>/health-insurance-costs
   git push space main
   ```
3. Le `Dockerfile` à la racine construit l'image et sert Streamlit. Pour HF, ajouter en tête
   du README du Space :
   ```yaml
   ---
   title: Health Insurance Costs
   sdk: docker
   app_port: 8501
   ---
   ```
