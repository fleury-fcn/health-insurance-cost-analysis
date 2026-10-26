# Phase 8 — Pipeline automatisé

## Une commande

```bash
make run                 # = insurance-costs run  (≈ 2 min sur un portable)
make run-fast            # recherche réduite, sans MLflow (≈ 30 s), pour la CI
```

| Étape | Module | Sorties |
|-------|--------|---------|
| Téléchargement + vérification SHA-256 | `data.py` | `data/raw/insurance.csv` |
| Audit + nettoyage | `data.py` | `data/processed/insurance_clean.csv`, `artifacts/data_quality.json` |
| EDA | `eda.py` | `artifacts/eda.json`, `artifacts/figures/eda_*` |
| Inférence | `inference.py` | `artifacts/inference.json` |
| Ablation | `features.py` | `artifacts/feature_ablation.json` |
| Modèles | `modeling.py` | `artifacts/metrics.json`, `models/best_model.joblib`, `models/model_card.json` |
| Interprétation | `interpretation.py` | `artifacts/interpretation.json`, figures SHAP/PDP/équité |
| Suivi | `tracking.py` | `artifacts/mlruns/` (MLflow) |
| Rapport | `report.py` | `reports/generated/report.html` |

Chaque exécution écrit `artifacts/run_summary.json` : commit git, empreinte des données,
versions Python/OS, durée de chaque étape, métriques du meilleur modèle.

## CLI (Typer)

```
insurance-costs run [--n-iter 30] [--no-mlflow]   # tout le pipeline
insurance-costs quality                            # données + audit
insurance-costs analyze                            # EDA + tests
insurance-costs train [--n-iter 30]                # modèles + interprétation
insurance-costs report                             # rapport depuis les artefacts
insurance-costs summary                            # dernière exécution
```

## Reproductibilité

- Seed unique (`config.RANDOM_STATE = 42`) propagée au split, aux CV, aux recherches
  aléatoires, aux modèles et aux bootstraps. LightGBM tourne en mode `deterministic`.
- Dépendances épinglées (`requirements.txt`), y compris les dépendances transitives
  sensibles (numba/llvmlite, pyparsing).
- Empreinte SHA-256 des données brutes : le pipeline s'arrête si la source change.
- **Vérifié** : deux exécutions successives produisent les mêmes métriques, hyperparamètres
  et modèle. Seules diffèrent les durées d'exécution et, ponctuellement, le 16e chiffre
  significatif d'une moyenne (arrondi flottant des calculs parallèles, ~1e-16).

## Suivi des expériences (MLflow)

Un run parent par exécution (tags : commit, empreinte des données, meilleur modèle),
puis un run enfant par modèle candidat (hyperparamètres retenus, métriques CV et test,
durée). Les artefacts (métriques, model card, modèle, figures) sont attachés au parent.

```bash
make mlflow   # http://127.0.0.1:5000
```

Le store est local (`artifacts/mlruns`, non versionné). Pour une équipe, il suffit de
définir `MLFLOW_TRACKING_URI` vers un serveur partagé.
