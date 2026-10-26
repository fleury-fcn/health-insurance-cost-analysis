PY ?= python
.PHONY: install install-dev run run-fast report test lint app api mlflow notebook docker clean

install:            ## Dépendances d'exécution + package
	$(PY) -m pip install -r requirements.txt && $(PY) -m pip install -e .

install-dev:        ## Dépendances de développement
	$(PY) -m pip install -r requirements-dev.txt && $(PY) -m pip install -e .

run:                ## Pipeline complet : données -> modèles -> rapport
	$(PY) -m insurance_costs.cli run

run-fast:           ## Pipeline rapide (recherche d'hyperparamètres réduite, sans MLflow)
	$(PY) -m insurance_costs.cli run --n-iter 3 --no-mlflow

report:             ## Régénère le rapport HTML + PDF à partir des artefacts
	$(PY) -m insurance_costs.cli report --pdf

test:
	$(PY) -m pytest

lint:
	$(PY) -m ruff check src app tests

app:                ## Application Streamlit
	$(PY) -m streamlit run app/streamlit_app.py

api:                ## API FastAPI (Swagger : http://127.0.0.1:8000/docs)
	$(PY) -m uvicorn app.api:app --reload

mlflow:             ## Interface MLflow (http://127.0.0.1:5000)
	$(PY) -m mlflow ui --backend-store-uri artifacts/mlruns

notebook:           ## Exécute le notebook de restitution
	$(PY) notebooks/build_notebook.py

docker:
	docker build -t insurance-costs . && docker run -p 8501:8501 insurance-costs

clean:              ## Supprime les sorties régénérables
	rm -rf artifacts/mlruns reports/generated/*.html artifacts/figures
