"""FastAPI service: `uvicorn app.api:app` then open http://127.0.0.1:8000/docs (Swagger)."""

from __future__ import annotations

import json
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import joblib  # noqa: E402
import pandas as pd  # noqa: E402
from fastapi import FastAPI, HTTPException, Query  # noqa: E402
from pydantic import BaseModel, ConfigDict, Field  # noqa: E402

from insurance_costs.config import BEST_MODEL, FEATURES, MODEL_DIR, PROCESSED_DATA  # noqa: E402
from insurance_costs.interpretation import explain_profile  # noqa: E402

STATE: dict = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    STATE["model"] = joblib.load(BEST_MODEL)
    STATE["card"] = json.loads((MODEL_DIR / "model_card.json").read_text(encoding="utf-8"))
    STATE["background"] = pd.read_csv(PROCESSED_DATA).drop(columns="charges")
    yield
    STATE.clear()


app = FastAPI(
    title="Health Insurance Cost API",
    version="2.0.0",
    description="Prédiction du coût médical annuel d'un assuré, avec explication SHAP "
                "optionnelle. Démonstration pédagogique : ne pas utiliser pour une décision "
                "tarifaire ou médicale individuelle.",
    lifespan=lifespan,
)


class Profile(BaseModel):
    model_config = ConfigDict(json_schema_extra={"examples": [{
        "age": 45, "sex": "male", "bmi": 32.5, "children": 2, "smoker": "yes",
        "region": "southeast"}]})

    age: int = Field(ge=18, le=100, description="Âge de l'assuré (ans)")
    sex: Literal["female", "male"]
    bmi: float = Field(gt=10, lt=80, description="Indice de masse corporelle (kg/m²)")
    children: int = Field(ge=0, le=20, description="Nombre d'enfants couverts")
    smoker: Literal["no", "yes"]
    region: Literal["northeast", "northwest", "southeast", "southwest"]


class Prediction(BaseModel):
    predicted_charges: float = Field(description="Coût annuel attendu (USD)")
    model: str
    typical_error: float = Field(description="Erreur absolue moyenne sur le jeu de test (USD)")
    base_value: float | None = Field(None, description="Coût moyen de référence (SHAP)")
    contributions: dict[str, float] | None = Field(
        None, description="Contribution SHAP de chaque variable (USD) ; base + somme = "
                          "prédiction")


@app.get("/health", tags=["service"])
def health() -> dict:
    return {"status": "ok", "model_loaded": "model" in STATE}


@app.get("/model", tags=["service"])
def model_card() -> dict:
    """Model card : modèle, hyperparamètres, métriques de test, usage prévu."""
    return STATE["card"]


@app.post("/predict", response_model=Prediction, tags=["prediction"])
def predict(profile: Profile,
            explain: bool = Query(False, description="Ajouter l'explication SHAP")
            ) -> Prediction:
    frame = pd.DataFrame([profile.model_dump()])[FEATURES]
    try:
        value = float(STATE["model"].predict(frame)[0])
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=500, detail=f"Prédiction impossible : {exc}") from exc
    out = Prediction(predicted_charges=round(value, 2), model=STATE["card"]["model"],
                     typical_error=round(STATE["card"]["test_metrics"]["mae"], 2))
    if explain:
        exp = explain_profile(STATE["model"], frame, STATE["background"])
        out.base_value = round(exp["base_value"], 2)
        out.contributions = {k: round(v, 2) for k, v in exp["contributions"].items()}
    return out
