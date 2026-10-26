from pathlib import Path

import joblib
from fastapi import FastAPI
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
MODEL = joblib.load(ROOT / "models" / "best_model.joblib")
app = FastAPI(title="Insurance Cost API", version="1.0.0")


class Profile(BaseModel):
    age: int = Field(ge=18, le=100)
    sex: str
    bmi: float = Field(gt=0, lt=100)
    children: int = Field(ge=0, le=20)
    smoker: str
    region: str


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict")
def predict(profile: Profile) -> dict:
    row = profile.model_dump()
    row["bmi_smoker"] = row["bmi"] * int(row["smoker"] == "yes")
    row["age_smoker"] = row["age"] * int(row["smoker"] == "yes")
    value = float(MODEL.predict([row])[0])
    return {"predicted_charges": round(value, 2), "currency": "dataset units"}
