import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "best_model.joblib"
DATA_PATH = ROOT / "data" / "processed" / "insurance_clean.csv"

st.set_page_config(page_title="Health insurance costs", page_icon="📊", layout="wide")
st.title("📊 Health insurance cost explorer")
st.caption("Démo pédagogique — aucune décision tarifaire individuelle.")

if not MODEL_PATH.exists() or not DATA_PATH.exists():
    st.warning("Lancez `make run` pour générer les données et le modèle.")
    st.stop()

df = pd.read_csv(DATA_PATH)
model = joblib.load(MODEL_PATH)
tab1, tab2 = st.tabs(["Explorer les données", "Prédire un profil"])
with tab1:
    st.dataframe(df.head(100), use_container_width=True)
    st.bar_chart(df.groupby("smoker")["charges"].mean())
    st.metric("Charge médiane", f"{df.charges.median():,.0f}")
with tab2:
    with st.form("profile"):
        age = st.slider("Âge", 18, 70, 35)
        bmi = st.number_input("IMC", 10.0, 60.0, 27.5)
        children = st.number_input("Enfants", 0, 10, 0)
        sex = st.selectbox("Sexe", ["female", "male"])
        smoker = st.selectbox("Fumeur", ["no", "yes"])
        region = st.selectbox("Région", sorted(df.region.unique()))
        submitted = st.form_submit_button("Prédire")
    if submitted:
        profile = pd.DataFrame([{
            "age": age, "sex": sex, "bmi": bmi, "children": children,
            "smoker": smoker, "region": region,
        }])
        profile["bmi_smoker"] = profile["bmi"] * (profile["smoker"] == "yes").astype(int)
        profile["age_smoker"] = profile["age"] * (profile["smoker"] == "yes").astype(int)
        prediction = float(model.predict(profile)[0])
        st.success(f"Coût prédit : **{prediction:,.0f}**")
        shap_plot = ROOT / "artifacts" / "shap_summary.png"
        if shap_plot.exists():
            st.image(str(shap_plot), caption="Importance globale SHAP")
        else:
            st.info("Installez l'extra `requirements-shap.txt` puis relancez le pipeline "
                    "pour générer l'explication SHAP. La prédiction ne constitue pas un "
                    "avis médical ou contractuel.")
