"""Streamlit demo: explore the data, compare models, predict and explain a profile."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))  # works on Streamlit Cloud without `pip install -e .`

import joblib  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from insurance_costs.config import (  # noqa: E402
    ARTIFACT_DIR,
    BEST_MODEL,
    CATEGORY_LEVELS,
    FIGURE_DIR,
    MODEL_DIR,
    PROCESSED_DATA,
)
from insurance_costs.interpretation import FEATURE_LABELS, explain_profile  # noqa: E402
from insurance_costs.viz import (  # noqa: E402
    BLUE,
    INK_2,
    ORANGE,
    SMOKER_COLORS,
    SMOKER_LABELS,
    load_plotly,
)

st.set_page_config(page_title="Coûts d'assurance maladie", page_icon="🩺", layout="wide")

REGION_LABELS = {"northeast": "Nord-Est", "northwest": "Nord-Ouest", "southeast": "Sud-Est",
                 "southwest": "Sud-Ouest"}
SEX_LABELS = {"female": "Femme", "male": "Homme"}


def money(x: float) -> str:
    return f"{x:,.0f} $".replace(",", "\u202f")


def signed(x: float) -> str:
    return f"{x:+,.0f} $".replace(",", "\u202f")


def md(text: str) -> str:
    """Escape dollar signs, which Streamlit markdown would render as LaTeX."""
    return text.replace("$", "\\$")


@st.cache_data
def load_data() -> pd.DataFrame:
    return pd.read_csv(PROCESSED_DATA)


@st.cache_data
def load_json(name: str) -> dict:
    return json.loads((ARTIFACT_DIR / name).read_text(encoding="utf-8"))


@st.cache_resource
def load_model():
    return joblib.load(BEST_MODEL)


@st.cache_resource
def figure(name: str) -> go.Figure:
    return load_plotly(name)


def show_figure(name: str) -> None:
    if (FIGURE_DIR / f"{name}.json").exists():
        st.plotly_chart(figure(name), use_container_width=True)


if not BEST_MODEL.exists() or not PROCESSED_DATA.exists():
    st.error("Artefacts absents : lancez `make run` pour générer données et modèle.")
    st.stop()

df = load_data()
model = load_model()
metrics = load_json("metrics.json")
interp = load_json("interpretation.json")
card = json.loads((MODEL_DIR / "model_card.json").read_text(encoding="utf-8"))
best = metrics["best_model"]
best_test = metrics["models"][best]["test"]

st.title("🩺 Coûts d'assurance maladie")
st.caption(md(f"Analyse et modélisation de {len(df)} assurés · modèle en production : "
              f"**{best}** (R² test {best_test['r2']:.3f}, erreur moyenne "
              f"{money(best_test['mae'])}) · démonstration pédagogique, aucune décision "
              "tarifaire ou médicale."))

tab_predict, tab_explore, tab_models, tab_explain, tab_about = st.tabs(
    ["🔮 Prédire un profil", "🔎 Explorer les données", "🏁 Comparer les modèles",
     "🧠 Interprétation & équité", "ℹ️ À propos"])

# ----------------------------------------------------------------------------- predict
with tab_predict:
    left, right = st.columns([1, 2], gap="large")
    with left:
        st.subheader("Profil de l'assuré")
        with st.form("profile"):
            age = st.slider("Âge", 18, 64, 40)
            bmi = st.slider("IMC (kg/m²)", 15.0, 55.0, 27.0, step=0.1, format="%.1f",
                            help="Indice de masse corporelle ; obésité à partir de 30.")
            children = st.number_input("Enfants à charge", 0, 5, 1)
            sex = st.radio("Sexe", CATEGORY_LEVELS["sex"], format_func=SEX_LABELS.get,
                           horizontal=True)
            smoker = st.radio("Fumeur", CATEGORY_LEVELS["smoker"],
                              format_func=lambda v: "Oui" if v == "yes" else "Non",
                              horizontal=True)
            region = st.selectbox("Région (États-Unis)", CATEGORY_LEVELS["region"],
                                  format_func=REGION_LABELS.get)
            st.form_submit_button("Estimer le coût", type="primary", use_container_width=True)
    profile = pd.DataFrame([{"age": age, "sex": sex, "bmi": bmi, "children": children,
                             "smoker": smoker, "region": region}])
    with right:
        with st.spinner("Calcul de l'explication SHAP…"):
            exp = explain_profile(model, profile, df.drop(columns="charges"))
        pred = exp["prediction"]
        similar = df[(df.smoker == smoker) & ((df.bmi >= 30) == (bmi >= 30))
                     & df.age.between(age - 5, age + 5)]
        c1, c2, c3 = st.columns(3)
        c1.metric("Coût annuel estimé", money(pred))
        c2.metric("Coût moyen du portefeuille", money(exp["base_value"]),
                  delta=money(pred - exp["base_value"]), delta_color="inverse")
        if len(similar) >= 5:
            c3.metric(f"Assurés similaires observés (n = {len(similar)})",
                      money(similar.charges.mean()),
                      help="Même statut tabagique, même côté du seuil d'obésité, ±5 ans.")
        st.caption(md(f"Erreur absolue moyenne du modèle sur des assurés jamais vus : "
                      f"± {money(best_test['mae'])}. Certains assurés subissent des chocs de "
                      "coût que les variables disponibles ne prévoient pas."))

        contrib = exp["contributions"]
        order = sorted(contrib, key=lambda k: abs(contrib[k]), reverse=True)
        values = {"age": f"{age} ans", "sex": SEX_LABELS[sex], "bmi": f"{bmi:.1f}",
                  "children": str(children), "smoker": "oui" if smoker == "yes" else "non",
                  "region": REGION_LABELS[region]}
        fig = go.Figure(go.Waterfall(
            orientation="h", measure=["absolute"] + ["relative"] * len(order) + ["total"],
            y=["Coût moyen"] + [f"{FEATURE_LABELS[k]} = {values[k]}" for k in order]
              + ["Prédiction"],
            x=[exp["base_value"]] + [contrib[k] for k in order] + [0],
            increasing=dict(marker=dict(color=ORANGE)),
            decreasing=dict(marker=dict(color=BLUE)),
            totals=dict(marker=dict(color=INK_2)),
            connector=dict(line=dict(color="#c3c2b7", width=1)),
            text=[money(exp["base_value"])] + [signed(contrib[k]) for k in order]
                 + [money(pred)],
            textposition="outside", hoverinfo="skip"))
        fig.update_layout(title="Pourquoi ce montant ? Contributions SHAP par variable",
                          yaxis=dict(autorange="reversed"), height=430, showlegend=False,
                          xaxis_title="$ par an", margin=dict(l=10, r=40, t=60, b=40),
                          xaxis=dict(range=[0, max(exp["base_value"], pred,
                                                   exp["base_value"] + sum(
                                                       v for v in contrib.values() if v > 0)
                                                   ) * 1.18]))
        st.plotly_chart(fig, use_container_width=True)
        top = order[0]
        st.info(md(f"**Lecture** : partant du coût moyen ({money(exp['base_value'])}), le "
                   f"facteur qui pèse le plus pour ce profil est "
                   f"**{FEATURE_LABELS[top].lower()}** ({signed(contrib[top])}). Orange = "
                   "augmente le coût, bleu = le réduit. Les contributions s'additionnent "
                   "exactement à la prédiction."))

# ----------------------------------------------------------------------------- explore
with tab_explore:
    f1, f2, f3, f4 = st.columns(4)
    smokers = f1.multiselect("Tabagisme", ["no", "yes"], ["no", "yes"],
                             format_func=SMOKER_LABELS.get)
    regions = f2.multiselect("Région", CATEGORY_LEVELS["region"], CATEGORY_LEVELS["region"],
                             format_func=REGION_LABELS.get)
    sexes = f3.multiselect("Sexe", CATEGORY_LEVELS["sex"], CATEGORY_LEVELS["sex"],
                           format_func=SEX_LABELS.get)
    ages = f4.slider("Âge", 18, 64, (18, 64))
    view = df[df.smoker.isin(smokers) & df.region.isin(regions) & df.sex.isin(sexes)
              & df.age.between(*ages)]
    if view.empty:
        st.warning("Aucun assuré ne correspond à ces filtres.")
    else:
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Assurés", f"{len(view):,}".replace(",", "\u202f"))
        k2.metric("Coût moyen", money(view.charges.mean()))
        k3.metric("Coût médian", money(view.charges.median()))
        k4.metric("Part de fumeurs", f"{view.smoker.eq('yes').mean():.0%}")
        plot_df = view.assign(Statut=view.smoker.map(SMOKER_LABELS))
        colors = {SMOKER_LABELS[k]: v for k, v in SMOKER_COLORS.items()}
        x_var = st.radio("Axe horizontal", ["age", "bmi"], horizontal=True,
                         format_func=lambda v: "Âge" if v == "age" else "IMC")
        fig = px.scatter(plot_df, x=x_var, y="charges", color="Statut",
                         color_discrete_map=colors, opacity=0.75,
                         labels={"age": "Âge", "bmi": "IMC", "charges": "Coûts annuels ($)"},
                         hover_data={"sex": True, "region": True, "children": True})
        fig.update_traces(marker=dict(size=8, line=dict(color="white", width=1)))
        if x_var == "bmi":
            fig.add_vline(x=30, line_dash="dot", line_color=INK_2, annotation_text="IMC 30")
        fig.update_layout(height=430, title="Coûts individuels")
        st.plotly_chart(fig, use_container_width=True)
        g1, g2 = st.columns(2)
        hist = px.histogram(plot_df, x="charges", color="Statut", nbins=40,
                            color_discrete_map=colors, barmode="overlay", opacity=0.7,
                            labels={"charges": "Coûts annuels ($)"})
        hist.update_layout(height=360, title="Distribution des coûts", yaxis_title="Assurés")
        g1.plotly_chart(hist, use_container_width=True)
        by_region = (plot_df.groupby(["region", "Statut"]).charges.mean().reset_index()
                     .assign(region=lambda d: d.region.map(REGION_LABELS)))
        bars = px.bar(by_region, x="region", y="charges", color="Statut", barmode="group",
                      color_discrete_map=colors,
                      labels={"region": "Région", "charges": "Coût moyen ($)"})
        bars.update_layout(height=360, title="Coût moyen par région et statut")
        g2.plotly_chart(bars, use_container_width=True)
        with st.expander("Données filtrées"):
            st.dataframe(view, use_container_width=True, hide_index=True)
            st.download_button("Télécharger (CSV)", view.to_csv(index=False).encode("utf-8"),
                               "assures_filtres.csv", "text/csv")

# ----------------------------------------------------------------------------- models
with tab_models:
    st.markdown(
        "**Protocole** : split 80/20 stratifié, hyperparamètres réglés en validation croisée "
        "5 plis sur le train, **sélection sur la RMSE de CV** (jamais sur le test), "
        "évaluation finale unique sur le test avec IC 95 % bootstrap.")
    rows = []
    for name in metrics["ranking"] + ["Baseline (moyenne)"]:
        m = metrics["models"][name]
        rows.append({"Modèle": ("✅ " if name == best else "") + name, "Famille": m["family"],
                     "RMSE CV": m["cv_rmse"], "RMSE test": m["test"]["rmse"],
                     "MAE test": m["test"]["mae"], "R² test": m["test"]["r2"],
                     "Gini": m["test"]["gini"], "Calibration": m["test"]["calibration_ratio"]})
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True,
                 column_config={c: st.column_config.NumberColumn(format="%.0f")
                                for c in ["RMSE CV", "RMSE test", "MAE test"]}
                 | {c: st.column_config.NumberColumn(format="%.3f")
                    for c in ["R² test", "Gini", "Calibration"]})
    cmp = metrics.get("best_vs_runner_up")
    if cmp:
        st.success(md(f"**{cmp['best']} vs {cmp['runner_up']}** : écart de RMSE "
                      f"{cmp['mean_diff']:+.0f} $, IC 95 % [{cmp['ci95'][0]:.0f} ; "
                      f"{cmp['ci95'][1]:.0f}]. Les meilleurs modèles sont à égalité "
                      "statistique ; le plus simple et le plus auditable est retenu."))
    show_figure("model_comparison")
    c1, c2 = st.columns(2)
    with c1:
        show_figure("model_cv_folds")
    with c2:
        show_figure("calibration_lorenz")
    show_figure("residuals")

# ----------------------------------------------------------------------------- explain
with tab_explain:
    for line in interp["insights"]:
        st.markdown(md(f"- {line}"))
    show_figure("importance")
    show_figure("partial_dependence")
    show_figure("shap_beeswarm")
    show_figure("fairness")

# ----------------------------------------------------------------------------- about
with tab_about:
    st.markdown(f"""
**Question** : quels facteurs expliquent les frais médicaux annuels d'un assuré, et comment
les prédire de façon fiable, interprétable et équitable ?

**Données** : *Medical Cost Personal Dataset* (Kaggle, licence ODbL), 1 338 assurés
américains simulés, 6 variables. Nettoyage : 1 doublon retiré.

**Méthode** : audit qualité → EDA → tests d'hypothèses (hypothèses vérifiées, tailles
d'effet, IC) → feature engineering (interactions tabagisme × obésité/IMC/âge) → 8 modèles
(baseline, Ridge, Lasso, cible log, GLM Gamma, Random Forest, XGBoost, LightGBM) →
SHAP, dépendance partielle et audit d'équité.

**Modèle déployé** : {card['model']} ({card['size_kb']} Ko), hyperparamètres
`{card['params']}`.

**Limites** : données simulées, pas de causalité, coût total sans séparation
fréquence/sévérité. Le sexe est un critère interdit en tarification dans l'UE : le
retirer ne dégrade pas le modèle.

[Rapport complet](https://fleury-fcn.github.io/health-insurance-cost-analysis/) ·
[Code source](https://github.com/fleury-fcn/health-insurance-cost-analysis) · API :
`make api` puis `/docs`.
""")
