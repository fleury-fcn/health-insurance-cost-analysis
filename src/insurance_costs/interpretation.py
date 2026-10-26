"""Model interpretation (permutation importance, SHAP, partial dependence) and fairness.

SHAP values are computed on the six *original* policyholder variables with an exact,
model-agnostic explainer: contributions stay readable for business users whatever the
model (engineered interactions are attributed back to smoking/BMI/age).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shap
from plotly.subplots import make_subplots
from sklearn.base import clone
from sklearn.inspection import permutation_importance

from .config import ARTIFACT_DIR, CATEGORY_LEVELS, FEATURES, RANDOM_STATE
from .eda import AGE_BANDS, AGE_LABELS
from .modeling import metrics
from .viz import (
    BLUE,
    INK_2,
    MUTED,
    ORANGE,
    SMOKER_COLORS,
    SMOKER_LABELS,
    dump,
    plt,
    save_plotly,
    save_png,
)

FEATURE_LABELS = {"age": "Âge", "sex": "Sexe", "bmi": "IMC", "children": "Enfants",
                  "smoker": "Tabagisme", "region": "Région"}
N_BACKGROUND = 100


# --------------------------------------------------------------------------- SHAP
def encode(X: pd.DataFrame) -> np.ndarray:
    """Raw policyholder table -> numeric matrix (categories as level indices)."""
    out = X[FEATURES].copy()
    for col, levels in CATEGORY_LEVELS.items():
        out[col] = out[col].map({lv: i for i, lv in enumerate(levels)})
    return out.to_numpy(dtype=float)


def decode(Z: np.ndarray) -> pd.DataFrame:
    df = pd.DataFrame(Z, columns=FEATURES)
    for col, levels in CATEGORY_LEVELS.items():
        df[col] = np.asarray(levels, dtype=object)[np.rint(df[col]).astype(int)]
    for col in ["age", "children"]:
        df[col] = np.rint(df[col]).astype(int)
    return df


def make_explainer(model, background: pd.DataFrame) -> shap.Explainer:
    bg = background.sample(min(N_BACKGROUND, len(background)), random_state=RANDOM_STATE)
    masker = shap.maskers.Independent(encode(bg), max_samples=N_BACKGROUND)
    return shap.explainers.Exact(lambda Z: model.predict(decode(Z)), masker,
                                 feature_names=FEATURES)


def explain_profile(model, profile: pd.DataFrame, background: pd.DataFrame) -> dict:
    """Local explanation of one profile: base value + additive contribution per variable."""
    values = make_explainer(model, background)(encode(profile), silent=True)
    contrib = {f: float(v) for f, v in zip(FEATURES, values.values[0], strict=True)}
    return {"base_value": float(values.base_values[0]),
            "prediction": float(model.predict(profile[FEATURES])[0]),
            "contributions": contrib}


def shap_global(model, X: pd.DataFrame, background: pd.DataFrame) -> tuple[np.ndarray, float]:
    values = make_explainer(model, background)(encode(X), silent=True)
    return values.values, float(np.mean(values.base_values))


# --------------------------------------------------------------------------- PDP
def conditional_pdp(model, X: pd.DataFrame, feature: str, grid: np.ndarray) -> dict:
    """Partial dependence of `feature`, computed separately for smokers and non-smokers."""
    out = {}
    for level in ["no", "yes"]:
        base = X.assign(smoker=level)
        out[level] = [float(model.predict(base.assign(**{feature: v})).mean()) for v in grid]
    return out


# --------------------------------------------------------------------------- fairness
def subgroup_errors(y: pd.Series, pred: np.ndarray, X: pd.DataFrame) -> dict:
    frame = pd.DataFrame({"y": y.to_numpy(), "pred": pred,
                          "age_band": pd.cut(X["age"], AGE_BANDS, labels=AGE_LABELS).to_numpy(),
                          "sex": X["sex"].to_numpy(), "region": X["region"].to_numpy(),
                          "smoker": X["smoker"].to_numpy()})
    out = {}
    rng = np.random.default_rng(RANDOM_STATE)
    for col in ["age_band", "sex", "region", "smoker"]:
        rows = {}
        for level, g in frame.groupby(col, observed=True):
            err = g["pred"] - g["y"]
            idx = rng.integers(0, len(g), (2000, len(g)))
            ratios = g["pred"].to_numpy()[idx].sum(1) / g["y"].to_numpy()[idx].sum(1)
            rows[str(level)] = {"n": int(len(g)), "mae": float(err.abs().mean()),
                                "rmse": float(np.sqrt((err**2).mean())),
                                "bias": float(err.mean()),
                                "calibration_ratio": float(g["pred"].sum() / g["y"].sum()),
                                "calibration_ci95": np.percentile(ratios, [2.5, 97.5]).tolist()}
        maes = [r["mae"] for r in rows.values()]
        out[col] = {"groups": rows, "mae_ratio_max_min": float(max(maes) / min(maes))}
    out["overall_calibration_ratio"] = float(frame["pred"].sum() / frame["y"].sum())
    return out


def without_sex(model, X_train, y_train, X_test, y_test) -> dict:
    """Refit the selected configuration with sex neutralised (EU unisex pricing rule)."""
    neutral = clone(model).fit(X_train.assign(sex="female"), y_train)
    pred = neutral.predict(X_test.assign(sex="female"))
    base = model.predict(X_test)
    by_sex = {s: {"mean_pred_with_sex": float(base[X_test.sex.eq(s)].mean()),
                  "mean_pred_unisex": float(pred[X_test.sex.eq(s)].mean()),
                  "mean_observed": float(y_test[X_test.sex.eq(s)].mean())}
              for s in ["female", "male"]}
    return {"test": metrics(y_test, pred), "by_sex": by_sex}


# --------------------------------------------------------------------------- runner
def run_interpretation(modeling: dict) -> dict:
    best = modeling["best_model"]
    model = modeling["_fitted"][best]
    X_train, y_train = modeling["_X_train"], modeling["_y_train"]
    X_test, y_test = modeling["_X_test"], modeling["_y_test"]
    compare = [m for m in [best, "XGBoost", "GLM Gamma (lien log)"] if m in modeling["_fitted"]]
    compare = list(dict.fromkeys(compare))
    result: dict = {"model": best, "compared_models": compare}

    # Permutation importance (held-out data, RMSE drop, 30 repeats)
    perm = {}
    for name in compare:
        r = permutation_importance(modeling["_fitted"][name], X_test, y_test,
                                   scoring="neg_root_mean_squared_error", n_repeats=30,
                                   random_state=RANDOM_STATE)
        perm[name] = {f: {"mean": float(m), "std": float(s)} for f, m, s in
                      zip(X_test.columns, r.importances_mean, r.importances_std, strict=True)}
    result["permutation_importance"] = perm

    # SHAP (global, test set)
    shap_values, base_value = shap_global(model, X_test, X_train)
    mean_abs = dict(zip(FEATURES, np.abs(shap_values).mean(axis=0).tolist(), strict=True))
    result["shap"] = {"base_value": base_value, "mean_abs": mean_abs,
                      "additivity_max_error": float(np.max(np.abs(
                          shap_values.sum(1) + base_value - model.predict(X_test))))}
    shap_by_model = {best: mean_abs}
    for name in compare:
        if name != best:
            v, _ = shap_global(modeling["_fitted"][name], X_test, X_train)
            shap_by_model[name] = dict(zip(FEATURES, np.abs(v).mean(0).tolist(), strict=True))
    result["shap_by_model"] = shap_by_model

    # Partial dependence, conditional on smoking
    grids = {"age": np.arange(18, 65, 2), "bmi": np.linspace(16, 50, 35),
             "children": np.arange(0, 6)}
    result["pdp"] = {name: {f: {"grid": g.tolist(), **conditional_pdp(
        modeling["_fitted"][name], X_test, f, g)} for f, g in grids.items()}
        for name in compare}

    # Fairness
    pred = modeling["_preds"][best]
    result["fairness"] = subgroup_errors(y_test, pred, X_test)
    result["unisex"] = without_sex(model, X_train, y_train, X_test, y_test)
    result["unisex"]["delta_rmse"] = (result["unisex"]["test"]["rmse"]
                                      - modeling["models"][best]["test"]["rmse"])

    result["figures"] = _figures(result, shap_values, X_test)
    _static(result, shap_values, X_test)
    result["insights"] = insights(result)
    dump(result, ARTIFACT_DIR / "interpretation.json")
    # Legacy-compatible summary consumed by the app
    dump({k: {g: v["mae"] for g, v in d["groups"].items()}
          for k, d in result["fairness"].items() if isinstance(d, dict)},
         ARTIFACT_DIR / "subgroup_errors.json")
    return result


# --------------------------------------------------------------------------- figures
def _figures(r: dict, shap_values: np.ndarray, X: pd.DataFrame) -> list[dict]:
    out = []
    order = sorted(FEATURES, key=lambda f: r["shap"]["mean_abs"][f])
    fig = make_subplots(rows=1, cols=2, subplot_titles=(
        f"SHAP : contribution moyenne absolue ({r['model']})",
        "Permutation : hausse de RMSE si la variable est brouillée"), horizontal_spacing=0.15)
    fig.add_bar(y=[FEATURE_LABELS[f] for f in order],
                x=[r["shap"]["mean_abs"][f] for f in order], orientation="h",
                marker=dict(color=BLUE, cornerradius=4), showlegend=False,
                hovertemplate="%{y}<br>%{x:,.0f} $<extra></extra>", row=1, col=1)
    perm = r["permutation_importance"][r["model"]]
    fig.add_bar(y=[FEATURE_LABELS[f] for f in order], x=[perm[f]["mean"] for f in order],
                error_x=dict(type="data", array=[perm[f]["std"] for f in order], color=INK_2),
                orientation="h", marker=dict(color=BLUE, cornerradius=4), showlegend=False,
                hovertemplate="%{y}<br>+%{x:,.0f} $ de RMSE<extra></extra>", row=1, col=2)
    fig.update_xaxes(title_text="$", row=1, col=1)
    fig.update_xaxes(title_text="$ de RMSE", row=1, col=2)
    fig.update_layout(title="Importance des variables", height=380)
    save_plotly(fig, "importance")
    out.append({"name": "importance", "title": "Importance des variables"})

    # Beeswarm-like strip: SHAP value per insured, coloured by feature value
    fig = go.Figure()
    rng = np.random.default_rng(RANDOM_STATE)
    for i, f in enumerate(order):
        raw = X[f]
        if f in CATEGORY_LEVELS:
            norm = raw.map({lv: k / max(1, len(CATEGORY_LEVELS[f]) - 1)
                            for k, lv in enumerate(CATEGORY_LEVELS[f])}).to_numpy()
        else:
            norm = ((raw - raw.min()) / (raw.max() - raw.min())).to_numpy()
        j = FEATURES.index(f)
        fig.add_scatter(x=shap_values[:, j], y=i + rng.uniform(-0.3, 0.3, len(X)),
                        mode="markers", showlegend=False,
                        marker=dict(size=6, color=norm, colorscale=[[0, BLUE], [1, ORANGE]],
                                    showscale=i == 0, line=dict(color="white", width=0.5),
                                    colorbar=dict(title="valeur", tickvals=[0, 1],
                                                  ticktext=["basse", "haute"], len=0.6)),
                        text=raw.astype(str),
                        hovertemplate=f"{FEATURE_LABELS[f]} = %{{text}}<br>contribution "
                                      "%{x:+,.0f} $<extra></extra>")
    fig.update_yaxes(tickvals=list(range(len(order))),
                     ticktext=[FEATURE_LABELS[f] for f in order])
    fig.add_vline(x=0, line_color=MUTED, line_dash="dot")
    fig.update_layout(title="Valeurs SHAP par assuré (jeu de test)",
                      xaxis_title="Contribution à la prédiction ($)", height=420)
    save_plotly(fig, "shap_beeswarm")
    out.append({"name": "shap_beeswarm", "title": "Valeurs SHAP individuelles"})

    pdp = r["pdp"][r["model"]]
    fig = make_subplots(rows=1, cols=3, subplot_titles=("Âge", "IMC", "Enfants"),
                        shared_yaxes=True)
    for c, f in enumerate(["age", "bmi", "children"], start=1):
        for level in ["no", "yes"]:
            fig.add_scatter(x=pdp[f]["grid"], y=pdp[f][level], mode="lines",
                            name=SMOKER_LABELS[level], legendgroup=level, showlegend=c == 1,
                            line=dict(color=SMOKER_COLORS[level], width=2), row=1, col=c,
                            hovertemplate=f"{FEATURE_LABELS[f]} = %{{x:.0f}}<br>%{{y:,.0f}} $"
                                          f"<extra>{SMOKER_LABELS[level]}</extra>")
    fig.update_yaxes(title_text="Coût moyen prédit ($)", row=1, col=1)
    fig.update_layout(title=f"Dépendance partielle selon le tabagisme ({r['model']})",
                      height=400)
    save_plotly(fig, "partial_dependence")
    out.append({"name": "partial_dependence", "title": "Dépendance partielle"})

    fig = make_subplots(rows=1, cols=4, shared_yaxes=True,
                        subplot_titles=("Âge", "Sexe", "Région", "Tabagisme"))
    for c, col in enumerate(["age_band", "sex", "region", "smoker"], start=1):
        groups = r["fairness"][col]["groups"]
        labels = [SMOKER_LABELS.get(g, g) if col == "smoker" else g for g in groups]
        fig.add_bar(x=labels, y=[v["mae"] for v in groups.values()], name="MAE",
                    marker=dict(color=BLUE, cornerradius=4), showlegend=c == 1,
                    customdata=[[v["n"], v["bias"]] for v in groups.values()],
                    hovertemplate="%{x}<br>MAE %{y:,.0f} $<br>biais %{customdata[1]:+,.0f} $"
                                  "<br>n = %{customdata[0]}<extra></extra>", row=1, col=c)
        fig.add_scatter(x=labels, y=[v["bias"] for v in groups.values()], name="Biais moyen",
                        mode="markers", marker=dict(color=ORANGE, size=10, symbol="diamond",
                                                    line=dict(color="white", width=1.5)),
                        showlegend=c == 1, hoverinfo="skip", row=1, col=c)
    fig.add_hline(y=0, line_color=MUTED)
    fig.update_yaxes(title_text="$", row=1, col=1)
    fig.update_layout(title="Équité : erreur absolue moyenne et biais (prédit − observé) "
                            "par sous-groupe", height=400)
    save_plotly(fig, "fairness")
    out.append({"name": "fairness", "title": "Erreurs par sous-groupe"})
    return out


def _static(r: dict, shap_values: np.ndarray, X: pd.DataFrame) -> None:
    order = sorted(FEATURES, key=lambda f: r["shap"]["mean_abs"][f])
    pdp = r["pdp"][r["model"]]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    vals = [r["shap"]["mean_abs"][f] for f in order]
    axes[0].barh([FEATURE_LABELS[f] for f in order], vals, color=BLUE, edgecolor="white")
    for i, v in enumerate(vals):
        axes[0].text(v, i, f" {v:,.0f} $".replace(",", " "), va="center", fontsize=9,
                     color=INK_2)
    axes[0].set(title="|SHAP| moyen ($)", xlim=(0, max(vals) * 1.25))
    axes[0].grid(axis="y", visible=False)
    for ax, f, label in [(axes[1], "bmi", "IMC"), (axes[2], "age", "Âge")]:
        for level in ["no", "yes"]:
            ax.plot(pdp[f]["grid"], pdp[f][level], color=SMOKER_COLORS[level], lw=2,
                    label=SMOKER_LABELS[level])
        ax.set(title=f"Dépendance partielle : {label}", xlabel=label,
               ylabel="Coût moyen prédit ($)")
    axes[1].legend(loc="upper left")
    fig.tight_layout()
    save_png(fig, "interpretation_overview")


def _money(x: float) -> str:
    return f"{x:,.0f} $".replace(",", " ")


def calibration_note(f: dict) -> str:
    """Flag subgroups whose calibration CI excludes the overall test-level ratio."""
    overall = f["overall_calibration_ratio"]
    flagged = [(g, v) for col in ["age_band", "sex", "region", "smoker"]
               for g, v in f[col]["groups"].items()
               if not v["calibration_ci95"][0] <= overall <= v["calibration_ci95"][1]]
    head = (f"Calibration (prédit / observé, {overall:.2f} sur l'ensemble du test) : ")
    if not flagged:
        return head + ("aucun sous-groupe ne s'écarte significativement du niveau global "
                       "(IC 95 % bootstrap).")
    label = {"yes": "fumeurs", "no": "non-fumeurs", "female": "femmes", "male": "hommes"}
    parts = [f"{label.get(g, g)} {v['calibration_ratio']:.2f} [IC {v['calibration_ci95'][0]:.2f} ; "
             f"{v['calibration_ci95'][1]:.2f}], n = {v['n']}" for g, v in flagged]
    age_flag = any(g in f["age_band"]["groups"] for g, _ in flagged)
    advice = (" C'est le point de vigilance équité : l'effet d'âge du modèle ne reproduit "
              "pas le niveau de coût de cette tranche. Pistes : recalibrer par tranche d'âge "
              "ou modéliser l'âge de façon non linéaire (spline)." if age_flag else
              " Ces groupes sont à surveiller en priorité ; pistes : recalibration ciblée.")
    return (head + "s'écartent significativement du niveau global : " + " ; ".join(parts)
            + "." + advice + " Les autres écarts restent dans le bruit d'échantillonnage.")


def insights(r: dict) -> list[str]:
    s = r["shap"]["mean_abs"]
    ranked = sorted(s, key=s.get, reverse=True)
    pdp = r["pdp"][r["model"]]["bmi"]
    grid = np.asarray(pdp["grid"])
    lo, hi = np.argmin(np.abs(grid - 28)), np.argmin(np.abs(grid - 32))
    f, u = r["fairness"], r["unisex"]
    worst_age = max(f["age_band"]["groups"], key=lambda g: f["age_band"]["groups"][g]["mae"])
    agree = {m: sorted(v, key=v.get, reverse=True)[:3] for m, v in r["shap_by_model"].items()}
    return [
        "Hiérarchie (SHAP) : " + " > ".join(FEATURE_LABELS[k] for k in ranked) +
        f". Le tabagisme pèse en moyenne {_money(s['smoker'])} par assuré, l'âge "
        f"{_money(s['age'])}, l'IMC {_money(s['bmi'])} ; sexe et région moins de "
        f"{_money(max(s['sex'], s['region']) + 1)}.",
        "Les modèles comparés s'accordent sur le top 3 : " + "; ".join(
            f"{m} → {', '.join(FEATURE_LABELS[x] for x in v)}" for m, v in agree.items()) + ".",
        f"Effet de seuil confirmé : passer d'un IMC de 28 à 32 ajoute "
        f"{_money(pdp['yes'][hi] - pdp['yes'][lo])} au coût prédit d'un fumeur contre "
        f"{_money(pdp['no'][hi] - pdp['no'][lo])} pour un non-fumeur.",
        f"Équité : l'erreur absolue varie d'un facteur {f['age_band']['mae_ratio_max_min']:.2f}"
        f" entre tranches d'âge (max : {worst_age}), {f['sex']['mae_ratio_max_min']:.2f} entre "
        f"sexes, {f['region']['mae_ratio_max_min']:.2f} entre régions. " + calibration_note(f),
        f"Tarification unisexe : neutraliser le sexe change la RMSE de "
        f"{u['delta_rmse']:+.0f} $ ({_money(u['test']['rmse'])}), un écart dans le bruit de "
        "validation. L'exigence réglementaire européenne ne coûte rien en précision.",
        f"Contrôle technique : les contributions SHAP s'additionnent exactement à la "
        f"prédiction (écart max {r['shap']['additivity_max_error']:.1e} $).",
    ]
