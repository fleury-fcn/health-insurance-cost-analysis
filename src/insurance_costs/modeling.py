"""Model benchmarking: tuned pipelines, honest evaluation and residual analysis.

Protocol
1. Fixed stratified 80/20 split (features.split). The test set is used once, at the end.
2. Hyper-parameters tuned by 5-fold CV on the training set only.
3. The production model is selected on CV RMSE (never on the test set).
4. Test metrics come with bootstrap confidence intervals.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import sklearn
from plotly.subplots import make_subplots
from scipy import stats
from scipy.stats import loguniform, randint, uniform
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, Ridge, TweedieRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_gamma_deviance,
    r2_score,
    root_mean_squared_error,
)
from sklearn.model_selection import GridSearchCV, KFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from .config import ARTIFACT_DIR, BEST_MODEL, CV_FOLDS, FEATURES, MODEL_DIR, RANDOM_STATE
from .features import SmearedLogRegressor, preprocessor, split
from .viz import (
    AQUA,
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

N_BOOT = 2000


@dataclass
class Candidate:
    name: str
    family: str
    pipeline: Pipeline
    grid: dict = field(default_factory=dict)
    random: bool = False
    n_iter: int = 30
    rationale: str = ""


def candidates(n_iter: int = 30) -> list[Candidate]:
    lin = lambda model: Pipeline([("prep", preprocessor(True, "linear")),  # noqa: E731
                                  ("model", model)])
    tree = lambda model: Pipeline([("prep", preprocessor(False, "tree")),  # noqa: E731
                                   ("model", model)])
    alphas = np.logspace(-3, 3, 13)
    return [
        Candidate("Baseline (moyenne)", "Référence", lin(DummyRegressor()),
                  rationale="Plancher de performance : prédire le coût moyen à tous."),
        Candidate("Ridge", "Linéaire", lin(Ridge()), {"model__alpha": alphas},
                  rationale="Linéaire pénalisé L2, stable, coefficients lisibles."),
        Candidate("Lasso", "Linéaire", lin(Lasso(max_iter=50_000)),
                  {"model__alpha": np.logspace(-1, 3, 13)},
                  rationale="Pénalité L1 : sélection automatique de variables."),
        Candidate("Ridge cible log", "Linéaire", lin(SmearedLogRegressor(Ridge())),
                  {"model__regressor__alpha": alphas},
                  rationale="Cible log + correction de Duan, pour tester l'échelle "
                            "multiplicative."),
        Candidate("GLM Gamma (lien log)", "GLM actuariel",
                  lin(TweedieRegressor(power=2, link="log", max_iter=5000)),
                  {"model__alpha": np.logspace(-4, 0, 9)},
                  rationale="Standard actuariel des montants : variance ∝ moyenne², effets "
                            "multiplicatifs lisibles comme des coefficients tarifaires."),
        Candidate("Random Forest", "Arbres", tree(RandomForestRegressor(
            random_state=RANDOM_STATE, n_jobs=1)), {
            "model__n_estimators": randint(200, 600),
            "model__max_depth": [3, 4, 5, 6, 8, None],
            "model__min_samples_leaf": randint(1, 30),
            "model__max_features": [0.5, 0.7, 1.0]},
            random=True, n_iter=n_iter,
            rationale="Bagging d'arbres : capte interactions et non-linéarités sans "
                      "feature engineering."),
        Candidate("XGBoost", "Boosting", tree(XGBRegressor(
            random_state=RANDOM_STATE, n_jobs=1, tree_method="hist")), {
            "model__n_estimators": randint(200, 900),
            "model__learning_rate": loguniform(0.01, 0.2),
            "model__max_depth": randint(2, 6),
            "model__min_child_weight": randint(1, 20),
            "model__subsample": uniform(0.6, 0.4),
            "model__colsample_bytree": uniform(0.6, 0.4),
            "model__reg_lambda": loguniform(0.1, 20)},
            random=True, n_iter=n_iter,
            rationale="Gradient boosting régularisé, souvent l'état de l'art sur données "
                      "tabulaires."),
        Candidate("LightGBM", "Boosting", tree(lgb.LGBMRegressor(
            random_state=RANDOM_STATE, n_jobs=1, verbose=-1, deterministic=True,
            force_row_wise=True)), {
            "model__n_estimators": randint(200, 900),
            "model__learning_rate": loguniform(0.01, 0.2),
            "model__num_leaves": randint(4, 32),
            "model__min_child_samples": randint(5, 60),
            "model__subsample": uniform(0.6, 0.4),
            "model__subsample_freq": [1],
            "model__colsample_bytree": uniform(0.6, 0.4),
            "model__reg_lambda": loguniform(0.1, 20)},
            random=True, n_iter=n_iter,
            rationale="Boosting par histogrammes, croissance par feuilles ; rapide et "
                      "léger à déployer."),
    ]


# --------------------------------------------------------------------------- metrics
def normalized_gini(y: np.ndarray, pred: np.ndarray) -> float:
    """Gini of the concentration curve, normalised by the oracle Gini (1 = perfect order)."""
    def gini(actual, score):
        order = np.argsort(-score, kind="mergesort")
        cum = np.cumsum(actual[order]) / actual.sum()
        return (cum.sum() / len(actual)) - (len(actual) + 1) / (2 * len(actual))
    y, pred = np.asarray(y), np.asarray(pred)
    if np.ptp(pred) == 0:  # a constant score carries no ranking information
        return 0.0
    return float(gini(y, pred) / gini(y, y))


def metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    y, pred = np.asarray(y), np.asarray(pred)
    safe = np.clip(pred, 1.0, None)  # gamma deviance needs strictly positive predictions
    return {
        "rmse": float(root_mean_squared_error(y, pred)),
        "mae": float(mean_absolute_error(y, pred)),
        "r2": float(r2_score(y, pred)),
        "gamma_deviance": float(mean_gamma_deviance(y, safe)),
        "calibration_ratio": float(pred.sum() / y.sum()),
        "gini": normalized_gini(y, pred),
        "share_negative_predictions": float((pred < 0).mean()),
    }


def bootstrap_metric_ci(y: np.ndarray, pred: np.ndarray, seed: int = RANDOM_STATE) -> dict:
    rng = np.random.default_rng(seed)
    y, pred = np.asarray(y), np.asarray(pred)
    idx = rng.integers(0, len(y), (N_BOOT, len(y)))
    err = y[idx] - pred[idx]
    rmse = np.sqrt((err**2).mean(axis=1))
    mae = np.abs(err).mean(axis=1)
    ss_tot = ((y[idx] - y[idx].mean(axis=1, keepdims=True)) ** 2).sum(axis=1)
    r2 = 1 - (err**2).sum(axis=1) / ss_tot
    q = lambda a: [float(v) for v in np.percentile(a, [2.5, 97.5])]  # noqa: E731
    return {"rmse": q(rmse), "mae": q(mae), "r2": q(r2)}


def paired_bootstrap(y, pred_a, pred_b, seed: int = RANDOM_STATE) -> dict:
    """CI of RMSE(a) - RMSE(b) on the same resamples: is the winner really better?"""
    rng = np.random.default_rng(seed)
    y, a, b = map(np.asarray, (y, pred_a, pred_b))
    idx = rng.integers(0, len(y), (N_BOOT, len(y)))
    diff = (np.sqrt(((y[idx] - a[idx]) ** 2).mean(1))
            - np.sqrt(((y[idx] - b[idx]) ** 2).mean(1)))
    lo, hi = np.percentile(diff, [2.5, 97.5])
    return {"mean_diff": float(diff.mean()), "ci95": [float(lo), float(hi)],
            "prob_a_better": float((diff < 0).mean())}


# --------------------------------------------------------------------------- training
def tune(candidate: Candidate, X, y, cv) -> tuple[Pipeline, dict]:
    scoring = {"rmse": "neg_root_mean_squared_error", "mae": "neg_mean_absolute_error",
               "r2": "r2"}
    if not candidate.grid:
        search = GridSearchCV(candidate.pipeline, {}, cv=cv, scoring=scoring, refit="rmse")
    elif candidate.random:
        search = RandomizedSearchCV(candidate.pipeline, candidate.grid, n_iter=candidate.n_iter,
                                    cv=cv, scoring=scoring, refit="rmse", n_jobs=-1,
                                    random_state=RANDOM_STATE)
    else:
        search = GridSearchCV(candidate.pipeline, candidate.grid, cv=cv, scoring=scoring,
                              refit="rmse", n_jobs=-1)
    search.fit(X, y)
    i = search.best_index_
    res = search.cv_results_
    folds = [float(-res[f"split{k}_test_rmse"][i]) for k in range(cv.get_n_splits())]
    params = {k.replace("model__", "").replace("regressor__", ""): (
        v.item() if hasattr(v, "item") else v) for k, v in search.best_params_.items()}
    return search.best_estimator_, {
        "cv_rmse": float(-res["mean_test_rmse"][i]), "cv_rmse_std": float(np.std(folds)),
        "cv_mae": float(-res["mean_test_mae"][i]), "cv_r2": float(res["mean_test_r2"][i]),
        "cv_fold_rmse": folds, "best_params": params, "n_configs": len(res["params"]),
    }


def train_and_evaluate(df: pd.DataFrame, n_iter: int = 30,
                       only: list[str] | None = None) -> dict:
    X_train, X_test, y_train, y_test = split(df)
    cv = KFold(CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    results, fitted, preds = {}, {}, {}
    for cand in candidates(n_iter):
        if only and cand.name not in only:
            continue
        start = time.perf_counter()
        model, cv_info = tune(cand, X_train, y_train, cv)
        pred = model.predict(X_test)
        results[cand.name] = {
            "family": cand.family, "rationale": cand.rationale, **cv_info,
            "test": metrics(y_test, pred), "test_ci95": bootstrap_metric_ci(y_test, pred),
            "fit_seconds": round(time.perf_counter() - start, 2),
        }
        fitted[cand.name], preds[cand.name] = model, pred

    eligible = {k: v for k, v in results.items() if not k.startswith("Baseline")} or results
    best = min(eligible, key=lambda k: eligible[k]["cv_rmse"])
    ranking = sorted(eligible, key=lambda k: eligible[k]["cv_rmse"])
    comparison = None
    if len(ranking) > 1:
        comparison = {"best": best, "runner_up": ranking[1],
                      **paired_bootstrap(y_test, preds[best], preds[ranking[1]])}
    glm = "GLM Gamma (lien log)"
    return {
        "best_model": best, "selection_metric": "cv_rmse", "ranking": ranking,
        "models": results, "best_vs_runner_up": comparison,
        "glm_coefficients": glm_coefficients(fitted[glm]) if glm in fitted else None,
        "split": {"n_train": len(X_train), "n_test": len(X_test),
                  "smoker_share_train": float(X_train.smoker.eq("yes").mean()),
                  "smoker_share_test": float(X_test.smoker.eq("yes").mean())},
        "_fitted": fitted, "_preds": preds, "_X_test": X_test, "_y_test": y_test,
        "_X_train": X_train, "_y_train": y_train,
    }


def glm_coefficients(model: Pipeline) -> dict:
    """GLM relativities exp(beta) per standard deviation (or vs reference level)."""
    names = model.named_steps["prep"].get_feature_names_out()
    glm = model.named_steps["model"]
    return {"intercept_base_cost": float(np.exp(glm.intercept_)),
            "relativities": {n: float(np.exp(c)) for n, c in
                             zip(names, glm.coef_, strict=True)}}


# --------------------------------------------------------------------------- residuals
def residual_analysis(y: pd.Series, pred: np.ndarray, X: pd.DataFrame) -> dict:
    resid = np.asarray(y) - pred
    smoker = X["smoker"].eq("yes").to_numpy()
    abs_r = np.abs(resid)
    hetero = stats.spearmanr(pred, abs_r)
    deciles = pd.qcut(pred, 10, labels=False, duplicates="drop")
    calib = (pd.DataFrame({"decile": deciles, "pred": pred, "obs": np.asarray(y)})
             .groupby("decile").mean())
    return {
        "mean_residual": float(resid.mean()), "median_residual": float(np.median(resid)),
        "skewness": float(stats.skew(resid)),
        "share_abs_error_gt_10k": float((abs_r > 10_000).mean()),
        "share_underpredicted_gt_10k": float((resid > 10_000).mean()),
        "heteroscedasticity_spearman": {"rho": float(hetero.statistic),
                                        "p_value": float(hetero.pvalue)},
        "mae_by_smoker": {"no": float(abs_r[~smoker].mean()), "yes": float(abs_r[smoker].mean())},
        "large_errors_smoker_share": float(smoker[abs_r > 10_000].mean())
        if (abs_r > 10_000).any() else 0.0,
        "calibration_by_decile": calib.round(1).to_dict(orient="list"),
    }


# --------------------------------------------------------------------------- figures
def figures(result: dict) -> list[dict]:
    models, best = result["models"], result["best_model"]
    y, X = result["_y_test"].to_numpy(), result["_X_test"]
    pred = result["_preds"][best]
    names = [n for n in sorted(models, key=lambda k: models[k]["cv_rmse"], reverse=True)]
    out = []

    fig = make_subplots(rows=1, cols=2, subplot_titles=("RMSE test (IC 95 %)",
                                                         "MAE test (IC 95 %)"),
                        shared_yaxes=True)
    for col, metric in enumerate(["rmse", "mae"], start=1):
        vals = [models[n]["test"][metric] for n in names]
        ci = [models[n]["test_ci95"][metric] for n in names]
        fig.add_bar(y=names, x=vals, orientation="h", showlegend=False,
                    marker=dict(color=[ORANGE if n == best else BLUE for n in names],
                                line=dict(color="white", width=2), cornerradius=4),
                    error_x=dict(type="data", symmetric=False, color=INK_2,
                                 array=[c[1] - v for c, v in zip(ci, vals, strict=True)],
                                 arrayminus=[v - c[0] for c, v in zip(ci, vals, strict=True)]),
                    hovertemplate="%{y}<br>%{x:,.0f} $<extra></extra>", row=1, col=col)
    fig.update_layout(title=f"Comparaison des modèles, meilleur en CV : {best}", height=430,
                      margin=dict(l=170))
    save_plotly(fig, "model_comparison")
    out.append({"name": "model_comparison", "title": "Comparaison des modèles"})

    fig = go.Figure()
    for n in names:
        fig.add_box(x=models[n]["cv_fold_rmse"], name=n, boxpoints="all", jitter=0.3,
                    marker=dict(color=ORANGE if n == best else BLUE, size=7),
                    line=dict(color=ORANGE if n == best else BLUE), showlegend=False,
                    hovertemplate="%{x:,.0f} $<extra>" + n + "</extra>")
    fig.update_layout(title="Stabilité : RMSE sur chacun des 5 plis de validation croisée",
                      xaxis_title="RMSE ($)", height=430, margin=dict(l=170))
    save_plotly(fig, "model_cv_folds")
    out.append({"name": "model_cv_folds", "title": "Stabilité en validation croisée"})

    resid = y - pred
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Observé vs prédit",
                                                         "Résidus vs prédit"))
    for level in ["no", "yes"]:
        m = X["smoker"].eq(level).to_numpy()
        common = dict(mode="markers", name=SMOKER_LABELS[level], legendgroup=level,
                      marker=dict(color=SMOKER_COLORS[level], size=8, opacity=0.75,
                                  line=dict(color="white", width=1)))
        fig.add_scatter(x=pred[m], y=y[m], **common, row=1, col=1,
                        hovertemplate="prédit %{x:,.0f} $<br>observé %{y:,.0f} $<extra></extra>")
        fig.add_scatter(x=pred[m], y=resid[m], **common, showlegend=False, row=1, col=2,
                        hovertemplate="prédit %{x:,.0f} $<br>résidu %{y:,.0f} $<extra></extra>")
    lim = [0, float(max(y.max(), pred.max()))]
    fig.add_scatter(x=lim, y=lim, mode="lines", line=dict(color=MUTED, dash="dot"),
                    showlegend=False, hoverinfo="skip", row=1, col=1)
    fig.add_hline(y=0, line_color=MUTED, line_dash="dot", row=1, col=2)
    fig.update_xaxes(title_text="Coût prédit ($)")
    fig.update_yaxes(title_text="Coût observé ($)", row=1, col=1)
    fig.update_yaxes(title_text="Résidu ($)", row=1, col=2)
    fig.update_layout(title=f"Diagnostic des résidus : {best} (jeu de test)", height=430)
    save_plotly(fig, "residuals")
    out.append({"name": "residuals", "title": "Analyse des résidus"})

    fig = make_subplots(rows=1, cols=2, subplot_titles=(
        "Calibration par décile de prédiction", "Courbe de concentration (Lorenz)"))
    deciles = pd.qcut(pred, 10, labels=False, duplicates="drop")
    cal = pd.DataFrame({"d": deciles, "p": pred, "o": y}).groupby("d").mean()
    fig.add_scatter(x=cal.index + 1, y=cal["o"], name="Observé", mode="lines+markers",
                    line=dict(color=BLUE, width=2), marker=dict(size=8), row=1, col=1)
    fig.add_scatter(x=cal.index + 1, y=cal["p"], name="Prédit", mode="lines+markers",
                    line=dict(color=ORANGE, width=2, dash="dash"), marker=dict(size=8),
                    row=1, col=1)
    share = np.arange(1, len(y) + 1) / len(y)
    for label, score, color, dash in [("Modèle", pred, ORANGE, "solid"),
                                      ("Oracle", y, AQUA, "dot")]:
        order = np.argsort(-score)
        fig.add_scatter(x=share, y=np.cumsum(y[order]) / y.sum(), mode="lines", name=label,
                        line=dict(color=color, width=2, dash=dash), row=1, col=2,
                        hovertemplate="%{x:.0%} des assurés<br>%{y:.0%} des coûts"
                                      "<extra>" + label + "</extra>")
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="Aléatoire",
                    line=dict(color=MUTED, dash="dot"), row=1, col=2, hoverinfo="skip")
    fig.update_xaxes(title_text="Décile de coût prédit", row=1, col=1)
    fig.update_yaxes(title_text="Coût moyen ($)", row=1, col=1)
    fig.update_xaxes(title_text="Part des assurés (triés par risque prédit)",
                     tickformat=".0%", row=1, col=2)
    fig.update_yaxes(title_text="Part cumulée des coûts", tickformat=".0%", row=1, col=2)
    fig.update_layout(title="Qualité actuarielle : calibration et pouvoir de segmentation",
                      height=430)
    save_plotly(fig, "calibration_lorenz")
    out.append({"name": "calibration_lorenz", "title": "Calibration et Lorenz"})

    # Static PNG for README
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    vals = [models[n]["test"]["rmse"] for n in names]
    axes[0].barh(names, vals, color=[ORANGE if n == best else BLUE for n in names],
                 edgecolor="white", linewidth=2)
    for i, v in enumerate(vals):
        axes[0].text(v, i, f" {v:,.0f}".replace(",", " "), va="center", color=INK_2,
                     fontsize=9)
    axes[0].set(title="RMSE sur le jeu de test ($)", xlim=(0, max(vals) * 1.18))
    axes[0].grid(axis="y", visible=False)
    for level in ["no", "yes"]:
        m = X["smoker"].eq(level).to_numpy()
        axes[1].scatter(pred[m], y[m], s=16, alpha=0.75, color=SMOKER_COLORS[level],
                        edgecolor="white", linewidth=0.4, label=SMOKER_LABELS[level])
        axes[2].scatter(pred[m], resid[m], s=16, alpha=0.75, color=SMOKER_COLORS[level],
                        edgecolor="white", linewidth=0.4)
    axes[1].plot(lim, lim, ls=":", color=MUTED)
    axes[1].set(title=f"Observé vs prédit : {best}", xlabel="Prédit ($)", ylabel="Observé ($)")
    axes[1].legend(loc="upper left")
    axes[2].axhline(0, ls=":", color=MUTED)
    axes[2].set(title="Résidus vs prédit", xlabel="Prédit ($)", ylabel="Résidu ($)")
    fig.tight_layout()
    save_png(fig, "model_overview")
    return out


# --------------------------------------------------------------------------- persistence
def save_best(result: dict) -> dict:
    best = result["best_model"]
    model = result["_fitted"][best]
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, BEST_MODEL, compress=3)
    card = {
        "model": best, "features": FEATURES, "target": "charges (USD/an)",
        "selection": "RMSE de validation croisée 5 plis (train)",
        "cv_rmse": result["models"][best]["cv_rmse"],
        "test_metrics": result["models"][best]["test"],
        "params": result["models"][best]["best_params"],
        "trained_on_rows": result["split"]["n_train"],
        "versions": {"scikit-learn": sklearn.__version__, "lightgbm": lgb.__version__,
                     "numpy": np.__version__, "pandas": pd.__version__},
        "size_kb": round(BEST_MODEL.stat().st_size / 1024, 1),
        "intended_use": "Démonstration pédagogique. Ne pas utiliser pour une décision "
                        "tarifaire ou médicale individuelle.",
    }
    dump(card, MODEL_DIR / "model_card.json")
    return card


def public(result: dict) -> dict:
    """Result without in-memory objects, JSON-serialisable."""
    return json.loads(json.dumps({k: v for k, v in result.items() if not k.startswith("_")},
                                 default=float))


def run_modeling(df: pd.DataFrame, n_iter: int = 30) -> dict:
    result = train_and_evaluate(df, n_iter=n_iter)
    best = result["best_model"]
    result["residuals"] = residual_analysis(result["_y_test"], result["_preds"][best],
                                            result["_X_test"])
    glm = "GLM Gamma (lien log)"
    if glm in result["_preds"]:
        result["residuals_glm"] = residual_analysis(result["_y_test"], result["_preds"][glm],
                                                    result["_X_test"])
    result["figures"] = figures(result)
    result["model_card"] = save_best(result)
    dump(public(result), ARTIFACT_DIR / "metrics.json")
    return result
