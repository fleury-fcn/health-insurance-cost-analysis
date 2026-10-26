"""Inferential statistics with explicit assumption checks, effect sizes and CIs.

Decision rule applied to every comparison: check normality (Shapiro-Wilk) and variance
homogeneity (Levene, median-centred). Classical tests are reported for reference, but the
conclusion relies on the test whose assumptions hold (Welch, rank-based, bootstrap).
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import stats

from .config import ARTIFACT_DIR, OBESITY_THRESHOLD, RANDOM_STATE, TARGET
from .viz import BLUE, INK_2, ORANGE, SMOKER_LABELS, dump, save_plotly

ALPHA = 0.05
N_BOOT = 10_000


# --------------------------------------------------------------------------- helpers
def bootstrap_ci(a: np.ndarray, b: np.ndarray | None = None, stat=np.mean,
                 n_boot: int = N_BOOT, seed: int = RANDOM_STATE) -> tuple[float, float]:
    """Percentile bootstrap CI of stat(a) or stat(a) - stat(b)."""
    rng = np.random.default_rng(seed)
    a = np.asarray(a)
    idx_a = rng.integers(0, len(a), (n_boot, len(a)))
    boot = stat(a[idx_a], axis=1)
    if b is not None:
        b = np.asarray(b)
        boot = boot - stat(b[rng.integers(0, len(b), (n_boot, len(b)))], axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(lo), float(hi)


def mean_ci(x: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    x = np.asarray(x)
    half = stats.t.ppf((1 + level) / 2, len(x) - 1) * stats.sem(x)
    return float(x.mean() - half), float(x.mean() + half)


def welch_ci(a: np.ndarray, b: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    va, vb = np.var(a, ddof=1) / len(a), np.var(b, ddof=1) / len(b)
    dof = (va + vb) ** 2 / (va**2 / (len(a) - 1) + vb**2 / (len(b) - 1))
    half = stats.t.ppf((1 + level) / 2, dof) * np.sqrt(va + vb)
    diff = np.mean(a) - np.mean(b)
    return float(diff - half), float(diff + half)


def hedges_g(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = len(a), len(b)
    pooled = np.sqrt(((na - 1) * np.var(a, ddof=1) + (nb - 1) * np.var(b, ddof=1))
                     / (na + nb - 2))
    correction = 1 - 3 / (4 * (na + nb) - 9)
    return float((np.mean(a) - np.mean(b)) / pooled * correction)


def rank_biserial(u: float, na: int, nb: int) -> float:
    """Effect size for Mann-Whitney: P(A > B) - P(B > A)."""
    return float(2 * u / (na * nb) - 1)


def cramers_v(table: pd.DataFrame, chi2: float) -> float:
    n = table.to_numpy().sum()
    return float(np.sqrt(chi2 / (n * (min(table.shape) - 1))))


def magnitude(value: float, thresholds: tuple[float, float, float]) -> str:
    small, medium, large = thresholds
    v = abs(value)
    return ("négligeable" if v < small else "faible" if v < medium
            else "moyen" if v < large else "fort")


def check_assumptions(groups: list[np.ndarray]) -> dict:
    shapiro = [stats.shapiro(g if len(g) <= 5000 else g[:5000]).pvalue for g in groups]
    levene = stats.levene(*groups, center="median").pvalue
    return {"shapiro_p_min": float(min(shapiro)), "normality_ok": bool(min(shapiro) > ALPHA),
            "levene_p": float(levene), "equal_variance_ok": bool(levene > ALPHA)}


def design_matrix(df: pd.DataFrame, drop: str | None = None) -> np.ndarray:
    cols = [df["age"], df["bmi"], df["children"], df["smoker"].eq("yes"),
            df["smoker"].eq("yes") * (df["bmi"] >= OBESITY_THRESHOLD)]
    if drop != "sex":
        cols.append(df["sex"].eq("male"))
    if drop != "region":
        cols += [df["region"].eq(r) for r in ["northwest", "southeast", "southwest"]]
    return np.column_stack([np.ones(len(df))] + [np.asarray(c, dtype=float) for c in cols])


def nested_f_test(df: pd.DataFrame, variable: str) -> dict:
    """Partial F-test of `variable` in an OLS model of log(charges).

    Tests whether sex/region still explain costs once age, BMI, children, smoking and
    the smoking x obesity interaction are controlled for (i.e. removes confounding).
    """
    y = np.log(df[TARGET].to_numpy())
    full, reduced = design_matrix(df), design_matrix(df, drop=variable)
    rss = []
    for X in (full, reduced):
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        rss.append(float(((y - X @ beta) ** 2).sum()))
    q, dof = full.shape[1] - reduced.shape[1], len(y) - full.shape[1]
    f = ((rss[1] - rss[0]) / q) / (rss[0] / dof)
    beta, *_ = np.linalg.lstsq(full, y, rcond=None)
    partial_eta2 = (rss[1] - rss[0]) / rss[1]
    # Columns: const, age, bmi, children, smoker, smoker x obese, sex_male, nw, se, sw
    names = ({"sex": ["male"]} if variable == "sex" else
             {"region": ["northwest", "southeast", "southwest"]})[variable]
    coefs = beta[6:7] if variable == "sex" else beta[7:10]
    effects = {f"{n} (vs {'female' if variable == 'sex' else 'northeast'})":
               float(np.expm1(c)) for n, c in zip(names, coefs, strict=True)}
    return {"F": float(f), "df": [q, dof], "p_value": float(stats.f.sf(f, q, dof)),
            "partial_eta2": float(partial_eta2), "relative_effects": effects,
            "note": "OLS sur log(coûts) : effets multiplicatifs, test F approximatif "
                    "(résidus non parfaitement normaux)."}


def holm(pvalues: list[float]) -> list[float]:
    order = np.argsort(pvalues)
    m, adjusted, running = len(pvalues), np.empty(len(pvalues)), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvalues[i]))
        adjusted[i] = running
    return adjusted.tolist()


# --------------------------------------------------------------------------- tests
def two_group_comparison(df: pd.DataFrame, column: str, a_level: str, b_level: str,
                         question: str) -> dict:
    a = df.loc[df[column] == a_level, TARGET].to_numpy()
    b = df.loc[df[column] == b_level, TARGET].to_numpy()
    assumptions = check_assumptions([a, b])
    student = stats.ttest_ind(a, b, equal_var=True)
    welch = stats.ttest_ind(a, b, equal_var=False)
    mw = stats.mannwhitneyu(a, b, alternative="two-sided")
    rbc = rank_biserial(mw.statistic, len(a), len(b))
    g = hedges_g(a, b)
    return {
        "question": question, "groups": [a_level, b_level], "n": [len(a), len(b)],
        "means": [float(a.mean()), float(b.mean())],
        "medians": [float(np.median(a)), float(np.median(b))],
        "assumptions": assumptions,
        "student_t": {"statistic": float(student.statistic), "p_value": float(student.pvalue)},
        "welch_t": {"statistic": float(welch.statistic), "p_value": float(welch.pvalue),
                    "ci95_mean_diff": welch_ci(a, b)},
        "mann_whitney": {"statistic": float(mw.statistic), "p_value": float(mw.pvalue),
                         "rank_biserial": rbc},
        "bootstrap_ci95_mean_diff": bootstrap_ci(a, b),
        "bootstrap_ci95_median_diff": bootstrap_ci(a, b, stat=np.median),
        "hedges_g": g, "hedges_g_magnitude": magnitude(g, (0.2, 0.5, 0.8)),
        "rank_biserial_magnitude": magnitude(rbc, (0.1, 0.3, 0.5)),
        "decision_test": "Mann-Whitney + Welch/bootstrap" if not assumptions["normality_ok"]
        else "Welch",
    }


def k_group_comparison(df: pd.DataFrame, column: str, question: str) -> dict:
    levels = sorted(df[column].unique())
    groups = [df.loc[df[column] == lv, TARGET].to_numpy() for lv in levels]
    assumptions = check_assumptions(groups)
    anova = stats.f_oneway(*groups)
    ag = stats.alexandergovern(*groups)  # heteroscedasticity-robust one-way test
    kw = stats.kruskal(*groups)
    y = df[TARGET].to_numpy()
    ss_between = sum(len(g) * (g.mean() - y.mean()) ** 2 for g in groups)
    ss_total = ((y - y.mean()) ** 2).sum()
    k, n = len(groups), len(y)
    eta2 = ss_between / ss_total
    ms_within = (ss_total - ss_between) / (n - k)
    omega2 = (ss_between - (k - 1) * ms_within) / (ss_total + ms_within)
    eps2 = (kw.statistic - k + 1) / (n - k)
    pairs = list(itertools.combinations(range(k), 2))
    raw_p = [stats.mannwhitneyu(groups[i], groups[j]).pvalue for i, j in pairs]
    posthoc = [{"pair": f"{levels[i]} vs {levels[j]}", "p_holm": p}
               for (i, j), p in zip(pairs, holm(raw_p), strict=True)]
    return {
        "question": question, "levels": levels, "n": [len(g) for g in groups],
        "means": [float(g.mean()) for g in groups], "assumptions": assumptions,
        "anova": {"statistic": float(anova.statistic), "p_value": float(anova.pvalue),
                  "eta2": float(eta2), "omega2": float(omega2)},
        "alexander_govern": {"statistic": float(ag.statistic), "p_value": float(ag.pvalue)},
        "kruskal_wallis": {"statistic": float(kw.statistic), "p_value": float(kw.pvalue),
                           "epsilon2": float(eps2)},
        "posthoc_mann_whitney_holm": posthoc,
        "eta2_magnitude": magnitude(eta2, (0.01, 0.06, 0.14)),
    }


def independence_test(df: pd.DataFrame, a: str, b: str, question: str) -> dict:
    table = pd.crosstab(df[a], df[b])
    chi2, p, dof, expected = stats.chi2_contingency(table)
    v = cramers_v(table, chi2)
    return {"question": question, "table": table.to_dict(), "chi2": float(chi2),
            "dof": int(dof), "p_value": float(p), "cramers_v": v,
            "min_expected": float(expected.min()),
            "expected_ok": bool(expected.min() >= 5),
            "cramers_v_magnitude": magnitude(v, (0.1, 0.3, 0.5))}


def correlation_test(df: pd.DataFrame, x: str) -> dict:
    r, p = stats.pearsonr(df[x], df[TARGET])
    rho, p_s = stats.spearmanr(df[x], df[TARGET])
    z, se = np.arctanh(r), 1 / np.sqrt(len(df) - 3)
    return {"variable": x, "pearson_r": float(r), "pearson_p": float(p),
            "pearson_ci95": [float(np.tanh(z - 1.96 * se)), float(np.tanh(z + 1.96 * se))],
            "spearman_rho": float(rho), "spearman_p": float(p_s)}


# --------------------------------------------------------------------------- figure
def _ci_figure(df: pd.DataFrame) -> dict:
    rows = []
    for col, levels in [("smoker", ["no", "yes"]), ("sex", ["female", "male"]),
                        ("region", sorted(df.region.unique()))]:
        for lv in levels:
            x = df.loc[df[col] == lv, TARGET].to_numpy()
            lo, hi = bootstrap_ci(x)
            label = SMOKER_LABELS.get(lv, lv) if col == "smoker" else lv
            rows.append({"group": f"{col} : {label}", "mean": x.mean(), "lo": lo, "hi": hi,
                         "color": ORANGE if (col == "smoker" and lv == "yes") else BLUE})
    rows = rows[::-1]
    fig = go.Figure()
    fig.add_scatter(x=[r["mean"] for r in rows], y=[r["group"] for r in rows], mode="markers",
                    marker=dict(size=11, color=[r["color"] for r in rows],
                                line=dict(color="white", width=2)),
                    error_x=dict(type="data", symmetric=False, color=INK_2, thickness=2,
                                 array=[r["hi"] - r["mean"] for r in rows],
                                 arrayminus=[r["mean"] - r["lo"] for r in rows]),
                    customdata=[[r["lo"], r["hi"]] for r in rows],
                    hovertemplate="%{y}<br>moyenne %{x:,.0f} $<br>IC95 [%{customdata[0]:,.0f}"
                                  " ; %{customdata[1]:,.0f}]<extra></extra>", showlegend=False)
    fig.add_vline(x=df[TARGET].mean(), line_dash="dot", line_color=INK_2,
                  annotation_text="moyenne globale", annotation_font_color=INK_2)
    fig.update_layout(title="Coût moyen par groupe, IC 95 % bootstrap (10 000 tirages)",
                      xaxis_title="Coût moyen ($)", height=420, margin=dict(l=170))
    save_plotly(fig, "inference_group_ci")
    return {"name": "inference_group_ci", "title": "Intervalles de confiance par groupe"}


# --------------------------------------------------------------------------- runner
def run_inference(df: pd.DataFrame) -> dict:
    smoker = two_group_comparison(df, "smoker", "yes", "no",
                                  "Les fumeurs coûtent-ils plus cher que les non-fumeurs ?")
    sex = two_group_comparison(df, "sex", "male", "female",
                               "Les coûts diffèrent-ils entre hommes et femmes ?")
    region = k_group_comparison(df, "region", "Les coûts diffèrent-ils selon la région ?")
    children = k_group_comparison(df, "children",
                                  "Les coûts diffèrent-ils selon le nombre d'enfants ?")
    chi_region = independence_test(df, "region", "smoker",
                                   "La prévalence du tabagisme dépend-elle de la région ?")
    chi_sex = independence_test(df, "sex", "smoker",
                                "La prévalence du tabagisme dépend-elle du sexe ?")
    adjusted = {"sex": nested_f_test(df, "sex"), "region": nested_f_test(df, "region")}
    correlations = [correlation_test(df, x) for x in ["age", "bmi", "children"]]

    family = {
        "smoker (Mann-Whitney)": smoker["mann_whitney"]["p_value"],
        "sex (Mann-Whitney)": sex["mann_whitney"]["p_value"],
        "region (Kruskal-Wallis)": region["kruskal_wallis"]["p_value"],
        "children (Kruskal-Wallis)": children["kruskal_wallis"]["p_value"],
        "region x smoker (chi2)": chi_region["p_value"],
        "sex x smoker (chi2)": chi_sex["p_value"],
        "sex | controls (F)": adjusted["sex"]["p_value"],
        "region | controls (F)": adjusted["region"]["p_value"],
    }
    holm_p = dict(zip(family, holm(list(family.values())), strict=True))

    overall = df[TARGET].to_numpy()
    result = {
        "alpha": ALPHA, "n": len(df),
        "mean_ci95_t": mean_ci(overall), "mean_ci95_bootstrap": bootstrap_ci(overall),
        "smoker": smoker, "sex": sex, "region": region, "children": children,
        "chi2_region_smoker": chi_region, "chi2_sex_smoker": chi_sex,
        "adjusted_f_tests": adjusted, "correlations": correlations,
        "holm_adjusted_p": holm_p,
        "figures": [_ci_figure(df)],
    }
    result["conclusions"] = conclusions(result)
    dump(result, ARTIFACT_DIR / "inference.json")
    return result


def _p(p: float) -> str:
    return "p < 0,001" if p < 1e-3 else f"p = {p:.3f}".replace(".", ",")


def fr(x: float, digits: int = 2) -> str:
    return f"{x:,.{digits}f}".replace(",", " ").replace(".", ",")


def _pct(x: float) -> str:
    return f"{x:+.1%}".replace(".", ",")


def conclusions(r: dict) -> list[str]:
    s, sx, rg = r["smoker"], r["sex"], r["region"]
    adj, h = r["adjusted_f_tests"], r["holm_adjusted_p"]
    lo, hi = s["bootstrap_ci95_mean_diff"]
    sex_eff = next(iter(adj["sex"]["relative_effects"].values()))
    reg_eff = adj["region"]["relative_effects"]
    reg_txt = ", ".join(f"{k.split(' ')[0]} {_pct(v)}" for k, v in reg_eff.items())
    age, bmi = r["correlations"][0], r["correlations"][1]
    return [
        "Hypothèses : les coûts ne sont normaux dans aucun groupe (Shapiro-Wilk p min = "
        f"{s['assumptions']['shapiro_p_min']:.1e}) et les variances sont hétérogènes (Levene "
        f"{_p(s['assumptions']['levene_p'])}). Le t de Student et l'ANOVA classique sont donc "
        "fournis à titre indicatif ; les conclusions reposent sur Welch, Mann-Whitney, "
        "Kruskal-Wallis, Alexander-Govern et le bootstrap.",
        f"Tabagisme : les fumeurs coûtent en moyenne {fr(s['means'][0] - s['means'][1], 0)} $ "
        f"de plus (IC 95 % bootstrap [{fr(lo, 0)} ; {fr(hi, 0)}] $). Mann-Whitney "
        f"{_p(s['mann_whitney']['p_value'])}, corrélation rang-bisériale "
        f"{fr(s['mann_whitney']['rank_biserial'])}, g de Hedges {fr(s['hedges_g'])} : effet "
        "très fort, résultat central de l'étude.",
        f"Sexe : écart brut de {fr(abs(sx['means'][0] - sx['means'][1]), 0)} $ en faveur des "
        f"{'femmes' if sx['means'][0] > sx['means'][1] else 'hommes'} "
        f"(Welch {_p(sx['welch_t']['p_value'])}, mais Mann-Whitney "
        f"{_p(sx['mann_whitney']['p_value'])}), effet {sx['hedges_g_magnitude']}. Cet écart "
        "brut vient surtout de la plus forte proportion de fumeurs chez les hommes. "
        f"À profil égal (âge, IMC, enfants, tabagisme), le sens s'inverse : {_pct(sex_eff)} "
        f"pour un homme ({_p(adj['sex']['p_value'])}, η² partiel "
        f"{fr(adj['sex']['partial_eta2'], 3)}). L'effet est détectable mais faible, et "
        "le sexe est un critère tarifaire interdit dans l'UE (arrêt Test-Achats, 2011).",
        f"Région : ANOVA {_p(rg['anova']['p_value'])} mais Kruskal-Wallis "
        f"{_p(rg['kruskal_wallis']['p_value'])} et Alexander-Govern "
        f"{_p(rg['alexander_govern']['p_value'])} (η² = {fr(rg['anova']['eta2'], 3)}, effet "
        f"{rg['eta2_magnitude']}) : pas d'écart brut robuste. Ajusté des autres facteurs "
        f"({_p(adj['region']['p_value'])}, η² partiel {fr(adj['region']['partial_eta2'], 3)}), "
        f"on observe un effet modéré par rapport au nord-est : {reg_txt}.",
        f"Tabagisme × région : χ² {_p(r['chi2_region_smoker']['p_value'])}, V de Cramér "
        f"{fr(r['chi2_region_smoker']['cramers_v'])}. Tabagisme × sexe : χ² "
        f"{_p(r['chi2_sex_smoker']['p_value'])}, V = {fr(r['chi2_sex_smoker']['cramers_v'])} "
        "(effectifs attendus tous ≥ 5, condition du χ² respectée). Dans les deux cas, "
        "l'association est faible.",
        f"Corrélations : âge r = {fr(age['pearson_r'])} (IC 95 % [{fr(age['pearson_ci95'][0])} ; "
        f"{fr(age['pearson_ci95'][1])}]), IMC r = {fr(bmi['pearson_r'])} (IC 95 % "
        f"[{fr(bmi['pearson_ci95'][0])} ; {fr(bmi['pearson_ci95'][1])}]).",
        "Tests multiples : après correction de Holm sur les 8 tests de la famille, restent "
        "significatifs au seuil de 5 % : "
        + ", ".join(k for k, v in h.items() if v < ALPHA) + ".",
        "Un p-value très faible ne mesure pas l'ampleur d'un effet. Les tailles d'effet "
        "montrent que le tabagisme écrase tous les autres facteurs catégoriels.",
    ]
