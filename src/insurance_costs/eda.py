"""Exploratory data analysis: univariate, bivariate and multivariate views.

Every figure is paired with an interpretation computed from the data, so the written
conclusions in the report can never drift away from the numbers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
from plotly.subplots import make_subplots
from scipy import stats

from .config import ARTIFACT_DIR, OBESITY_THRESHOLD, TARGET
from .viz import (
    AQUA,
    BLUE,
    DIVERGING,
    INK_2,
    MUTED,
    ORANGE,
    SEQUENTIAL,
    SMOKER_COLORS,
    SMOKER_LABELS,
    dump,
    money,
    plt,
    save_plotly,
    save_png,
)

AGE_BANDS = [17, 29, 39, 49, 64]
AGE_LABELS = ["18-29", "30-39", "40-49", "50-64"]


def add_segments(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["age_band"] = pd.cut(out["age"], AGE_BANDS, labels=AGE_LABELS)
    out["obese"] = np.where(out["bmi"] >= OBESITY_THRESHOLD, "IMC ≥ 30", "IMC < 30")
    out["risk_group"] = np.select(
        [out["smoker"].eq("no"), out["bmi"] < OBESITY_THRESHOLD],
        ["Non-fumeur", "Fumeur, IMC < 30"], "Fumeur, IMC ≥ 30")
    return out


def encoded(df: pd.DataFrame) -> pd.DataFrame:
    """Numeric view of the dataset for correlation analysis."""
    out = df[["age", "bmi", "children", TARGET]].copy()
    out["smoker"] = df["smoker"].eq("yes").astype(int)
    out["sex_male"] = df["sex"].eq("male").astype(int)
    for region in sorted(df["region"].unique()):
        out[f"region_{region}"] = df["region"].eq(region).astype(int)
    return out


def _univariate(df: pd.DataFrame, figures: list, findings: dict) -> None:
    y = df[TARGET]
    skew, log_skew = float(stats.skew(y)), float(stats.skew(np.log(y)))
    top10_share = float(y.nlargest(int(len(y) * 0.1)).sum() / y.sum())
    findings["target"] = {
        "mean": float(y.mean()), "median": float(y.median()), "std": float(y.std()),
        "min": float(y.min()), "max": float(y.max()), "skewness": skew,
        "kurtosis": float(stats.kurtosis(y)), "log_skewness": log_skew,
        "top10_share": top10_share,
        "q": {str(q): float(y.quantile(q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9, 0.99)},
    }
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Échelle naturelle ($)",
                                                         "Échelle logarithmique"))
    fig.add_histogram(x=y, nbinsx=50, marker_color=BLUE, name="Coûts",
                      hovertemplate="%{x}<br>%{y} assurés<extra></extra>", row=1, col=1)
    fig.add_histogram(x=np.log(y), nbinsx=50, marker_color=BLUE, name="log(coûts)",
                      hovertemplate="log = %{x:.2f}<br>%{y} assurés<extra></extra>",
                      row=1, col=2)
    for x, label in [(y.mean(), "moyenne"), (y.median(), "médiane")]:
        fig.add_vline(x=x, line_dash="dot", line_color=INK_2, row=1, col=1,
                      annotation_text=label, annotation_font_color=INK_2)
    fig.update_layout(title="Distribution des coûts annuels", showlegend=False, height=380)
    fig.update_yaxes(title_text="Nombre d'assurés", row=1, col=1)
    save_plotly(fig, "eda_target_distribution")
    figures.append({"name": "eda_target_distribution", "section": "univariate",
                    "title": "Distribution des coûts", "caption": (
        f"Les coûts sont fortement asymétriques à droite (skewness {skew:.2f}) : la moyenne "
        f"({money(y.mean())}) dépasse nettement la médiane ({money(y.median())}), et les 10 % "
        f"d'assurés les plus coûteux concentrent {top10_share:.0%} de la charge totale. "
        f"Après passage au logarithme, la distribution devient quasi symétrique (skewness "
        f"{log_skew:.2f}) mais reste multimodale, signe de sous-populations distinctes. "
        "Cette forme motive une cible log, le GLM Gamma à lien log, et le suivi de la MAE "
        "en plus de la RMSE.")})

    fig = make_subplots(rows=1, cols=3, subplot_titles=("Âge", "IMC", "Enfants à charge"))
    fig.add_histogram(x=df["age"], nbinsx=47, marker_color=BLUE, row=1, col=1)
    fig.add_histogram(x=df["bmi"], nbinsx=40, marker_color=BLUE, row=1, col=2)
    counts = df["children"].value_counts().sort_index()
    fig.add_bar(x=counts.index, y=counts.values, marker_color=BLUE, row=1, col=3)
    fig.add_vline(x=OBESITY_THRESHOLD, line_dash="dot", line_color=INK_2, row=1, col=2,
                  annotation_text="obésité", annotation_font_color=INK_2)
    fig.update_traces(hovertemplate="%{x}<br>%{y} assurés<extra></extra>")
    fig.update_layout(title="Distribution des variables numériques", showlegend=False,
                      height=340)
    save_plotly(fig, "eda_numeric_features")
    young = float((df["age"] <= 19).mean())
    findings["features"] = {
        "share_age_18_19": young, "bmi_mean": float(df["bmi"].mean()),
        "share_obese": float((df["bmi"] >= OBESITY_THRESHOLD).mean()),
        "bmi_skewness": float(stats.skew(df["bmi"])),
        "share_smokers": float(df["smoker"].eq("yes").mean()),
        "share_male": float(df["sex"].eq("male").mean()),
        "region_shares": df["region"].value_counts(normalize=True).round(3).to_dict(),
    }
    figures.append({"name": "eda_numeric_features", "section": "univariate",
                    "title": "Variables explicatives numériques", "caption": (
        f"L'âge est quasi uniforme entre 18 et 64 ans, avec un pic à 18-19 ans "
        f"({young:.1%} des assurés), artefact probable de la simulation. L'IMC suit une loi "
        f"proche de la normale (moyenne {df['bmi'].mean():.1f}) ; "
        f"{(df['bmi'] >= OBESITY_THRESHOLD).mean():.0%} des assurés sont obèses (IMC ≥ 30). "
        "Le nombre d'enfants décroît de 0 à 5. Les variables catégorielles sont équilibrées "
        f"(hommes {df['sex'].eq('male').mean():.0%}, régions de 24 à 27 %), sauf le tabagisme : "
        f"{df['smoker'].eq('yes').mean():.0%} de fumeurs.")})


def _bivariate(df: pd.DataFrame, figures: list, findings: dict) -> None:
    group_stats = {}
    for col in ["smoker", "sex", "region", "children"]:
        g = df.groupby(col, observed=True)[TARGET]
        group_stats[col] = {str(k): {"mean": float(v.mean()), "median": float(v.median()),
                                     "n": int(len(v))} for k, v in g}
    findings["groups"] = group_stats
    s = group_stats["smoker"]
    ratio = s["yes"]["mean"] / s["no"]["mean"]
    findings["smoker_ratio"] = ratio

    fig = make_subplots(rows=1, cols=3, subplot_titles=("Tabagisme", "Sexe", "Région"),
                        shared_yaxes=True)
    for i, col in enumerate(["smoker", "sex", "region"], start=1):
        for level in sorted(df[col].unique()):
            color = SMOKER_COLORS[level] if col == "smoker" else BLUE
            label = SMOKER_LABELS.get(level, level) if col == "smoker" else level
            fig.add_box(y=df.loc[df[col] == level, TARGET], name=label, marker_color=color,
                        boxmean=True, showlegend=False, row=1, col=i,
                        hovertemplate=f"{label}<br>%{{y:,.0f}} $<extra></extra>")
    fig.update_yaxes(title_text="Coûts annuels ($)", row=1, col=1)
    fig.update_layout(title="Coûts selon les variables catégorielles", height=400)
    save_plotly(fig, "eda_costs_by_category")
    sx, rg = group_stats["sex"], group_stats["region"]
    top_region = max(rg, key=lambda k: rg[k]["mean"])
    figures.append({"name": "eda_costs_by_category", "section": "bivariate",
                    "title": "Coûts par variable catégorielle", "caption": (
        f"Le tabagisme domine : un fumeur coûte en moyenne {money(s['yes']['mean'])} contre "
        f"{money(s['no']['mean'])} pour un non-fumeur, soit ×{ratio:.1f}. Les écarts par sexe "
        f"(hommes {money(sx['male']['mean'])}, femmes {money(sx['female']['mean'])}) et par "
        f"région (maximum : {top_region}, {money(rg[top_region]['mean'])}) sont modestes. La "
        "phase 4 teste s'ils sont significatifs et s'ils persistent une fois le tabagisme "
        "pris en compte (le sud-est compte plus de fumeurs et d'obèses).")})

    fig = make_subplots(rows=1, cols=2, subplot_titles=("Âge", "IMC"), shared_yaxes=True)
    for level in ["no", "yes"]:
        d = df[df["smoker"] == level]
        for col_i, x in enumerate(["age", "bmi"], start=1):
            fig.add_scatter(x=d[x], y=d[TARGET], mode="markers", name=SMOKER_LABELS[level],
                            legendgroup=level, showlegend=col_i == 1,
                            marker=dict(color=SMOKER_COLORS[level], size=8, opacity=0.7,
                                        line=dict(color="white", width=1)),
                            hovertemplate=f"{x} = %{{x}}<br>%{{y:,.0f}} $<extra>"
                                          f"{SMOKER_LABELS[level]}</extra>", row=1, col=col_i)
    fig.add_vline(x=OBESITY_THRESHOLD, line_dash="dot", line_color=INK_2, row=1, col=2,
                  annotation_text="IMC 30", annotation_font_color=INK_2)
    fig.update_yaxes(title_text="Coûts annuels ($)", row=1, col=1)
    fig.update_layout(title="Coûts en fonction de l'âge et de l'IMC", height=420)
    save_plotly(fig, "eda_costs_vs_numeric")
    slope = stats.linregress(df.loc[df.smoker == "no", "age"],
                             df.loc[df.smoker == "no", TARGET]).slope
    obese = df["bmi"] >= OBESITY_THRESHOLD
    sm = df["smoker"] == "yes"
    jump_smokers = df.loc[sm & obese, TARGET].mean() - df.loc[sm & ~obese, TARGET].mean()
    jump_non = df.loc[~sm & obese, TARGET].mean() - df.loc[~sm & ~obese, TARGET].mean()
    findings["age_slope_non_smokers"] = float(slope)
    findings["obesity_jump"] = {"smokers": float(jump_smokers), "non_smokers": float(jump_non)}
    figures.append({"name": "eda_costs_vs_numeric", "section": "bivariate",
                    "title": "Âge, IMC et coûts", "caption": (
        f"Les coûts croissent avec l'âge de façon régulière (≈ {money(slope)} par année chez "
        "les non-fumeurs), en trois bandes parallèles : non-fumeurs, fumeurs non obèses et "
        "fumeurs obèses. L'IMC n'a presque pas d'effet chez les non-fumeurs (+"
        f"{money(jump_non)} au-delà de 30) mais crée une rupture chez les fumeurs (+"
        f"{money(jump_smokers)}). C'est une **interaction** : l'effet de l'IMC dépend du "
        "tabagisme. Un modèle additif ne peut pas la capturer sans un terme dédié "
        "(phase 5).")})

    enc = encoded(df)
    pearson, spearman = enc.corr(method="pearson"), enc.corr(method="spearman")
    findings["correlation_with_target"] = {
        "pearson": pearson[TARGET].drop(TARGET).round(3).to_dict(),
        "spearman": spearman[TARGET].drop(TARGET).round(3).to_dict()}
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Pearson (linéaire)",
                                                         "Spearman (rangs)"),
                        horizontal_spacing=0.18)
    for i, mat in enumerate([pearson, spearman], start=1):
        fig.add_heatmap(z=mat.values, x=mat.columns, y=mat.index, zmin=-1, zmax=1,
                        colorscale=DIVERGING, text=np.round(mat.values, 2),
                        texttemplate="%{text}", textfont=dict(size=9),
                        showscale=i == 2, hovertemplate="%{y} × %{x}<br>r = %{z:.2f}"
                                                        "<extra></extra>", row=1, col=i)
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(title="Matrices de corrélation (variables encodées)", height=520)
    save_plotly(fig, "eda_correlations")
    # Variance inflation factors (reference region dropped to avoid perfect collinearity)
    X = enc.drop(columns=[TARGET, "region_southwest"])
    vif = dict(zip(X.columns, np.diag(np.linalg.inv(X.corr().values)), strict=True))
    findings["vif"] = {k: round(float(v), 3) for k, v in vif.items()}
    pc = findings["correlation_with_target"]["pearson"]
    sc = findings["correlation_with_target"]["spearman"]
    figures.append({"name": "eda_correlations", "section": "bivariate",
                    "title": "Corrélations", "caption": (
        f"Corrélation avec les coûts : tabagisme r = {pc['smoker']:.2f}, âge r = "
        f"{pc['age']:.2f}, IMC r = {pc['bmi']:.2f}, les autres variables sont proches de 0. "
        f"En rangs (Spearman), l'âge devient le premier facteur (ρ = {sc['age']:.2f}) : "
        "il ordonne régulièrement les coûts, tandis que le tabagisme crée les écarts les "
        "plus importants en montant. Les variables explicatives sont peu corrélées entre "
        f"elles (VIF max {max(vif.values()):.2f}, très en dessous du seuil usuel de 5) : "
        "il n'y a pas de multicolinéarité à traiter pour les modèles linéaires.")})


def _multivariate(df: pd.DataFrame, figures: list, findings: dict) -> None:
    seg = add_segments(df)
    pivot = seg.pivot_table(index="risk_group", columns="age_band", values=TARGET,
                            aggfunc="mean", observed=True)
    order = ["Non-fumeur", "Fumeur, IMC < 30", "Fumeur, IMC ≥ 30"]
    pivot = pivot.loc[order]
    counts = seg.groupby("risk_group")[TARGET].agg(["mean", "count"]).loc[order]
    findings["risk_groups"] = {k: {"mean": float(v["mean"]), "n": int(v["count"])}
                               for k, v in counts.iterrows()}
    fig = go.Figure()
    for group, color in zip(order, [BLUE, AQUA, ORANGE], strict=True):
        fig.add_bar(x=pivot.columns.astype(str), y=pivot.loc[group], name=group,
                    marker=dict(color=color, line=dict(color="white", width=2)),
                    hovertemplate=f"{group}<br>%{{x}} ans<br>%{{y:,.0f}} $<extra></extra>")
    fig.update_layout(barmode="group", title="Coût moyen par segment de risque et âge",
                      xaxis_title="Tranche d'âge", yaxis_title="Coût moyen ($)", height=400)
    save_plotly(fig, "eda_risk_segments")
    r = findings["risk_groups"]
    figures.append({"name": "eda_risk_segments", "section": "multivariate",
                    "title": "Segments de risque", "caption": (
        "En combinant trois variables, on obtient une segmentation tarifaire lisible : "
        f"non-fumeurs ({money(r['Non-fumeur']['mean'])} en moyenne, n = "
        f"{r['Non-fumeur']['n']}), fumeurs non obèses ({money(r['Fumeur, IMC < 30']['mean'])}"
        f", n = {r['Fumeur, IMC < 30']['n']}) et fumeurs obèses "
        f"({money(r['Fumeur, IMC ≥ 30']['mean'])}, n = {r['Fumeur, IMC ≥ 30']['n']}). Dans "
        "chaque segment, l'âge ajoute un gradient régulier. Ce couple tabagisme × obésité, "
        "puis l'âge, structure la suite de la modélisation.")})

    resid_df = df.copy()
    resid_df["log_charges"] = np.log(resid_df[TARGET])
    heat = resid_df.assign(bmi_band=pd.cut(df["bmi"], [0, 18.5, 25, 30, 35, 100],
                                           labels=["<18.5", "18.5-25", "25-30", "30-35",
                                                   "≥35"]),
                           age_band=pd.cut(df["age"], AGE_BANDS, labels=AGE_LABELS))
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Non-fumeurs", "Fumeurs"),
                        shared_yaxes=True)
    for i, level in enumerate(["no", "yes"], start=1):
        p = heat[heat.smoker == level].pivot_table(index="bmi_band", columns="age_band",
                                                   values=TARGET, aggfunc="median",
                                                   observed=False)
        fig.add_heatmap(z=p.values, x=p.columns.astype(str), y=p.index.astype(str),
                        colorscale=[[k / 5, c] for k, c in enumerate(SEQUENTIAL)],
                        zmin=0, zmax=float(df[TARGET].quantile(0.95)), showscale=i == 2,
                        text=np.vectorize(lambda v: "" if np.isnan(v) else f"{v/1000:.0f}k")(
                            p.values), texttemplate="%{text}",
                        hovertemplate="IMC %{y}, %{x} ans<br>médiane %{z:,.0f} $"
                                      "<extra></extra>", row=1, col=i)
    fig.update_layout(title="Coût médian par IMC × âge, selon le tabagisme", height=380)
    fig.update_yaxes(title_text="Classe d'IMC", row=1, col=1)
    save_plotly(fig, "eda_heatmap_bmi_age")
    figures.append({"name": "eda_heatmap_bmi_age", "section": "multivariate",
                    "title": "Interaction IMC × âge × tabagisme", "caption": (
        "La carte de chaleur confirme l'interaction à trois variables. Chez les non-fumeurs, "
        "la couleur varie surtout de gauche à droite (effet de l'âge). Chez les fumeurs, "
        "elle bascule nettement au-dessus de 30 d'IMC, et ce quel que soit l'âge. Un modèle "
        "à arbres capture cette structure automatiquement ; un modèle linéaire ou un GLM "
        "doit la recevoir comme variable construite.")})


def _static_figures(df: pd.DataFrame) -> None:
    """A few PNG figures for the README and the notebook."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    sns.histplot(df[TARGET], bins=50, ax=axes[0], color=BLUE, edgecolor="white")
    axes[0].axvline(df[TARGET].median(), color=INK_2, ls=":", lw=1.5)
    axes[0].set(title="Distribution des coûts (asymétrie à droite)", xlabel="Coûts ($)",
                ylabel="Assurés")
    for level in ["no", "yes"]:
        d = df[df.smoker == level]
        axes[1].scatter(d["age"], d[TARGET], s=16, alpha=0.7, color=SMOKER_COLORS[level],
                        edgecolor="white", linewidth=0.4, label=SMOKER_LABELS[level])
        axes[2].scatter(d["bmi"], d[TARGET], s=16, alpha=0.7, color=SMOKER_COLORS[level],
                        edgecolor="white", linewidth=0.4, label=SMOKER_LABELS[level])
    axes[1].set(title="Trois bandes de coûts selon l'âge", xlabel="Âge", ylabel="Coûts ($)")
    axes[2].axvline(OBESITY_THRESHOLD, color=MUTED, ls=":", lw=1.5)
    axes[2].set(title="Rupture à IMC 30 chez les fumeurs", xlabel="IMC", ylabel="Coûts ($)")
    axes[1].legend(loc="upper left")
    fig.tight_layout()
    save_png(fig, "eda_overview")

    seg = add_segments(df)
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(data=seg, x="age_band", y=TARGET, hue="risk_group", errorbar=("ci", 95),
                hue_order=["Non-fumeur", "Fumeur, IMC < 30", "Fumeur, IMC ≥ 30"],
                palette=[BLUE, AQUA, ORANGE], ax=ax, edgecolor="white", linewidth=1.5)
    ax.set(title="Coût moyen par segment (IC 95 %)", xlabel="Tranche d'âge",
           ylabel="Coût moyen ($)")
    ax.legend(title=None)
    save_png(fig, "eda_segments")


def run_eda(df: pd.DataFrame) -> dict:
    figures: list[dict] = []
    findings: dict = {}
    _univariate(df, figures, findings)
    _bivariate(df, figures, findings)
    _multivariate(df, figures, findings)
    _static_figures(df)
    result = {"findings": findings, "figures": figures}
    dump(result, ARTIFACT_DIR / "eda.json")
    return result
