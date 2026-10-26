"""EDA, inferential statistics and visualisations."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats

from .config import ARTIFACT_DIR, REPORT_DIR


def run_analysis(df: pd.DataFrame) -> dict:
    ARTIFACT_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.histplot(df["charges"], kde=True, ax=axes[0], color="#2563eb")
    axes[0].set_title("Distribution des charges")
    sns.boxplot(data=df, x="smoker", y="charges", ax=axes[1], palette="Set2")
    axes[1].set_title("Charges selon le statut tabagique")
    fig.tight_layout()
    fig.savefig(ARTIFACT_DIR / "eda_distributions.png", dpi=150)
    plt.close(fig)

    smokers = df.loc[df.smoker == "yes", "charges"]
    non_smokers = df.loc[df.smoker == "no", "charges"]
    t_stat, t_p = stats.ttest_ind(smokers, non_smokers, equal_var=False)
    u_stat, u_p = stats.mannwhitneyu(smokers, non_smokers, alternative="two-sided")
    pooled_sd = ((smokers.var() + non_smokers.var()) / 2) ** 0.5
    effect = float((smokers.mean() - non_smokers.mean()) / pooled_sd)
    anova = stats.f_oneway(*(g["charges"] for _, g in df.groupby("region")))
    contingency = pd.crosstab(df["region"], df["smoker"])
    chi2, chi_p, _, _ = stats.chi2_contingency(contingency)
    result = {
        "n": int(len(df)),
        "mean_charges": float(df.charges.mean()),
        "median_charges": float(df.charges.median()),
        "skewness": float(df.charges.skew()),
        "smoker_mean": float(smokers.mean()),
        "non_smoker_mean": float(non_smokers.mean()),
        "welch_t": {"statistic": float(t_stat), "p_value": float(t_p)},
        "mann_whitney": {"statistic": float(u_stat), "p_value": float(u_p)},
        "cohens_d": effect,
        "anova_region": {"statistic": float(anova.statistic), "p_value": float(anova.pvalue)},
        "chi2_region_smoker": {"statistic": float(chi2), "p_value": float(chi_p)},
    }
    (ARTIFACT_DIR / "statistics.json").write_text(json.dumps(result, indent=2))
    return result
