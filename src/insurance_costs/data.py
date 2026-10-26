"""Data acquisition, quality audit and documented cleaning."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd

from .config import (
    ARTIFACT_DIR,
    CATEGORY_LEVELS,
    DATA_SHA256,
    DATA_URL,
    NUMERIC,
    PROCESSED_DATA,
    RAW_DATA,
    TARGET,
)

EXPECTED_COLUMNS = ["age", "sex", "bmi", "children", "smoker", "region", "charges"]

# Plausibility bounds. Values outside are data errors, not rare risks.
PLAUSIBLE_RANGES = {
    "age": (18, 100),  # adult policyholders
    "bmi": (10, 80),  # beyond these values a BMI is physiologically implausible
    "children": (0, 20),
    "charges": (0, 1e7),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_raw(path: Path = RAW_DATA, verify: bool = True) -> pd.DataFrame:
    """Download the raw file once, verify its checksum and load it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with urlopen(DATA_URL, timeout=30) as response:
            path.write_bytes(response.read())
    if verify and path == RAW_DATA and sha256(path) != DATA_SHA256:
        raise ValueError(
            f"Empreinte inattendue pour {path}: la source a changé, résultats non reproductibles."
        )
    return pd.read_csv(path)


def _iqr_outliers(series: pd.Series) -> dict:
    q1, q3 = series.quantile([0.25, 0.75])
    iqr = q3 - q1
    low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    mask = (series < low) | (series > high)
    return {"lower_fence": float(low), "upper_fence": float(high), "count": int(mask.sum()),
            "share": float(mask.mean())}


def _robust_z_outliers(series: pd.Series, threshold: float = 3.5) -> int:
    """Modified z-score (Iglewicz & Hoaglin) using the median absolute deviation."""
    median = series.median()
    mad = (series - median).abs().median()
    if mad == 0:
        return 0
    z = 0.6745 * (series - median) / mad
    return int((z.abs() > threshold).sum())


def audit(df: pd.DataFrame) -> dict:
    """Return a structured quality report of a raw dataframe."""
    present = [c for c in EXPECTED_COLUMNS if c in df.columns]
    report: dict = {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "missing_columns": sorted(set(EXPECTED_COLUMNS) - set(df.columns)),
        "unexpected_columns": sorted(set(df.columns) - set(EXPECTED_COLUMNS)),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "duplicates": int(df.duplicated().sum()),
        "duplicate_rows": df[df.duplicated(keep=False)].to_dict(orient="records"),
        "missing_by_column": {c: int(v) for c, v in df[present].isna().sum().items()},
        "missing_values": int(df[present].isna().sum().sum()),
    }
    report["unexpected_categories"] = {
        col: sorted(set(df[col].dropna().astype(str).str.strip().str.lower()) - set(levels))
        for col, levels in CATEGORY_LEVELS.items() if col in df
    }
    report["out_of_range"] = {
        col: int(((pd.to_numeric(df[col], errors="coerce") < lo)
                  | (pd.to_numeric(df[col], errors="coerce") > hi)).sum())
        for col, (lo, hi) in PLAUSIBLE_RANGES.items() if col in df
    }
    report["non_numeric"] = {
        col: int(pd.to_numeric(df[col], errors="coerce").isna().sum() - df[col].isna().sum())
        for col in NUMERIC + [TARGET] if col in df
    }
    numeric = [c for c in NUMERIC + [TARGET] if c in df]
    report["describe"] = json.loads(df[numeric].describe().round(3).to_json())
    report["outliers_iqr"] = {c: _iqr_outliers(df[c]) for c in numeric}
    report["outliers_robust_z"] = {c: _robust_z_outliers(df[c]) for c in numeric}
    return report


def clean_data(df: pd.DataFrame, output_path: Path | None = PROCESSED_DATA
               ) -> tuple[pd.DataFrame, list[dict]]:
    """Clean a raw dataframe and return it with a log of every decision taken."""
    missing = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes: {sorted(missing)}")
    log: list[dict] = []

    def step(action: str, reason: str, before: int, after: int) -> None:
        log.append({"action": action, "reason": reason, "rows_removed": before - after,
                    "rows_after": after})

    clean = df[EXPECTED_COLUMNS].copy()
    n = len(clean)
    for col in CATEGORY_LEVELS:
        clean[col] = clean[col].astype(str).str.strip().str.lower()
    for col in NUMERIC + [TARGET]:
        clean[col] = pd.to_numeric(clean[col], errors="coerce")
    step("Normalisation des types et des libellés",
         "Libellés en minuscules sans espaces ; conversion numérique stricte.", n, len(clean))

    n = len(clean)
    clean = clean.drop_duplicates()
    step("Suppression des doublons exacts",
         "Sans identifiant, deux lignes identiques au centime près sur les 7 variables sont "
         "presque certainement une double saisie : la garder sur-pondérerait ce profil.",
         n, len(clean))

    n = len(clean)
    clean = clean.dropna()
    step("Suppression des lignes incomplètes",
         "Aucune imputation : la part de manquants est nulle ou négligeable, et imputer la "
         "cible ou ses déterminants principaux introduirait un biais.", n, len(clean))

    n = len(clean)
    valid = pd.Series(True, index=clean.index)
    for col, (lo, hi) in PLAUSIBLE_RANGES.items():
        valid &= clean[col].between(lo, hi)
    for col, levels in CATEGORY_LEVELS.items():
        valid &= clean[col].isin(levels)
    clean = clean[valid]
    step("Exclusion des valeurs impossibles",
         "Hors bornes de plausibilité ou modalité inconnue : erreurs de saisie.", n, len(clean))

    log.append({"action": "Conservation des valeurs extrêmes de coûts", "rows_removed": 0,
                "rows_after": len(clean),
                "reason": "Les coûts élevés sont le risque que l'assureur couvre. Les "
                          "retirer sous-estimerait la prime pure. Ils sont traités par la "
                          "modélisation (cible log, GLM Gamma, métriques robustes)."})

    clean["age"] = clean["age"].astype(int)
    clean["children"] = clean["children"].astype(int)
    clean = clean.reset_index(drop=True)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        clean.to_csv(output_path, index=False)
    return clean, log


def load_clean(path: Path = PROCESSED_DATA) -> pd.DataFrame:
    return pd.read_csv(path)


def run_quality(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Audit, clean and persist the quality report."""
    report = audit(raw)
    clean, log = clean_data(raw)
    report["cleaning_log"] = log
    report["rows_after_cleaning"] = int(len(clean))
    fence = report["outliers_iqr"][TARGET]["upper_fence"]
    extreme = clean[clean[TARGET] > fence]
    report["charge_outlier_profile"] = {
        "count": int(len(extreme)),
        "share_smokers": float((extreme["smoker"] == "yes").mean()),
        "share_smokers_overall": float((clean["smoker"] == "yes").mean()),
        "share_obese": float((extreme["bmi"] >= 30).mean()),
    }
    report["post_cleaning_checks"] = {
        "duplicates": int(clean.duplicated().sum()),
        "missing_values": int(clean.isna().sum().sum()),
        "min_charges": float(clean[TARGET].min()),
        "share_obese": float(np.mean(clean["bmi"] >= 30)),
    }
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACT_DIR / "data_quality.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return clean, report
