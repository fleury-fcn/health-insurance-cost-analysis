from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_DATA = ROOT / "data" / "raw" / "insurance.csv"
PROCESSED_DATA = ROOT / "data" / "processed" / "insurance_clean.csv"
REPORT_DIR = ROOT / "reports" / "generated"
MODEL_DIR = ROOT / "models"
ARTIFACT_DIR = ROOT / "artifacts"
DATA_URL = (
    "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/"
    "master/insurance.csv"
)
