"""Central configuration: paths, seed and data source metadata."""

from pathlib import Path

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

ROOT = Path(__file__).resolve().parents[2]
RAW_DATA = ROOT / "data" / "raw" / "insurance.csv"
PROCESSED_DATA = ROOT / "data" / "processed" / "insurance_clean.csv"
REPORT_DIR = ROOT / "reports" / "generated"
MODEL_DIR = ROOT / "models"
BEST_MODEL = MODEL_DIR / "best_model.joblib"
ARTIFACT_DIR = ROOT / "artifacts"
FIGURE_DIR = ARTIFACT_DIR / "figures"
MLFLOW_DIR = ARTIFACT_DIR / "mlruns"

DATA_URL = (
    "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/"
    "master/insurance.csv"
)
DATA_SHA256 = "505c1cbc2e63d0363bac59501563df2530aadf4cdb9cfee226f4ef32f5468281"

TARGET = "charges"
NUMERIC = ["age", "bmi", "children"]
CATEGORICAL = ["sex", "smoker", "region"]
FEATURES = ["age", "sex", "bmi", "children", "smoker", "region"]
CATEGORY_LEVELS = {
    "sex": ["female", "male"],
    "smoker": ["no", "yes"],
    "region": ["northeast", "northwest", "southeast", "southwest"],
}
OBESITY_THRESHOLD = 30.0  # WHO definition of obesity (BMI >= 30)
