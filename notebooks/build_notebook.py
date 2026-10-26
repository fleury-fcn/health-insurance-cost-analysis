"""Build and execute the walkthrough notebook (`make notebook`).

The notebook is a showcase: every number comes from the package or from the artifacts of
the last `make run`, so it cannot drift from the report.
"""

from pathlib import Path

import nbformat
from nbclient import NotebookClient

HERE = Path(__file__).resolve().parent
TARGET = HERE / "01_walkthrough.ipynb"

md = nbformat.v4.new_markdown_cell
code = nbformat.v4.new_code_cell

cells = [
    md("# Coûts d'assurance maladie : visite guidée\n\n"
       "Ce notebook présente les résultats du pipeline (`make run`). Il n'entraîne rien : il "
       "appelle le package `insurance_costs` et lit les artefacts. La source de vérité reste "
       "le code testé dans `src/`. Le rapport complet est publié sur "
       "[GitHub Pages](https://fleury-fcn.github.io/health-insurance-cost-analysis/)."),
    code("import json, sys\nfrom pathlib import Path\nimport pandas as pd\n"
         "from IPython.display import Image, Markdown, display\n"
         "ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()\n"
         "sys.path.insert(0, str(ROOT / 'src'))\n"
         "from insurance_costs.data import load_clean\n"
         "from insurance_costs.config import ARTIFACT_DIR, FIGURE_DIR\n"
         "load = lambda name: json.loads((ARTIFACT_DIR / name).read_text(encoding='utf-8'))\n"
         "pd.set_option('display.float_format', '{:,.2f}'.format)\n"
         "df = load_clean()\ndf.head()"),
    md("## 1. Qualité des données\nUn doublon exact supprimé, aucune valeur manquante ni "
       "impossible. Les coûts extrêmes sont conservés : ce sont surtout des fumeurs."),
    code("q = load('data_quality.json')\n"
         "pd.DataFrame(q['cleaning_log'])[['action', 'rows_removed', 'reason']]"),
    code("q['charge_outlier_profile']"),
    md("## 2. Exploration\nTrois bandes de coûts selon l'âge, et une rupture à IMC 30 "
       "réservée aux fumeurs : c'est une **interaction**."),
    code("display(Image(FIGURE_DIR / 'eda_overview.png'))\n"
         "display(Image(FIGURE_DIR / 'eda_segments.png'))"),
    code("df.groupby(['smoker', df.bmi >= 30]).charges.agg(['count', 'mean', 'median'])"),
    md("## 3. Inférence\nHypothèses vérifiées (non-normalité, hétéroscédasticité), donc "
       "tests robustes, tailles d'effet et IC."),
    code("for c in load('inference.json')['conclusions']:\n    display(Markdown('- ' + c))"),
    md("## 4. Feature engineering (ablation en CV)"),
    code("pd.DataFrame(load('feature_ablation.json')).T[['cv_rmse', 'cv_r2']]"),
    md("## 5. Modèles\nSélection sur la RMSE de validation croisée, évaluation unique sur "
       "le test."),
    code("m = load('metrics.json')\n"
         "rows = {n: {'cv_rmse': v['cv_rmse'], **{k: v['test'][k] for k in "
         "['rmse', 'mae', 'r2', 'gini', 'calibration_ratio']}} for n, v in m['models'].items()}\n"
         "print('Meilleur modèle :', m['best_model'])\n"
         "pd.DataFrame(rows).T.sort_values('cv_rmse')"),
    code("display(Image(FIGURE_DIR / 'model_overview.png'))\nm['best_vs_runner_up']"),
    md("## 6. Interprétation et équité"),
    code("display(Image(FIGURE_DIR / 'interpretation_overview.png'))\n"
         "for i in load('interpretation.json')['insights']:\n    display(Markdown('- ' + i))"),
    md("## 7. Expliquer un profil\nLa même fonction alimente l'app Streamlit et l'API."),
    code("import joblib\nfrom insurance_costs.config import BEST_MODEL\n"
         "from insurance_costs.interpretation import explain_profile\n"
         "model = joblib.load(BEST_MODEL)\n"
         "profile = pd.DataFrame([dict(age=45, sex='male', bmi=32.5, children=2, "
         "smoker='yes', region='southeast')])\n"
         "exp = explain_profile(model, profile, df.drop(columns='charges'))\n"
         "print(f\"Prédiction : {exp['prediction']:,.0f} $ (base {exp['base_value']:,.0f} $)\")\n"
         "pd.Series(exp['contributions']).sort_values(key=abs, ascending=False)"
         ".to_frame('contribution ($)')"),
]

nb = nbformat.v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
    "language_info": {"name": "python"}})
NotebookClient(nb, timeout=300, kernel_name="python3",
               resources={"metadata": {"path": str(HERE)}}).execute()
# Drop stderr streams (library warnings): they are noise and embed local file paths
for cell in nb.cells:
    if cell.cell_type == "code":
        cell.outputs = [o for o in cell.outputs
                        if not (o.get("output_type") == "stream" and o.get("name") == "stderr")]
if str(HERE.parent) in nbformat.writes(nb):
    raise RuntimeError("Le notebook contient un chemin local : publication refusée.")
nbformat.write(nb, TARGET)
print(f"Notebook exécuté : {TARGET.relative_to(HERE.parent)}")
