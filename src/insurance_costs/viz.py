"""Shared visual identity for Plotly (report/app) and matplotlib (static PNG) charts.

Categorical slots follow a CVD-validated order (blue, orange, aqua); only the first
three are used on scatter-type charts, where every pair must stay distinguishable.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import plotly.io as pio  # noqa: E402

from .config import FIGURE_DIR  # noqa: E402

BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
SERIES = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA, "#008300", "#6250d6", "#e34948"]
SMOKER_COLORS = {"no": BLUE, "yes": ORANGE}
SMOKER_LABELS = {"no": "Non-fumeur", "yes": "Fumeur"}
INK, INK_2, MUTED, GRID, AXIS, SURFACE = (
    "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb")
# Diverging blue <-> red with neutral grey midpoint (correlations)
DIVERGING = [[0.0, "#1c5cab"], [0.25, "#86b6ef"], [0.5, "#f0efec"],
             [0.75, "#f09a96"], [1.0, "#b8302f"]]
SEQUENTIAL = ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"]
FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'

pio.templates["insurance"] = go.layout.Template(layout=dict(
    font=dict(family=FONT, color=INK, size=13),
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    colorway=SERIES,
    title=dict(font=dict(size=15, color=INK), x=0, xanchor="left", y=0.98, yanchor="top"),
    xaxis=dict(gridcolor=GRID, linecolor=AXIS, zeroline=False, tickfont=dict(color=INK_2),
               title=dict(font=dict(color=INK_2)), automargin=True),
    yaxis=dict(gridcolor=GRID, linecolor=AXIS, zeroline=False, tickfont=dict(color=INK_2),
               title=dict(font=dict(color=INK_2)), automargin=True),
    # Legend top-right, above subplot titles, so it never collides with them
    legend=dict(orientation="h", yanchor="bottom", y=1.1, xanchor="right", x=1,
                font=dict(color=INK_2)),
    hoverlabel=dict(bgcolor="white", font=dict(family=FONT, color=INK), bordercolor=GRID),
    margin=dict(l=60, r=20, t=95, b=50),
    bargap=0.25,
))
pio.templates.default = "insurance"

plt.rcParams.update({
    "font.family": "sans-serif", "axes.edgecolor": AXIS, "axes.labelcolor": INK_2,
    "axes.titlecolor": INK, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "xtick.color": INK_2, "ytick.color": INK_2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "white",
    "axes.facecolor": "white", "legend.frameon": False, "savefig.dpi": 150,
    "savefig.bbox": "tight",
})


def save_plotly(fig: go.Figure, name: str, directory: Path = FIGURE_DIR) -> Path:
    """Persist a Plotly figure as JSON (embedded later in the report and the app)."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.json"
    path.write_text(fig.to_json(), encoding="utf-8")
    return path


def load_plotly(name: str, directory: Path = FIGURE_DIR) -> go.Figure:
    return pio.from_json((directory / f"{name}.json").read_text(encoding="utf-8"))


def save_png(fig: plt.Figure, name: str, directory: Path = FIGURE_DIR) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def money(x: float) -> str:
    return f"{x:,.0f} $".replace(",", " ")


def dump(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=float),
                    encoding="utf-8")
