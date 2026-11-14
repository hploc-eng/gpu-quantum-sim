"""Publication-ready, headless Matplotlib defaults."""
from __future__ import annotations
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

COLORS = {"cpu": "#2563EB", "gpu": "#DC6A31", "fci": "#174A74", "hf": "#7B8190"}


def setup_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 130, "savefig.dpi": 220,
        "font.family": "DejaVu Sans", "font.size": 11,
        "axes.titlesize": 13, "axes.labelsize": 11,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.axisbelow": True,
        "grid.alpha": 0.21, "grid.linewidth": 0.7,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "legend.frameon": False, "savefig.bbox": "tight",
    })


def save_figure(fig: object, path: object) -> None:
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
