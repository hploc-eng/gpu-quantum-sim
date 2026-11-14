"""Render an analytical memory-scaling figure; no CUDA-Q run is required.

This chart is an exact statevector byte count, NOT a measured runtime or
VRAM allocation. Its data is calculated as 2^n * bytes_per_complex_amplitude.
"""
from __future__ import annotations
from pathlib import Path

from src.ghz_benchmark import estimated_memory_bytes


def make_memory_plot(path: Path = Path("results/statevector_memory.png")) -> None:
    from matplotlib import pyplot as plt
    from src.plot_style import COLORS, save_figure, setup_style
    setup_style()
    plt.rcParams["svg.fonttype"] = "none"  # keep text selectable and SVG compact
    n = list(range(10, 35))
    fp32 = [estimated_memory_bytes(k, "fp32") / 2**30 for k in n]
    fp64 = [estimated_memory_bytes(k, "fp64") / 2**30 for k in n]
    fig, ax = plt.subplots(figsize=(8.5, 4.9))
    ax.plot(n, fp32, "o-", markersize=2.7, lw=2, color=COLORS["gpu"],
            label="Complex64 · 8 bytes/amplitude")
    ax.plot(n, fp64, "o-", markersize=2.7, lw=2, color=COLORS["cpu"],
            label="Complex128 · 16 bytes/amplitude")
    ax.axhline(16, color="#708090", ls="--", lw=1.2, label="16 GiB reference capacity")
    ax.annotate("30 qubits = 8 GiB / 16 GiB", (30, 16), (22, 90),
                fontsize=9, arrowprops={"arrowstyle": "->", "color": "#667085"})
    ax.set_yscale("log", base=2)
    ax.set(xlabel="Number of qubits", ylabel="Dense statevector alone (GiB; log₂ scale)",
           title="Exponential memory cost of exact statevector simulation")
    ax.legend(loc="upper left", fontsize=9)
    fig.text(0.11, -0.012, "Analytical 2ⁿ × bytes/amplitude; excludes runtime overhead and workspace.",
             fontsize=8.2, color="#667085")
    path.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, path)


if __name__ == "__main__":
    make_memory_plot()
    make_memory_plot(Path("results/statevector_memory.svg"))
