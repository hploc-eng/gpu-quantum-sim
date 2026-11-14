"""Render genuine plot functions with controlled test-only input; no reported data."""
from pathlib import Path
from src.ghz_benchmark import plot_results
from src.memory_plot import make_memory_plot


def test_empty_gpu_comparison_is_clearly_supported(tmp_path: Path) -> None:
    # This tests layout, not measured simulator performance.
    plot_results([], tmp_path)
    assert (tmp_path / "ghz_benchmark.png").stat().st_size > 1000
    assert (tmp_path / "ghz_speedup.png").stat().st_size > 1000


def test_analytical_memory_plot(tmp_path: Path) -> None:
    output = tmp_path / "memory.png"
    make_memory_plot(output)
    assert output.stat().st_size > 1000
