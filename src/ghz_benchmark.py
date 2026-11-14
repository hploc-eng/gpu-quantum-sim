"""Benchmark CUDA-Q GHZ sampling on qpp-cpu and NVIDIA cuStateVec.

Normal mode: n=20..30, one untimed warm-up, five timed samples per target.
Quick mode: n=8, 10, 12 for a fast CPU-only smoke benchmark.

Runtime measures kernel execution and sampling, *not* isolated gate throughput.
CPU backend uses complex128 (16 bytes/amplitude); NVIDIA default is
complex64 (8 bytes/amplitude). The comparison is intentionally unequal
precision and is identified as such in the output.
"""
from __future__ import annotations

import argparse
import csv
import logging
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.system_info import (available_ram_bytes, detect_cudaq_gpu,
                             free_vram_bytes, write_metadata)

LOG = logging.getLogger(__name__)
GIB = 2**30
CSV_FIELDS = ["target", "n", "median_s", "std_s", "est_mem_gb"]


def estimated_memory_bytes(n: int, precision: str = "fp32") -> int:
    """Dense statevector bytes: 2**n complex amplitudes (8/16 bytes).

    Deliberately excludes runtime, temporary buffers, and fragmentation;
    `fits_memory` therefore enforces an additional safety fraction.
    """
    if n < 0 or n > 62:
        raise ValueError("n must be between 0 and 62")
    if precision not in ("fp32", "fp64"):
        raise ValueError("precision must be 'fp32' or 'fp64'")
    return (1 << n) * (8 if precision == "fp32" else 16)


def fits_memory(n: int, precision: str, free_bytes: int | None,
                fraction: float = 0.65) -> bool:
    """Reject states with insufficient memory, leaving a safety margin."""
    if not 0 < fraction < 1:
        raise ValueError("fraction must be strictly between 0 and 1")
    return free_bytes is not None and estimated_memory_bytes(n, precision) <= free_bytes * fraction


try:
    import cudaq
except ImportError:  # The estimator and its tests run without the quantum stack.
    cudaq = None


# The kernel is defined at module scope so CUDA-Q's AST compiler can inspect
# its source and its annotation types (especially across target switches).
if cudaq is not None:
    @cudaq.kernel
    def ghz(n: int):
        """Prepare and measure an n-qubit GHZ state with an adjacent CNOT chain."""
        q = cudaq.qvector(n)
        h(q[0])
        for i in range(n - 1):
            x.ctrl(q[i], q[i + 1])
        mz(q)
else:
    ghz = None


def import_cudaq() -> Any:
    if cudaq is None:
        raise SystemExit(
            "CUDA-Q is not installed. Install requirements.txt and run `python -m src.check_api`."
        )
    return cudaq


def ghz_kernel() -> Any:
    """Return the real module-scope CUDA-Q GHZ kernel."""
    import_cudaq()
    return ghz


def check_ghz_correctness(cudaq: Any, kernel: Any, n: int = 8,
                          shots: int = 1024, tolerance: float = 0.1) -> dict[str, int]:
    """Check support and empirical 50:50 statistics on a small GHZ circuit."""
    if n < 2 or shots <= 0:
        raise ValueError("n>=2 and shots>0 are required")
    cudaq.set_target("qpp-cpu")
    cudaq.set_random_seed(1979)
    result = cudaq.sample(kernel, n, shots_count=shots)
    counts = {str(key): int(value) for key, value in result.items()}
    expected = {"0" * n, "1" * n}
    invalid = set(counts) - expected
    if invalid:
        raise AssertionError(f"Unexpected bitstrings for GHZ({n}): {invalid}")
    for bitstring in expected:
        probability = counts.get(bitstring, 0) / shots
        if abs(probability - 0.5) > tolerance:
            raise AssertionError(f"GHZ support imbalance: {bitstring}: {probability:.3f}")
    return counts


@dataclass(frozen=True)
class BenchmarkRow:
    """One qubit count on one simulator backend."""
    target: str
    n: int
    median_s: float
    std_s: float
    est_mem_gb: float


def measure_target(cudaq: Any, kernel: Any, target: str, n_values: list[int],
                   memory_fraction: float, repetitions: int = 5,
                   timing_sink: list[tuple[str, int, int, float]] | None = None) -> list[BenchmarkRow]:
    """Measure kernel+sampling runtime; skip oversized statevectors safely."""
    cudaq.set_target(target)
    precision = "fp32" if target == "nvidia" else "fp64"
    out: list[BenchmarkRow] = []
    for n in n_values:
        free = free_vram_bytes() if target == "nvidia" else available_ram_bytes()
        estimate = estimated_memory_bytes(n, precision)
        if not fits_memory(n, precision, free, memory_fraction):
            LOG.warning("SKIP %-7s n=%d: state %.2f GiB, available %s GiB; safety fraction %.2f",
                        target, n, estimate / GIB,
                        f"{free / GIB:.2f}" if free is not None else "unknown", memory_fraction)
            continue
        try:
            cudaq.sample(kernel, n, shots_count=1024)  # Untimed compilation + warmup
            elapsed = []
            for _ in range(repetitions):
                t0 = time.perf_counter()
                cudaq.sample(kernel, n, shots_count=1024)
                elapsed.append(time.perf_counter() - t0)
        except (RuntimeError, MemoryError) as exc:
            LOG.warning("SKIP %-7s n=%d: simulator failed: %s", target, n, exc)
            continue
        row = BenchmarkRow(
            target=target, n=n, median_s=statistics.median(elapsed),
            std_s=statistics.stdev(elapsed) if len(elapsed) > 1 else 0.0,
            est_mem_gb=estimate / GIB,
        )
        if timing_sink is not None:
            timing_sink.extend((target, n, i + 1, seconds) for i, seconds in enumerate(elapsed))
        LOG.info("%-7s %2d qubits: median=%.5f s, std=%.5f s, state=%.2f GiB",
                 target, n, row.median_s, row.std_s, row.est_mem_gb)
        out.append(row)
    return out


def write_csv(rows: list[BenchmarkRow], file: Path) -> None:
    file.parent.mkdir(parents=True, exist_ok=True)
    with file.open("w", newline="", encoding="utf8") as fp:
        writer = csv.DictWriter(fp, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in sorted(rows, key=lambda item: (item.target, item.n)):
            writer.writerow({k: getattr(row, k) for k in CSV_FIELDS})


def write_raw_timings(records: list[tuple[str, int, int, float]], path: Path) -> None:
    """Archive every timed repetition behind the reported median/std."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf8") as fp:
        writer = csv.writer(fp)
        writer.writerow(["target", "n", "run", "elapsed_s"])
        writer.writerows(records)


def plot_results(rows: list[BenchmarkRow], out_dir: Path) -> None:
    """Write two figures; an unavailable GPU yields an explicitly empty speedup plot."""
    from matplotlib import pyplot as plt
    from src.plot_style import COLORS, save_figure, setup_style
    setup_style()
    fig, ax = plt.subplots(figsize=(8.1, 4.9))
    for target, name, color in (("qpp-cpu", "Q++ CPU · FP64", COLORS["cpu"]),
                                ("nvidia", "NVIDIA cuStateVec · FP32", COLORS["gpu"])):
        subset = sorted((r for r in rows if r.target == target), key=lambda r: r.n)
        if subset:
            ax.errorbar([r.n for r in subset], [r.median_s for r in subset],
                        yerr=[r.std_s for r in subset], marker="o", capsize=2,
                        lw=2, label=name, color=color)
    ax.set_yscale("log")
    ax.set_xlabel("Number of qubits")
    ax.set_ylabel("Kernel + 1,024-shot sampling time (seconds, log scale)")
    ax.set_title("GHZ statevector simulation · median of five runs")
    if rows:
        ax.legend(loc="upper left")
    fig.text(0.11, -0.015, "One compilation warm-up per (target, n); CPU and GPU have different precisions.",
             fontsize=8, color="#667085")
    save_figure(fig, out_dir / "ghz_benchmark.png")

    cpu = {r.n: r.median_s for r in rows if r.target == "qpp-cpu"}
    gpu = {r.n: r.median_s for r in rows if r.target == "nvidia"}
    both = sorted(set(cpu) & set(gpu))
    fig, ax = plt.subplots(figsize=(8.1, 4.9))
    if both:
        ax.plot(both, [cpu[n] / gpu[n] for n in both], "o-", lw=2.2, color=COLORS["gpu"])
        ax.axhline(1.0, color="#667085", ls="--", linewidth=1.1)
    else:
        ax.text(0.5, 0.5, "No matched CPU/GPU measurements\nGPU unavailable or memory-limited",
                transform=ax.transAxes, ha="center", va="center", color="#667085")
    ax.set_xlabel("Number of qubits")
    ax.set_ylabel("Speedup (CPU time / GPU time)")
    ax.set_title("GPU speedup at matched qubit counts")
    save_figure(fig, out_dir / "ghz_speedup.png")


def run(max_qubits: int = 30, quick: bool = False, result_dir: Path = Path("results"),
        memory_fraction: float = 0.65) -> list[BenchmarkRow]:
    """Run correctness check, CPU benchmark, and optionally GPU benchmark."""
    cudaq = import_cudaq()
    if not cudaq.has_target("qpp-cpu"):
        raise RuntimeError("CUDA-Q 'qpp-cpu' target is unavailable")
    kernel = ghz_kernel()
    correct = check_ghz_correctness(cudaq, kernel)
    LOG.info("GHZ(8) passed: %s", correct)
    values = [8, 10, 12] if quick else list(range(20, max_qubits + 1))
    if not values:
        raise ValueError("max-qubits must be >= 20 (unless --quick)")
    raw_timings: list[tuple[str, int, int, float]] = []
    rows = measure_target(cudaq, kernel, "qpp-cpu", values, memory_fraction,
                          timing_sink=raw_timings)
    gpu_ok, reason = detect_cudaq_gpu(cudaq)
    if gpu_ok:
        LOG.info("GPU present: %s", reason)
        try:
            rows += measure_target(cudaq, kernel, "nvidia", values, memory_fraction,
                                   timing_sink=raw_timings)
        except RuntimeError as exc:
            LOG.warning("NVIDIA backend unavailable despite detection: %s", exc)
    else:
        LOG.warning("GPU SKIPPED: %s. CPU results remain valid.", reason)
    result_dir.mkdir(parents=True, exist_ok=True)
    write_csv(rows, result_dir / "ghz_benchmark.csv")
    write_raw_timings(raw_timings, result_dir / "ghz_timings.csv")
    plot_results(rows, result_dir)
    write_metadata(result_dir / "metadata.json", "ghz", quick=quick,
                   shots=1024, repetitions=5, warmups=1,
                   qubit_counts=values, memory_fraction=memory_fraction,
                   cpu_precision="fp64", gpu_precision="fp32",
                   gpu_status=reason)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-qubits", type=int, default=30)
    parser.add_argument("--quick", action="store_true", help="Benchmark 8, 10, 12 qubits on available targets")
    parser.add_argument("--memory-fraction", type=float, default=0.65)
    parser.add_argument("--result-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    rows = run(max_qubits=args.max_qubits, quick=args.quick,
               result_dir=args.result_dir, memory_fraction=args.memory_fraction)
    print(f"GHZ benchmark complete: {len(rows)} measured (target, n) pairs.")


if __name__ == "__main__":
    main()
