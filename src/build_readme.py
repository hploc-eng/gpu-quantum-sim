"""Rebuild the README measurement block from real experiment CSVs, never guesses.

Usage: python -m src.build_readme
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

START = "<!-- AUTO_RESULTS_START -->"
END = "<!-- AUTO_RESULTS_END -->"


def load_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf8") as stream:
        return list(csv.DictReader(stream))


def result_summary(results: Path) -> str:
    """Generate an evidence-based summary and a conditional résumé statement."""
    ghz = load_csv(results / "ghz_benchmark.csv")
    pes = load_csv(results / "h2_pes.csv")
    meta_path = results / "metadata.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    ghz_meta = meta.get("ghz", {})
    vqe_meta = meta.get("vqe", {})
    cpu = {int(row["n"]): float(row["median_s"]) for row in ghz if row["target"] == "qpp-cpu"}
    gpu = {int(row["n"]): float(row["median_s"]) for row in ghz if row["target"] == "nvidia"}
    both = sorted(set(cpu) & set(gpu))
    lines = [START, "### Measured results (generated directly from CSVs)", ""]
    if ghz:
        overall_n = max(int(row["n"]) for row in ghz)
        qualifiers = ", quick subset" if ghz_meta.get("settings", {}).get("quick") else ", full requested sweep"
        lines.append(f"- **Largest completed statevector:** **{overall_n} qubits** ({', '.join(sorted(set(r['target'] for r in ghz)))}{qualifiers}).")
    else:
        lines.append("- **Largest completed statevector:** not measured yet (no GHZ CSV available).")
    if both:
        n = both[-1]
        ratio = cpu[n] / gpu[n]
        lines.append(f"- **GPU speedup at largest matched n:** **{ratio:.2f}× at {n} qubits** "
                     "(Q++ CPU FP64 versus NVIDIA cuStateVec FP32, median times).")
    else:
        lines.append("- **GPU speedup:** not measurable yet (no matched CPU/GPU measurements).")
    if pes:
        best = min(pes, key=lambda r: float(r["E_vqe"]))
        worst = max(float(r["abs_error_mHa"]) for r in pes)
        quick = vqe_meta.get("settings", {}).get("quick", False)
        label = "quick subset" if quick else "full 0.30–2.50 Å sweep"
        lines.append(f"- **H₂ UCCSD VQE:** minimum sampled energy **{float(best['E_vqe']):.6f} Ha** "
                     f"at **{float(best['R']):.3f} Å**; maximum "
                     f"**{worst:.4f} mHa** discrepancy from FCI "
                     f"over {len(pes)} bond lengths ({label}).")
        lines.append(f"- **Chemical accuracy across sampled geometries:** "
                     f"**{'PASS' if worst <= 1.6 else 'NOT MET'}** "
                     "(threshold 1.6 mHa).")
    else:
        lines.append("- **H₂ VQE error:** not measured yet (no potential-energy CSV available).")
    hardware = ghz_meta.get("gpu") or vqe_meta.get("gpu")
    cpu_info = ghz_meta.get("platform") or vqe_meta.get("platform")
    if cpu_info:
        lines.append(f"- **Execution environment:** {cpu_info}.")
    if hardware:
        lines.append(f"- **NVIDIA device:** {hardware}.")
    elif meta:
        lines.append("- **NVIDIA device:** none detected / unavailable in recorded run.")

    lines += ["", "**Résumé bullet (only when supported by measured results):**", ""]
    if both and pes:
        n = both[-1]
        worst = max(float(r["abs_error_mHa"]) for r in pes)
        biggest_gpu_n = max(gpu)
        lines.append(
            f"> Implemented GPU-accelerated statevector simulation up to {biggest_gpu_n} qubits "
            f"using NVIDIA CUDA-Q/cuStateVec ({cpu[n]/gpu[n]:.2f}× over Q++ CPU "
            f"at {n} matched qubits; FP32 GPU vs FP64 CPU); built "
            f"four-qubit H₂ UCCSD VQE with {worst:.3f} mHa maximum "
            f"error versus exact STO-3G FCI across {len(pes)} sampled bond lengths."
        )
    else:
        lines.append("> Not generated: requires measured CUDA-Q CPU/GPU rows *and* H₂ VQE rows.")
    lines.extend(["", END])
    return "\n".join(lines)


def update_readme(readme_path: Path, result_dir: Path) -> None:
    content = readme_path.read_text(encoding="utf8")
    if START not in content or END not in content:
        raise RuntimeError("README missing AUTO_RESULTS markers")
    before, remainder = content.split(START, 1)
    _, after = remainder.split(END, 1)
    readme_path.write_text(before + result_summary(result_dir) + after, encoding="utf8")


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    update_readme(repo / "README.md", repo / "results")
    print("README results block refreshed using existing real CSVs only.")


if __name__ == "__main__":
    main()
