"""CUDA-Q VQE for the full four-qubit H2/STO-3G Jordan-Wigner Hamiltonian.

Chemistry: PySCF via OpenFermion-PySCF (including nuclear repulsion).
Ansatz: *first-order factorized UCCSD* with two spin-conserving singles and
one double excitation, mapped by Jordan-Wigner and enacted by CUDA-Q exp_pauli.
This explicit construction avoids reliance on version-dependent chemistry / UCC
convenience helpers, and makes the excitation operators fully inspectable.

Observe uses exact analytic statevector expectations (no sampling noise).
SciPy L-BFGS-B uses deterministic seeds and warm-starts successive geometries.
"""
from __future__ import annotations

import argparse
import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
from scipy.optimize import minimize

from src.system_info import detect_cudaq_gpu, write_metadata

LOG = logging.getLogger(__name__)
ACCURACY_MHA = 1.6
N_QUBITS = 4
N_ELECTRONS = 2
CSV_FIELDS = ["R", "E_vqe", "E_fci", "E_hf", "abs_error_mHa"]


@dataclass(frozen=True)
class MolecularSystem:
    """Hamiltonian coefficients and independent classical reference energies."""
    bond_angstrom: float
    pauli_coefficients: dict[str, float]
    e_hf: float
    e_fci: float
    n_electrons: int
    n_qubits: int


def build_molecule(bond_angstrom: float) -> MolecularSystem:
    """Perform true ab initio H2/STO-3G RHF and FCI at one bond length.

    OpenFermion-PySCF saves an intermediary HDF5; use a temp directory so
    runs do not silently accumulate files under site-packages.
    """
    if not 0.15 <= bond_angstrom <= 5.0:
        raise ValueError("H2 bond length must be between 0.15 and 5 Angstrom")
    from openfermion import MolecularData, get_fermion_operator, jordan_wigner
    from openfermionpyscf import run_pyscf

    geometry = [("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, float(bond_angstrom)))]
    with TemporaryDirectory(prefix="gpu_quantum_h2_") as temp:
        molecule = MolecularData(
            geometry=geometry, basis="sto-3g", multiplicity=1, charge=0,
            filename=str(Path(temp) / "h2_sto3g"),
        )
        molecule = run_pyscf(molecule, run_scf=True, run_fci=True, verbose=False)
        if molecule.n_qubits != N_QUBITS or molecule.n_electrons != N_ELECTRONS:
            raise RuntimeError(f"Expected 4 qubits/2 electrons: {molecule.n_qubits}/{molecule.n_electrons}")
        fermions = get_fermion_operator(molecule.get_molecular_hamiltonian())
        qubits = jordan_wigner(fermions)
        qubits.compress(abs_tol=1e-12)
        pauli_terms: dict[str, float] = {}
        for term, coefficient in qubits.terms.items():
            if abs(complex(coefficient).imag) > 1e-9:
                raise ValueError(f"Non-Hermitian Hamiltonian term: {term}: {coefficient}")
            word = pauli_word(term, N_QUBITS)
            pauli_terms[word] = float(complex(coefficient).real)
        hf_energy = float(molecule.hf_energy)
        fci_energy = float(molecule.fci_energy)
    return MolecularSystem(float(bond_angstrom), pauli_terms, hf_energy, fci_energy,
                           N_ELECTRONS, N_QUBITS)


def pauli_word(term: tuple[tuple[int, str], ...], n_qubits: int) -> str:
    """Convert OpenFermion's qubit-indexed term into CUDA-Q word order."""
    word = ["I"] * n_qubits
    for index, gate in term:
        if index < 0 or index >= n_qubits or gate not in {"X", "Y", "Z"}:
            raise ValueError(f"Invalid Pauli factor: {(index, gate)}")
        word[index] = gate
    return "".join(word)


def build_uccsd_generators() -> tuple[list[str], list[float], list[int]]:
    """Map spin-conserving H2 UCCSD excitation generators by Jordan-Wigner.

    With OpenFermion's interleaved spin ordering, the HF reference |1100>
    occupies spin orbitals 0(alpha),1(beta), and the virtual ones are
    2(alpha),3(beta). The ansatz is a product of the exponentials of
    (a†_2 a_0 - h.c.), (a†_3 a_1 - h.c.) and
    (a†_2 a†_3 a_1 a_0 - h.c.).

    OpenFermion JW yields K = sum i*c_j*P_j for anti-Hermitian K.
    CUDA-Q exp_pauli(angle, q, P) = exp(i*angle*P), so angle=theta*c_j.
    Within each excitation the constituent Pauli strings commute, hence
    their product implements exp(theta*K) exactly.
    """
    from openfermion import FermionOperator, hermitian_conjugated, jordan_wigner

    excitation_terms = [
        ((2, 1), (0, 0)),       # alpha single 0 -> 2
        ((3, 1), (1, 0)),       # beta single 1 -> 3
        ((2, 1), (3, 1), (1, 0), (0, 0)),  # paired double
    ]
    words: list[str] = []
    weights: list[float] = []
    owners: list[int] = []
    for parameter_idx, operator_indices in enumerate(excitation_terms):
        excitation = FermionOperator(operator_indices, 1.0)
        anti_hermitian = excitation - hermitian_conjugated(excitation)
        mapped = jordan_wigner(anti_hermitian)
        mapped.compress(abs_tol=1e-12)
        for term, coefficient in sorted(mapped.terms.items()):
            c = complex(coefficient)
            if abs(c.real) > 1e-10:
                raise ValueError(f"UCCSD generator must be anti-Hermitian: {term} {c}")
            if abs(c.imag) < 1e-12:
                continue
            words.append(pauli_word(term, N_QUBITS))
            weights.append(float(c.imag))
            owners.append(parameter_idx)
    if set(owners) != {0, 1, 2}:
        raise RuntimeError("UCCSD pool unexpectedly missing an excitation")
    return words, weights, owners


def to_cudaq_hamiltonian(cudaq: Any, coefficients: dict[str, float]) -> Any:
    """Convert all Jordan-Wigner terms (including identity) to SpinOperator."""
    total = None
    for word, coefficient in sorted(coefficients.items()):
        term = float(coefficient) * cudaq.SpinOperator.from_word(word)
        total = term if total is None else total + term
    if total is None:
        raise ValueError("Empty Hamiltonian")
    # Current CUDA-Q exposes `term_count`; older releases used get_term_count().
    count = int(total.term_count) if hasattr(total, "term_count") else int(total.get_term_count())
    if count != 15:
        raise AssertionError(f"Expected H2/STO-3G Hamiltonian to have 15 Pauli terms, got {count}")
    LOG.info("H2/STO-3G JW Hamiltonian: %d Pauli terms including identity", count)
    return total


try:
    import cudaq
except ImportError:
    cudaq = None


# This is a genuine UCCSD kernel: the Pauli words and coefficients are produced
# by JW-transforming fermionic excitations, not by handcoded HEA rotations.
# Module scope is important because CUDA-Q compiles kernel function ASTs.
if cudaq is not None:
    @cudaq.kernel
    def uccsd_ansatz(theta: list[float], weights: list[float],
                     words: list[cudaq.pauli_word], owners: list[int]):
        """First-order factorized UCCSD with Hartree-Fock reference |1100>."""
        q = cudaq.qvector(4)
        x(q[0])
        x(q[1])
        for i in range(len(words)):
            exp_pauli(theta[owners[i]] * weights[i], q, words[i])
else:
    uccsd_ansatz = None


def make_ansatz(cudaq_module: Any) -> Any:
    """Return the kernel; fail clearly if CUDA-Q is absent."""
    if uccsd_ansatz is None:
        raise RuntimeError("CUDA-Q not installed; cannot construct UCCSD ansatz")
    return uccsd_ansatz


@dataclass
class VQEResult:
    R: float
    E_vqe: float
    E_fci: float
    E_hf: float
    abs_error_mHa: float
    theta: list[float]
    converged: bool
    iterations: int
    trace: list[float]


def solve_vqe(cudaq: Any, ansatz: Any, system: MolecularSystem,
              x0: np.ndarray | None = None, maxiter: int = 200,
              max_restarts: int = 3) -> VQEResult:
    """Optimize the UCCSD circuit with exact cudaq.observe energies.

    A reference mismatch triggers deterministic restarts; FCI is *never*
    substituted for VQE. Failed optimizer convergence is disclosed.
    """
    hamiltonian = to_cudaq_hamiltonian(cudaq, system.pauli_coefficients)
    words, weights, owners = build_uccsd_generators()
    zeros = np.zeros(3, dtype=float)

    def energy(theta: np.ndarray) -> float:
        return float(cudaq.observe(ansatz, hamiltonian,
                                   np.asarray(theta, dtype=float).tolist(),
                                   weights, words, owners).expectation())

    ref = energy(zeros)
    if abs(ref - system.e_hf) > 1e-5:
        raise AssertionError(
            f"HF reference mismatch at R={system.bond_angstrom}: "
            f"CUDA-Q {ref:.9f} Ha vs PySCF {system.e_hf:.9f} Ha. "
            "Check spin-orbital order, Hamiltonian conversion and exp_pauli convention."
        )

    def minimize_from(initial: np.ndarray) -> tuple[Any, list[float]]:
        trace = [energy(initial)]
        def record_iteration(theta: np.ndarray) -> None:
            trace.append(energy(theta))
        res = minimize(
            energy, np.asarray(initial, dtype=float).copy(), method="L-BFGS-B",
            callback=record_iteration,
            options={"maxiter": maxiter, "ftol": 1e-13, "gtol": 1e-8,
                     "maxls": 30, "maxfun": maxiter * 15},
        )
        if not np.isclose(trace[-1], res.fun, atol=1e-11, rtol=0):
            trace.append(float(res.fun))
        return res, trace

    initial = np.asarray(x0 if x0 is not None else zeros, dtype=float)
    if initial.shape != (3,):
        raise ValueError("UCCSD requires three excitation parameters")
    best, trace = minimize_from(initial)
    error = 1000 * abs(float(best.fun) - system.e_fci)
    if error > ACCURACY_MHA:
        LOG.warning("R=%.3f Å: first optimization error %.4f mHa; trying deterministic restarts",
                    system.bond_angstrom, error)
        rng = np.random.default_rng(1979 + round(system.bond_angstrom * 1000))
        for start in [zeros, *[rng.uniform(-0.25, 0.25, size=3) for _ in range(max_restarts)]]:
            candidate, candidate_trace = minimize_from(start)
            if float(candidate.fun) < float(best.fun):
                best, trace = candidate, candidate_trace
            if 1000 * abs(float(best.fun) - system.e_fci) < 0.1:
                break
    result = VQEResult(
        R=system.bond_angstrom,
        E_vqe=float(best.fun), E_fci=system.e_fci, E_hf=system.e_hf,
        abs_error_mHa=1000 * abs(float(best.fun) - system.e_fci),
        theta=list(map(float, best.x)), converged=bool(best.success),
        iterations=int(best.nit), trace=trace,
    )
    LOG.info("R=%.3f Å E_vqe=%.9f E_fci=%.9f |error|=%.4f mHa (%s)",
             result.R, result.E_vqe, result.E_fci, result.abs_error_mHa,
             "converged" if result.converged else "optimizer warning")
    return result


def hydrogen_atom_energy() -> float:
    """H atom UHF energy in *the same* STO-3G basis, for 2 E_H asymptote."""
    from pyscf import gto, scf
    atom = gto.M(atom="H 0 0 0", basis="sto-3g", charge=0,
                 spin=1, unit="Angstrom", verbose=0)
    hf = scf.UHF(atom)
    hf.verbose = 0
    hf.kernel()
    if not hf.converged:
        raise RuntimeError("Hydrogen atom UHF did not converge")
    return float(hf.e_tot)


def write_pes_csv(results: list[VQEResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf8") as fp:
        writer = csv.DictWriter(fp, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in sorted(results, key=lambda r: r.R):
            writer.writerow({field: getattr(row, field) for field in CSV_FIELDS})


def write_convergence_csv(result: VQEResult, path: Path) -> None:
    with path.open("w", newline="", encoding="utf8") as fp:
        w = csv.writer(fp)
        w.writerow(["iteration", "E_vqe", "E_fci"])
        for i, energy in enumerate(result.trace):
            w.writerow([i, energy, result.E_fci])


def plot_convergence(result: VQEResult, out_dir: Path) -> None:
    from matplotlib import pyplot as plt
    from src.plot_style import COLORS, save_figure, setup_style
    setup_style()
    fig, ax = plt.subplots(figsize=(8.1, 4.9))
    ax.plot(range(len(result.trace)), result.trace, "o-", color=COLORS["gpu"], ms=4,
            lw=1.7, label="UCCSD VQE")
    ax.axhline(result.E_fci, color=COLORS["fci"], ls="--", lw=1.5, label="Exact FCI")
    ax.set(xlabel="Optimizer iteration", ylabel="Energy (Hartree)",
           title=f"H₂ VQE convergence at R = {result.R:.3f} Å")
    ax.legend()
    save_figure(fig, out_dir / "vqe_convergence.png")


def plot_pes(results: list[VQEResult], atom_energy: float, out_dir: Path) -> None:
    from matplotlib import pyplot as plt
    from src.plot_style import COLORS, save_figure, setup_style
    setup_style()
    rows = sorted(results, key=lambda r: r.R)
    r = [x.R for x in rows]
    setup_style()
    fig, ax = plt.subplots(figsize=(8.5, 5.3))
    ax.plot(r, [x.E_fci for x in rows], "-", color=COLORS["fci"],
            lw=2.2, label="FCI · exact in STO-3G")
    ax.plot(r, [x.E_hf for x in rows], "--", color=COLORS["hf"],
            lw=1.7, label="Hartree–Fock")
    ax.scatter(r, [x.E_vqe for x in rows], s=18, color=COLORS["gpu"],
               zorder=5, label="UCCSD VQE")
    ax.axhline(2 * atom_energy, color="#3E8D73", ls=":", lw=1.6,
               label="2 × E(H atom), STO-3G")
    ax.set(xlabel="H–H bond length (Å)", ylabel="Total energy (Hartree)",
           title="H₂ potential energy surface · 4-qubit Jordan–Wigner")
    ax.legend(fontsize=9, ncol=2)
    save_figure(fig, out_dir / "h2_pes.png")

    fig, ax = plt.subplots(figsize=(8.5, 4.7))
    ax.axhspan(0, ACCURACY_MHA, color="#D9F1E8", alpha=0.85,
               label="Chemical accuracy (≤ 1.6 mHa)")
    ax.plot(r, [x.abs_error_mHa for x in rows], "o-", color=COLORS["gpu"], ms=3.5)
    ax.set_ylim(bottom=0)
    ax.set(xlabel="H–H bond length (Å)", ylabel="|E(VQE) − E(FCI)| (mHa)",
           title="UCCSD VQE error vs. exact FCI")
    ax.legend(loc="upper left")
    save_figure(fig, out_dir / "h2_error.png")


def sweep_distances() -> list[float]:
    """Inclusive 0.30 to 2.50 Å on an exactly rounded 0.05 Å grid."""
    return [round(0.30 + 0.05 * index, 2) for index in range(45)]


def run(quick: bool = False, result_dir: Path = Path("results"),
        target: str = "cpu", maxiter: int = 200) -> list[VQEResult]:
    """Optimize equilibrium first, then a warm-started molecular bond sweep."""
    try:
        import cudaq
    except ImportError as exc:
        raise SystemExit("CUDA-Q not installed. Install requirements.txt before chemistry VQE.") from exc
    if target == "gpu":
        available, reason = detect_cudaq_gpu(cudaq)
        if not available:
            LOG.warning("GPU SKIPPED (%s); using qpp-cpu instead", reason)
            target = "cpu"
    cudaq.set_target("nvidia" if target == "gpu" else "qpp-cpu",
                     **({"option": "fp64"} if target == "gpu" else {}))
    cudaq.set_random_seed(1979)
    ansatz = make_ansatz(cudaq)
    result_dir.mkdir(parents=True, exist_ok=True)

    at_reference = solve_vqe(cudaq, ansatz, build_molecule(0.735),
                             maxiter=maxiter, max_restarts=1 if quick else 3)
    write_convergence_csv(at_reference, result_dir / "vqe_convergence.csv")
    plot_convergence(at_reference, result_dir)

    r_values = [0.70, 0.735, 0.80] if quick else sweep_distances()
    solution: list[VQEResult] = []
    previous = None
    for r in r_values:
        if r == 0.735:  # reuse real, independently optimized reference measurement
            result = at_reference
        else:
            result = solve_vqe(cudaq, ansatz, build_molecule(r), x0=previous,
                               maxiter=maxiter, max_restarts=1 if quick else 3)
        solution.append(result)
        previous = np.asarray(result.theta, dtype=float)

    e_h = hydrogen_atom_energy()
    write_pes_csv(solution, result_dir / "h2_pes.csv")
    plot_pes(solution, e_h, result_dir)
    best = min(solution, key=lambda item: item.E_vqe)
    max_error = max(x.abs_error_mHa for x in solution)
    write_metadata(result_dir / "metadata.json", "vqe", quick=quick,
                   optimizer="SciPy L-BFGS-B", maxiter=maxiter, rng_seed=1979,
                   R_start=min(r_values), R_end=max(r_values), points=len(r_values),
                   chemistry="H2/STO-3G, JW 4 qubits, 2 electrons",
                   ansatz="first-order factorized UCCSD: 2 singles + 1 double",
                   observable="cudaq.observe, exact expectation (no shots)",
                   simulator="qpp-cpu" if target == "cpu" else "nvidia fp64",
                   h_atom_energy_hartree=e_h,
                   equilibrium_grid_angstrom=best.R,
                   min_grid_energy_hartree=best.E_vqe,
                   max_error_mHa=max_error)
    print(f"H2 min sampled energy = {best.E_vqe:.9f} Ha at R={best.R:.3f} Å")
    print(f"Maximum VQE/FCI error = {max_error:.4f} mHa ({len(solution)} bond lengths)")
    if max_error > ACCURACY_MHA:
        raise RuntimeError(f"Chemical accuracy NOT met: max error {max_error:.3f} mHa. "
                           "CSV/plots saved for diagnosis; do not claim success.")
    return solution


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="Three short-distance geometries only")
    parser.add_argument("--target", choices=("cpu", "gpu"), default="cpu",
                        help="VQE defaults to CPU FP64 for validation")
    parser.add_argument("--maxiter", type=int, default=200)
    parser.add_argument("--result-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run(quick=args.quick, result_dir=args.result_dir,
        target=args.target, maxiter=args.maxiter)


if __name__ == "__main__":
    main()
