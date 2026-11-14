"""Hamiltonian, generator, and true 0.735 Å VQE/FCI integration tests."""

import pytest

from src.h2_vqe import ACCURACY_MHA, N_QUBITS, pauli_word, sweep_distances


def test_pauli_word_mapping() -> None:
    assert pauli_word((), N_QUBITS) == "IIII"
    assert pauli_word(((0, "X"), (3, "Y")), N_QUBITS) == "XIIY"
    with pytest.raises(ValueError):
        pauli_word(((5, "X"),), N_QUBITS)


def test_sweep_grid() -> None:
    grid = sweep_distances()
    assert len(grid) == 45
    assert grid[0] == 0.30
    assert grid[-1] == 2.50
    assert grid[9] == 0.75


@pytest.mark.integration
def test_h2_jw_pauli_count_and_three_excitations() -> None:
    cudaq = pytest.importorskip("cudaq")
    pytest.importorskip("pyscf")
    pytest.importorskip("openfermion")
    pytest.importorskip("openfermionpyscf")
    from src.h2_vqe import build_molecule, build_uccsd_generators, to_cudaq_hamiltonian

    mol = build_molecule(0.735)
    assert mol.n_qubits == 4
    assert mol.n_electrons == 2
    assert len(mol.pauli_coefficients) == 15
    assert "IIII" in mol.pauli_coefficients
    cudaq_ham = to_cudaq_hamiltonian(cudaq, mol.pauli_coefficients)
    count = cudaq_ham.term_count if hasattr(cudaq_ham, "term_count") else cudaq_ham.get_term_count()
    assert count == 15
    words, weights, owners = build_uccsd_generators()
    assert len(words) == len(weights) == len(owners)
    assert set(owners) == {0, 1, 2}
    assert all(4 == len(word) for word in words)


@pytest.mark.integration
@pytest.mark.slow
def test_h2_vqe_0735_within_chemical_accuracy() -> None:
    cudaq = pytest.importorskip("cudaq")
    pytest.importorskip("pyscf")
    pytest.importorskip("openfermion")
    pytest.importorskip("openfermionpyscf")
    from src.h2_vqe import build_molecule, make_ansatz, solve_vqe

    cudaq.set_target("qpp-cpu")
    system = build_molecule(0.735)
    result = solve_vqe(cudaq, make_ansatz(cudaq), system, maxiter=200)
    assert result.E_vqe >= system.e_fci - 1e-5   # variational principle
    assert result.abs_error_mHa <= ACCURACY_MHA
    assert abs(result.E_vqe - (-1.1373)) < 0.01
