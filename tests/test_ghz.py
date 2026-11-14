"""Real circuit validation when CUDA-Q is installed; no synthetic substitutes."""
import pytest

from src.ghz_benchmark import (check_ghz_correctness, estimated_memory_bytes,
                               fits_memory, ghz_kernel)


def test_memory_estimate_complex64_and_complex128() -> None:
    assert estimated_memory_bytes(20, "fp32") == (2**20) * 8
    assert estimated_memory_bytes(30, "fp64") == (2**30) * 16
    assert estimated_memory_bytes(0, "fp32") == 8


def test_memory_guard_and_validation() -> None:
    assert fits_memory(20, "fp32", 256 * 2**20)
    assert not fits_memory(30, "fp64", 8 * 2**30)
    assert not fits_memory(20, "fp32", None)
    with pytest.raises(ValueError):
        estimated_memory_bytes(-1)
    with pytest.raises(ValueError):
        estimated_memory_bytes(10, "fp16")
    with pytest.raises(ValueError):
        fits_memory(2, "fp32", 100, fraction=1.2)


@pytest.mark.integration
def test_ghz_8_qubits_has_correct_support_and_balance() -> None:
    cudaq = pytest.importorskip("cudaq")
    counts = check_ghz_correctness(cudaq, ghz_kernel(), n=8, shots=1024)
    assert set(counts).issubset({"0" * 8, "1" * 8})
    assert sum(counts.values()) == 1024
