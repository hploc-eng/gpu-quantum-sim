"""README only reports values derived from actual CSV rows supplied to it."""
from pathlib import Path
from src.build_readme import result_summary, START, END


def test_missing_results_never_claim_values(tmp_path: Path) -> None:
    summary = result_summary(tmp_path)
    assert START in summary and END in summary
    assert "not measured yet" in summary
    assert "not measurable yet" in summary


def test_csv_summary_uses_measured_rows(tmp_path: Path) -> None:
    (tmp_path / "ghz_benchmark.csv").write_text(
        "target,n,median_s,std_s,est_mem_gb\n"
        "qpp-cpu,20,2.0,0.1,0.016\n"
        "nvidia,20,1.0,0.1,0.008\n")
    # Controlled test fixture, never shipped as a published result.
    (tmp_path / "h2_pes.csv").write_text(
        "R,E_vqe,E_fci,E_hf,abs_error_mHa\n0.7,-1.12,-1.121,-1.10,1.0\n")
    summary = result_summary(tmp_path)
    assert "2.00× at 20 qubits" in summary
    assert "1.0000 mHa" in summary
    assert "**PASS**" in summary
