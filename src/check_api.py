"""Inspect the installed CUDA-Q API rather than assuming a documentation version.

Usage: python -m src.check_api
"""
from __future__ import annotations
import importlib
import inspect
import platform
import sys


def describe(obj: object, label: str) -> None:
    print(f"  {label}: {type(obj).__name__}")
    try:
        print("    signature:", inspect.signature(obj))
    except (ValueError, TypeError):
        print("    signature: unavailable (compiled binding)")
    doc = inspect.getdoc(obj)
    if doc:
        print("    doc:", doc.replace("\n", " ")[:260])


def main() -> None:
    print("Python:", sys.version.split()[0], "Platform:", platform.platform())
    try:
        cudaq = importlib.import_module("cudaq")
    except ImportError as exc:
        raise SystemExit(f"CUDA-Q is NOT installed: {exc}. Install requirements.txt first.") from exc
    print("CUDA-Q version:", getattr(cudaq, "__version__", "unknown"))
    print("CUDA-Q module:", getattr(cudaq, "__file__", "unknown"))
    for name in ("kernel", "set_target", "get_targets", "has_target", "sample", "observe",
                 "num_available_gpus", "SpinOperator", "set_random_seed", "optimizers"):
        member = getattr(cudaq, name, None)
        if member is None:
            print(f"  MISSING: cudaq.{name}")
        else:
            describe(member, "cudaq." + name)
    kernels = getattr(cudaq, "kernels", None)
    if kernels is not None:
        print("  cudaq.kernels:", [k for k in dir(kernels) if "ucc" in k or "parameter" in k])
        if hasattr(kernels, "uccsd"):
            describe(kernels.uccsd, "cudaq.kernels.uccsd")
    else:
        print("  cudaq.kernels: unavailable; project uses explicit JW-UCCSD generators")
    spin = getattr(cudaq, "SpinOperator", None)
    if spin is not None:
        print("  SpinOperator.from_word:", hasattr(spin, "from_word"))
        print("  SpinOperator.term_count:", hasattr(spin, "term_count"))
    try:
        print("  get_targets:", [str(x) for x in cudaq.get_targets()])
    except Exception as exc:
        print("  get_targets failed:", exc)
    try:
        print("  available NVIDIA GPUs:", cudaq.num_available_gpus())
    except Exception as exc:
        print("  GPU count failed:", exc)
    try:
        cudaq.set_target("qpp-cpu")
        print("  set_target(qpp-cpu): OK")
    except Exception as exc:
        raise SystemExit(f"CPU target not usable: {exc}") from exc
    try:
        print("  has_target(nvidia):", cudaq.has_target("nvidia"))
    except Exception as exc:
        print("  has_target failed:", exc)
    print("SciPy optimization: scipy.optimize.minimize(method='L-BFGS-B')")


if __name__ == "__main__":
    main()
