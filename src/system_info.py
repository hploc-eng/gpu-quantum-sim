"""Collect reproducibility metadata and safely check simulator resources."""
from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def available_ram_bytes() -> int:
    """Return a conservative memory budget accounting for Linux cgroup limits.

    This is *available* memory, not physical total memory. For containerized
    runtimes (Colab/Docker), subtract estimated current cgroup usage from max.
    """
    candidates: list[int] = []
    if sys.platform.startswith("linux"):
        try:
            for line in Path("/proc/meminfo").read_text().splitlines():
                if line.startswith("MemAvailable:"):
                    candidates.append(int(line.split()[1]) * 1024)
                    break
        except (OSError, ValueError):
            pass
        try:
            limit = Path("/sys/fs/cgroup/memory.max").read_text().strip()
            usage = int(Path("/sys/fs/cgroup/memory.current").read_text())
            if limit != "max":
                candidates.append(max(0, int(limit) - usage))
        except (OSError, ValueError):
            pass
    if not candidates:
        try:
            import psutil  # optional: not needed on Colab Linux
            candidates.append(int(psutil.virtual_memory().available))
        except ImportError as exc:
            raise RuntimeError("Cannot determine available RAM safely") from exc
    return min(candidates)


def free_vram_bytes() -> int | None:
    """Free VRAM on the first NVIDIA GPU, using nvidia-smi (no extra package)."""
    if not shutil.which("nvidia-smi"):
        return None
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True, timeout=8,
        )
        mib = int(result.stdout.strip().splitlines()[0].strip())
        return mib * 1024 * 1024
    except (subprocess.SubprocessError, ValueError, IndexError):
        return None


def gpu_description() -> str | None:
    """Identify NVIDIA hardware without assuming there is a GPU."""
    if not shutil.which("nvidia-smi"):
        return None
    try:
        p = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
             "--format=csv,noheader"],
            capture_output=True, text=True, check=True, timeout=8,
        )
        return p.stdout.strip().splitlines()[0].strip() or None
    except (subprocess.SubprocessError, IndexError):
        return None


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def collect_metadata() -> dict[str, Any]:
    """Capture runtime and package information; never infer GPU capability."""
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "memory_available_gib_at_start": round(available_ram_bytes() / 2**30, 3),
        "gpu": gpu_description(),
        "gpu_free_gib_at_start": (
            round(v / 2**30, 3) if (v := free_vram_bytes()) is not None else None
        ),
        "packages": {name: package_version(name) for name in (
            "cudaq", "cuda-quantum-cu12", "cuda-quantum-cu13", "numpy",
            "scipy", "pyscf", "openfermion", "openfermionpyscf")},
    }


def write_metadata(path: Path, experiment: str, **details: Any) -> None:
    """Store per-experiment environment settings, preserving other runs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    previous = json.loads(path.read_text()) if path.is_file() else {}
    previous[experiment] = {**collect_metadata(), "settings": details}
    path.write_text(json.dumps(previous, indent=2, sort_keys=True) + "\n")


def detect_cudaq_gpu(cudaq: Any) -> tuple[bool, str]:
    """Require hardware, available CUDA-Q target, *and* a VRAM query."""
    try:
        if cudaq.num_available_gpus() < 1:
            return False, "CUDA-Q reports no NVIDIA GPUs"
        if not cudaq.has_target("nvidia"):
            return False, "CUDA-Q 'nvidia' target is unavailable"
        if free_vram_bytes() is None:
            return False, "Cannot safely inspect free VRAM with nvidia-smi"
        return True, "NVIDIA GPU and CUDA-Q 'nvidia' target detected"
    except (AttributeError, RuntimeError) as exc:
        return False, f"GPU capability check failed: {exc}"
