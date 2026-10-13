"""Compare CPU and CUDA with identical physics, grid, and solver settings."""

from __future__ import annotations

import argparse
import platform
import time
from pathlib import Path

import numpy as np

from sheath_pic.config import Project
from sheath_pic.constants import E_CHARGE
from sheath_pic.solver import run_fields
from sheath_pic.storage import save_field, write_json
from sheath_pic.transport import build_transport
from sheath_pic.worker import make_waveform


def relative_error(a, b):
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-30))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path)
    parser.add_argument("--nr", type=int, default=24)
    parser.add_argument("--nz", type=int, default=40)
    parser.add_argument("--cycles", type=int, default=4)
    parser.add_argument("--output", type=Path, default=Path("artifacts/cuda-benchmark"))
    args = parser.parse_args()
    project = Project.load(args.project) if args.project else Project(base_dir=str(Path.cwd()))
    project.geometry.radial_cells, project.geometry.axial_cells = args.nr, args.nz
    project.numerics.minimum_cycles = project.numerics.maximum_cycles = args.cycles
    project.validate()
    directory = args.output.resolve()
    if any((directory / filename).exists() for filename in ("cpu.npz", "cuda.npz", "report.json")):
        raise ValueError("既存ベンチマークの上書きはしません。新しい保存先を指定してください。")
    directory.mkdir(parents=True, exist_ok=True)
    table = build_transport(project)
    waveform = make_waveform(project)
    results, timings = {}, {}
    for backend in ("cpu", "cuda"):
        project.numerics.backend = backend
        started = time.perf_counter()
        last_print = [started - 30]

        def progress(value, message, mode=backend, state=last_print):
            now = time.perf_counter()
            if now - state[0] > 10:
                print(f"{mode}: {value:.0%} / {message}", flush=True)
                state[0] = now

        results[backend] = run_fields(project, table, waveform, progress)
        timings[backend] = time.perf_counter() - started
        save_field(directory / f"{backend}.npz", results[backend])
        print(f"{backend}: {timings[backend]:.3f} 秒", flush=True)
    cpu, cuda = results["cpu"], results["cuda"]
    errors = {
        name: relative_error(getattr(cpu, name), getattr(cuda, name))
        for name in ("potential_v", "electron_density_m3", "ion_density_m3")
    }
    content_cpu = cpu.electron_energy_density_j_m3
    content_cuda = cuda.electron_energy_density_j_m3
    if content_cpu is None or content_cuda is None:
        content_cpu = E_CHARGE * cpu.electron_density_m3 * cpu.mean_energy_ev
        content_cuda = E_CHARGE * cuda.electron_density_m3 * cuda.mean_energy_ev
    errors["electron_energy_density"] = relative_error(content_cpu, content_cuda)
    dense = (cpu.electron_density_m3 > cpu.electron_density_m3.max() * 1e-3) & (
        cuda.electron_density_m3 > cuda.electron_density_m3.max() * 1e-3
    )
    errors["dense_mean_energy_ev"] = relative_error(
        cpu.mean_energy_ev[dense], cuda.mean_energy_ev[dense]
    )
    report = {
        "project": project.to_dict(),
        "platform": platform.platform(),
        "cpu_seconds": timings["cpu"],
        "cuda_seconds": timings["cuda"],
        "speedup": timings["cpu"] / timings["cuda"],
        "relative_l2_differences": errors,
        "backend": {key: result.diagnostics["backend"] for key, result in results.items()},
        "steps": {key: result.diagnostics["steps"] for key, result in results.items()},
        "periodic_converged": {key: result.converged for key, result in results.items()},
        "timing_scope": "field calculation including backend initialization; excludes DC and I/O",
    }
    write_json(directory / "report.json", report)
    print(f"速度比: {report['speedup']:.2f} 倍 / 差: {errors}", flush=True)
    print(directory / "report.json", flush=True)


if __name__ == "__main__":
    main()
