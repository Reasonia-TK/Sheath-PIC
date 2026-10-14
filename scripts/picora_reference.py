"""Run an installed PICORA-2D source checkout as an independent PIC-MCC reference.

Use PICORA's existing uv environment with --no-sync; no source or dependency changes.
All inputs, outputs, and runtime caches belong to the Sheath-PIC workspace.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import sys
import time
from functools import cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sheath_pic.config import Project  # noqa: E402
from sheath_pic.constants import E_CHARGE, EPSILON_0, K_B, M_AR, M_E  # noqa: E402
from sheath_pic.inputs import Waveform, file_hash  # noqa: E402
from sheath_pic.storage import write_json  # noqa: E402


def source_fingerprint(backend: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted((backend / "es_sim").rglob("*")):
        if path.suffix in {".py", ".cu", ".cuh"}:
            digest.update(path.relative_to(backend).as_posix().encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def align_collector_angle():
    """Align the diagnostic to the 3D polar angle; all PIC dynamics remain identical.

    The native collector records a signed in-plane angle and drops v_theta / v_z.
    Compile a diagnostic-only variant in this process without editing PICORA files.
    """
    from es_sim.device import cuda

    source = cuda._read_source(cuda.KERNEL_DIR / "pic.cu")
    original = "rec_a[o] = atan2(vt, fabs(vn)) * 57.29577951308232;"
    polar = (
        "rec_a[o] = acos(fmin(1.0, fabs(vn) / "
        "fmax(sqrt(ux * ux + uy * uy + uz * uz), 1e-300))) * 57.29577951308232;"
    )
    if source.count(original) != 1:
        raise ValueError("PICORAのコレクタ実装が変わっています。角度定義を再確認してください。")
    aligned = source.replace(original, polar)
    stock_load = cuda.load_module

    @cache
    def diagnostic_module(name, defines=()):
        if name != "pic":
            return stock_load(name, defines)
        module = cuda.cp.RawModule(
            code=aligned, options=cuda._OPTIONS + tuple(f"-D{d}" for d in defines)
        )
        module.compile()
        return module

    cuda.load_module = diagnostic_module
    return {
        "native": "signed meridional angle atan2(v_t_in_plane, abs(v_n))",
        "comparison": "3D polar angle acos(abs(v_n)/speed), 0..90 degrees",
        "change_scope": "collector angle diagnostic only; deposit/Poisson/push/MCC unchanged",
        "original_expanded_kernel_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "diagnostic_expanded_kernel_sha256": hashlib.sha256(aligned.encode()).hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--picora", type=Path, required=True)
    parser.add_argument("--project", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mesh-size", type=float, default=0.001)
    parser.add_argument(
        "--plate-depth",
        type=float,
        default=0.0001,
        help="Depth of wafer conductor below z=0; annulus reflects at this depth",
    )
    parser.add_argument("--macros", type=int, default=100000)
    parser.add_argument("--dt", type=float)
    parser.add_argument("--cycles", type=int, default=4)
    parser.add_argument("--average-cycles", type=int, default=1)
    parser.add_argument("--steps", type=int, help="Short pilot; overrides --cycles")
    parser.add_argument("--seed", type=int, default=20261014)
    parser.add_argument(
        "--prescribed",
        choices=("static", "rf"),
        help="Test-charge planar PIC; validates ions in a prescribed sheath only",
    )
    parser.add_argument("--no-collisions", action="store_true")
    parser.add_argument("--particle-budget", type=int, default=2000000)
    parser.add_argument(
        "--omit-final-particles",
        action="store_true",
        help="Save fields and diagnostics without the large surviving-particle arrays",
    )
    parser.add_argument(
        "--birth-phase",
        type=float,
        default=0.0,
        help="RF phase fraction for prescribed test-ion injection",
    )
    args = parser.parse_args()
    if args.mesh_size <= 0 or (args.dt is not None and args.dt <= 0):
        raise ValueError("格子幅と時間刻みは正の値が必要です。")
    if not 100 <= args.macros <= args.particle_budget or not 0 <= args.birth_phase < 1:
        raise ValueError("粒子数・粒子上限・注入位相を確認してください。")
    if (
        args.cycles < 1
        or not 1 <= args.average_cycles <= args.cycles
        or (args.steps is not None and args.steps < 1)
    ):
        raise ValueError("周期数・平均周期数・ステップ数を確認してください。")
    backend = args.picora.resolve() / "backend"
    if not (backend / "es_sim/gpic/simulation.py").is_file():
        raise ValueError("PICORA-2Dのソースディレクトリが見つかりません。")
    directory = args.output.resolve()
    if (directory / "picora-project.json").exists():
        raise ValueError("既存のPIC結果は上書きしません。新しい保存先を指定してください。")
    directory.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("CUPY_CACHE_DIR", str(ROOT / ".cache/cupy-picora"))
    os.environ.setdefault("NUMBA_CACHE_DIR", str(ROOT / ".cache/numba-picora"))
    sys.path.insert(0, str(backend))
    from es_sim import __version__ as picora_version
    from es_sim.gpic import make_pic_simulation
    from es_sim.lxcat import parse_lxcat
    from es_sim.particles import MP
    from es_sim.project2d import Project2d

    angle_diagnostic = align_collector_angle()

    project = Project.load(args.project) if args.project else Project(base_dir=str(ROOT))
    p, n, g = project.plasma, project.numerics, project.geometry
    if p.density_mode != "self_consistent" or p.density_profile != "uniform":
        raise ValueError("このPIC照合は一様初期密度・自己整合モードから開始します。")
    if p.electron_sticking != 1:
        raise ValueError("PIC側の完全吸収境界と電子吸収率が一致しません。")
    if n.waveform_csv:
        wave = Waveform.from_csv(
            project.path(n.waveform_csv),
            n.frequency_hz,
            n.waveform_time_column,
            n.waveform_voltage_column,
            n.waveform_time_scale,
            n.waveform_voltage_scale,
        )
    else:
        wave = Waveform.demo(n.frequency_hz, n.demo_negative_peak_v)
    if args.prescribed == "static":
        wave.voltage_v[:] = -n.demo_negative_peak_v
    # Both solvers replay the same piecewise-linear voltage, including the periodic seam.
    phases = np.arange(512) / 512
    wafer_v = wave.at((phases + args.birth_phase) * wave.period_s)
    dt = args.dt or 0.1 / np.sqrt(p.density_m3 * E_CHARGE**2 / (EPSILON_0 * M_E))
    steps_per_cycle = round(wave.period_s / dt)
    # Adjust by < 1 step so completed runs contain an exact integer number of RF periods.
    if args.dt is None:
        dt = wave.period_s / steps_per_cycle
    if args.prescribed and args.dt is None:
        dt = 1e-10
    total_steps = args.steps or round(args.cycles * wave.period_s / dt)
    avg_steps = min(total_steps, round(args.average_cycles * wave.period_s / dt))
    e_path, i_path = project.path(project.electron_xsec), project.path(project.ion_xsec)
    electrons, e_warnings = parse_lxcat(e_path.read_text(encoding="utf-8-sig"), "electron")
    ions, i_warnings = parse_lxcat(i_path.read_text(encoding="utf-8-sig"), "ion")
    rc, radius, gap = g.cathode_radius_m, g.chamber_radius_m, g.gap_m
    if not 0 < args.plate_depth < gap * 0.1:
        raise ValueError("PICの電極厚さは0より大きくギャップの10%未満にしてください。")
    boundaries = [
        {"edges": [0, 3], "type": "symmetry"},
        {"edges": [1, 2], "type": "dirichlet", "voltage": 0, "see_gamma": p.ground_secondary_yield},
    ]
    collectors = [{"p1": [0, 0], "p2": [rc, 0], "label": "Wafer total", "tol": 1e-8}]
    for left, right in zip(np.linspace(0, rc, 4)[:-1], np.linspace(0, rc, 4)[1:], strict=True):
        collectors.append(
            {"p1": [left, 0], "p2": [right, 0], "label": f"r={left:g}..{right:g} m", "tol": 1e-8}
        )
    payload = {
        "coord": "rz_x0",
        "geometry": {
            # Stock PICORA allows four outer vertices and one BC per outer side.
            # Keep the wafer surface at z=0 using an embedded conductor. The reflecting
            # annulus is recessed by plate_depth; explicitly record this mismatch.
            "domain": {
                "polygon": [
                    [0, -args.plate_depth],
                    [radius, -args.plate_depth],
                    [radius, gap],
                    [0, gap],
                ]
            },
            "boundaries": boundaries,
            "regions": [
                {
                    "id": "wafer",
                    "type": "conductor",
                    "voltage": 0,
                    "polygon": [[0, -args.plate_depth], [rc, -args.plate_depth], [rc, 0], [0, 0]],
                    "voltage_waveform": {
                        "freq_hz": n.frequency_hz,
                        "phase": phases.tolist(),
                        "v": wafer_v.tolist(),
                    },
                    "see_gamma": p.cathode_secondary_yield,
                }
            ],
        },
        "mesh": {"size": args.mesh_size, "mode": "cartesian"},
        "pic": {
            "initial_plasma": {
                "density": p.density_m3,
                "te_ev": p.initial_mean_energy_ev * 2 / 3,
                "ti_ev": K_B * p.gas_temperature_k / E_CHARGE,
                # PICORA multiplies this input by proton mass, not atomic mass.
                "ion_mass_amu": M_AR / MP,
                "seed": args.seed,
            },
            "n_macro": args.macros,
            "dt": dt,
            "n_steps": total_steps,
            "frame_every": max(1000, total_steps // 200),
            "avg_steps": avg_steps,
            "phase_bins": n.stored_phases,
            "ion_subcycle": 1,
            "radial_weighting": True,
            "collectors": collectors,
            "mcc": {
                "gas": {
                    "name": "Ar",
                    "pressure_pa": p.pressure_pa,
                    "temperature_k": p.gas_temperature_k,
                },
                "electron_processes": [value.model_dump() for value in electrons],
                "ion_processes": [value.model_dump() for value in ions],
                "ion_energy_frame": "com" if p.ion_energy_frame == "center_of_mass" else "lab",
                "ionization_split": "half",
                "seed": args.seed,
            },
            "convergence": {"enabled": True, "tol": 0.01, "hold": 3, "stop": False},
        },
    }
    pic_project = Project2d.model_validate(payload)
    injection = None
    if args.prescribed:
        from sheath_pic.verification import beam_injection

        width = 4 * args.mesh_size
        prescribed = project.to_dict()
        prescribed["numerics"]["particles"] = args.macros
        injection = beam_injection(Project.from_dict(prescribed))
        # PIC coordinates (x,y,vz) map to (lateral,normal,second lateral).
        injection.position_m[:, 0] = width / 2
        injection.position_m[:, 1] = 0
        payload["coord"] = "xy"
        payload["geometry"] = {
            "domain": {"polygon": [[0, 0], [width, 0], [width, gap], [0, gap]]},
            "boundaries": [
                {
                    "edges": [0],
                    "type": "dirichlet",
                    "voltage": 0,
                    "voltage_waveform": {
                        "freq_hz": n.frequency_hz,
                        "phase": phases.tolist(),
                        "v": wafer_v.tolist(),
                    },
                },
                {"edges": [2], "type": "dirichlet", "voltage": 0},
                {"edges": [1, 3], "type": "symmetry"},
            ],
        }
        payload["pic"]["initial_plasma"]["density"] = 1.0
        payload["pic"]["dt"] = dt
        payload["pic"]["radial_weighting"] = False
        payload["pic"]["collectors"] = [
            {"p1": [0, 0], "p2": [width, 0], "label": "Prescribed sheath ions", "tol": 1e-8}
        ]
        payload["pic"]["mcc"]["electron_processes"] = []
        payload["pic"]["convergence"]["enabled"] = False
        payload["pic"]["avg_steps"] = total_steps
        avg_steps = total_steps
        payload["pic"]["frame_every"] = 256
        pic_project = Project2d.model_validate(payload)
    if args.no_collisions:
        pic_project.pic.mcc = None
    write_json(directory / "picora-project.json", pic_project.model_dump(mode="json"))
    project.save(directory / "sheath-project.json")
    started = time.perf_counter()
    last_print = [started - 60]
    stop_reason = [None]

    def progress(frame):
        now = time.perf_counter()
        if now - last_print[0] > 10 or frame["step"] >= total_steps:
            print(
                f"PIC {frame['step']}/{total_steps} steps / "
                f"{frame['t'] / wave.period_s:.4g} cycles / {now - started:.1f} s",
                flush=True,
            )
            last_print[0] = now
            write_json(
                directory / "progress.json",
                {
                    "steps": frame["step"],
                    "total_steps": total_steps,
                    "simulated_cycles": frame["t"] / wave.period_s,
                    "elapsed_s": now - started,
                    "diagnostics": frame["diag"],
                    "convergence": frame["convergence"],
                },
            )
        if args.prescribed and frame["diag"]["n_i"] == 0:
            stop_reason[0] = "All injected test ions have left the domain"

    def should_stop():
        if stop_reason[0]:
            return True
        if sim.step_count and sim.step_count % 256 == 0:
            counts = sim._cnt.get(stream=sim._stream)[:2]
            if int(counts.sum()) > args.particle_budget:
                stop_reason[0] = "PIC particle budget exceeded; reference not qualified"
                return True
        return False

    try:
        sim = make_pic_simulation(pic_project, device="cuda")
        if injection is not None:
            empty = np.array([], dtype=float)
            sim.set_particles("electron", **dict.fromkeys(("x", "y", "vx", "vy", "vz", "w"), empty))
            original = sim.get_particles("ion")
            initial_v = injection.velocity_ms.copy()
            # set_particles expects staggered velocities; apply the initial half kick.
            initial_ez = float(wafer_v[0]) / gap
            initial_v[:, 2] -= 0.5 * E_CHARGE / M_AR * initial_ez * sim.dt
            sim.set_particles(
                "ion",
                x=injection.position_m[:, 0],
                y=injection.position_m[:, 2],
                vx=initial_v[:, 0],
                vy=initial_v[:, 2],
                vz=initial_v[:, 1],
                w=original["w"],
            )
            # Freeze initialization weighting: charges are tiny but not zero, so collectors
            # retain physical weights and the normal PIC deposit/Poisson/push path runs.
        print(
            f"PICORA {picora_version} / grid {sim.grid.nx}x{sim.grid.ny} / "
            f"dt={sim.dt:.6g} / nodes={sim.n_nodes} / {total_steps} steps",
            flush=True,
        )
        sim.run_batch(callback=progress, should_stop=should_stop, store_frames=False)
        arrays = {
            "nodes_m": sim.mesh.nodes,
            "triangles": sim.mesh.triangles,
            "r_m": sim.grid.xs,
            "z_m": sim.grid.ys,
        }
        for key, value in sim.fields.items():
            if isinstance(value, np.ndarray):
                arrays[f"average_{key}"] = value
        if sim.cycle:
            for key in ("phi", "n_e", "n_i", "te_ev", "ion_rate"):
                arrays[f"phase_{key}"] = sim.cycle[key]
            arrays["phase_counts"] = sim._cyc_count.get()
            arrays["phase_s"] = (
                (np.arange(sim.cycle["bins"]) + 0.5) * wave.period_s / sim.cycle["bins"]
            )
        collector_info = []
        for index, collection in enumerate(sim.collector_results or []):
            for key in ("energies_ev", "angles_deg", "weights"):
                arrays[f"collector_{index}_{key}"] = collection[key]
            collector_info.append(
                {k: v for k, v in collection.items() if not isinstance(v, np.ndarray)}
            )
        # Preserve kinetic states for follow-up diagnostics without writing into PICORA.
        if not args.omit_final_particles:
            for species in ("electron", "ion"):
                for key, value in sim.get_particles(species).items():
                    arrays[f"final_{species}_{key}"] = value
        report = {
            "status": "completed",
            "reference_qualified": False,
            "qualification_note": "PICの格子・時間・粒子・周期収束を別途確認するまで比較候補。",
            "reference_scope": "prescribed planar sheath test ions"
            if args.prescribed
            else "full axisymmetric CCP",
            "prescribed": args.prescribed,
            "birth_phase": args.birth_phase,
            "collisions_enabled": not args.no_collisions,
            "angle_diagnostic": angle_diagnostic,
            "stop_reason": stop_reason[0],
            "particle_budget": args.particle_budget,
            "final_particle_states_saved": not args.omit_final_particles,
            "picora_version": picora_version,
            "source_directory": str(args.picora.resolve()),
            "source_sha256": source_fingerprint(backend),
            "versions": {
                key: importlib.metadata.version(key) for key in ("numpy", "scipy", "cupy-cuda13x")
            },
            "source_electron_xsec_sha256": file_hash(e_path),
            "source_ion_xsec_sha256": file_hash(i_path),
            "collision_import_notes": e_warnings + i_warnings,
            "elapsed_s": time.perf_counter() - started,
            "setup_s": sim.setup_s,
            "steps": sim.step_count,
            "simulated_cycles": sim.t / wave.period_s,
            "dt_s": sim.dt,
            "omega_pe_dt": np.sqrt(
                pic_project.pic.initial_plasma.density * E_CHARGE**2 / (EPSILON_0 * M_E)
            )
            * sim.dt,
            "grid": {
                "nr": sim.grid.nx,
                "nz": sim.grid.ny,
                "dr_m": sim.grid.dx,
                "dz_m": sim.grid.dy,
            },
            "geometry_difference": {
                "reflecting_annulus_recess_m": 0 if args.prescribed else args.plate_depth,
                "ground_cathode_area_ratio": 1
                if args.prescribed
                else (np.pi * radius**2 + 2 * np.pi * radius * (gap + args.plate_depth))
                / (np.pi * rc**2),
                "prescribed_plane": bool(args.prescribed),
            },
            "initial_debye_m": float(
                np.sqrt(EPSILON_0 * p.initial_mean_energy_ev * 2 / 3 / (E_CHARGE * p.density_m3))
            ),
            "physical_ion_mass_kg": sim.m_ion,
            "initial_macros_per_species": args.macros,
            "convergence": sim.conv.result() if sim.conv is not None else None,
            "warnings": sim.warnings,
            "collectors": collector_info,
            "timing": sim.timing,
            "history_stride": max(1, sim.step_count // 2000),
            "history": {
                key: values[:: max(1, sim.step_count // 2000)]
                for key, values in sim.history.items()
            },
            "density_mode": "negligible charge test ions" if args.prescribed else "self_consistent",
            "waveform": wave.source,
            "average_steps": sim.fields["avg_steps"],
        }
        np.savez_compressed(
            directory / "reference.npz", **arrays, metadata=json.dumps(report, ensure_ascii=False)
        )
        write_json(directory / "report.json", report)
        print(f"PIC 完了 {report['elapsed_s']:.2f} s / {directory / 'reference.npz'}", flush=True)
    except Exception as exc:
        write_json(
            directory / "failure.json",
            {"error": f"{type(exc).__name__}: {exc}", "elapsed_s": time.perf_counter() - started},
        )
        raise


if __name__ == "__main__":
    main()
