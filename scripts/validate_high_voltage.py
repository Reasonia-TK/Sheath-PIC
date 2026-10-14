"""Reproducible 10 kV verification and PICORA comparisons; never equate completion with accuracy."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
from scipy.stats import wasserstein_distance

from sheath_pic.config import Geometry, Project
from sheath_pic.constants import E_CHARGE
from sheath_pic.inputs import Waveform
from sheath_pic.mesh import AxisymmetricMesh
from sheath_pic.particles import run_particles
from sheath_pic.solver import run_fields
from sheath_pic.storage import (
    load_field,
    plasma_observables,
    save_field,
    save_particles,
    write_json,
)
from sheath_pic.transport import build_transport
from sheath_pic.verification import (
    beam_injection,
    no_reactions,
    planar_field,
    rf_impact_reference,
    static_impact_reference,
    vacuum_cylinder_potential,
)
from sheath_pic.worker import make_waveform

ROOT = Path(__file__).resolve().parents[1]


def source_hash():
    digest = hashlib.sha256()
    for path in sorted((ROOT / "src/sheath_pic").glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def l2_difference(reference, other, weights=None):
    if weights is None:
        weights = np.ones_like(reference)
    return float(
        np.sqrt(
            np.sum(weights * (reference - other) ** 2) / max(np.sum(weights * reference**2), 1e-60)
        )
    )


def orbit_project():
    project = Project(base_dir=str(ROOT))
    project.geometry.cathode_radius_m = project.geometry.chamber_radius_m
    project.geometry.radial_cells = 4
    project.numerics.particles = 192
    project.numerics.steps_per_period = 4096
    project.numerics.stored_phases = 8
    return project


def analytical(directory):
    report = {"scope": "prescribed analytical fields and Poisson kernel, not a full CCP solution"}
    poisson = []
    for cells in (12, 24, 48, 96):
        g = Geometry(radial_cells=cells, axial_cells=cells)
        g.cathode_radius_m = g.chamber_radius_m
        mesh = AxisymmetricMesh(g)
        zero = np.zeros(mesh.shape)
        actual = mesh.solve_poisson(zero, zero, -10000)
        exact = vacuum_cylinder_potential(mesh, 10000)
        smooth = mesh.R < g.chamber_radius_m * 0.8
        poisson.append(
            {"cells": cells, "relative_l2": l2_difference(exact[smooth], actual[smooth])}
        )
    report["cylindrical_poisson"] = poisson
    static = []
    for exponent, label in ((1, "uniform"), (4 / 3, "child_langmuir"), (2, "quadratic")):
        for nz in (8, 16, 32, 64, 128):
            project = orbit_project()
            project.geometry.axial_cells = nz
            injection = beam_injection(project)
            expected, angle = static_impact_reference(project, injection, exponent)
            result = run_particles(
                project,
                planar_field(project, exponent, mesh_gradient=True),
                no_reactions(),
                collisions=False,
                injection=injection,
            )
            static.append(
                {
                    "profile": label,
                    "nz": nz,
                    "max_relative_energy_error": float(
                        max(abs(result.arrival_energy_ev / expected - 1))
                    ),
                    "max_angle_error_deg": float(max(abs(result.arrival_angle_deg - angle))),
                }
            )
    report["static_orbits"] = static
    rf = []
    for phases in (16, 32, 64, 128, 256, 512):
        project = orbit_project()
        project.numerics.stored_phases = phases
        injection = beam_injection(project, rf_phases=True)
        expected, angle, times = rf_impact_reference(project, injection)
        result = run_particles(
            project,
            planar_field(project, rf=True),
            no_reactions(),
            collisions=False,
            injection=injection,
        )
        phase_error = (
            np.remainder(result.arrival_phase_s - times + 0.5 / 400e3, 1 / 400e3) - 0.5 / 400e3
        )
        rf.append(
            {
                "phases": phases,
                "relative_l2_energy_error": l2_difference(expected, result.arrival_energy_ev),
                "max_relative_energy_error": float(
                    max(abs(result.arrival_energy_ev / expected - 1))
                ),
                "max_angle_error_deg": float(max(abs(result.arrival_angle_deg - angle))),
                "max_arrival_time_error_s": float(max(abs(phase_error))),
            }
        )
    report["rf_orbits"] = rf
    report["passed"] = (
        poisson[-1]["relative_l2"] < 1e-4
        and all(row["max_relative_energy_error"] < 1e-3 for row in static if row["nz"] == 128)
        and rf[-1]["relative_l2_energy_error"] < 1e-4
        and rf[-1]["max_angle_error_deg"] < 0.002
    )
    write_json(directory / "analytic.json", report)
    return report


def field_summary(field, mesh):
    obs = plasma_observables(field, mesh)
    volume = mesh.volume
    ne = field.electron_density_m3
    raw_energy = field.electron_energy_density_j_m3
    waveform = Waveform(
        field.waveform_time_s,
        field.waveform_voltage_v,
        1 / field.project["numerics"]["frequency_hz"],
        "saved waveform",
    )
    return {
        "diagnostics": field.diagnostics,
        "history": field.history,
        "bulk_present_phases": int(obs["bulk_present"].sum()),
        "plasma_potential_v": obs["plasma_potential_v"],
        "sheath_voltage_v": waveform.at(field.phase_s) - obs["plasma_potential_v"],
        "phase_mean_density_m3": np.sum(ne * volume[None], axis=(1, 2)) / volume.sum(),
        "electron_weighted_energy_ev": np.sum(raw_energy * volume[None], axis=(1, 2))
        / (E_CHARGE * np.maximum(np.sum(ne * volume[None], axis=(1, 2)), 1)),
    }


def field_cases(directory, backend, reuse=None):
    base = Project(base_dir=str(ROOT))
    base.numerics.backend = backend
    base.numerics.minimum_cycles = base.numerics.maximum_cycles = 4
    base.numerics.steps_per_period = 1024
    base.numerics.stored_phases = 64
    base.numerics.newton_tolerance = 2e-8
    base.numerics.newton_iterations = 60
    table = build_transport(base)
    table.save(directory / "transport.npz")
    setups = [
        ("grid24x40", 24, 40, 1024, 1e-8, 3e16),
        ("grid48x80", 48, 80, 1024, 1e-8, 3e16),
        ("grid72x120", 72, 120, 1024, 1e-8, 3e16),
        ("time128", 24, 40, 128, 1e-8, 3e16),
        ("time4096", 24, 40, 4096, 1e-8, 3e16),
        ("floor1e-10", 24, 40, 1024, 1e-10, 3e16),
        ("floor1e-6", 24, 40, 1024, 1e-6, 3e16),
        ("density1e16", 24, 40, 1024, 1e-8, 1e16),
        ("density1e17", 24, 40, 1024, 1e-8, 1e17),
    ]
    fields, summaries = {}, {}
    for label, nr, nz, steps, floor, density in setups:
        project = copy.deepcopy(base)
        project.geometry.radial_cells, project.geometry.axial_cells = nr, nz
        project.numerics.steps_per_period = steps
        project.numerics.electron_density_floor_fraction = floor
        project.plasma.density_m3 = density
        if reuse:
            path = reuse / label / "field.npz"
            if not path.is_file():
                summaries[label] = {"error": "saved field absent"}
                continue
            field = load_field(path)
            if field.diagnostics.get("field_equations_version") != 2:
                raise ValueError("この比較は改訂版の場計算に限定しています。")
            if field.diagnostics.get("transport_cache_key") != table.metadata["cache_key"]:
                raise ValueError("保存場の電子物性が現在の計算条件と異なります。")
            fields[label] = field
            summaries[label] = field_summary(
                field, AxisymmetricMesh(Project.from_dict(field.project).geometry)
            )
            continue
        folder = directory / label
        folder.mkdir()
        project.save(folder / "project.json")
        last_print = [0.0]

        def progress(value, message, name=label, state=last_print):
            now = time.monotonic()
            if now - state[0] > 10 or value == 1:
                print(f"{name}: {value:.0%} / {message}", flush=True)
                write_json(
                    directory / "active-case.json",
                    {"case": name, "fraction": value, "message": message},
                )
                state[0] = now

        try:
            field = run_fields(project, table, make_waveform(project), progress)
            save_field(folder / "field.npz", field)
            fields[label] = field
            summaries[label] = field_summary(field, AxisymmetricMesh(project.geometry))
            write_json(folder / "summary.json", summaries[label])
        except (ArithmeticError, RuntimeError) as exc:
            summaries[label] = {"error": f"{type(exc).__name__}: {exc}"}
            write_json(folder / "failure.json", summaries[label])
        write_json(directory / "field-progress.json", summaries)
    from scipy.interpolate import RegularGridInterpolator

    comparisons = []
    for baseline, other in (
        ("grid24x40", "grid48x80"),
        ("grid48x80", "grid72x120"),
        ("time128", "grid24x40"),
        ("grid24x40", "time4096"),
        ("grid24x40", "floor1e-10"),
        ("grid24x40", "floor1e-6"),
    ):
        if baseline not in fields or other not in fields:
            continue
        a, b = fields[baseline], fields[other]
        mesh_a = AxisymmetricMesh(Project.from_dict(a.project).geometry)
        mesh_b = AxisymmetricMesh(Project.from_dict(b.project).geometry)
        points = np.column_stack((mesh_a.R.ravel(), mesh_a.Z.ravel()))
        errors = {}
        for name in (
            "potential_v",
            "electron_density_m3",
            "ion_density_m3",
            "electron_energy_density_j_m3",
        ):
            reference = getattr(a, name)
            if name == "potential_v":
                interpolated = np.array(
                    [
                        RegularGridInterpolator(
                            (mesh_b.r, mesh_b.z), frame, bounds_error=False, fill_value=None
                        )(points).reshape(mesh_a.shape)
                        for frame in getattr(b, name)
                    ]
                )
            else:
                # Densities and energy density are FV cell averages. Remap by exact
                # cylindrical-volume overlap instead of comparing point interpolants.
                def overlap(a_edges, b_edges):
                    amount = np.maximum(
                        np.minimum(a_edges[1:, None], b_edges[None, 1:])
                        - np.maximum(a_edges[:-1, None], b_edges[None, :-1]),
                        0,
                    )
                    return amount / np.diff(a_edges)[:, None]

                wr = overlap(mesh_a.r_edges**2, mesh_b.r_edges**2)
                wz = overlap(mesh_a.z_edges, mesh_b.z_edges)
                interpolated = np.einsum("ij,pjk,lk->pil", wr, getattr(b, name), wz)
            errors[name] = l2_difference(reference, interpolated, mesh_a.volume[None])
        va, vb = (
            np.array(summaries[baseline]["plasma_potential_v"]),
            np.array(summaries[other]["plasma_potential_v"]),
        )
        valid = np.isfinite(va) & np.isfinite(vb)
        comparisons.append(
            {
                "baseline": baseline,
                "other": other,
                "volume_weighted_l2_differences": errors,
                "common_bulk_phases": int(valid.sum()),
                "plasma_potential_rms_difference_v": float(
                    np.sqrt(np.mean((va[valid] - vb[valid]) ** 2))
                )
                if valid.any()
                else None,
                "periodic_steady_comparison": a.converged and b.converged,
            }
        )
    report = {
        "scope": "four-cycle transients; periodic convergence required for final CCP accuracy",
        "cases": summaries,
        "comparisons": comparisons,
        "transport_range_ev": table.energy_range,
        "transport_metadata": table.metadata,
        "comparison_method": (
            "potential: cell-centre linear interpolation; "
            "densities: conservative cylindrical-volume overlap"
        ),
        "reused_field_directory": str(reuse) if reuse else None,
        "physical_accuracy_qualified": False,
    }
    write_json(directory / "fields.json", report)
    return report


def distribution_comparison(pic_energy, pic_angle, pic_weight, mc, seed):
    weight = pic_weight / pic_weight.sum()
    if not np.allclose(weight, 1 / len(weight), rtol=1e-6, atol=0):
        raise ValueError("この多項分布の統計基準は等重みのPIC試験粒子に限定しています。")
    e, a = mc.arrival_energy_ev, mc.arrival_angle_deg
    n_eff = 1 / np.sum(weight**2)
    report = {
        "pic_samples": len(pic_energy),
        "pic_effective_samples": float(n_eff),
        "mc_samples": len(e),
        "mc_timed_out": mc.diagnostics["timed_out"],
        "cdf_energy_wasserstein_ev": float(wasserstein_distance(pic_energy, e, u_weights=weight)),
        "cdf_angle_wasserstein_deg": float(wasserstein_distance(pic_angle, a, u_weights=weight)),
    }
    for label, x, y in (("energy_ev", pic_energy, e), ("angle_deg", pic_angle, a)):
        mean_x, mean_y = np.sum(weight * x), y.mean()
        variance_x = np.sum(weight * (x - mean_x) ** 2)
        standard_error = np.sqrt(variance_x / n_eff + y.var(ddof=1) / len(y))
        report[label] = {
            "pic_mean": float(mean_x),
            "mc_mean": float(mean_y),
            "difference": float(mean_y - mean_x),
            "difference_standard_error": float(standard_error),
        }
    bins = (np.linspace(0, 11000, 111), np.linspace(0, 90, 46))
    p, _ = np.histogramdd(np.column_stack((pic_energy, pic_angle)), bins=bins, weights=weight)
    q, _ = np.histogramdd(np.column_stack((e, a)), bins=bins)
    q /= len(e)
    if not np.isclose(p.sum(), 1) or not np.isclose(q.sum(), 1):
        raise ValueError("比較用IAEDFのビン範囲を超える粒子があります。")
    pooled = ((p + q) / 2).ravel()
    rng = np.random.default_rng(seed)
    effective_n = round(n_eff)
    null_differences = []
    for _ in range(300):
        one = rng.multinomial(effective_n, pooled) / effective_n
        two = rng.multinomial(len(e), pooled) / len(e)
        null_differences.append(0.5 * np.sum(abs(one - two)))
    tv = float(0.5 * np.sum(abs(p - q)))
    noise99 = float(np.quantile(null_differences, 0.99))
    report["joint_total_variation"] = tv
    report["joint_sampling_noise_q99"] = noise99
    report["joint_excess_over_noise_q99"] = max(tv - noise99, 0)
    report["joint_bins"] = {"energy_ev": 100, "angle_deg": 2}
    # Statistical comparison with an explicit 1 percentage point discretization allowance.
    report["joint_within_sampling_and_numerical_tolerance"] = tv <= noise99 + 0.01
    return report, bins, p, q


def picora_cases(directory, source):
    report = {
        "scope": "PICORA prescribed test ions; full plasma potential remains unqualified",
        "cases": {},
    }
    scenarios = [("static", 0.0), ("rf", 0.0), ("rf", 0.25), ("rf", 0.5), ("rf", 0.75)]
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment.pop("VIRTUAL_ENV", None)
    for mode, phase in scenarios:
        label = f"{mode}-{phase:g}"
        folder = directory / label
        command = [
            "uv",
            "run",
            "--project",
            str(source / "backend"),
            "--no-sync",
            "--offline",
            "python",
            "-B",
            str(ROOT / "scripts/picora_reference.py"),
            "--picora",
            str(source),
            "--output",
            str(folder),
            "--prescribed",
            mode,
            "--birth-phase",
            str(phase),
            "--cycles",
            "5",
            "--macros",
            "30000",
            "--mesh-size",
            "0.001",
        ]
        subprocess.run(command, cwd=ROOT, env=environment, check=True)
        with np.load(folder / "reference.npz", allow_pickle=False) as data:
            reference_report = json.loads(str(data["metadata"]))
            pic_e = data["collector_0_energies_ev"].copy()
            pic_a = data["collector_0_angles_deg"].copy()
            pic_w = data["collector_0_weights"].copy()
        if reference_report["collectors"][0]["truncated"]:
            raise ValueError("PICコレクタの先頭サンプルだけの結果は精度比較に使用しません。")
        project = orbit_project()
        project.numerics.particles = 30000
        project.numerics.stored_phases = 256
        project.numerics.seed = 20261014
        project.numerics.particle_max_periods = 5
        injection = beam_injection(project)
        injection.time_s[:] = phase / project.numerics.frequency_hz
        field = planar_field(project, rf=mode == "rf")
        started = time.perf_counter()
        mc = run_particles(
            project,
            field,
            no_reactions(),
            injection=injection,
            cell_step_fraction=0.125,
            collision_step_hazard=0.025,
        )
        save_particles(folder / "sheath-particles.npz", mc)
        comparison, bins, p, q = distribution_comparison(pic_e, pic_a, pic_w, mc, 20261014)
        comparison["pic_report"] = reference_report
        comparison["mc_seconds"] = time.perf_counter() - started
        comparison["mc_diagnostics"] = mc.diagnostics
        write_json(folder / "comparison.json", comparison)
        np.savez_compressed(
            folder / "comparison-histograms.npz",
            energy_edges_ev=bins[0],
            angle_edges_deg=bins[1],
            pic_iaedf_probability=p,
            mc_iaedf_probability=q,
        )
        report["cases"][label] = comparison
        write_json(directory / "picora.json", report)
        print(
            f"{label}: IAEDF TV={comparison['joint_total_variation']:.4g} / "
            f"noise99={comparison['joint_sampling_noise_q99']:.4g}",
            flush=True,
        )
    report["full_ccp_accuracy_qualified"] = False
    return report


def mc_resolution(directory, references):
    report = {
        "scope": "GUI default MC settings versus refined MC in prescribed PIC fields",
        "cases": {},
    }
    for mode, phase in (("static", 0.0), ("rf", 0.0), ("rf", 0.25), ("rf", 0.5), ("rf", 0.75)):
        label = f"{mode}-{phase:g}"
        with np.load(references / label / "reference.npz", allow_pickle=False) as data:
            pic_e, pic_a, pic_w = (
                data[f"collector_0_{name}"].copy()
                for name in ("energies_ev", "angles_deg", "weights")
            )
        project = orbit_project()
        project.numerics.particles = 30000
        project.numerics.steps_per_period = 128
        project.numerics.stored_phases = 64
        project.numerics.seed = 20261014
        project.numerics.particle_max_periods = 5
        injection = beam_injection(project)
        injection.time_s[:] = phase / project.numerics.frequency_hz
        result = run_particles(
            project, planar_field(project, rf=mode == "rf"), no_reactions(), injection=injection
        )
        comparison, _, _, _ = distribution_comparison(pic_e, pic_a, pic_w, result, 20261014)
        comparison["mc_diagnostics"] = result.diagnostics
        report["cases"][label] = comparison
        save_particles(directory / f"{label}-default.npz", result)
        write_json(directory / "mc-resolution.json", report)
        print(f"Default MC {label}: TV={comparison['joint_total_variation']:.4g}", flush=True)
    return report


def transport_refinement(directory):
    project = Project(base_dir=str(ROOT))
    baseline = build_transport(project)
    report = {
        "scope": "Boltzpmp DC convergence and mean-energy coefficient interpolation",
        "cases": {},
    }
    for label, points, angles in (("en72", 72, 24), ("angles48", 36, 48)):
        refined = copy.deepcopy(project)
        refined.transport.en_points, refined.transport.angle_cells = points, angles
        table = build_transport(refined)
        table.save(directory / f"transport-{label}.npz")
        low = max(baseline.energy_range[0], table.energy_range[0])
        high = min(baseline.energy_range[1], table.energy_range[1])
        energy = np.geomspace(low, high, 1000)
        a, b = baseline.coefficients(energy), table.coefficients(energy)
        metrics = {}
        for key in a:
            active = b[key] > max(float(b[key].max()) * 1e-6, 1e-300)
            metrics[key] = {
                "relative_l2": l2_difference(b[key], a[key]),
                "max_relative_above_one_millionth_peak": float(
                    np.max(abs(a[key][active] / b[key][active] - 1))
                )
                if active.any()
                else None,
            }
        if points == project.transport.en_points:
            metrics["mean_energy_ev"] = {
                "relative_l2": l2_difference(table.mean_energy_ev, baseline.mean_energy_ev),
                "max_relative": float(max(abs(baseline.mean_energy_ev / table.mean_energy_ev - 1))),
            }
        report["cases"][label] = {
            "metrics": metrics,
            "energy_range_ev": table.energy_range,
            "max_tail_ratio": max(table.metadata["tail_ratios"]),
            "metadata": table.metadata,
        }
    write_json(directory / "transport-refinement.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite",
        choices=("analytic", "fields", "picora", "mc-resolution", "transport", "all"),
        default="analytic",
    )
    parser.add_argument("--backend", choices=("cpu", "cuda", "auto"), default="cuda")
    parser.add_argument("--picora", type=Path)
    parser.add_argument("--reuse-fields", type=Path)
    parser.add_argument("--references", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.suite in ("picora", "all") and args.picora is None:
        raise ValueError("--picoraでPICORA-2Dのソースフォルダーを指定してください。")
    if args.suite == "mc-resolution" and args.references is None:
        raise ValueError("--referencesで指定シースのPIC比較フォルダーを指定してください。")
    directory = args.output.resolve()
    if directory.exists() and any(directory.iterdir()):
        raise ValueError("検証結果を上書きしません。新しい保存先を指定してください。")
    directory.mkdir(parents=True, exist_ok=True)
    report = {
        "source_sha256": source_hash(),
        "suite": args.suite,
        "condition": "Ar / 10 mTorr / 400 kHz / negative peak 10 kV / gap 50 mm",
        "accuracy_qualified": False,
    }
    for name, callback in (
        ("analytic", lambda: analytical(directory)),
        ("fields", lambda: field_cases(directory, args.backend, args.reuse_fields)),
        ("picora", lambda: picora_cases(directory, args.picora.resolve())),
        ("mc-resolution", lambda: mc_resolution(directory, args.references)),
        ("transport", lambda: transport_refinement(directory)),
    ):
        if args.suite == name or (args.suite == "all" and name != "mc-resolution"):
            report[name] = callback()
            write_json(directory / "report.json", report)
    print(directory / "report.json", flush=True)


if __name__ == "__main__":
    main()
