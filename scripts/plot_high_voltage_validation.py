"""Create standalone scientific figures and compact metrics from saved validation outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import NullFormatter, StrMethodFormatter

from sheath_pic.storage import write_json


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--fields", type=Path, help="Directory containing fields.json and report.json"
    )
    parser.add_argument(
        "--transport", type=Path, help="Directory containing transport-refinement.json"
    )
    args = parser.parse_args()
    source, output = args.data.resolve(), args.output.resolve()
    field_directory = args.fields.resolve() if args.fields else source / "final-fields"
    transport_directory = args.transport.resolve() if args.transport else source / "final-transport"
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.grid": True,
            "grid.alpha": 0.2,
            "figure.facecolor": "#faf7f2",
            "axes.facecolor": "#faf7f2",
            "savefig.facecolor": "#faf7f2",
        }
    )
    colors = ("#b96b4e", "#4b7e80", "#8b74a4")
    analytic = read_json(source / "analytic/analytic.json")
    fields = read_json(field_directory / "fields.json")
    transport = read_json(transport_directory / "transport-refinement.json")
    pic = read_json(source / "picora-comparison/picora.json")
    default_mc = read_json(source / "mc-resolution/mc-resolution.json")
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), constrained_layout=True)
    rows = analytic["cylindrical_poisson"]
    axes[0].loglog(
        [row["cells"] for row in rows],
        [row["relative_l2"] * 100 for row in rows],
        "o-",
        color=colors[0],
    )
    axes[0].set(
        xlabel="Cells per direction",
        ylabel="Relative L2 error (%)",
        title="10 kV cylindrical Poisson",
    )
    for profile, color in (("child_langmuir", colors[0]), ("quadratic", colors[1])):
        rows = [r for r in analytic["static_orbits"] if r["profile"] == profile]
        axes[1].loglog(
            [r["nz"] for r in rows],
            [r["max_relative_energy_error"] * 100 for r in rows],
            "o-",
            label=profile.replace("_", " "),
            color=color,
        )
    axes[1].set(
        xlabel="Axial cells", ylabel="Max impact-energy error (%)", title="Prescribed static sheath"
    )
    axes[1].legend(fontsize=9)
    rows = analytic["rf_orbits"]
    axes[2].loglog(
        [r["phases"] for r in rows],
        [r["max_relative_energy_error"] * 100 for r in rows],
        "o-",
        color=colors[0],
        label="max energy error",
    )
    axes[2].loglog(
        [r["phases"] for r in rows],
        [r["relative_l2_energy_error"] * 100 for r in rows],
        "s--",
        color=colors[1],
        label="L2 energy error",
    )
    axes[2].set(
        xlabel="Stored RF phases", ylabel="Impact-energy error (%)", title="400 kHz / 10 kV RF"
    )
    axes[2].legend(fontsize=9)
    for axis, ticks in zip(
        axes, ((12, 24, 48, 96), (8, 16, 32, 64, 128), (16, 32, 64, 128, 256, 512)), strict=True
    ):
        axis.set_xticks(ticks)
        axis.xaxis.set_major_formatter(StrMethodFormatter("{x:.0f}"))
        axis.xaxis.set_minor_formatter(NullFormatter())
        axis.tick_params(axis="x", labelsize=9)
    fig.savefig(output / "analytic-convergence.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for label, color in zip(("grid24x40", "grid48x80", "grid72x120"), colors, strict=True):
        result = fields["cases"][label]
        history = result["history"]
        cycles = [r["cycle"] for r in history]
        axes[0, 0].plot(
            cycles,
            [r["mean_electron_density_m3"] / 1e16 for r in history],
            "o-",
            label=label.replace("grid", ""),
            color=color,
        )
        axes[0, 1].plot(cycles, [r["mean_electron_energy_ev"] for r in history], "o-", color=color)
        phases = np.arange(len(result["plasma_potential_v"])) / len(result["plasma_potential_v"])
        axes[1, 0].plot(phases, result["plasma_potential_v"], color=color)
        axes[1, 1].plot(
            cycles[1:], [r["periodic_error"] * 100 for r in history[1:]], "o-", color=color
        )
    axes[0, 0].set(
        xlabel="Completed RF cycles",
        ylabel="Mean ne (1e16 m^-3)",
        title="Uniform initial ne = 3e16 m^-3",
    )
    axes[0, 0].legend(title="r x z cells")
    axes[0, 1].set(
        xlabel="Completed RF cycles",
        ylabel="Electron-weighted mean energy (eV)",
        title="Energy evolution",
    )
    axes[1, 0].set(
        xlabel="RF phase",
        ylabel="Representative plasma potential (V)",
        title="Fourth cycle / quasineutral mask",
    )
    axes[1, 1].set(
        xlabel="Completed RF cycles",
        ylabel="Period-to-period change (%)",
        title="Periodic steady state: not reached",
    )
    axes[1, 1].axhline(2, color="#777", linestyle="--", label="default 2% target")
    fig.savefig(output / "field-convergence.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(3, 3, figsize=(12, 9), constrained_layout=True)
    for row, label in enumerate(("static-0", "rf-0", "rf-0.5")):
        with np.load(source / "picora-comparison" / label / "comparison-histograms.npz") as data:
            energy = data["energy_edges_ev"] / 1000
            angle = data["angle_edges_deg"]
            p, q = data["pic_iaedf_probability"], data["mc_iaedf_probability"]
        for probability, color, name in ((p, colors[0], "PICORA"), (q, colors[1], "Test-ion MC")):
            axes[row, 0].stairs(
                probability.sum(axis=1) / np.diff(energy), energy, color=color, label=name
            )
            axes[row, 1].stairs(probability.sum(axis=0) / np.diff(angle), angle, color=color)
        difference = (q - p).T
        image = axes[row, 2].pcolormesh(
            energy, angle, difference, cmap="RdBu_r", vmin=-0.005, vmax=0.005, rasterized=True
        )
        fig.colorbar(image, ax=axes[row, 2], label="MC - PIC probability per bin", shrink=0.85)
        axes[row, 0].set(xlabel="Impact energy (keV)", ylabel="IEDF (1/keV)", title=label)
        axes[row, 1].set(xlabel="Polar angle (degree)", ylabel="IADF (1/degree)", xlim=(0, 20))
        axes[row, 2].set(
            xlabel="Impact energy (keV)",
            ylabel="Polar angle (degree)",
            ylim=(0, 30),
            title=f"IAEDF / TV={pic['cases'][label]['joint_total_variation']:.3f}",
        )
    axes[0, 0].legend(fontsize=9)
    fig.suptitle(
        "Prescribed planar sheath / Ar 10 mTorr, 300 K / 30,000 ions per code\n"
        "3D polar-angle diagnostic; electrons and plasma-potential feedback excluded",
        fontsize=12,
    )
    fig.savefig(output / "picora-ion-comparison.png", dpi=180)
    plt.close(fig)
    full_pic_trials = {}
    for folder in sorted(source.glob("picora-*")):
        report_path = folder / "report.json"
        if report_path.is_file():
            trial = read_json(report_path)
            if trial.get("reference_scope") != "full axisymmetric CCP":
                continue
            last = {
                key: values[-1]
                for key, values in trial["history"].items()
                if isinstance(values, list) and values
            }
            full_pic_trials[folder.name] = {
                key: trial[key]
                for key in (
                    "reference_qualified",
                    "simulated_cycles",
                    "steps",
                    "dt_s",
                    "omega_pe_dt",
                    "elapsed_s",
                    "grid",
                    "initial_debye_m",
                    "initial_macros_per_species",
                    "geometry_difference",
                    "warnings",
                )
            }
            full_pic_trials[folder.name]["last_sampled_diagnostics"] = last
        elif (folder / "failure.json").is_file() and (folder / "progress.json").is_file():
            full_pic_trials[folder.name] = {
                "reference_qualified": False,
                "failure": read_json(folder / "failure.json"),
                "last_progress": read_json(folder / "progress.json"),
            }
    metrics = {
        "scope": "Numerical verification; full CCP accuracy remains unqualified",
        "analytic": analytic,
        "field_comparisons": fields["comparisons"],
        "field_cases": {
            label: {
                "periodic_error": row["diagnostics"]["periodic_error"],
                "completed_cycles": row["diagnostics"]["completed_cycles"],
                "final_mean_ne_m3": row["history"][-1]["mean_electron_density_m3"],
                "final_mean_energy_ev": row["history"][-1]["mean_electron_energy_ev"],
                "minimum_debye_m": row["diagnostics"]["minimum_debye_m"],
                "elapsed_total_s": row["diagnostics"]["elapsed_total_s"],
                "particle_balance_error": row["diagnostics"]["electron_balance_relative_error"],
                "energy_balance_error": row["diagnostics"][
                    "electron_energy_balance_relative_error"
                ],
            }
            for label, row in fields["cases"].items()
            if "diagnostics" in row
        },
        "picora_ions": {
            label: {
                key: value
                for key, value in row.items()
                if key not in {"pic_report", "mc_diagnostics"}
            }
            for label, row in pic["cases"].items()
        },
        "default_mc_comparisons": {
            label: {key: value for key, value in row.items() if key != "mc_diagnostics"}
            for label, row in default_mc["cases"].items()
        },
        "transport_refinement": {
            label: {key: value for key, value in row.items() if key != "metadata"}
            for label, row in transport["cases"].items()
        },
        "transport_cache_key": fields["transport_metadata"]["cache_key"],
        "source_sha256": read_json(field_directory / "report.json")["source_sha256"],
        "execution_source_sha256": {
            "analytic": read_json(source / "analytic/report.json")["source_sha256"],
            "picora_ions": read_json(source / "picora-comparison/report.json")["source_sha256"],
            "fields": read_json(field_directory / "report.json")["source_sha256"],
            "transport": read_json(transport_directory / "report.json")["source_sha256"],
        },
        "picora_source_sha256": next(iter(pic["cases"].values()))["pic_report"]["source_sha256"],
        "full_pic_trials": full_pic_trials,
        "full_ccp_accuracy_qualified": False,
    }
    write_json(output / "metrics.json", metrics)
    print(output, flush=True)


if __name__ == "__main__":
    main()
