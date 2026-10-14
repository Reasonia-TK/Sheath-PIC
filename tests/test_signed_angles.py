import json
import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from sheath_pic.config import Project
from sheath_pic.constants import E_CHARGE, M_AR
from sheath_pic.particles import ParticleInjection, ParticleResult, distributions, run_particles
from sheath_pic.storage import export_distributions, load_particles, save_particles
from sheath_pic.verification import no_reactions, planar_field

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def signed_result():
    project = Project()
    project.numerics.particles = 500
    project.numerics.angle_bins = 18
    energy = np.linspace(100, 9000, 500)
    signed = np.resize([-90, -30, 0, 30, 90], 500).astype(float)
    radius = np.linspace(0, project.geometry.cathode_radius_m, 500)
    arrays = distributions(energy, abs(signed), radius, project, 1e18)
    return ParticleResult(
        *arrays,
        energy,
        abs(signed),
        radius,
        np.zeros(500),
        np.zeros(500, dtype=int),
        {
            "injected": 500,
            "arrived": 500,
            "field_periodic_converged": True,
            "mean_energy_ev": float(energy.mean()),
            "mean_angle_deg": float(abs(signed).mean()),
            "mean_meridional_angle_deg": float(signed.mean()),
            "lost_ground": 0,
            "timed_out": 0,
            "mean_collision_count": 0,
            "seed": 20261014,
            "source": "synthetic signed-angle validation",
            "warnings": [],
        },
        arrival_meridional_angle_deg=signed,
    )


def test_signed_impact_angles_follow_local_meridian_and_ignore_azimuthal_velocity():
    project = Project(base_dir=str(ROOT))
    project.geometry.cathode_radius_m = project.geometry.chamber_radius_m
    project.geometry.radial_cells, project.geometry.axial_cells = 4, 8
    project.numerics.particles, project.numerics.stored_phases = 192, 8
    phi = np.arange(192) / 192 * 2 * np.pi
    radius = 0.05
    position = np.column_stack((radius * np.cos(phi), radius * np.sin(phi), np.full(192, 0.04)))
    radial = np.resize([-10000, 0, 10000], 192)
    azimuthal = np.resize([0, 10000, 10000], 192)
    velocity = np.column_stack(
        (
            radial * np.cos(phi) - azimuthal * np.sin(phi),
            radial * np.sin(phi) + azimuthal * np.cos(phi),
            np.full(192, -1000),
        )
    )
    acceleration = E_CHARGE / M_AR * 10000 / project.geometry.gap_m
    time = 0.08 / (1000 + np.sqrt(1000**2 + 0.08 * acceleration))
    impact = position + velocity * time
    normal_speed = 1000 + acceleration * time
    impact_radial = np.sum(velocity[:, :2] * impact[:, :2], axis=1) / np.linalg.norm(
        impact[:, :2], axis=1
    )
    expected = np.degrees(np.arctan2(impact_radial, normal_speed))
    result = run_particles(
        project,
        planar_field(project),
        no_reactions(),
        collisions=False,
        injection=ParticleInjection(position, velocity, np.zeros(192)),
    )
    assert result.diagnostics["arrived"] == 192
    assert result.arrival_meridional_angle_deg == pytest.approx(expected, abs=1e-10)
    assert result.arrival_meridional_angle_deg.min() < 0 < result.arrival_meridional_angle_deg.max()
    assert np.all(abs(result.arrival_meridional_angle_deg) <= result.arrival_angle_deg + 1e-10)
    assert np.max(result.arrival_angle_deg - abs(result.arrival_meridional_angle_deg)) > 1


def test_signed_pdf_preserves_energy_marginal_ring_counts_and_angular_resolution(signed_result):
    result = signed_result
    view = result.angular_distribution("signed_meridional")
    de, da = np.diff(result.energy_edges_ev), np.diff(view.angle_edges_deg)
    assert view.angle_edges_deg[[0, -1]] == pytest.approx([-90, 90])
    assert da == pytest.approx(np.diff(result.angle_edges_deg)[0])
    assert np.sum(view.iaedf * de[:, None] * da[None, :]) == pytest.approx(1)
    assert np.sum(view.iadf * da) == pytest.approx(1)
    assert np.sum(view.iaedf * da[None, :], axis=1) == pytest.approx(result.iedf)
    assert view.counts.sum(axis=(1, 2)) == pytest.approx(result.counts.sum(axis=(1, 2)))
    assert np.sum(view.radial_iaedf * de[None, :, None] * da[None, None, :], axis=(1, 2)) == (
        pytest.approx(1)
    )
    centers = 0.5 * (view.angle_edges_deg[1:] + view.angle_edges_deg[:-1])
    assert np.sum(view.iadf[centers < 0] * da[centers < 0]) == pytest.approx(0.4)


def test_signed_angles_survive_npz_and_export_and_legacy_results_remain_polar(
    signed_result, tmp_path
):
    result = signed_result
    path = tmp_path / "particles.npz"
    save_particles(path, result)
    restored = load_particles(path)
    assert np.array_equal(
        restored.arrival_meridional_angle_deg, result.arrival_meridional_angle_deg
    )
    export_distributions(tmp_path / "csv", restored)
    signed_csv = np.genfromtxt(tmp_path / "csv/iadf_signed.csv", names=True, delimiter=",")
    assert np.sum(signed_csv["pdf_per_degree"] * 5) == pytest.approx(1)
    arrivals = np.genfromtxt(tmp_path / "csv/arrivals.csv", names=True, delimiter=",")
    assert arrivals["signed_meridional_angle_deg"] == pytest.approx(
        result.arrival_meridional_angle_deg
    )
    assert arrivals["angle_deg"] == pytest.approx(result.arrival_angle_deg)
    metadata = json.loads((tmp_path / "csv/particle_diagnostics.json").read_text(encoding="utf-8"))
    assert metadata["exported_angle_definitions"]["iadf_signed.csv / iaedf_signed.csv"] == (
        "signed_meridional"
    )
    save_particles(path, replace(result, arrival_meridional_angle_deg=None))
    legacy = load_particles(path)
    assert legacy.arrival_meridional_angle_deg is None
    assert np.array_equal(legacy.angular_distribution().iaedf, result.iaedf)
    with pytest.raises(ValueError, match="再実行"):
        legacy.angular_distribution("signed_meridional")
    export_distributions(tmp_path / "legacy-csv", legacy)
    assert not (tmp_path / "legacy-csv/iadf_signed.csv").exists()


def test_gui_angle_switch_ring_view_and_legacy_fallback(signed_result, tmp_path):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mpl-cache"))
    from PySide6.QtWidgets import QApplication

    from sheath_pic.gui import MainWindow

    app = QApplication.instance() or QApplication([])
    window = MainWindow(tmp_path)
    path = tmp_path / "particles.npz"
    save_particles(path, signed_result)
    window.load_outputs({"particles": str(path)})
    window.angle_combo.setCurrentIndex(window.angle_combo.findData("signed_meridional"))
    app.processEvents()
    axes = window.figures["particles"][0].axes
    assert axes[1].get_xlim() == pytest.approx((-90, 90))
    assert axes[2].get_ylim() == pytest.approx((-90, 90))
    assert "Signed meridional" in axes[1].get_xlabel()
    assert "断面内" in window.particle_summary.text()
    window.radial_combo.setCurrentIndex(1)
    app.processEvents()
    axes = window.figures["particles"][0].axes
    assert np.sum(axes[0].lines[0].get_ydata() * np.diff(signed_result.energy_edges_ev)) == (
        pytest.approx(1)
    )
    line = axes[0].lines[0]
    assert line.get_xdata()[line.get_ydata() > 0].max() < 1000
    saved = json.loads(window.settings_path.read_text(encoding="utf-8"))
    assert saved["particle_angle"] == "signed_meridional"
    save_particles(path, replace(signed_result, arrival_meridional_angle_deg=None))
    window.load_outputs({"particles": str(path)})
    assert window.angle_combo.currentData() == "polar"
    assert not window.angle_combo.model().item(1).isEnabled()
    assert "再実行" in window.angle_help.text()
    window.close()
    save_particles(path, signed_result)
    reopened = MainWindow(tmp_path)
    reopened.load_outputs({"particles": str(path)})
    assert reopened.angle_combo.currentData() == "signed_meridional"
    reopened.close()
