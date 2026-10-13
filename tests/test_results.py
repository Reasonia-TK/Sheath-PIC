import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from sheath_pic.config import Project
from sheath_pic.constants import E_CHARGE, K_B
from sheath_pic.inputs import Waveform
from sheath_pic.mesh import AxisymmetricMesh
from sheath_pic.particles import distributions, run_particles
from sheath_pic.solver import FieldResult, run_fields
from sheath_pic.storage import (
    field_fingerprint,
    load_checkpoint,
    load_field,
    plasma_observables,
    save_checkpoint,
    save_field,
)
from sheath_pic.transport import build_transport
from sheath_pic.worker import archive_inputs, make_waveform

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def dc_table():
    project = Project(base_dir=str(ROOT))
    project.transport.en_points = 8
    project.transport.angle_cells = 16
    return build_transport(project)


def test_real_boltzpmp_dc_and_eedf_normalization(dc_table):
    assert dc_table.metadata["calculation"] == "DC"
    assert dc_table.mean_energy_ev.min() > 0
    assert (dc_table.mobility_m2_vs > 0).all()
    assert dc_table.ionization_m3_s[-1] > dc_table.ionization_m3_s[0]
    assert np.trapezoid(dc_table.eedf, dc_table.eedf_energy_ev, axis=1) == pytest.approx(1)
    assert max(dc_table.metadata["tail_ratios"]) <= 1e-5


def test_distributions_are_marginals_and_flux_weights_use_ring_area():
    project = Project()
    project.numerics.particles = 1000
    rng = np.random.default_rng(30)
    energy = rng.uniform(1, 10000, 1000)
    angle = rng.uniform(0, 89, 1000)
    radius = project.geometry.cathode_radius_m * np.sqrt(rng.random(1000))
    ee, aa, rr, counts, joint, iedf, iadf, _, flux = distributions(
        energy, angle, radius, project, 1e18
    )
    de, da = np.diff(ee), np.diff(aa)
    assert np.sum(joint * de[:, None] * da[None, :]) == pytest.approx(1)
    assert np.sum(iedf * de) == pytest.approx(1)
    assert np.sum(iadf * da) == pytest.approx(1)
    assert np.sum(flux * np.pi * np.diff(rr**2)) == pytest.approx(1e18)
    assert counts.sum() == 1000


def test_10kv_coupled_solver_budgets_and_checkpoint_restart(dc_table, tmp_path):
    project = Project(base_dir=str(ROOT))
    project.geometry.radial_cells = 4
    project.geometry.axial_cells = 6
    project.numerics.minimum_cycles = 1
    project.numerics.maximum_cycles = 1
    project.numerics.steps_per_period = 32
    project.numerics.stored_phases = 8
    project.numerics.newton_iterations = 60
    project.numerics.periodic_tolerance = 1e-12
    wave = Waveform.demo(400e3, 10000)
    states = []
    field = run_fields(project, dc_table, wave, checkpoint=lambda value: states.append(value))
    assert np.isfinite(field.potential_v).all()
    assert field.electron_density_m3.min() >= 0
    assert field.ion_density_m3.min() >= 0
    assert field.diagnostics["ion_balance_relative_error"] < 1e-10
    assert field.diagnostics["electron_balance_relative_error"] < 1e-4
    assert field.diagnostics["electron_energy_balance_relative_error"] < 1e-8
    path = tmp_path / "field.npz"
    save_field(path, field)
    restored = load_field(path)
    assert np.array_equal(restored.potential_v, field.potential_v)
    checkpoint_path = tmp_path / "checkpoint.npz"
    save_checkpoint(checkpoint_path, states[-1], "fingerprint")
    state = load_checkpoint(checkpoint_path, "fingerprint")
    with pytest.raises(ValueError, match="一致"):
        load_checkpoint(checkpoint_path, "changed")
    project.numerics.maximum_cycles = 2
    continued = run_fields(project, dc_table, wave, restart=state)
    uninterrupted = run_fields(project, dc_table, wave)
    assert continued.potential_v == pytest.approx(uninterrupted.potential_v, rel=1e-10)
    assert continued.diagnostics["electron_energy_balance_relative_error"] < 1e-8


def synthetic_constant_field(project):
    mesh = AxisymmetricMesh(project.geometry)
    shape = (project.numerics.stored_phases, *mesh.shape)
    n = np.full(shape, 1e16)
    potential = np.broadcast_to(-1000 + 1000 * mesh.Z / project.geometry.gap_m, shape).copy()
    return FieldResult(
        np.arange(shape[0]) / shape[0] / project.numerics.frequency_hz,
        potential,
        np.zeros(shape),
        np.full(shape, -1000 / project.geometry.gap_m),
        n,
        n.copy(),
        np.full(shape, 4.5),
        np.zeros(shape),
        np.zeros(shape),
        {"periodic_converged": True},
        [],
        project.to_dict(),
        np.linspace(0, 1 / project.numerics.frequency_hz, 3),
        np.full(3, -1000),
    )


def test_test_particles_obey_collisionless_energy_conservation_and_seed(dc_table):
    project = Project(base_dir=str(ROOT))
    project.geometry.cathode_radius_m = project.geometry.chamber_radius_m
    project.geometry.radial_cells = 4
    project.geometry.axial_cells = 8
    project.numerics.particles = 1000
    project.numerics.particle_max_periods = 3
    project.numerics.steps_per_period = 128
    project.numerics.stored_phases = 8
    # Uniform volume birth in a constant field implies uniform impact energy over 0..1000 eV.
    field = synthetic_constant_field(project)
    result = run_particles(project, field, dc_table, collisions=False)
    repeated = run_particles(project, field, dc_table, collisions=False)
    assert result.arrival_energy_ev == pytest.approx(repeated.arrival_energy_ev, rel=0, abs=0)
    assert result.diagnostics["mean_energy_ev"] == pytest.approx(500, abs=35)
    thermal_ev = 1.5 * K_B * project.plasma.gas_temperature_k / E_CHARGE
    assert result.arrival_energy_ev.max() <= 1000 + 10 * thermal_ev
    assert result.diagnostics["injected"] == (
        result.diagnostics["arrived"]
        + result.diagnostics["lost_ground"]
        + result.diagnostics["timed_out"]
    )


def test_no_quasineutral_bulk_is_not_reported_as_zero_volts():
    project = Project()
    field = synthetic_constant_field(project)
    field.electron_density_m3[:] = 0
    obs = plasma_observables(field, AxisymmetricMesh(project.geometry))
    assert not obs["bulk_present"].any()
    assert np.isnan(obs["plasma_potential_v"]).all()


def test_external_pair_removal_competes_with_particle_wall_arrival(dc_table):
    project = Project(base_dir=str(ROOT))
    project.geometry.cathode_radius_m = project.geometry.chamber_radius_m
    project.geometry.radial_cells = 4
    project.geometry.axial_cells = 8
    project.plasma.density_m3 = 1e16
    project.plasma.density_mode = "maintained"
    project.numerics.particles = 1000
    project.numerics.particle_max_periods = 3
    project.numerics.stored_phases = 8
    field = synthetic_constant_field(project)
    field.electron_density_m3 *= 2
    field.ion_density_m3 *= 2
    table = replace(dc_table, ionization_m3_s=np.zeros_like(dc_table.ionization_m3_s))
    result = run_particles(project, field, table, collisions=False)
    diagnostics = result.diagnostics
    assert diagnostics["lost_external"] > 0
    assert diagnostics["arrived"] > 0
    assert diagnostics["injected"] == (
        diagnostics["arrived"]
        + diagnostics["lost_ground"]
        + diagnostics["lost_external"]
        + diagnostics["timed_out"]
    )


def test_only_particle_parameters_can_change_without_invalidating_field():
    project = Project()
    one = field_fingerprint(project.to_dict(), "table", "wave")
    project.numerics.particles *= 2
    project.numerics.maximum_cycles += 1
    assert field_fingerprint(project.to_dict(), "table", "wave") == one
    project.plasma.pressure_pa *= 2
    assert field_fingerprint(project.to_dict(), "table", "wave") != one


def test_project_rejects_nan_and_invalid_phase_resolution():
    project = Project()
    project.plasma.density_m3 = float("nan")
    with pytest.raises(ValueError, match="有限"):
        project.validate()
    project.plasma.density_m3 = 1e16
    project.numerics.stored_phases = 256
    with pytest.raises(ValueError, match="保存位相"):
        project.validate()


def test_archived_project_is_relocatable_and_column_mapping_is_part_of_field_identity(tmp_path):
    project = Project.load(ROOT / "examples" / "quick-project.json")
    project.plasma.density_profile = "csv"
    project.plasma.density_csv = "examples/density_demo.csv"
    before = make_waveform(project)
    archived = tmp_path / "original"
    archive_inputs(project, archived)
    relocated = tmp_path / "different" / "result"
    shutil.copytree(archived, relocated)
    loaded = Project.load(relocated / "project.json")
    assert loaded.path(loaded.electron_xsec).parent == relocated / "inputs"
    assert make_waveform(loaded).voltage_v == pytest.approx(before.voltage_v)
    mesh = AxisymmetricMesh(loaded.geometry)
    density = mesh.initial_density(loaded.plasma, loaded.path(loaded.plasma.density_csv))
    assert np.isfinite(density).all() and density.max() > density.min() > 0
    original = field_fingerprint(loaded.to_dict(), "table", before.source_hash)
    loaded.numerics.waveform_voltage_column = "different_voltage_column"
    assert field_fingerprint(loaded.to_dict(), "table", before.source_hash) != original


def test_maintained_nonuniform_density_and_secondary_sources_preserve_budgets(dc_table):
    project = Project(base_dir=str(ROOT))
    project.geometry.radial_cells = 4
    project.geometry.axial_cells = 6
    project.plasma.density_profile = "expression"
    project.plasma.density_mode = "maintained"
    project.plasma.density_expression = "ne0 * (1 + 0.2*cos(pi*r/R))"
    project.plasma.cathode_secondary_yield = 0.01
    project.plasma.ground_secondary_yield = 0.01
    project.numerics.minimum_cycles = 1
    project.numerics.maximum_cycles = 1
    project.numerics.steps_per_period = 32
    project.numerics.stored_phases = 8
    project.numerics.newton_iterations = 60
    field = run_fields(project, dc_table, Waveform.demo(400e3, 10000))
    diagnostics = field.diagnostics
    assert diagnostics["electron_secondary_created"] > 0
    assert diagnostics["ion_balance_relative_error"] < 1e-10
    assert diagnostics["electron_balance_relative_error"] < 1e-4
    assert diagnostics["electron_energy_balance_relative_error"] < 1e-8
    assert diagnostics["density_mode"] == "maintained"
    assert np.isfinite(field.mean_energy_ev).all()
