from pathlib import Path

import numpy as np
import pytest

from sheath_pic.config import Geometry, Project
from sheath_pic.constants import E_CHARGE, EPSILON_0, K_B, M_AR
from sheath_pic.mesh import AxisymmetricMesh
from sheath_pic.particles import electric_cartesian, run_particles, surface_events
from sheath_pic.transport import elastic_collision_solver, elastic_energy_loss
from sheath_pic.verification import (
    beam_injection,
    no_reactions,
    planar_field,
    rf_impact_reference,
    static_impact_reference,
    vacuum_cylinder_potential,
)
from sheath_pic.xsections import scatter_equal_mass

ROOT = Path(__file__).resolve().parents[1]


def orbit_project(nz=40, phases=8):
    project = Project(base_dir=str(ROOT))
    project.geometry.cathode_radius_m = project.geometry.chamber_radius_m
    project.geometry.radial_cells = 4
    project.geometry.axial_cells = nz
    project.numerics.particles = 192
    project.numerics.steps_per_period = 2048
    project.numerics.stored_phases = phases
    return project


def test_cell_centred_fields_reproduce_quadratics_at_axis_and_metal():
    g = Geometry(radial_cells=8, axial_cells=12)
    g.cathode_radius_m = g.chamber_radius_m
    mesh = AxisymmetricMesh(g)
    phi = -10000 * (1 - mesh.Z / g.gap_m) ** 2
    _, ez = mesh.electric_field(phi, -10000)
    assert ez == pytest.approx(-20000 / g.gap_m * (1 - mesh.Z / g.gap_m), abs=1e-8)
    radial_phi = 10000 * (1 - mesh.R**2 / g.chamber_radius_m**2)
    er, _ = mesh.electric_field(radial_phi, 0)
    assert er == pytest.approx(20000 * mesh.R / g.chamber_radius_m**2, abs=1e-8)
    field = planar_field(orbit_project(), exponent=2, mesh_gradient=True)
    sample = np.array([[0, 0, 0], [0.1, 0, 0.05]])
    actual = electric_cartesian(
        AxisymmetricMesh(orbit_project().geometry), field, sample, 0, 2.5e-6
    )
    assert actual[:, 2] == pytest.approx([-400000, 0], abs=1e-8)


def test_surface_hit_uses_accelerated_time_and_cylinder_intersection():
    project = orbit_project()
    acceleration = E_CHARGE / M_AR * 10000 / project.geometry.gap_m
    x = np.array([[0.01, 0, 0.04], [0.25, 0, 0.025]])
    v = np.array([[0, 0, -1000], [200000, 0, 0]])
    a = np.array([[0, 0, -acceleration], [1e10, 0, 0]])
    dt = np.full(2, 2e-6)
    times, surfaces = surface_events(project, x, v, a, dt)
    expected_z = 0.08 / (1000 + np.sqrt(1000**2 + 0.08 * acceleration))
    distance = project.geometry.chamber_radius_m - x[1, 0]
    expected_r = 2 * distance / (200000 + np.sqrt(200000**2 + 2e10 * distance))
    assert surfaces.tolist() == [1, 3]
    assert times == pytest.approx([expected_z, expected_r], rel=1e-12)


@pytest.mark.parametrize("voltage", [1000, 10000, 20000])
def test_uniform_field_impact_energy_and_angle_per_ion(voltage):
    project = orbit_project(nz=8)
    project.numerics.demo_negative_peak_v = voltage
    injection = beam_injection(project)
    energy, angle = static_impact_reference(project, injection)
    result = run_particles(
        project, planar_field(project), no_reactions(), collisions=False, injection=injection
    )
    assert result.diagnostics["arrived"] == project.numerics.particles
    assert result.arrival_energy_ev == pytest.approx(energy, rel=1e-12)
    assert result.arrival_angle_deg == pytest.approx(angle, abs=1e-10)


def test_child_langmuir_orbits_converge_in_space():
    errors = []
    for nz in (8, 16, 32, 64):
        project = orbit_project(nz)
        injection = beam_injection(project)
        field = planar_field(project, exponent=4 / 3, mesh_gradient=True)
        expected, _ = static_impact_reference(project, injection, exponent=4 / 3)
        result = run_particles(
            project, field, no_reactions(), collisions=False, injection=injection
        )
        errors.append(np.max(abs(result.arrival_energy_ev / expected - 1)))
    assert np.all(np.diff(errors) < 0)
    assert errors[-1] < 3e-4
    assert errors[0] / errors[-1] > 15


def test_rf_phase_interpolation_converges_to_exact_time_dependent_orbits():
    errors = []
    for phases in (16, 32, 64, 128, 256):
        project = orbit_project(phases=phases)
        injection = beam_injection(project, rf_phases=True)
        expected, angle, _ = rf_impact_reference(project, injection)
        result = run_particles(
            project,
            planar_field(project, rf=True),
            no_reactions(),
            collisions=False,
            injection=injection,
        )
        errors.append(
            np.linalg.norm(result.arrival_energy_ev - expected) / np.linalg.norm(expected)
        )
        if phases == 256:
            assert max(abs(result.arrival_angle_deg - angle)) < 0.002
    assert np.all(np.diff(errors) < 0)
    assert errors[-1] < 1e-4
    assert errors[0] / errors[-1] > 100


def test_cylindrical_10kv_poisson_converges_to_independent_bessel_series():
    errors = []
    for cells in (12, 24, 48):
        g = Geometry(radial_cells=cells, axial_cells=cells)
        g.cathode_radius_m = g.chamber_radius_m
        mesh = AxisymmetricMesh(g)
        zero = np.zeros(mesh.shape)
        numerical = mesh.solve_poisson(zero, zero, -10000)
        exact = vacuum_cylinder_potential(mesh, 10000)
        # Exclude the discontinuous bottom/side corner, where the exact gradient is singular.
        smooth = mesh.R < g.chamber_radius_m * 0.8
        errors.append(
            np.linalg.norm(numerical[smooth] - exact[smooth]) / np.linalg.norm(exact[smooth])
        )
    assert errors[0] / errors[1] > 3
    assert errors[1] / errors[2] > 3
    assert errors[-1] < 5e-4


def test_cold_charge_exchange_matches_analytic_survival_and_iedf(tmp_path):
    project = orbit_project()
    project.numerics.particles = 6000
    project.numerics.seed = 813
    injection = beam_injection(project)
    injection.velocity_ms[:, :2] = 0
    # A cold gas at the number density of 10 mTorr / 300 K gives an exact 1D reference.
    gas_density = project.plasma.pressure_pa / (K_B * project.plasma.gas_temperature_k)
    project.plasma.gas_temperature_k = 1e-5
    project.plasma.pressure_pa = gas_density * K_B * project.plasma.gas_temperature_k
    distance = injection.position_m[0, 2]
    optical_depth = 2.0
    sigma = optical_depth / (gas_density * distance)
    path = tmp_path / "constant-cx.txt"
    path.write_text(
        f"PROCESS: Backscat\nCOLUMNS: Energy (eV) | Cross section (m2)\n"
        f"------\n0 {sigma}\n10000 {sigma}\n------\n"
        "PROCESS: Isotropic\nCOLUMNS: Energy (eV) | Cross section (m2)\n"
        "------\n0 0\n10000 0\n------\n",
        encoding="utf-8",
    )
    project.ion_xsec = str(path)
    result = run_particles(
        project,
        planar_field(project),
        no_reactions(),
        injection=injection,
        cell_step_fraction=0.125,
        collision_step_hazard=0.025,
    )
    survival = np.count_nonzero(result.arrival_collisions == 0) / project.numerics.particles
    expected = np.exp(-optical_depth)
    standard_error = np.sqrt(expected * (1 - expected) / project.numerics.particles)
    assert abs(survival - expected) < 5 * standard_error
    # Continuous part of the CX IEDF, below the atom at full potential energy.
    full_energy = 10000 * distance / project.geometry.gap_m
    probes = np.linspace(0.1, 0.9, 9) * full_energy
    measured = np.mean(result.arrival_energy_ev[:, None] <= probes, axis=0)
    exact = -np.expm1(-optical_depth * probes / full_energy)
    assert max(abs(measured - exact)) < 0.035
    assert result.diagnostics["timed_out"] == 0


def test_isotropic_equal_mass_joint_energy_angle_distribution():
    speed = np.sqrt(2 * E_CHARGE * 10000 / M_AR)
    incoming = np.tile([0, 0, -speed], (20000, 1))
    outgoing = scatter_equal_mass(
        incoming, np.zeros_like(incoming), np.zeros(20000, bool), np.random.default_rng(66)
    )
    fraction = np.sum(outgoing**2, axis=1) / speed**2
    cosine_squared = outgoing[:, 2] ** 2 / np.sum(outgoing**2, axis=1)
    assert fraction == pytest.approx(cosine_squared, abs=1e-12)
    assert fraction.mean() == pytest.approx(0.5, abs=0.007)
    assert np.mean(fraction < 0.25) == pytest.approx(0.25, abs=0.015)


def test_collision_before_first_wall_hit_is_retained(tmp_path):
    project = orbit_project()
    project.numerics.particles = 1000
    injection = beam_injection(project)
    injection.position_m[:, 2] = 1e-7
    injection.velocity_ms[:, :2] = 0
    gas_density = project.plasma.pressure_pa / (K_B * project.plasma.gas_temperature_k)
    project.plasma.gas_temperature_k = 1e-5
    project.plasma.pressure_pa = gas_density * K_B * project.plasma.gas_temperature_k
    sigma = 0.04 / (gas_density * 1e-7)
    path = tmp_path / "last-step-cx.txt"
    path.write_text(
        f"PROCESS: Backscat\nCOLUMNS: Energy (eV) | Cross section (m2)\n"
        f"------\n0 {sigma}\n10000 {sigma}\n------\n"
        "PROCESS: Isotropic\nCOLUMNS: Energy (eV) | Cross section (m2)\n"
        "------\n0 0\n10000 0\n------\n",
        encoding="utf-8",
    )
    project.ion_xsec = str(path)
    result = run_particles(project, planar_field(project), no_reactions(), injection=injection)
    collided = np.count_nonzero(result.arrival_collisions > 0)
    # The wall lies inside the first attempted step. Suppressing collisions for any
    # wall-crossing step incorrectly yields zero, instead of 1000*(1-exp(-0.04)).
    assert 15 < collided < 70
    assert result.diagnostics["arrived"] == 1000


def test_axisymmetric_radial_orbits_match_harmonic_force_reference():
    project = orbit_project()
    mesh = AxisymmetricMesh(project.geometry)
    field = planar_field(project)
    frequency = 5e5
    field.electric_r_vm[:] = -M_AR / E_CHARGE * frequency**2 * mesh.R
    injection = beam_injection(project)
    injection.position_m[:, 0] = np.resize([0.001, 0.025, 0.08], project.numerics.particles)
    injection.velocity_ms[:, 0] = 2000
    injection.velocity_ms[:, 1] = 2500
    acceleration = E_CHARGE / M_AR * 10000 / project.geometry.gap_m
    tau = (
        2
        * injection.position_m[:, 2]
        / (
            -injection.velocity_ms[:, 2]
            + np.sqrt(
                injection.velocity_ms[:, 2] ** 2 + 2 * acceleration * injection.position_m[:, 2]
            )
        )
    )
    sine, cosine = np.sin(frequency * tau)[:, None], np.cos(frequency * tau)[:, None]
    position = (
        injection.position_m[:, :2] * cosine + injection.velocity_ms[:, :2] * sine / frequency
    )
    transverse_v = (
        injection.velocity_ms[:, :2] * cosine - frequency * injection.position_m[:, :2] * sine
    )
    axial_v = injection.velocity_ms[:, 2] - acceleration * tau
    energy = 0.5 * M_AR / E_CHARGE * (np.sum(transverse_v**2, axis=1) + axial_v**2)
    angle = np.degrees(np.arctan2(np.linalg.norm(transverse_v, axis=1), abs(axial_v)))
    result = run_particles(project, field, no_reactions(), collisions=False, injection=injection)
    assert result.arrival_radius_m == pytest.approx(np.linalg.norm(position, axis=1), abs=1e-8)
    assert result.arrival_energy_ev == pytest.approx(energy, rel=2e-7)
    assert result.arrival_angle_deg == pytest.approx(angle, abs=1e-5)


def test_child_langmuir_charge_and_potential_are_consistent():
    # Verify the continuum reference independently of a discretization.
    voltage, h = 10000, 0.05
    z = np.linspace(0.05 * h, 0.9 * h, 5000)
    potential = -voltage * (1 - z / h) ** (4 / 3)
    curvature = np.gradient(np.gradient(potential, z, edge_order=2), z, edge_order=2)
    density = EPSILON_0 / E_CHARGE * (4 * voltage / (9 * h**2)) * (1 - z / h) ** (-2 / 3)
    assert curvature[2:-2] == pytest.approx(-E_CHARGE * density[2:-2] / EPSILON_0, rel=2e-6)


def test_elastic_energy_moment_matches_cold_recoil_analytic_law():
    from types import SimpleNamespace

    import boltzpmp as bp

    mass_ratio = 1e-3
    sigma = 1e-19
    section = bp.CrossSection(
        "ELASTIC",
        "Ar",
        "constant sigma",
        mass_ratio=mass_ratio,
        data=np.array([[0, sigma], [20, sigma]]),
    )
    mixture = bp.Mixture([bp.Gas("Ar", 1, [section], mass_amu=39.948)], N=1e20, T_K=300)
    solver = bp.PMSolver(mixture, energy_grid=np.linspace(0, 20, 401), n_theta=8, gas_heating=False)
    state = np.zeros(solver.mesh.shape)
    index = np.argmin(abs(solver.mesh.eps_c - 10))
    state[index] = solver.mesh.w_theta
    result = SimpleNamespace(n=state.ravel(), mesh=solver.mesh)
    energy = solver.mesh.eps_c[index]
    from sheath_pic.constants import M_E

    expected = 2 * mass_ratio * sigma * np.sqrt(2 * E_CHARGE * energy / M_E) * energy
    assert elastic_energy_loss(result, solver) == pytest.approx(expected, rel=1e-9)


def test_elastic_only_operator_matches_full_collision_energy_budget_and_angular_refinement():
    import boltzpmp as bp

    sections = bp.parse_lxcat(ROOT / "xsec/Argon_電子衝突断面積.txt")
    mixture = bp.Mixture(
        [bp.Gas("Ar", 1, sections, mass_amu=39.948)], p_Pa=1.3332236842105263, T_K=300
    )
    losses = []
    for angles in (24, 48):
        solver = bp.PMSolver(
            mixture,
            energy_grid=bp.graded_energy_grid(400, 0.01, 2),
            n_theta=angles,
            gas_heating=True,
            superelastic=False,
        )
        result = solver.solve_dc(100, scheme="limiter")
        loss = elastic_energy_loss(result, elastic_collision_solver(bp, solver))
        losses.append(loss)
        full_rate = np.array(solver._core_solver.collision_apply(result.n.tolist())).reshape(
            result.mesh.shape
        )
        full_loss = -np.sum(full_rate * result.mesh.eps_c[:, None]) / mixture.N
        inelastic = sum(
            section.threshold
            * sum(rate for label, rate in result.rate_coefficients.items() if section.name in label)
            for section in sections
            if section.kind in {"EXCITATION", "IONIZATION"}
        )
        assert full_loss == pytest.approx(loss + inelastic, rel=1e-8)
    assert abs(losses[0] / losses[1] - 1) < 0.002
