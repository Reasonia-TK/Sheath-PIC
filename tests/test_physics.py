from pathlib import Path

import numpy as np
import pytest

from sheath_pic.config import Geometry
from sheath_pic.constants import E_CHARGE, M_AR
from sheath_pic.mesh import AxisymmetricMesh
from sheath_pic.solver import bernoulli, electron_operator, ion_step, solve_electron_poisson
from sheath_pic.xsections import IonCrossSections, scatter_equal_mass

ROOT = Path(__file__).resolve().parents[1]


def test_phelps_ion_parser_and_range():
    table = IonCrossSections.load(ROOT / "xsec/Argon_イオン衝突断面積.txt")
    assert len(table.isotropic.energy_ev) == 114
    assert table.max_energy_ev == 10000
    with pytest.raises(ValueError, match="上限"):
        table.isotropic.at(10001)
    speed = np.sqrt(2 * E_CHARGE * 100 / M_AR)
    assert table.collision_energy(speed) == pytest.approx(50)


def test_equal_mass_scattering_conserves_energy_and_backscatter_exchanges_velocity():
    rng = np.random.default_rng(11)
    v = rng.normal(0, 1000, (2000, 3))
    gas = rng.normal(0, 200, v.shape)
    outgoing = scatter_equal_mass(v, gas, np.zeros(len(v), dtype=bool), rng)
    recoil = v + gas - outgoing
    assert np.sum(outgoing**2 + recoil**2, axis=1) == pytest.approx(np.sum(v**2 + gas**2, axis=1))
    back = scatter_equal_mass(v, gas, np.ones(len(v), dtype=bool), rng)
    assert back == pytest.approx(gas)


def test_sg_equilibrium_and_large_voltage_are_finite():
    assert np.isfinite(bernoulli(np.array([-10000, 0, 10000]))).all()
    x = np.linspace(-10, 10, 100)
    assert bernoulli(-x) == pytest.approx(np.exp(x) * bernoulli(x))


def test_electron_poisson_solves_positive_density_and_charge_balance():
    mesh = AxisymmetricMesh(Geometry(radial_cells=6, axial_cells=8))
    n = np.full(mesh.shape, 1e16)
    phi = mesh.solve_poisson(n, n, -20)
    potential, density, op, _, residual = solve_electron_poisson(
        mesh,
        n,
        n,
        phi,
        np.full(mesh.shape, 4.5),
        np.full(mesh.shape, 2e4),
        np.zeros(mesh.shape),
        2e-9,
        -20,
        1e16,
        1e-8,
        80,
        1,
        True,
    )
    assert residual < 1e-8
    assert density.min() > 0
    loss = np.sum(op.boundary_rate * density.ravel()[mesh.boundary_cell] * mesh.boundary_area)
    assert np.sum((density - n) * mesh.volume) == pytest.approx(-2e-9 * loss, rel=1e-5)
    rhs = mesh.poisson_rhs(n, density, -20)
    assert np.linalg.norm(mesh.poisson @ potential.ravel() - rhs) / np.linalg.norm(rhs) < 1e-5


def test_ion_finite_volume_particle_budget():
    mesh = AxisymmetricMesh(Geometry(radial_cells=6, axial_cells=8))
    n = np.full(mesh.shape, 1e16)
    zero = np.zeros(mesh.shape)
    cross = IonCrossSections.load(ROOT / "xsec/Argon_イオン衝突断面積.txt")
    result, _, _, loss = ion_step(mesh, n, zero, zero, zero, zero, zero, 1e-9, cross, 3.2e20, 300)
    assert np.sum(result * mesh.volume) == pytest.approx(
        np.sum(n * mesh.volume) - 1e-9 * np.sum(loss * mesh.boundary_area),
        rel=1e-12,
    )


def test_electron_operator_jacobian_matches_finite_difference():
    mesh = AxisymmetricMesh(Geometry(radial_cells=4, axial_cells=4))
    rng = np.random.default_rng(3)
    phi = rng.normal(0, 4, mesh.shape)
    energy = np.full(mesh.shape, 4.5)
    mu = np.full(mesh.shape, 2e4)
    op = electron_operator(mesh, phi, energy, mu, 1e-9, -100)
    epsilon = 1e-5
    # Check a face by perturbing its j potential while holding all coefficients fixed.
    chosen = 4
    shifted = phi.copy().ravel()
    shifted[mesh.face_j[chosen]] += epsilon
    plus = electron_operator(mesh, shifted.reshape(mesh.shape), energy, mu, 1e-9, -100)
    derivative = (plus.flux_i[chosen] - op.flux_i[chosen]) / epsilon
    assert derivative == pytest.approx(op.flux_derivative[chosen, 0], rel=1e-4)
