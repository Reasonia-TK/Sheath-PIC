"""Independent analytical references for numerical verification, not discharge models.

The planar benchmarks prescribe their side boundaries; they do not represent the
grounded cylindrical CCP. Child-Langmuir is used only in its cold collisionless limit.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.special import j0, jn_zeros, jv

from .config import Project
from .constants import E_CHARGE, M_AR
from .mesh import AxisymmetricMesh
from .particles import ParticleInjection
from .solver import FieldResult
from .transport import TransportTable


def no_reactions() -> TransportTable:
    zero = np.zeros(2)
    return TransportTable(
        np.array([1, 1000]),
        np.array([0.01, 100]),
        np.ones(2),
        zero,
        zero.copy(),
        zero.copy(),
        np.array([0, 1]),
        np.ones((2, 2)),
        {"calculation": "prescribed-field verification; no reactions"},
    )


def planar_field(project: Project, exponent=1.0, rf=False, mesh_gradient=False) -> FieldResult:
    """phi=-U(1-z/H)^p, with analytical E or the actual FV gradient reconstruction."""
    mesh = AxisymmetricMesh(project.geometry)
    n = project.numerics
    phases = np.arange(n.stored_phases) / n.stored_phases / n.frequency_hz
    voltage = np.full(len(phases), n.demo_negative_peak_v)
    if rf:
        voltage *= 0.5 * (1 - np.cos(2 * np.pi * n.frequency_hz * phases))
    potential = -voltage[:, None, None] * (1 - mesh.Z[None] / project.geometry.gap_m) ** exponent
    ez = (
        -exponent
        * voltage[:, None, None]
        / project.geometry.gap_m
        * (1 - mesh.Z[None] / project.geometry.gap_m) ** (exponent - 1)
    )
    if mesh_gradient:
        ez = np.array(
            [mesh.electric_field(phi, -v)[1] for phi, v in zip(potential, voltage, strict=True)]
        )
    density = np.full(potential.shape, project.plasma.density_m3)
    zero = np.zeros_like(potential)
    return FieldResult(
        phases,
        potential,
        zero.copy(),
        ez,
        density,
        density.copy(),
        np.full(potential.shape, 4.5),
        zero.copy(),
        zero.copy(),
        {"periodic_converged": True, "field_model": "prescribed analytical planar verification"},
        [],
        project.to_dict(),
        np.r_[phases, 1 / n.frequency_hz],
        -np.r_[voltage, voltage[0]],
    )


def beam_injection(project: Project, rf_phases=False) -> ParticleInjection:
    """3 eV axial energy plus 0, 10, 100 eV transverse energy, starting at z=0.95H."""
    n = project.numerics.particles
    position = np.zeros((n, 3))
    position[:, 0] = min(0.02, project.geometry.cathode_radius_m / 4)
    position[:, 2] = project.geometry.gap_m * 0.95
    velocity = np.zeros_like(position)
    velocity[:, 0] = np.sqrt(2 * E_CHARGE / M_AR * np.resize([0, 10, 100], n))
    velocity[:, 2] = -np.sqrt(2 * E_CHARGE * 3 / M_AR)
    times = np.zeros(n)
    if rf_phases:
        times = np.arange(n) / n / project.numerics.frequency_hz
    return ParticleInjection(position, velocity, times)


def static_impact_reference(project, injection, exponent=1.0):
    initial = 0.5 * M_AR / E_CHARGE * np.sum(injection.velocity_ms**2, axis=1)
    transverse = 0.5 * M_AR / E_CHARGE * np.sum(injection.velocity_ms[:, :2] ** 2, axis=1)
    gain = project.numerics.demo_negative_peak_v * (
        1 - (1 - injection.position_m[:, 2] / project.geometry.gap_m) ** exponent
    )
    energy = initial + gain
    angle = np.degrees(np.arcsin(np.sqrt(transverse / energy)))
    return energy, angle


def rf_impact_reference(project, injection):
    """Exact integrated orbit in spatially uniform E(t)=-U[1-cos(omega t)]/(2H)."""
    omega = 2 * np.pi * project.numerics.frequency_hz
    scale = E_CHARGE * project.numerics.demo_negative_peak_v / (2 * M_AR * project.geometry.gap_m)
    end_times, velocities = [], []
    for position, velocity, t0 in zip(
        injection.position_m, injection.velocity_ms, injection.time_s, strict=True
    ):
        phase0 = omega * t0

        def height(tau, position=position, velocity=velocity, phase0=phase0):
            return (
                position[2]
                + velocity[2] * tau
                - scale
                * (
                    0.5 * tau**2
                    + (np.cos(phase0 + omega * tau) - np.cos(phase0)) / omega**2
                    + tau * np.sin(phase0) / omega
                )
            )

        tau = brentq(
            height,
            0,
            project.numerics.particle_max_periods / project.numerics.frequency_hz,
            xtol=1e-19,
            rtol=1e-13,
        )
        final_v = velocity.copy()
        final_v[2] -= scale * (tau - (np.sin(phase0 + omega * tau) - np.sin(phase0)) / omega)
        velocities.append(final_v)
        end_times.append(t0 + tau)
    velocities = np.array(velocities)
    energy = 0.5 * M_AR / E_CHARGE * np.sum(velocities**2, axis=1)
    angle = np.degrees(np.arccos(abs(velocities[:, 2]) / np.linalg.norm(velocities, axis=1)))
    return energy, angle, np.array(end_times)


def vacuum_cylinder_potential(mesh: AxisymmetricMesh, negative_voltage: float, modes=None):
    """Bessel series: full bottom Dirichlet -U, top/side ground, regular on the axis."""
    # The series converges slowly near the discontinuous bottom/side corner. Resolve
    # the first cell centre independently of the FV refinement (k_max z_min > 15).
    modes = modes or max(384, 64 * len(mesh.z))
    alpha = jn_zeros(0, modes)
    k = alpha / mesh.geometry.chamber_radius_m
    coefficient = 2 / (alpha * jv(1, alpha))
    h, z = mesh.geometry.gap_m, mesh.z
    # Exponential form avoids overflow of sinh at large mode number.
    ratio = (
        np.exp(-k[:, None] * z)
        * (-np.expm1(-2 * k[:, None] * (h - z)))
        / (-np.expm1(-2 * k[:, None] * h))
    )
    return -negative_voltage * (j0(mesh.r[:, None] * k) * coefficient) @ ratio
