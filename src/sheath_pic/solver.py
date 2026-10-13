from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.sparse import bmat, coo_matrix, diags, eye
from scipy.sparse.linalg import spsolve

from .config import Project
from .constants import E_CHARGE, EPSILON_0, K_B, M_AR, M_E
from .inputs import Waveform
from .mesh import AxisymmetricMesh
from .transport import Progress, TransportTable
from .xsections import IonCrossSections


def bernoulli(x):
    """Stable Bernoulli function x / (exp(x)-1), including |x| >> 1."""
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    small, high, low = abs(x) < 1e-4, x > 50, x < -50
    middle = ~(small | high | low)
    out[small] = 1 - x[small] / 2 + x[small] ** 2 / 12 - x[small] ** 4 / 720
    out[high] = x[high] * np.exp(-x[high])
    out[low] = -x[low]
    out[middle] = x[middle] / np.expm1(x[middle])
    return out


def bernoulli_derivative(x):
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    small, high, low = abs(x) < 1e-4, x > 50, x < -50
    middle = ~(small | high | low)
    out[small] = -0.5 + x[small] / 6 - x[small] ** 3 / 180
    out[high] = (1 - x[high]) * np.exp(-x[high])
    out[low] = -1
    em1 = np.expm1(x[middle])
    out[middle] = (em1 - x[middle] * (em1 + 1)) / em1**2
    return out


@dataclass
class FieldResult:
    phase_s: np.ndarray
    potential_v: np.ndarray
    electric_r_vm: np.ndarray
    electric_z_vm: np.ndarray
    electron_density_m3: np.ndarray
    ion_density_m3: np.ndarray
    mean_energy_ev: np.ndarray
    ion_velocity_r_ms: np.ndarray
    ion_velocity_z_ms: np.ndarray
    diagnostics: dict
    history: list[dict]
    project: dict
    waveform_time_s: np.ndarray
    waveform_voltage_v: np.ndarray

    @property
    def converged(self) -> bool:
        return bool(self.diagnostics["periodic_converged"])


@dataclass
class ElectronOperator:
    matrix: object
    flux_i: np.ndarray
    flux_j: np.ndarray
    flux_derivative: np.ndarray
    boundary_rate: np.ndarray
    boundary_derivative: np.ndarray
    limited_fraction: float


def electron_operator(
    mesh: AxisymmetricMesh,
    potential: np.ndarray,
    mean_energy: np.ndarray,
    mobility: np.ndarray,
    dt: float,
    wafer_voltage: float,
    sticking: float = 1.0,
    limit_flux: bool = True,
) -> ElectronOperator:
    phi, energy, mu = potential.ravel(), mean_energy.ravel(), mobility.ravel()
    i, j = mesh.face_i, mesh.face_j
    te = np.maximum((energy[i] + energy[j]) / 3, 0.01)
    face_mu = 0.5 * (mu[i] + mu[j])
    delta = phi[j] - phi[i]
    limited = np.zeros_like(delta, dtype=bool)
    if limit_flux:
        # Free-streaming closure limits drift flux, not electron trajectories or ion energy.
        speed_limit = np.sqrt(8 * E_CHARGE * te / (np.pi * M_E))
        maximum_mu = speed_limit * mesh.face_distance / np.maximum(abs(delta), 1e-20)
        limited = face_mu > maximum_mu
        face_mu = np.minimum(face_mu, maximum_mu)
    diffusion_over_distance = face_mu * te / mesh.face_distance
    peclet = delta / te
    a = diffusion_over_distance * bernoulli(-peclet)
    b = diffusion_over_distance * bernoulli(peclet)
    da = -diffusion_over_distance / te * bernoulli_derivative(-peclet)
    db = diffusion_over_distance / te * bernoulli_derivative(peclet)
    da[limited] -= a[limited] / delta[limited]
    db[limited] -= b[limited] / delta[limited]
    bc = mesh.boundary_cell
    te_b = np.maximum(2 * energy[bc] / 3, 0.01)
    difference = mesh.boundary_voltage(wafer_voltage) - phi[bc]
    wall_rate = (
        sticking
        * np.sqrt(E_CHARGE * te_b / (2 * np.pi * M_E))
        * np.exp(
            np.clip(difference / te_b, -700, 0),
        )
    )
    wall_derivative = np.where(difference < 0, -wall_rate / te_b, 0.0)
    vi, vj = mesh.volume.ravel()[i], mesh.volume.ravel()[j]
    fa = dt * mesh.face_area
    ba = dt * mesh.boundary_area / mesh.volume.ravel()[bc]
    rows = np.concatenate((i, i, j, j, bc))
    cols = np.concatenate((i, j, i, j, bc))
    values = np.concatenate((fa * a / vi, -fa * b / vi, -fa * a / vj, fa * b / vj, ba * wall_rate))
    matrix = (
        eye(mesh.size, format="csr")
        + coo_matrix(
            (values, (rows, cols)),
            shape=(mesh.size, mesh.size),
        ).tocsr()
    )
    return ElectronOperator(
        matrix, a, b, np.column_stack((da, db)), wall_rate, wall_derivative, float(limited.mean())
    )


def solve_electron_poisson(
    mesh: AxisymmetricMesh,
    ion_density: np.ndarray,
    old_electron_density: np.ndarray,
    initial_potential: np.ndarray,
    mean_energy: np.ndarray,
    mobility: np.ndarray,
    electron_source: np.ndarray,
    dt: float,
    wafer_voltage: float,
    density_scale: float,
    tolerance: float,
    maximum_iterations: int,
    sticking: float,
    limit_flux: bool,
) -> tuple[np.ndarray, np.ndarray, ElectronOperator, int, float]:
    """Coupled Newton solve of implicit SG electron continuity and Poisson.

    The density is a dependent variable. Applying Poisson to a prescribed neutral charge
    density would only solve Laplace's equation and cannot produce a physical floating bulk.
    """
    vol = mesh.volume.ravel()
    phi = initial_potential.ravel().copy()
    density = np.maximum(old_electron_density.ravel() / density_scale, 1e-18)
    rhs_e = (old_electron_density.ravel() + dt * electron_source.ravel()) / density_scale
    if np.any(rhs_e < 0):
        raise ArithmeticError("電子源によって負の密度が生成されました。")
    poisson_ion = mesh.poisson_rhs(ion_density, np.zeros(mesh.shape), wafer_voltage)
    charge_scale = E_CHARGE / EPSILON_0 * density_scale * vol
    row_scale = 1 / (mesh.poisson.diagonal() * np.maximum(mean_energy.ravel(), 0.03) + charge_scale)
    jp_phi = diags(row_scale) @ mesh.poisson
    jp_n = diags(row_scale * charge_scale)
    i, j, bc = mesh.face_i, mesh.face_j, mesh.boundary_cell

    def evaluate(p, n):
        op = electron_operator(
            mesh,
            p.reshape(mesh.shape),
            mean_energy,
            mobility,
            dt,
            wafer_voltage,
            sticking,
            limit_flux,
        )
        rp = row_scale * (mesh.poisson @ p - poisson_ion + charge_scale * n)
        rn = op.matrix @ n - rhs_e
        residual = np.concatenate((rp, rn))
        return op, residual, float(np.max(abs(residual)))

    op, residual, norm = evaluate(phi, density)
    for iteration in range(maximum_iterations):
        if norm < tolerance:
            return (
                phi.reshape(mesh.shape),
                (density * density_scale).reshape(mesh.shape),
                op,
                iteration,
                norm,
            )
        da, db = op.flux_derivative.T
        derivative = da * density[i] - db * density[j]
        fi = dt * mesh.face_area * derivative / vol[i]
        fj = dt * mesh.face_area * derivative / vol[j]
        fb = dt * mesh.boundary_area / vol[bc] * op.boundary_derivative * density[bc]
        jn_phi = coo_matrix(
            (
                np.concatenate((-fi, fi, fj, -fj, fb)),
                (
                    np.concatenate((i, i, j, j, bc)),
                    np.concatenate((i, j, i, j, bc)),
                ),
            ),
            shape=(mesh.size, mesh.size),
        ).tocsr()
        jacobian = bmat([[jp_phi, jp_n], [jn_phi, op.matrix]], format="csc")
        delta = spsolve(jacobian, -residual)
        if not np.all(np.isfinite(delta)):
            raise ArithmeticError("電子・Poisson連成の線形解が有限ではありません。")
        dp, dn = delta[: mesh.size], delta[mesh.size :]
        negative = dn < 0
        alpha = (
            min(1.0, float(np.min(-0.98 * density[negative] / dn[negative])))
            if np.any(negative)
            else 1.0
        )
        accepted = False
        for _ in range(16):
            pn, nn = phi + alpha * dp, density + alpha * dn
            new_op, new_residual, new_norm = evaluate(pn, nn)
            if new_norm < norm * (1 - 1e-4 * alpha) or new_norm < tolerance:
                phi, density, op, residual, norm = pn, nn, new_op, new_residual, new_norm
                accepted = True
                break
            alpha *= 0.5
        if not accepted:
            raise ArithmeticError(
                f"電子・Poisson連成のNewton反復が停滞しました（残差 {norm:.3g}）。"
            )
    raise ArithmeticError(
        f"電子・Poisson連成が{maximum_iterations}反復で収束しません（{norm:.3g}）。"
    )


def ion_step(
    mesh: AxisymmetricMesh,
    density: np.ndarray,
    momentum_r: np.ndarray,
    momentum_z: np.ndarray,
    er: np.ndarray,
    ez: np.ndarray,
    source: np.ndarray,
    dt: float,
    ion_xsec: IonCrossSections,
    neutral_density: float,
    gas_temperature: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Conservative axisymmetric isothermal ion Euler update and implicit collision drag."""
    ci2 = K_B * gas_temperature / M_AR
    ci = np.sqrt(ci2)
    n = density.ravel()
    ur = np.divide(momentum_r.ravel(), n, out=np.zeros_like(n), where=n > 0)
    uz = np.divide(momentum_z.ravel(), n, out=np.zeros_like(n), where=n > 0)
    state = np.column_stack((n, momentum_r.ravel(), momentum_z.ravel()))
    derivative = np.zeros_like(state)
    i, j = mesh.face_i, mesh.face_j
    pressure = ci2 * n
    # Per-face physical fluxes; pressure acts only on the normal momentum component.
    ui = np.where(mesh.face_axis == 0, ur[i], uz[i])
    uj = np.where(mesh.face_axis == 0, ur[j], uz[j])
    left, right = state[i] * ui[:, None], state[j] * uj[:, None]
    components = mesh.face_axis + 1
    left[np.arange(len(i)), components] += pressure[i]
    right[np.arange(len(j)), components] += pressure[j]
    speed = np.maximum(abs(ui), abs(uj)) + ci
    flux = 0.5 * (left + right) - 0.5 * speed[:, None] * (state[j] - state[i])
    amount = mesh.face_area[:, None] * flux
    for component in range(3):
        np.add.at(derivative[:, component], i, -amount[:, component])
        np.add.at(derivative[:, component], j, amount[:, component])
    bc = mesh.boundary_cell
    bn = np.where(mesh.boundary_axis == 0, ur[bc], uz[bc]) * mesh.boundary_sign
    # Absorbing boundary: no incoming ion population; outgoing advection is conservative.
    outgoing = np.maximum(bn, 0) + 0.5 * ci * np.exp(-np.maximum(bn, 0) / ci)
    wall_flux = n[bc] * outgoing
    boundary = state[bc] * outgoing[:, None] * mesh.boundary_area[:, None]
    for component in range(3):
        np.add.at(derivative[:, component], bc, -boundary[:, component])
    derivative /= mesh.volume.ravel()[:, None]
    derivative[:, 0] += source.ravel()
    removal = np.minimum(source.ravel(), 0) / np.maximum(n, 1e-30)
    derivative[:, 1:] += removal[:, None] * state[:, 1:]
    # Axisymmetric radial pressure source cancels the cylindrical pressure-flux divergence.
    derivative[:, 1] += ci2 * n / mesh.R.ravel()
    updated = state + dt * derivative
    if np.min(updated[:, 0]) < -1e-9 * max(n.max(), 1):
        raise ArithmeticError("イオン輸送で負の密度が生じました。時間刻みを縮小します。")
    updated[:, 0] = np.maximum(updated[:, 0], 0)
    updated[:, 1] += dt * E_CHARGE / M_AR * updated[:, 0] * er.ravel()
    updated[:, 2] += dt * E_CHARGE / M_AR * updated[:, 0] * ez.ravel()
    speed_new = np.linalg.norm(updated[:, 1:], axis=1) / np.maximum(updated[:, 0], 1e-30)
    collision_frequency = ion_xsec.momentum_frequency(speed_new, neutral_density, gas_temperature)
    updated[:, 1:] /= 1 + dt * collision_frequency[:, None]
    updated[updated[:, 0] == 0, 1:] = 0
    return (*(updated[:, k].reshape(mesh.shape) for k in range(3)), wall_flux)


def electron_energy_step(
    mesh: AxisymmetricMesh,
    old_energy_density: np.ndarray,
    electron_density: np.ndarray,
    mean_energy: np.ndarray,
    potential: np.ndarray,
    op: ElectronOperator,
    source_energy_ev_m3_s: np.ndarray,
    coefficients: dict,
    neutral_density: float,
    gas_temperature: float,
    dt: float,
    wafer_voltage: float,
    density_scale: float,
) -> tuple[np.ndarray, dict]:
    """Implicit Maxwell moment closure, with explicit reporting of its approximations."""
    i, j, bc, vol = mesh.face_i, mesh.face_j, mesh.boundary_cell, mesh.volume.ravel()
    phi, ne = potential.ravel(), electron_density.ravel()
    scaled_ne = ne / density_scale
    particle_flux = op.flux_i * scaled_ne[i] - op.flux_j * scaled_ne[j]
    # Work in eV * n_ref / s. The elementary charge cancels after conversion from joules.
    face_work = mesh.face_area * particle_flux * (phi[j] - phi[i])
    work = np.zeros(mesh.size)
    np.add.at(work, i, 0.5 * face_work)
    np.add.at(work, j, 0.5 * face_work)
    wall_flux = op.boundary_rate * scaled_ne[bc]
    wall_work = mesh.boundary_area * wall_flux * (mesh.boundary_voltage(wafer_voltage) - phi[bc])
    np.add.at(work, bc, wall_work)
    work /= vol
    old_w = old_energy_density.ravel() / (E_CHARGE * density_scale)
    inelastic_per_electron = neutral_density * (
        11.5 * coefficients["excitation_m3_s"].ravel()
        + 15.8 * coefficients["ionization_m3_s"].ravel()
    )
    elastic_per_electron = neutral_density * coefficients["elastic_loss_ev_m3_s"].ravel()
    inelastic_rate = inelastic_per_electron / np.maximum(mean_energy.ravel(), 0.01)
    elastic_rate = elastic_per_electron / np.maximum(mean_energy.ravel(), 0.01)
    work_loss_rate = np.maximum(-work, 0) / np.maximum(old_w, 1e-30)
    loss = inelastic_rate + elastic_rate + work_loss_rate
    # Enthalpy transport uses 5/3; Maxwell wall losses use 2 Te = 4/3 <eps>.
    interior = op.matrix - eye(mesh.size, format="csr")
    wall_diagonal = np.bincount(
        bc, weights=dt * mesh.boundary_area / vol[bc] * op.boundary_rate, minlength=mesh.size
    )
    matrix = (
        eye(mesh.size, format="csr")
        + (5 / 3) * interior
        - diags(wall_diagonal / 3)
        + diags(dt * loss)
    )
    rhs = old_w + dt * (np.maximum(work, 0) + source_energy_ev_m3_s.ravel() / density_scale)
    new_w = spsolve(matrix, rhs)
    if not np.all(np.isfinite(new_w)) or np.min(new_w) < -1e-8:
        raise ArithmeticError("電子エネルギー式で非物理的な解が生じました。")
    thermal_floor = 1.5 * K_B * gas_temperature / E_CHARGE * scaled_ne
    correction = np.maximum(thermal_floor - new_w, 0)
    new_w += correction
    diagnostics = {
        "field_work_w": float(
            E_CHARGE
            * density_scale
            * np.sum(
                (np.maximum(work, 0) - work_loss_rate * (new_w - correction)) * vol,
            )
        ),
        "field_work_raw_w": float(E_CHARGE * density_scale * np.sum(work * vol)),
        "inelastic_loss_w": float(
            E_CHARGE
            * density_scale
            * np.sum(
                inelastic_rate * (new_w - correction) * vol,
            )
        ),
        "elastic_loss_w": float(
            E_CHARGE
            * density_scale
            * np.sum(
                elastic_rate * (new_w - correction) * vol,
            )
        ),
        "wall_energy_loss_w": float(
            E_CHARGE
            * density_scale
            * np.sum(
                4 / 3 * op.boundary_rate * (new_w - correction)[bc] * mesh.boundary_area,
            )
        ),
        "thermal_floor_energy_j": float(E_CHARGE * density_scale * np.sum(correction * vol)),
    }
    return (new_w * E_CHARGE * density_scale).reshape(mesh.shape), diagnostics


def _period_error(current, previous) -> float:
    scale = max(float(np.linalg.norm(previous)), 1e-30)
    return float(np.linalg.norm(current - previous) / scale)


def run_fields(
    project: Project,
    table: TransportTable,
    waveform: Waveform,
    progress: Progress | None = None,
    cancelled: Callable[[], bool] | None = None,
    checkpoint: Callable[[dict], None] | None = None,
    restart: dict | None = None,
) -> FieldResult:
    project.validate()
    progress = progress or (lambda *_: None)
    cancelled = cancelled or (lambda: False)
    mesh = AxisymmetricMesh(project.geometry)
    p, cfg = project.plasma, project.numerics
    neutral_density = p.pressure_pa / (K_B * p.gas_temperature_k)
    ion_xsec = IonCrossSections.load(project.path(project.ion_xsec), p.ion_energy_frame)
    target = mesh.initial_density(p, project.path(p.density_csv) if p.density_csv else None)
    scale = max(float(target.max()), p.density_m3)
    ne = target.copy()
    ni = target.copy()
    pr = np.zeros(mesh.shape)
    pz = np.zeros(mesh.shape)
    w = E_CHARGE * ne * p.initial_mean_energy_ev
    phi = mesh.solve_poisson(ni, ne, float(waveform.at(0)))
    history = []
    start_cycle = 0
    counters = {
        "ion_created": 0.0,
        "electron_created": 0.0,
        "ion_wall_loss": 0.0,
        "electron_wall_loss": 0.0,
        "electron_secondary_created": 0.0,
        "thermal_floor_energy_j": 0.0,
        "field_work_j": 0.0,
        "source_energy_j": 0.0,
        "inelastic_loss_j": 0.0,
        "elastic_loss_j": 0.0,
        "wall_energy_loss_j": 0.0,
        "field_work_raw_j": 0.0,
        "maximum_newton_residual": 0.0,
        "maximum_limited_face_fraction": 0.0,
        "maximum_table_outside_mass_fraction": 0.0,
        "steps": 0,
        "rejected_steps": 0,
    }
    initial_ions = float(np.sum(ni * mesh.volume))
    initial_electrons = float(np.sum(ne * mesh.volume))
    initial_energy = float(np.sum(w * mesh.volume))
    if restart:
        for name in ("ne", "ni", "pr", "pz", "w", "phi"):
            if restart[name].shape != mesh.shape:
                raise ValueError("チェックポイントの格子が現在の条件と一致しません。")
        ne, ni, pr, pz, w, phi = (
            restart[name].copy() for name in ("ne", "ni", "pr", "pz", "w", "phi")
        )
        start_cycle = int(restart["cycle"])
        history = restart["history"]
        counters = restart["counters"]
        initial_ions, initial_electrons = restart["initial_ions"], restart["initial_electrons"]
        initial_energy = restart["initial_energy"]
    period = waveform.period_s
    phases = np.arange(cfg.stored_phases) * period / cfg.stored_phases
    lower_energy, upper_energy = table.energy_range
    thermal_energy = 1.5 * K_B * p.gas_temperature_k / E_CHARGE
    start_time = time.monotonic()
    converged = False
    stop_reason = "最大周期数に到達"
    final_frames = None

    def save_checkpoint(cycle):
        if checkpoint:
            checkpoint(
                {
                    "cycle": cycle,
                    "ne": ne,
                    "ni": ni,
                    "pr": pr,
                    "pz": pz,
                    "w": w,
                    "phi": phi,
                    "history": history,
                    "counters": counters,
                    "initial_ions": initial_ions,
                    "initial_electrons": initial_electrons,
                    "initial_energy": initial_energy,
                }
            )

    save_checkpoint(start_cycle)
    for cycle in range(start_cycle, cfg.maximum_cycles):
        begin_state = [array.copy() for array in (ne, ni, w, phi)]
        time_in_cycle = 0.0
        phase_index = 0
        frames = {name: [] for name in ("phi", "ne", "ni", "energy", "ur", "uz")}
        last_progress = time.monotonic() - 1
        while time_in_cycle < period * (1 - 1e-12):
            if cancelled():
                raise InterruptedError("場計算を中断しました。直近の完了周期から再開できます。")
            current_time = cycle * period + time_in_cycle
            old_v = float(waveform.at(current_time))
            energy = np.maximum(w / (E_CHARGE * np.maximum(ne, scale * 1e-18)), thermal_energy)
            while (
                phase_index < len(phases) and phases[phase_index] <= time_in_cycle + period * 1e-12
            ):
                for name, array in zip(
                    frames,
                    (phi, ne, ni, energy, pr / np.maximum(ni, 1e-30), pz / np.maximum(ni, 1e-30)),
                    strict=True,
                ):
                    frames[name].append(array.copy())
                phase_index += 1
            er, ez = mesh.electric_field(phi, old_v)
            speed = np.sqrt(pr**2 + pz**2) / np.maximum(ni, scale * 1e-18)
            acceleration = E_CHARGE / M_AR * np.sqrt(er**2 + ez**2)
            active = ni > scale * 1e-10
            cell_h = np.minimum(mesh.dr[:, None], mesh.dz[None, :])
            advective_dt = (
                float(
                    np.min(
                        cell_h[active] / (speed[active] + np.sqrt(K_B * p.gas_temperature_k / M_AR))
                    )
                )
                * cfg.cfl
            )
            acceleration_dt = float(
                np.min(
                    np.sqrt(
                        cfg.cfl * cell_h[active] / np.maximum(acceleration[active], 1),
                    )
                )
            )
            coefficients = table.coefficients(energy)
            source_ionization = ne * neutral_density * coefficients["ionization_m3_s"]
            source_maintenance = np.zeros(mesh.shape)
            if p.density_mode == "maintained":
                # Charge-neutral pair supply/removal, explicitly included in the energy budget.
                bulk = abs(ni - ne) / np.maximum(ni + ne, scale * 1e-12) < 0.1
                source_maintenance = (
                    (target - 0.5 * (ni + ne)) / (p.density_relaxation_periods * period) * bulk
                )
                source_maintenance = np.maximum(
                    source_maintenance, -0.1 * np.minimum(ne, ni) / period
                )
            pair_source = source_ionization + source_maintenance
            dt = min(
                period / cfg.steps_per_period, advective_dt, acceleration_dt, period - time_in_cycle
            )
            reaction_frequency = float(np.max(neutral_density * coefficients["ionization_m3_s"]))
            if reaction_frequency > 0:
                dt = min(dt, 0.05 / reaction_frequency)
            if phase_index < len(phases):
                dt = min(dt, phases[phase_index] - time_in_cycle)
            if dt < period * 1e-10:
                raise RuntimeError(
                    "必要な時間刻みが小さすぎます。条件と物性範囲を確認してください。"
                )
            for attempt in range(10):
                new_time = current_time + dt
                voltage = float(waveform.at(new_time))
                try:
                    new_ni, new_pr, new_pz, ion_wall_flux = ion_step(
                        mesh,
                        ni,
                        pr,
                        pz,
                        er,
                        ez,
                        pair_source,
                        dt,
                        ion_xsec,
                        neutral_density,
                        p.gas_temperature_k,
                    )
                    yields = np.where(
                        mesh.boundary_cathode, p.cathode_secondary_yield, p.ground_secondary_yield
                    )
                    emitted_flux = yields * ion_wall_flux
                    source_secondary = np.zeros(mesh.size)
                    np.add.at(
                        source_secondary,
                        mesh.boundary_cell,
                        emitted_flux * mesh.boundary_area / mesh.volume.ravel()[mesh.boundary_cell],
                    )
                    electron_source = pair_source + source_secondary.reshape(mesh.shape)
                    new_phi, new_ne, operator, iterations, residual = solve_electron_poisson(
                        mesh,
                        new_ni,
                        ne,
                        phi,
                        energy,
                        coefficients["mobility_m2_vs"],
                        electron_source,
                        dt,
                        voltage,
                        scale,
                        cfg.newton_tolerance,
                        cfg.newton_iterations,
                        p.electron_sticking,
                        p.electron_flux_limit,
                    )
                    source_energy = (
                        np.maximum(source_maintenance, 0) * p.source_mean_energy_ev
                        + np.minimum(source_maintenance, 0) * energy
                    )
                    # Emitted electrons are a fluid source approximation, not a resolved beam.
                    source_se_energy = np.zeros(mesh.size)
                    gain = np.maximum(
                        new_phi.ravel()[mesh.boundary_cell] - mesh.boundary_voltage(voltage), 0
                    )
                    np.add.at(
                        source_se_energy,
                        mesh.boundary_cell,
                        emitted_flux
                        * (p.secondary_energy_ev + gain)
                        * mesh.boundary_area
                        / mesh.volume.ravel()[mesh.boundary_cell],
                    )
                    source_energy += source_se_energy.reshape(mesh.shape)
                    new_w, energy_stats = electron_energy_step(
                        mesh,
                        w,
                        new_ne,
                        energy,
                        new_phi,
                        operator,
                        source_energy,
                        coefficients,
                        neutral_density,
                        p.gas_temperature_k,
                        dt,
                        voltage,
                        scale,
                    )
                    break
                except ArithmeticError:
                    counters["rejected_steps"] += 1
                    dt *= 0.5
                    if attempt == 9:
                        raise
            ni, pr, pz, ne, phi, w = new_ni, new_pr, new_pz, new_ne, new_phi, new_w
            ion_loss = float(np.sum(ion_wall_flux * mesh.boundary_area))
            electron_loss = float(
                np.sum(operator.boundary_rate * ne.ravel()[mesh.boundary_cell] * mesh.boundary_area)
            )
            secondary_creation = float(np.sum(emitted_flux * mesh.boundary_area))
            source_number = float(np.sum(pair_source * mesh.volume))
            counters["ion_created"] += source_number * dt
            counters["electron_created"] += source_number * dt
            counters["electron_secondary_created"] += secondary_creation * dt
            counters["ion_wall_loss"] += ion_loss * dt
            counters["electron_wall_loss"] += electron_loss * dt
            counters["source_energy_j"] += (
                float(np.sum(source_energy * mesh.volume)) * E_CHARGE * dt
            )
            for key in (
                "field_work",
                "field_work_raw",
                "inelastic_loss",
                "elastic_loss",
                "wall_energy_loss",
            ):
                counters[f"{key}_j"] += energy_stats[f"{key}_w"] * dt
            counters["thermal_floor_energy_j"] += energy_stats["thermal_floor_energy_j"]
            counters["maximum_newton_residual"] = max(counters["maximum_newton_residual"], residual)
            counters["maximum_limited_face_fraction"] = max(
                counters["maximum_limited_face_fraction"],
                operator.limited_fraction,
            )
            new_energy = w / (E_CHARGE * np.maximum(ne, scale * 1e-18))
            outside = (new_energy < lower_energy) | (new_energy > upper_energy)
            outside_fraction = float(
                np.sum(ne[outside] * mesh.volume[outside]) / max(np.sum(ne * mesh.volume), 1)
            )
            counters["maximum_table_outside_mass_fraction"] = max(
                counters["maximum_table_outside_mass_fraction"],
                outside_fraction,
            )
            counters["steps"] += 1
            time_in_cycle += dt
            if time.monotonic() - last_progress > 0.4:
                progress(
                    (cycle + time_in_cycle / period) / cfg.maximum_cycles,
                    f"場計算 {cycle + 1}/{cfg.maximum_cycles}周期 / "
                    f"位相 {time_in_cycle / period:.1%} / Newton {iterations}反復",
                )
                last_progress = time.monotonic()
        # All stored samples belong to the same completed cycle; no synthetic interpolation.
        if any(len(values) != cfg.stored_phases for values in frames.values()):
            raise RuntimeError("周期場の位相保存に失敗しました。")
        final_frames = {name: np.asarray(values) for name, values in frames.items()}
        errors = {
            name: _period_error(array, previous)
            for name, array, previous in zip(
                ("ne", "ni", "energy", "potential"), (ne, ni, w, phi), begin_state, strict=True
            )
        }
        periodic_error = max(errors.values())
        number_e = float(np.sum(ne * mesh.volume))
        number_i = float(np.sum(ni * mesh.volume))
        history.append(
            {
                "cycle": cycle + 1,
                "periodic_error": periodic_error,
                **errors,
                "mean_electron_density_m3": number_e / mesh.volume.sum(),
                "mean_ion_density_m3": number_i / mesh.volume.sum(),
                "mean_electron_energy_ev": float(
                    np.sum(w * mesh.volume) / (E_CHARGE * max(number_e, 1))
                ),
            }
        )
        save_checkpoint(cycle + 1)
        if cycle + 1 >= cfg.minimum_cycles and periodic_error < cfg.periodic_tolerance:
            converged, stop_reason = True, "周期定常の許容値を満たしました"
            break
        if number_e < initial_electrons * 1e-4:
            stop_reason = "電子密度低下：このモデル・供給条件では放電が維持されていません"
            break
    if final_frames is None:
        raise ValueError(
            "再開する場合は最大周期数をチェックポイントの完了周期数より増やしてください。"
        )
    final_ions = float(np.sum(ni * mesh.volume))
    final_electrons = float(np.sum(ne * mesh.volume))
    ion_expected = initial_ions + counters["ion_created"] - counters["ion_wall_loss"]
    electron_expected = (
        initial_electrons
        + counters["electron_created"]
        + counters["electron_secondary_created"]
        - counters["electron_wall_loss"]
    )
    energy_expected = (
        initial_energy
        + counters["field_work_j"]
        + counters["source_energy_j"]
        - counters["inelastic_loss_j"]
        - counters["elastic_loss_j"]
        - counters["wall_energy_loss_j"]
        + counters["thermal_floor_energy_j"]
    )
    debye = np.sqrt(
        EPSILON_0
        * np.maximum(2 * final_frames["energy"] / 3, 0.01)
        / (E_CHARGE * np.maximum(final_frames["ne"], scale * 1e-12))
    )
    dense = final_frames["ne"] > scale * 0.01
    minimum_debye = float(np.min(debye[dense])) if dense.any() else float("nan")
    warnings_list = [
        "電子はDC物性＋Maxwell輸送モーメント近似。非局所EEDFは解いていません。",
        "イオン場は等温流体・衝突摩擦近似。テスト粒子密度は場にフィードバックしません。",
        "断面積のイオンエネルギー軸（重心系/実験室系）は利用者による確認が必要です。",
        "格子・時間刻み・粒子数の収束確認と実験照合は別途必要です。",
    ]
    work_difference = abs(counters["field_work_j"] - counters["field_work_raw_j"]) / max(
        abs(counters["field_work_j"]) + abs(counters["field_work_raw_j"]),
        initial_energy * 1e-8,
    )
    if work_difference > 0.1:
        warnings_list.append(
            "電子仕事の陰的線形化補正が大きいです。時間刻み・格子の収束を確認してください。"
        )
    if not converged:
        warnings_list.append("周期未収束：この場の周期再生による粒子分布は探索用です。")
    if np.isfinite(minimum_debye) and min(mesh.dr.min(), mesh.dz.min()) > minimum_debye:
        warnings_list.append("最小Debye長より格子が粗く、シース遷移層の解像度が不足しています。")
    if counters["maximum_table_outside_mass_fraction"] > 0.001:
        warnings_list.append(
            "電子物性テーブル外で端点係数を使用した領域があります。物性範囲を拡張してください。"
        )
    if counters["maximum_limited_face_fraction"] > 0:
        warnings_list.append("電子流束制限を適用しました。低圧の自由飛行を近似する数値閉包です。")
    if p.cathode_secondary_yield or p.ground_secondary_yield:
        warnings_list.append("二次電子は流体源近似です。高電圧でのビーム輸送・電離は未検証です。")
    if p.density_mode == "maintained":
        warnings_list.append(
            "密度維持モード：外部の電荷中性粒子供給と供給エネルギーを仮定しています。"
        )
    if waveform.seam_error_v > max(np.ptp(waveform.voltage_v) * 0.01, 1):
        warnings_list.append("波形の周期端が連続ではありません。最初の1周期を再生しています。")
    diagnostics = dict(
        counters,
        periodic_converged=converged,
        stop_reason=stop_reason,
        completed_cycles=len(history),
        periodic_error=history[-1]["periodic_error"],
        elapsed_s=time.monotonic() - start_time,
        ion_balance_relative_error=abs(final_ions - ion_expected) / max(initial_ions, 1),
        electron_balance_relative_error=abs(final_electrons - electron_expected)
        / max(initial_electrons, 1),
        electron_energy_balance_relative_error=abs(float(np.sum(w * mesh.volume)) - energy_expected)
        / max(
            initial_energy + abs(counters["field_work_j"]) + abs(counters["source_energy_j"]), 1e-30
        ),
        minimum_debye_m=minimum_debye,
        work_linearization_relative_difference=work_difference,
        radial_spacing_m=float(mesh.dr.min()),
        axial_spacing_m=float(mesh.dz.min()),
        ground_cathode_area_ratio=mesh.ground_area / mesh.cathode_area,
        ion_xsec_sha256=ion_xsec.source_hash,
        electron_xsec_sha256=table.metadata["source_sha256"],
        transport_cache_key=table.metadata["cache_key"],
        boltzpmp_version=table.metadata["boltzpmp_version"],
        density_mode=p.density_mode,
        warnings=warnings_list,
        field_model="axisymmetric ion Euler / SG electron continuity + energy / coupled Poisson",
        bottom_annulus_boundary="insulating, zero normal particle and electric flux",
        source_waveform=waveform.source,
        source_waveform_sha256=waveform.source_hash,
    )
    fields_r, fields_z = [], []
    for phase, potential in zip(phases, final_frames["phi"], strict=True):
        fr, fz = mesh.electric_field(potential, float(waveform.at(phase)))
        fields_r.append(fr)
        fields_z.append(fz)
    progress(1.0, stop_reason)
    return FieldResult(
        phases,
        final_frames["phi"],
        np.asarray(fields_r),
        np.asarray(fields_z),
        final_frames["ne"],
        final_frames["ni"],
        final_frames["energy"],
        final_frames["ur"],
        final_frames["uz"],
        diagnostics,
        history,
        project.to_dict(),
        waveform.time_s,
        waveform.voltage_v,
    )
