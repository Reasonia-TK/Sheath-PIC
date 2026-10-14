from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .config import Project
from .constants import E_CHARGE, K_B, M_AR
from .mesh import AxisymmetricMesh
from .solver import FieldResult
from .transport import Progress, TransportTable
from .xsections import IonCrossSections, scatter_equal_mass


@dataclass
class ParticleInjection:
    """Prescribed ions for reproducible orbit benchmarks (SI units)."""

    position_m: np.ndarray
    velocity_ms: np.ndarray
    time_s: np.ndarray
    source_rate_s: float | None = None

    def validate(self, project: Project) -> None:
        n = project.numerics.particles
        for value, shape in (
            (self.position_m, (n, 3)),
            (self.velocity_ms, (n, 3)),
            (self.time_s, (n,)),
        ):
            if np.shape(value) != shape or not np.all(np.isfinite(value)):
                raise ValueError(
                    "指定注入の位置・速度・時刻は粒子数と一致する有限の配列が必要です。"
                )
        position = np.asarray(self.position_m)
        if (
            np.any(np.linalg.norm(position[:, :2], axis=1) >= project.geometry.chamber_radius_m)
            or np.any(position[:, 2] <= 0)
            or np.any(position[:, 2] >= project.geometry.gap_m)
        ):
            raise ValueError("指定注入の位置は計算領域の内部にしてください。")
        if self.source_rate_s is not None and (
            not np.isfinite(self.source_rate_s) or self.source_rate_s < 0
        ):
            raise ValueError("指定注入の供給率は0以上の有限な値が必要です。")


@dataclass
class ParticleResult:
    energy_edges_ev: np.ndarray
    angle_edges_deg: np.ndarray
    radius_edges_m: np.ndarray
    counts: np.ndarray
    iaedf: np.ndarray
    iedf: np.ndarray
    iadf: np.ndarray
    radial_iaedf: np.ndarray
    radial_flux_m2_s: np.ndarray
    arrival_energy_ev: np.ndarray
    arrival_angle_deg: np.ndarray
    arrival_radius_m: np.ndarray
    arrival_phase_s: np.ndarray
    arrival_collisions: np.ndarray
    diagnostics: dict
    arrival_particle_index: np.ndarray | None = None


def interpolate_periodic(
    mesh: AxisymmetricMesh,
    frames: np.ndarray,
    time_s,
    radius,
    height,
    period_s: float,
    extrapolate: bool = False,
) -> np.ndarray:
    """Trilinear interpolation in phase, radius, and height; periodic only in time."""
    phase = np.remainder(time_s, period_s) / period_s * len(frames)
    it = np.floor(phase).astype(int) % len(frames)
    ft = phase - np.floor(phase)
    r = np.clip(
        radius,
        0 if extrapolate else mesh.r[0],
        mesh.geometry.chamber_radius_m if extrapolate else mesh.r[-1],
    )
    z = np.clip(
        height, 0 if extrapolate else mesh.z[0], mesh.geometry.gap_m if extrapolate else mesh.z[-1]
    )
    ir = np.clip(np.searchsorted(mesh.r, r) - 1, 0, len(mesh.r) - 2)
    iz = np.clip(np.searchsorted(mesh.z, z) - 1, 0, len(mesh.z) - 2)
    fr = (r - mesh.r[ir]) / (mesh.r[ir + 1] - mesh.r[ir])
    fz = (z - mesh.z[iz]) / (mesh.z[iz + 1] - mesh.z[iz])

    def sample(k):
        return (
            (1 - fr) * (1 - fz) * frames[k, ir, iz]
            + fr * (1 - fz) * frames[k, ir + 1, iz]
            + (1 - fr) * fz * frames[k, ir, iz + 1]
            + fr * fz * frames[k, ir + 1, iz + 1]
        )

    return (1 - ft) * sample(it) + ft * sample((it + 1) % len(frames))


def electric_cartesian(
    mesh: AxisymmetricMesh, field: FieldResult, position: np.ndarray, time_s, period_s: float
) -> np.ndarray:
    radius = np.linalg.norm(position[:, :2], axis=1)
    sampled_r = np.maximum(radius, mesh.r[0])
    er = interpolate_periodic(
        mesh, field.electric_r_vm, time_s, sampled_r, position[:, 2], period_s, extrapolate=True
    )
    er *= np.minimum(radius / mesh.r[0], 1)
    ez = interpolate_periodic(
        mesh, field.electric_z_vm, time_s, sampled_r, position[:, 2], period_s, extrapolate=True
    )
    axis = radius < mesh.r[0]
    if axis.any():
        sampled_t = np.broadcast_to(time_s, radius.shape)[axis]
        second = interpolate_periodic(
            mesh,
            field.electric_z_vm,
            sampled_t,
            mesh.r[1],
            position[axis, 2],
            period_s,
            extrapolate=True,
        )
        ez[axis] += (
            (radius[axis] ** 2 - mesh.r[0] ** 2)
            / (mesh.r[1] ** 2 - mesh.r[0] ** 2)
            * (second - ez[axis])
        )
    # Odd normal field at the ideal reflecting annulus; E_z=0 exactly on that face.
    annulus = (radius > mesh.geometry.cathode_radius_m) & (position[:, 2] < mesh.z[0])
    if annulus.any():
        sampled_t = np.broadcast_to(time_s, radius.shape)[annulus]
        first = interpolate_periodic(
            mesh,
            field.electric_z_vm,
            sampled_t,
            sampled_r[annulus],
            mesh.z[0],
            period_s,
            extrapolate=True,
        )
        ez[annulus] = first * np.clip(position[annulus, 2] / mesh.z[0], 0, 1)
    direction = np.divide(
        position[:, :2],
        radius[:, None],
        out=np.zeros_like(position[:, :2]),
        where=radius[:, None] > 1e-20,
    )
    return np.column_stack((er[:, None] * direction, ez))


def surface_events(project, position, velocity, acceleration, dt):
    """First hit on the local Verlet parabola; time is not a chord-length fraction."""
    hit_time = np.full(len(dt), np.inf)
    surface = np.zeros(len(dt), dtype=int)  # bottom=1, top=2, cylindrical side=3
    aa, bb = 0.5 * acceleration[:, 2], velocity[:, 2]
    for boundary, kind, direction in ((0.0, 1, -1), (project.geometry.gap_m, 2, 1)):
        cc = position[:, 2] - boundary
        discriminant = bb**2 - 4 * aa * cc
        q = -0.5 * (bb + np.copysign(np.sqrt(np.maximum(discriminant, 0)), bb))
        candidates = [
            np.divide(q, aa, out=np.full(len(dt), np.inf), where=aa != 0),
            np.divide(cc, q, out=np.full(len(dt), np.inf), where=q != 0),
            np.divide(-cc, bb, out=np.full(len(dt), np.inf), where=(aa == 0) & (bb != 0)),
        ]
        for tau in candidates:
            with np.errstate(invalid="ignore"):
                outgoing = (bb + 2 * aa * tau) * direction > 0
            choose = (discriminant >= 0) & (tau >= 0) & (tau <= dt) & outgoing & (tau < hit_time)
            hit_time[choose], surface[choose] = tau[choose], kind
    end = position + velocity * dt[:, None] + 0.5 * acceleration * dt[:, None] ** 2
    delta = end[:, :2] - position[:, :2]
    qa = np.sum(delta**2, axis=1)
    qb = 2 * np.sum(position[:, :2] * delta, axis=1)
    qc = np.sum(position[:, :2] ** 2, axis=1) - project.geometry.chamber_radius_m**2
    fraction = np.divide(
        -qb + np.sqrt(np.maximum(qb**2 - 4 * qa * qc, 0)),
        2 * qa,
        out=np.full(len(dt), np.inf),
        where=qa > 0,
    )
    side = (fraction >= 0) & (fraction <= 1)
    if side.any():
        tau = dt[side] * fraction[side]
        x, v, a, upper = position[side, :2], velocity[side, :2], acceleration[side, :2], dt[side]
        for _ in range(8):
            contact = x + v * tau[:, None] + 0.5 * a * tau[:, None] ** 2
            value = np.sum(contact**2, axis=1) - project.geometry.chamber_radius_m**2
            derivative = 2 * np.sum(contact * (v + a * tau[:, None]), axis=1)
            tau -= np.divide(value, derivative, out=np.zeros_like(tau), where=derivative != 0)
            tau = np.clip(tau, 0, upper)
        ids = np.flatnonzero(side)
        choose = tau < hit_time[side]
        hit_time[ids[choose]], surface[ids[choose]] = tau[choose], 3
    return hit_time, surface


def distributions(
    energy: np.ndarray,
    angle: np.ndarray,
    radius: np.ndarray,
    project: Project,
    source_rate_s: float | None,
):
    cfg = project.numerics
    maximum = max(float(energy.max()) * (1 + 1e-9), 1.0)
    energy_edges = np.linspace(0, maximum, cfg.energy_bins + 1)
    angle_edges = np.linspace(0, 90, cfg.angle_bins + 1)
    radial_edges = np.linspace(0, project.geometry.cathode_radius_m, cfg.radial_bins + 1)
    counts, _ = np.histogramdd(
        np.column_stack((radius, energy, angle)), bins=(radial_edges, energy_edges, angle_edges)
    )
    global_counts = counts.sum(axis=0)
    if int(global_counts.sum()) != len(energy):
        raise RuntimeError("分布の範囲外に到達粒子がありました。")
    de, da = np.diff(energy_edges), np.diff(angle_edges)
    joint = global_counts / len(energy) / de[:, None] / da[None, :]
    iedf = np.sum(joint * da[None, :], axis=1)
    iadf = np.sum(joint * de[:, None], axis=0)
    radial_total = counts.sum(axis=(1, 2))
    radial_joint = (
        np.divide(
            counts,
            radial_total[:, None, None],
            out=np.zeros_like(counts),
            where=radial_total[:, None, None] > 0,
        )
        / de[None, :, None]
        / da[None, None, :]
    )
    area = np.pi * np.diff(radial_edges**2)
    flux = (
        np.full(cfg.radial_bins, np.nan)
        if source_rate_s is None
        else (radial_total * source_rate_s / cfg.particles / area)
    )
    return energy_edges, angle_edges, radial_edges, counts, joint, iedf, iadf, radial_joint, flux


def run_particles(
    project: Project,
    field: FieldResult,
    table: TransportTable,
    progress: Progress | None = None,
    cancelled: Callable[[], bool] | None = None,
    allow_unconverged: bool = False,
    collisions: bool = True,
    injection: ParticleInjection | None = None,
    cell_step_fraction: float = 0.25,
    collision_step_hazard: float = 0.05,
) -> ParticleResult:
    if not field.converged and not allow_unconverged:
        raise ValueError(
            "周期未収束の場です。場計算を継続するか、探索用の未収束場を許可してください。"
        )
    progress = progress or (lambda *_: None)
    cancelled = cancelled or (lambda: False)
    project.validate()
    if not np.isfinite(cell_step_fraction) or not 0 < cell_step_fraction <= 0.5:
        raise ValueError("粒子のセル通過刻み係数は0より大きく0.5以下にしてください。")
    if not np.isfinite(collision_step_hazard) or not 0 < collision_step_hazard <= 0.1:
        raise ValueError("粒子の衝突ハザード刻みは0より大きく0.1以下にしてください。")
    if injection is not None:
        injection.validate(project)
    mesh = AxisymmetricMesh(project.geometry)
    cfg, p = project.numerics, project.plasma
    rng = np.random.default_rng(cfg.seed)
    cross_sections = IonCrossSections.load(project.path(project.ion_xsec), p.ion_energy_frame)
    neutral_density = p.pressure_pa / (K_B * p.gas_temperature_k)
    period = 1 / cfg.frequency_hz
    rates = table.coefficients(field.mean_energy_ev)["ionization_m3_s"]
    source = field.electron_density_m3 * neutral_density * rates
    removal_frequency = np.zeros_like(source)
    if p.density_mode == "maintained":
        target = mesh.initial_density(p, project.path(p.density_csv) if p.density_csv else None)
        bulk = (
            abs(field.ion_density_m3 - field.electron_density_m3)
            / np.maximum(
                field.ion_density_m3 + field.electron_density_m3,
                1,
            )
            < 0.1
        )
        maintenance = (
            (target - 0.5 * (field.ion_density_m3 + field.electron_density_m3))
            / (p.density_relaxation_periods * period)
            * bulk
        )
        maintenance = np.maximum(
            maintenance,
            -0.1 * np.minimum(field.electron_density_m3, field.ion_density_m3) / period,
        )
        source += np.maximum(maintenance, 0)
        removal_frequency = -np.minimum(maintenance, 0) / np.maximum(field.ion_density_m3, 1)
    has_volume_removal = bool(np.any(removal_frequency > 0))
    probability = source * mesh.volume[None, :, :]
    source_rate = float(np.mean(np.sum(probability, axis=(1, 2))))
    warnings_list = []
    source_name = "位相別の電子衝突電離＋指定密度維持源"
    if source_rate <= 1e-10 and injection is None:
        probability = field.ion_density_m3 * mesh.volume[None, :, :]
        source_rate = None
        source_name = "背景イオン数分布（供給源未確定・相対分布のみ）"
        warnings_list.append(
            "イオン生成源がほぼ0のため、背景分布からの探索です。絶対フラックスは未定義です。"
        )
    thermal_speed = np.sqrt(K_B * p.gas_temperature_k / M_AR)
    if injection is None:
        flat = probability.ravel()
        if not np.all(np.isfinite(flat)) or flat.sum() <= 0:
            raise ValueError("粒子を注入できる正のイオン供給源がありません。")
        selection = rng.choice(len(flat), cfg.particles, p=flat / flat.sum())
        phase_index, ir, iz = np.unravel_index(selection, probability.shape)
        radius = np.sqrt(rng.uniform(mesh.r_edges[ir] ** 2, mesh.r_edges[ir + 1] ** 2))
        azimuth = rng.uniform(0, 2 * np.pi, cfg.particles)
        height = rng.uniform(mesh.z_edges[iz], mesh.z_edges[iz + 1])
        position = np.column_stack((radius * np.cos(azimuth), radius * np.sin(azimuth), height))
        velocity = rng.normal(0, thermal_speed, (cfg.particles, 3))
        times = (phase_index + rng.random(cfg.particles)) * period / len(field.phase_s)
    else:
        position = np.array(injection.position_m, dtype=float, copy=True)
        velocity = np.array(injection.velocity_ms, dtype=float, copy=True)
        times = np.array(injection.time_s, dtype=float, copy=True)
        source_rate = injection.source_rate_s
        source_name = "指定粒子注入（軌道・衝突の検証用）"
    birth = times.copy()
    active = np.ones(cfg.particles, dtype=bool)
    collision_count = np.zeros(cfg.particles, dtype=int)
    arrival = np.zeros(cfg.particles, dtype=bool)
    hit_energy = np.zeros(cfg.particles)
    hit_angle = np.zeros(cfg.particles)
    hit_radius = np.zeros(cfg.particles)
    hit_phase = np.zeros(cfg.particles)
    lost_ground = 0
    lost_external = 0
    timed_out = 0
    reflecting_annulus_hits = 0
    steps = 0
    progress(0, f"2D3V粒子を{cfg.particles:,}個注入しました。")
    base_dt = period / max(cfg.steps_per_period, cfg.stored_phases * 4)
    minimum_h = min(mesh.dr.min(), mesh.dz.min())
    while active.any():
        if cancelled():
            raise InterruptedError("粒子計算を中断しました。seedを保持して再実行できます。")
        ids = np.flatnonzero(active)
        x, v, t = position[ids].copy(), velocity[ids].copy(), times[ids].copy()
        e0 = electric_cartesian(mesh, field, x, t, period)
        a0 = E_CHARGE / M_AR * e0
        speed = np.linalg.norm(v, axis=1)
        acceleration = np.linalg.norm(a0, axis=1)
        dt = np.minimum(base_dt, cell_step_fraction * minimum_h / np.maximum(speed, thermal_speed))
        dt = np.minimum(dt, np.sqrt(cell_step_fraction * minimum_h / np.maximum(acceleration, 1)))
        dt = np.minimum(dt, cfg.particle_max_periods * period - (t - birth[ids]))
        collision_time = np.full(len(ids), np.inf)
        if collisions:
            neutral_velocity = rng.normal(0, thermal_speed, v.shape)
            # A midpoint hazard accounts for acceleration during the step. Event times
            # compete with the wall: collisions before the last wall step must not vanish.
            for _ in range(3):
                relative_speed = np.linalg.norm(
                    v + 0.5 * a0 * dt[:, None] - neutral_velocity, axis=1
                )
                collision_energy = cross_sections.collision_energy(relative_speed)
                sigma_iso = cross_sections.isotropic.at(collision_energy)
                sigma_back = cross_sections.backscatter.at(collision_energy)
                frequency = neutral_density * relative_speed * (sigma_iso + sigma_back)
                shortened = np.minimum(dt, collision_step_hazard / np.maximum(frequency, 1))
                if np.array_equal(shortened, dt):
                    break
                dt = shortened
            wait = -np.log(np.maximum(rng.random(len(ids)), np.finfo(float).tiny))
            collision_time = np.divide(wait, frequency, out=collision_time, where=frequency > 0)
        removal_time = np.full(len(ids), np.inf)
        if has_volume_removal:
            volume_hazard = interpolate_periodic(
                mesh, removal_frequency, t, np.linalg.norm(x[:, :2], axis=1), x[:, 2], period
            )
            dt = np.minimum(dt, collision_step_hazard / np.maximum(volume_hazard, 1))
            wait = -np.log(np.maximum(rng.random(len(ids)), np.finfo(float).tiny))
            removal_time = np.divide(wait, volume_hazard, out=removal_time, where=volume_hazard > 0)
        wall_time, surface = surface_events(project, x, v, a0, dt)
        at_surface = (wall_time <= dt) & (wall_time <= collision_time) & (wall_time <= removal_time)
        removed = (
            (removal_time <= dt) & (removal_time < wall_time) & (removal_time <= collision_time)
        )
        occurrence = (
            (collision_time <= dt) & (collision_time < wall_time) & (collision_time < removal_time)
        )
        residence_dt = np.minimum.reduce((dt, wall_time, collision_time, removal_time))
        half_v = v + 0.5 * a0 * residence_dt[:, None]
        next_position = x + half_v * residence_dt[:, None]
        next_time = t + residence_dt
        next_position[at_surface & (surface == 1), 2] = 0
        next_position[at_surface & (surface == 2), 2] = project.geometry.gap_m
        e1 = electric_cartesian(mesh, field, next_position, next_time, period)
        next_velocity = half_v + 0.5 * E_CHARGE / M_AR * e1 * residence_dt[:, None]
        active[ids[removed]] = False
        lost_external += int(removed.sum())
        contact_radius = np.linalg.norm(next_position[:, :2], axis=1)
        wafer = at_surface & (surface == 1) & (contact_radius <= project.geometry.cathode_radius_m)
        annulus = at_surface & (surface == 1) & ~wafer
        ground = at_surface & ((surface == 2) | (surface == 3))
        if wafer.any():
            hit_v = next_velocity[wafer]
            hit_ids = ids[wafer]
            magnitude = np.linalg.norm(hit_v, axis=1)
            hit_energy[hit_ids] = 0.5 * M_AR * magnitude**2 / E_CHARGE
            hit_angle[hit_ids] = np.degrees(
                np.arccos(
                    np.clip(
                        abs(hit_v[:, 2]) / np.maximum(magnitude, 1e-30),
                        0,
                        1,
                    )
                )
            )
            hit_radius[hit_ids] = contact_radius[wafer]
            hit_phase[hit_ids] = np.remainder(next_time[wafer], period)
            arrival[hit_ids] = True
            active[hit_ids] = False
        if ground.any():
            active[ids[ground]] = False
            lost_ground += int(ground.sum())
        if annulus.any():
            next_velocity[annulus, 2] = abs(next_velocity[annulus, 2])
            reflecting_annulus_hits += int(annulus.sum())
        if occurrence.any():
            back = rng.random(int(occurrence.sum())) < (
                sigma_back[occurrence]
                / np.maximum(sigma_iso[occurrence] + sigma_back[occurrence], 1e-40)
            )
            next_velocity[occurrence] = scatter_equal_mass(
                next_velocity[occurrence],
                neutral_velocity[occurrence],
                back,
                rng,
            )
            collision_count[ids[occurrence]] += 1
        position[ids], velocity[ids], times[ids] = next_position, next_velocity, next_time
        expired = next_time - birth[ids] >= cfg.particle_max_periods * period * (1 - 1e-12)
        expired &= active[ids]
        active[ids[expired]] = False
        timed_out += int(expired.sum())
        steps += 1
        if steps % 100 == 0:
            progress(
                1 - active.mean(),
                f"粒子計算 / Wafer到達 {arrival.sum():,} / "
                f"追跡中 {active.sum():,} / 接地面 {lost_ground:,}",
            )
        if steps > 200000:
            raise RuntimeError(
                "粒子追跡のステップ上限です。粒子の追跡時間と格子を確認してください。"
            )
    if not arrival.any():
        raise RuntimeError(
            "Wafer到達粒子がありません。場・供給源・最大追跡時間を確認してください。"
        )
    arrays = distributions(
        hit_energy[arrival], hit_angle[arrival], hit_radius[arrival], project, source_rate
    )
    if not field.converged:
        warnings_list.append("未収束場を周期再生した探索結果です。")
    if timed_out:
        warnings_list.append(
            f"{timed_out:,}粒子が追跡時間上限に達しました。分布に打切りの影響があります。"
        )
    if has_volume_removal:
        warnings_list.append(
            "密度維持モードの外部対除去を確率的な体積損失として粒子追跡に含めました。"
        )
    warnings_list.append("IADF/IAEDFの角度密度は度あたり。立体角あたりの密度ではありません。")
    diagnostics = {
        "injected": cfg.particles,
        "arrived": int(arrival.sum()),
        "lost_ground": lost_ground,
        "lost_external": lost_external,
        "timed_out": timed_out,
        "reflecting_annulus_hits": reflecting_annulus_hits,
        "seed": cfg.seed,
        "source": source_name,
        "source_rate_s": source_rate,
        "absolute_flux_available": source_rate is not None,
        "collisions": collisions,
        "particle_model_version": 2,
        "cell_step_fraction": cell_step_fraction,
        "collision_step_hazard": collision_step_hazard,
        "mean_energy_ev": float(hit_energy[arrival].mean()),
        "mean_angle_deg": float(hit_angle[arrival].mean()),
        "mean_collision_count": float(collision_count[arrival].mean()),
        "field_periodic_converged": field.converged,
        "warnings": warnings_list,
        "normalization": "flux sampled ions; PDF per eV / per degree; rings sampled by volume",
        "wafer_material": "Si",
        "ground_material": "Si",
    }
    progress(1, f"粒子計算完了 / Wafer到達 {arrival.sum():,}")
    return ParticleResult(
        *arrays,
        hit_energy[arrival],
        hit_angle[arrival],
        hit_radius[arrival],
        hit_phase[arrival],
        collision_count[arrival],
        diagnostics,
        np.flatnonzero(arrival),
    )
