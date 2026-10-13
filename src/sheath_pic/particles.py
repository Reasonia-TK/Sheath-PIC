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


def interpolate_periodic(
    mesh: AxisymmetricMesh, frames: np.ndarray, time_s, radius, height, period_s: float
) -> np.ndarray:
    """Trilinear interpolation in phase, radius, and height; periodic only in time."""
    phase = np.remainder(time_s, period_s) / period_s * len(frames)
    it = np.floor(phase).astype(int) % len(frames)
    ft = phase - np.floor(phase)
    r = np.clip(radius, mesh.r[0], mesh.r[-1])
    z = np.clip(height, mesh.z[0], mesh.z[-1])
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
    er = interpolate_periodic(mesh, field.electric_r_vm, time_s, radius, position[:, 2], period_s)
    ez = interpolate_periodic(mesh, field.electric_z_vm, time_s, radius, position[:, 2], period_s)
    direction = np.divide(
        position[:, :2],
        radius[:, None],
        out=np.zeros_like(position[:, :2]),
        where=radius[:, None] > 1e-20,
    )
    return np.column_stack((er[:, None] * direction, ez))


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
) -> ParticleResult:
    if not field.converged and not allow_unconverged:
        raise ValueError(
            "周期未収束の場です。場計算を継続するか、探索用の未収束場を許可してください。"
        )
    progress = progress or (lambda *_: None)
    cancelled = cancelled or (lambda: False)
    project.validate()
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
    if source_rate <= 1e-10:
        probability = field.ion_density_m3 * mesh.volume[None, :, :]
        source_rate = None
        source_name = "背景イオン数分布（供給源未確定・相対分布のみ）"
        warnings_list.append(
            "イオン生成源がほぼ0のため、背景分布からの探索です。絶対フラックスは未定義です。"
        )
    flat = probability.ravel()
    if not np.all(np.isfinite(flat)) or flat.sum() <= 0:
        raise ValueError("粒子を注入できる正のイオン供給源がありません。")
    selection = rng.choice(len(flat), cfg.particles, p=flat / flat.sum())
    phase_index, ir, iz = np.unravel_index(selection, probability.shape)
    radius = np.sqrt(rng.uniform(mesh.r_edges[ir] ** 2, mesh.r_edges[ir + 1] ** 2))
    azimuth = rng.uniform(0, 2 * np.pi, cfg.particles)
    height = rng.uniform(mesh.z_edges[iz], mesh.z_edges[iz + 1])
    position = np.column_stack((radius * np.cos(azimuth), radius * np.sin(azimuth), height))
    thermal_speed = np.sqrt(K_B * p.gas_temperature_k / M_AR)
    velocity = rng.normal(0, thermal_speed, (cfg.particles, 3))
    times = (phase_index + rng.random(cfg.particles)) * period / len(field.phase_s)
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
        dt = np.minimum(base_dt, 0.25 * minimum_h / np.maximum(speed, thermal_speed))
        dt = np.minimum(dt, np.sqrt(0.25 * minimum_h / np.maximum(acceleration, 1)))
        # Hazard is sampled with a thermal neutral; the small-probability step resolves collisions.
        neutral_velocity = rng.normal(0, thermal_speed, v.shape)
        relative_speed = np.linalg.norm(v - neutral_velocity, axis=1)
        energy_cm = cross_sections.collision_energy(relative_speed)
        sigma_iso = cross_sections.isotropic.at(energy_cm)
        sigma_back = cross_sections.backscatter.at(energy_cm)
        frequency = neutral_density * relative_speed * (sigma_iso + sigma_back)
        if collisions:
            dt = np.minimum(dt, 0.05 / np.maximum(frequency, 1))
        if has_volume_removal:
            volume_hazard = interpolate_periodic(
                mesh, removal_frequency, t, np.linalg.norm(x[:, :2], axis=1), x[:, 2], period
            )
            dt = np.minimum(dt, 0.05 / np.maximum(volume_hazard, 1))
        dt = np.minimum(dt, cfg.particle_max_periods * period - (t - birth[ids]))
        half_v = v + 0.5 * a0 * dt[:, None]
        next_position = x + half_v * dt[:, None]
        next_time = t + dt
        e1 = electric_cartesian(mesh, field, next_position, next_time, period)
        next_velocity = half_v + 0.5 * E_CHARGE / M_AR * e1 * dt[:, None]
        # Find first physical surface intersection, including cylinder curvature.
        delta = next_position - x
        crossing = np.full(len(ids), np.inf)
        surface = np.zeros(len(ids), dtype=int)  # 1 bottom, 2 top, 3 side
        for boundary_z, kind in ((0.0, 1), (project.geometry.gap_m, 2)):
            moving = delta[:, 2] < 0 if kind == 1 else delta[:, 2] > 0
            fraction = np.divide(
                boundary_z - x[:, 2], delta[:, 2], out=np.full(len(ids), np.inf), where=moving
            )
            choose = moving & (fraction >= 0) & (fraction <= 1) & (fraction < crossing)
            crossing[choose], surface[choose] = fraction[choose], kind
        qa = np.sum(delta[:, :2] ** 2, axis=1)
        qb = 2 * np.sum(x[:, :2] * delta[:, :2], axis=1)
        qc = np.sum(x[:, :2] ** 2, axis=1) - project.geometry.chamber_radius_m**2
        discriminant = np.maximum(qb * qb - 4 * qa * qc, 0)
        side_fraction = np.divide(
            -qb + np.sqrt(discriminant), 2 * qa, out=np.full(len(ids), np.inf), where=qa > 0
        )
        choose = (side_fraction >= 0) & (side_fraction <= 1) & (side_fraction < crossing)
        crossing[choose], surface[choose] = side_fraction[choose], 3
        removed = np.zeros(len(ids), dtype=bool)
        if has_volume_removal:
            # Compete with the first wall hit so a removed ion cannot also reach a wall.
            residence_dt = dt * np.minimum(crossing, 1)
            removed = rng.random(len(ids)) < -np.expm1(-volume_hazard * residence_dt)
            active[ids[removed]] = False
            lost_external += int(removed.sum())
            surface[removed] = 0
        at_surface = surface > 0
        contact = x + delta * np.minimum(crossing, 1)[:, None]
        contact_radius = np.linalg.norm(contact[:, :2], axis=1)
        wafer = (surface == 1) & (contact_radius <= project.geometry.cathode_radius_m)
        annulus = (surface == 1) & ~wafer
        ground = (surface == 2) | (surface == 3)
        if wafer.any():
            hit_v = v[wafer] + a0[wafer] * (dt[wafer] * crossing[wafer])[:, None]
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
            hit_phase[hit_ids] = np.remainder(t[wafer] + dt[wafer] * crossing[wafer], period)
            arrival[hit_ids] = True
            active[hit_ids] = False
        if ground.any():
            active[ids[ground]] = False
            lost_ground += int(ground.sum())
        if annulus.any():
            next_position[annulus, 2] = abs(next_position[annulus, 2])
            next_velocity[annulus, 2] = abs(next_velocity[annulus, 2])
            reflecting_annulus_hits += int(annulus.sum())
        if collisions:
            occurrence = rng.random(len(ids)) < -np.expm1(-frequency * dt)
            occurrence &= ~at_surface & ~removed
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
    )
