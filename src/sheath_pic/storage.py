from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .particles import ParticleResult
from .solver import FieldResult


def json_value(value):
    if isinstance(value, dict):
        return {key: json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(json_value(value), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def _save_npz(path: Path, arrays: dict, metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            **arrays,
            metadata=json.dumps(json_value(metadata), ensure_ascii=False, allow_nan=False),
        )
    temporary.replace(path)


def save_field(path: Path, result: FieldResult) -> None:
    metadata_names = {"diagnostics", "history", "project"}
    _save_npz(
        path,
        {
            name: getattr(result, name)
            for name in result.__dataclass_fields__
            if name not in metadata_names and getattr(result, name) is not None
        },
        {name: getattr(result, name) for name in metadata_names},
    )


def load_field(path: Path) -> FieldResult:
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata"]))
        return FieldResult(
            **metadata, **{name: data[name].copy() for name in data.files if name != "metadata"}
        )


def save_particles(path: Path, result: ParticleResult) -> None:
    _save_npz(
        path,
        {
            name: getattr(result, name)
            for name in result.__dataclass_fields__
            if name != "diagnostics" and getattr(result, name) is not None
        },
        {"diagnostics": result.diagnostics},
    )


def load_particles(path: Path) -> ParticleResult:
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata"]))
        return ParticleResult(
            **metadata, **{name: data[name].copy() for name in data.files if name != "metadata"}
        )


def save_checkpoint(path: Path, state: dict, fingerprint: str) -> None:
    arrays = {key: state[key] for key in ("ne", "ni", "pr", "pz", "w", "phi")}
    metadata = {key: value for key, value in state.items() if key not in arrays}
    metadata["fingerprint"] = fingerprint
    _save_npz(path, arrays, metadata)


def load_checkpoint(path: Path, fingerprint: str) -> dict:
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata"]))
        if metadata.pop("fingerprint") != fingerprint:
            raise ValueError("再開時の物理条件・格子・物性がチェックポイントと一致しません。")
        return dict(metadata, **{key: data[key].copy() for key in data.files if key != "metadata"})


def field_fingerprint(project: dict, transport_key: str, waveform_hash: str = "") -> str:
    copy = json.loads(json.dumps(project))
    copy.pop("base_dir", None)
    for key in ("electron_xsec", "ion_xsec", "name"):
        copy.pop(key, None)
    p = copy["plasma"]
    p.pop("density_csv", None)
    n = copy["numerics"]
    for key in ("waveform_time_column", "waveform_voltage_column"):
        if not n[key]:
            n.pop(key)
    for key in (
        "maximum_cycles",
        "minimum_cycles",
        "periodic_tolerance",
        "particles",
        "particle_max_periods",
        "radial_bins",
        "energy_bins",
        "angle_bins",
        "seed",
        "waveform_csv",
        "backend",
    ):
        n.pop(key, None)
    payload = {
        "project": copy,
        "transport": transport_key,
        "waveform": waveform_hash,
        "field_equations_version": 2,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def plasma_observables(field: FieldResult, mesh) -> dict:
    ne, ni = field.electron_density_m3, field.ion_density_m3
    minimum = max(float(ne.max()) * 1e-3, 1)
    mask = (abs(ni - ne) / np.maximum(ni + ne, minimum) < 0.05) & (ne > minimum)
    mask[:, :, 0] = False
    mask[:, :, -1] = False
    mask[:, -1, :] = False
    weight = mask * ne * mesh.volume[None, :, :]
    norm = weight.sum(axis=(1, 2))
    vp = np.divide(
        np.sum(weight * field.potential_v, axis=(1, 2)),
        norm,
        out=np.full(len(norm), np.nan),
        where=norm > 0,
    )
    local_norm = weight.sum(axis=2)
    local_vp = np.divide(
        np.sum(weight * field.potential_v, axis=2),
        local_norm,
        out=np.full(local_norm.shape, np.nan),
        where=local_norm > 0,
    )
    sheath_edge = np.full(local_norm.shape, np.nan)
    for phase in range(len(mask)):
        for radial in range(mesh.geometry.radial_cells):
            locations = np.flatnonzero(mask[phase, radial])
            if len(locations):
                sheath_edge[phase, radial] = mesh.z[locations[0]]
    return {
        "plasma_potential_v": vp,
        "radial_plasma_potential_v": local_vp,
        "sheath_edge_m": sheath_edge,
        "quasineutral_mask": mask,
        "bulk_present": norm > 0,
    }


def export_distributions(directory: Path, result: ParticleResult) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    energy = 0.5 * (result.energy_edges_ev[1:] + result.energy_edges_ev[:-1])
    radius = 0.5 * (result.radius_edges_m[1:] + result.radius_edges_m[:-1])
    np.savetxt(
        directory / "iedf.csv",
        np.column_stack((energy, result.iedf)),
        delimiter=",",
        header="energy_eV,pdf_per_eV",
        comments="",
    )
    exported_angles = {}
    for mode, suffix, column in (
        ("polar", "", "angle_deg"),
        ("signed_meridional", "_signed", "signed_meridional_angle_deg"),
    ):
        if mode == "signed_meridional" and result.arrival_meridional_angle_deg is None:
            continue
        angular = result.angular_distribution(mode)
        angle = 0.5 * (angular.angle_edges_deg[1:] + angular.angle_edges_deg[:-1])
        np.savetxt(
            directory / f"iadf{suffix}.csv",
            np.column_stack((angle, angular.iadf)),
            delimiter=",",
            header=f"{column},pdf_per_degree",
            comments="",
        )
        ee, aa = np.meshgrid(energy, angle, indexing="ij")
        np.savetxt(
            directory / f"iaedf{suffix}.csv",
            np.column_stack((ee.ravel(), aa.ravel(), angular.iaedf.ravel())),
            delimiter=",",
            header=f"energy_eV,{column},pdf_per_eV_per_degree",
            comments="",
        )
        exported_angles[f"iadf{suffix}.csv / iaedf{suffix}.csv"] = mode
    np.savetxt(
        directory / "radial_flux.csv",
        np.column_stack((radius * 1000, result.radial_flux_m2_s)),
        delimiter=",",
        header="radius_mm,ion_flux_m2_s",
        comments="",
    )
    arrivals = [
        result.arrival_energy_ev,
        result.arrival_angle_deg,
        result.arrival_radius_m * 1000,
        result.arrival_phase_s,
        result.arrival_collisions,
    ]
    arrival_header = "energy_eV,angle_deg,radius_mm,phase_s,collisions"
    if result.arrival_meridional_angle_deg is not None:
        arrivals.append(result.arrival_meridional_angle_deg)
        arrival_header += ",signed_meridional_angle_deg"
    np.savetxt(
        directory / "arrivals.csv",
        np.column_stack(arrivals),
        delimiter=",",
        header=arrival_header,
        comments="",
    )
    write_json(
        directory / "particle_diagnostics.json",
        {**result.diagnostics, "exported_angle_definitions": exported_angles},
    )
