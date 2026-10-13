from __future__ import annotations

import argparse
import json
import shutil
import sys
import traceback
from pathlib import Path

import numpy as np
import scipy

from . import __version__
from .config import Project
from .inputs import Waveform, file_hash
from .mesh import AxisymmetricMesh
from .particles import run_particles
from .solver import run_fields
from .storage import (
    export_distributions,
    field_fingerprint,
    load_checkpoint,
    load_field,
    plasma_observables,
    save_checkpoint,
    save_field,
    save_particles,
    write_json,
)
from .transport import build_transport


def emit(kind: str, **values) -> None:
    print(json.dumps(dict(event=kind, **values), ensure_ascii=False), flush=True)


def make_waveform(project: Project) -> Waveform:
    n = project.numerics
    if not n.waveform_csv:
        return Waveform.demo(n.frequency_hz, n.demo_negative_peak_v)
    return Waveform.from_csv(
        project.path(n.waveform_csv),
        n.frequency_hz,
        n.waveform_time_column,
        n.waveform_voltage_column,
        n.waveform_time_scale,
        n.waveform_voltage_scale,
    )


def archive_inputs(project: Project, directory: Path) -> None:
    inputs = directory / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    portable = Project.from_dict(project.to_dict())
    portable.base_dir = ""
    for name, value in (
        ("electron_xsec", project.electron_xsec),
        ("ion_xsec", project.ion_xsec),
        ("waveform", project.numerics.waveform_csv),
        ("density", project.plasma.density_csv),
    ):
        if value:
            source = project.path(value)
            target = inputs / f"{name}{source.suffix}"
            if source.resolve() != target.resolve():
                shutil.copy2(source, target)
            relative = target.relative_to(directory).as_posix()
            if name in {"electron_xsec", "ion_xsec"}:
                setattr(portable, name, relative)
            elif name == "waveform":
                portable.numerics.waveform_csv = relative
            else:
                portable.plasma.density_csv = relative
    write_json(directory / "project.json", portable.to_dict())


def execute(manifest: dict) -> dict:
    project = Project.from_dict(manifest["project"])
    project.validate()
    directory = Path(manifest["directory"]).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    action = manifest.get("action", "all")
    if action not in {"transport", "field", "particles", "all"}:
        raise ValueError("未知の計算アクションです。")
    if any((directory / name).exists() for name in ("field.npz", "particles.npz")):
        raise ValueError("保存先に計算結果があります。新しい結果フォルダーを指定してください。")
    cancel_path = directory / "cancel.request"
    cancelled = cancel_path.exists

    def report(value, message):
        emit("progress", value=float(value), message=message)

    archive_inputs(project, directory)
    stage_weights = {"transport": (0, 0.2), "field": (0.2, 0.65), "particles": (0.85, 0.15)}

    def stage_progress(stage):
        offset, width = stage_weights[stage] if action == "all" else (0, 1)
        return lambda value, message: report(offset + width * value, message)

    emit("stage", name="電子物性")
    table = build_transport(project, stage_progress("transport"), cancelled)
    table.save(directory / "transport.npz")
    if action == "transport":
        return {"transport": str(directory / "transport.npz")}
    waveform = make_waveform(project)
    wave_hash = waveform.source_hash or file_hash(project.path(project.electron_xsec)) + str(
        project.numerics.demo_negative_peak_v,
    )
    fingerprint = field_fingerprint(project.to_dict(), table.metadata["cache_key"], wave_hash)
    fingerprint += file_hash(project.path(project.ion_xsec))
    if project.plasma.density_csv:
        fingerprint += file_hash(project.path(project.plasma.density_csv))
    if action != "particles":
        emit("stage", name="2次元RF場")
        restart = None
        if manifest.get("resume"):
            restart = load_checkpoint(Path(manifest["resume"]), fingerprint)
        field = run_fields(
            project,
            table,
            waveform,
            stage_progress("field"),
            cancelled,
            lambda state: save_checkpoint(directory / "checkpoint.npz", state, fingerprint),
            restart,
        )
        field.diagnostics["fingerprint"] = fingerprint
        field.diagnostics["application_version"] = __version__
        field.diagnostics["runtime_versions"] = {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        }
        save_field(directory / "field.npz", field)
        write_json(directory / "field_diagnostics.json", field.diagnostics)
    else:
        field = load_field(Path(manifest["field"]))
        if field.diagnostics.get("fingerprint") != fingerprint:
            raise ValueError(
                "場生成時の装置・プラズマ・波形条件と一致しません。場を再計算してください。"
            )
        save_field(directory / "field.npz", field)
    obs = plasma_observables(field, AxisymmetricMesh(project.geometry))
    np.savetxt(
        directory / "plasma_potential.csv",
        np.column_stack(
            (
                field.phase_s,
                obs["plasma_potential_v"],
                obs["bulk_present"].astype(int),
            )
        ),
        delimiter=",",
        header="phase_s,plasma_potential_V,quasineutral_bulk_present",
        comments="",
    )
    output = {"field": str(directory / "field.npz"), "transport": str(directory / "transport.npz")}
    emit("result", **output)
    if action == "field":
        return output
    emit("stage", name="2D3Vテスト粒子")
    particles = run_particles(
        project,
        field,
        table,
        stage_progress("particles"),
        cancelled,
        allow_unconverged=manifest.get("allow_unconverged", False),
    )
    particles.diagnostics["field_fingerprint"] = fingerprint
    save_particles(directory / "particles.npz", particles)
    export_distributions(directory / "csv", particles)
    output["particles"] = str(directory / "particles.npz")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    try:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8-sig"))
        output = execute(manifest)
        emit("complete", **output)
    except InterruptedError as exc:
        emit("cancelled", message=str(exc))
        sys.exit(2)
    except Exception as exc:
        emit("error", message=str(exc), details=traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
