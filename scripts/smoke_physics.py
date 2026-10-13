"""Small end-to-end calculation; output is a numerical smoke check, not validation data."""

from pathlib import Path

from sheath_pic.config import Project
from sheath_pic.inputs import Waveform
from sheath_pic.mesh import AxisymmetricMesh
from sheath_pic.particles import run_particles
from sheath_pic.solver import run_fields
from sheath_pic.storage import plasma_observables, save_field, save_particles, write_json
from sheath_pic.transport import build_transport


def main():
    project = Project(base_dir=str(Path.cwd()))
    project.geometry.radial_cells = 6
    project.geometry.axial_cells = 10
    project.numerics.maximum_cycles = 2
    project.numerics.minimum_cycles = 2
    project.numerics.steps_per_period = 64
    project.numerics.stored_phases = 16
    project.numerics.newton_iterations = 60
    project.numerics.particles = 1000
    project.numerics.particle_max_periods = 10
    project.transport.en_points = 10
    project.transport.angle_cells = 16
    table = build_transport(project, lambda p, text: print(text, flush=True))
    print("table energy range", table.energy_range, flush=True)
    wave = Waveform.demo(project.numerics.frequency_hz, 10000)
    field = run_fields(project, table, wave, lambda p, text: print(text, flush=True))
    target = Path("artifacts/smoke")
    save_field(target / "field.npz", field)
    write_json(target / "diagnostics.json", field.diagnostics)
    obs = plasma_observables(field, AxisymmetricMesh(project.geometry))
    print("potential min/max", field.potential_v.min(), field.potential_v.max(), flush=True)
    print("plasma potential", obs["plasma_potential_v"], flush=True)
    print(
        "budgets",
        field.diagnostics["ion_balance_relative_error"],
        field.diagnostics["electron_balance_relative_error"],
        flush=True,
    )
    particles = run_particles(
        project, field, table, lambda p, text: print(text, flush=True), allow_unconverged=True
    )
    save_particles(target / "particles.npz", particles)
    print(particles.diagnostics, flush=True)


if __name__ == "__main__":
    main()
