from pathlib import Path

import numpy as np

from sheath_pic.config import Project
from sheath_pic.inputs import Waveform


def main():
    directory = Path("examples")
    directory.mkdir(exist_ok=True)
    wave = Waveform.demo(400e3, 10000)
    np.savetxt(
        directory / "wafer_demo.csv",
        np.column_stack((wave.time_s * 1e6, wave.voltage_v / 1000)),
        delimiter=",",
        header="time_us,voltage_kV",
        comments="",
    )
    r, z = np.meshgrid(np.linspace(0, 290, 13), np.linspace(0, 50, 11), indexing="ij")
    density = 3e16 * (1 + 0.2 * np.cos(np.pi * r / 290))
    np.savetxt(
        directory / "density_demo.csv",
        np.column_stack((r.ravel(), z.ravel(), density.ravel())),
        delimiter=",",
        header="r_mm,z_mm,ne_m3",
        comments="",
    )
    project = Project(base_dir="..")
    project.numerics.waveform_csv = "examples/wafer_demo.csv"
    project.numerics.waveform_time_column = "time_us"
    project.numerics.waveform_voltage_column = "voltage_kV"
    project.numerics.waveform_time_scale = 1e-6
    project.numerics.waveform_voltage_scale = 1000
    project.geometry.radial_cells = 8
    project.geometry.axial_cells = 12
    project.numerics.maximum_cycles = 4
    project.numerics.minimum_cycles = 2
    project.numerics.steps_per_period = 64
    project.numerics.stored_phases = 16
    project.numerics.particles = 2000
    project.transport.en_points = 12
    project.transport.angle_cells = 16
    # The example is relocatable relative to its own project file.
    import json

    (directory / "quick-project.json").write_text(
        json.dumps(project.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
