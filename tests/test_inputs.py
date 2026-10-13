import numpy as np
import pytest

from sheath_pic.config import Geometry
from sheath_pic.inputs import Waveform, density_expression, density_from_csv
from sheath_pic.mesh import AxisymmetricMesh


def test_waveform_units_and_periodicity(tmp_path):
    path = tmp_path / "wave.csv"
    path.write_text("t_us,v_kV\n0,0\n0.625,-5\n1.25,-10\n1.875,-5\n2.5,0\n", encoding="utf-8-sig")
    wave = Waveform.from_csv(path, 400e3, "t_us", "v_kV", 1e-6, 1000)
    assert wave.at(1.25e-6) == pytest.approx(-10000)
    assert wave.at(3.75e-6) == pytest.approx(-10000)
    assert wave.seam_error_v == 0


def test_waveform_rejects_duplicate_time():
    with pytest.raises(ValueError, match="重複"):
        Waveform(np.array([0, 0, 1]), np.zeros(3), 1, "test")


def test_density_expression_is_not_arbitrary_python():
    r, z = np.meshgrid(np.linspace(0, 1, 3), np.linspace(0, 1, 4), indexing="ij")
    value = density_expression("ne0*(1+0.2*cos(pi*r/R))", r, z, 1e16, 1, 1)
    assert value[0, 0] == pytest.approx(1.2e16)
    for expression in ("__import__('os').system('bad')", "r.__class__", "10**100000", "[1,2]"):
        with pytest.raises(ValueError):
            density_expression(expression, r, z, 1e16, 1, 1)


def test_density_csv_interpolates_and_does_not_extrapolate(tmp_path):
    path = tmp_path / "density.csv"
    path.write_text("r_mm,z_mm,ne_m3\n0,0,1e16\n0,50,2e16\n100,0,3e16\n100,50,4e16\n")
    value = density_from_csv(path, np.array([[0.05]]), np.array([[0.025]]))
    assert value[0, 0] == pytest.approx(2.5e16)
    value = density_from_csv(path, np.array([[0.15]]), np.array([[0.025]]))
    assert np.isnan(value[0, 0])


def test_mesh_area_and_poisson_gauss_law():
    mesh = AxisymmetricMesh(Geometry(radial_cells=12, axial_cells=16))
    assert mesh.cathode_area == pytest.approx(np.pi * 0.15**2)
    assert mesh.ground_area / mesh.cathode_area == pytest.approx(5)
    ni = np.full(mesh.shape, 1e16)
    ne = ni * 0.999
    phi = mesh.solve_poisson(ni, ne, -100)
    rhs = mesh.poisson_rhs(ni, ne, -100)
    assert np.linalg.norm(mesh.poisson @ phi.ravel() - rhs) / np.linalg.norm(rhs) < 1e-12
    assert np.isfinite(phi).all()
