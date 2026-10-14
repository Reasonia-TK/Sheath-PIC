import importlib.util
from pathlib import Path

import numpy as np
import pytest
from scipy.sparse import diags

from sheath_pic.config import Project
from sheath_pic.inputs import Waveform
from sheath_pic.linear import LinearBackend
from sheath_pic.mesh import AxisymmetricMesh
from sheath_pic.solver import run_fields, solve_electron_poisson
from sheath_pic.storage import field_fingerprint, load_checkpoint, save_checkpoint
from sheath_pic.transport import build_transport

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def cuda_available():
    if importlib.util.find_spec("cupy") is None:
        pytest.skip("CUDA追加依存が未導入")
    with LinearBackend("auto") as backend:
        if backend.name != "cuda":
            pytest.skip(f"CUDA GPUが利用できません: {backend.unavailable_reason}")


def test_cuda_unavailable_is_visible_and_explicit_cuda_never_silently_falls_back(monkeypatch):
    import sys
    from types import SimpleNamespace

    monkeypatch.setitem(sys.modules, "cupy", None)
    with LinearBackend("auto") as backend:
        assert backend.name == "cpu"
        assert backend.unavailable_reason
    with pytest.raises(RuntimeError, match="初期化"):
        LinearBackend("cuda")
    messages = []

    def calculation(*args):
        # A fallback must be explained before the potentially long discharge solve.
        backend = args[-1]
        assert backend.name == "cpu"
        assert any(backend.unavailable_reason in message for message in messages)
        return SimpleNamespace(diagnostics={})

    monkeypatch.setattr("sheath_pic.solver._run_fields", calculation)
    project = Project()
    project.numerics.backend = "auto"
    result = run_fields(
        project, None, None, progress=lambda value, message: messages.append(message)
    )
    assert result.diagnostics["backend"]["actual"] == "cpu"


def test_backend_choice_does_not_change_physical_field_identity():
    project = Project()
    project.numerics.backend = "cpu"
    fingerprint = field_fingerprint(project.to_dict(), "table", "wave")
    project.numerics.backend = "cuda"
    assert field_fingerprint(project.to_dict(), "table", "wave") == fingerprint
    project.numerics.electron_density_floor_fraction *= 10
    assert field_fingerprint(project.to_dict(), "table", "wave") != fingerprint


def test_cuda_fp64_and_symbolic_reuse(cuda_available):
    a = diags([-np.ones(19), np.full(20, 4.0), -np.ones(19)], [-1, 0, 1], format="csr")
    b = np.arange(20, dtype=np.float64)
    with LinearBackend("cuda") as backend:
        backend.register("test", a)
        first = backend.solve("test", a, b)
        assert first.dtype == np.float64
        assert a @ first == pytest.approx(b, rel=1e-12, abs=1e-12)
        a.data *= 1.25
        second = backend.solve("test", a, b)
        assert a @ second == pytest.approx(b, rel=1e-12, abs=1e-12)
        assert backend.metadata()["symbolic_analyses"] == 1
        assert backend.metadata()["linear_solve_count"] == 2


def test_cuda_field_budgets_and_cpu_checkpoint_restart(cuda_available, tmp_path):
    project = Project(base_dir=str(ROOT))
    project.geometry.radial_cells, project.geometry.axial_cells = 4, 6
    project.numerics.minimum_cycles = project.numerics.maximum_cycles = 1
    project.numerics.steps_per_period, project.numerics.stored_phases = 32, 8
    project.numerics.newton_tolerance = 1e-9
    project.numerics.newton_iterations = 80
    project.transport.en_points, project.transport.angle_cells = 8, 16
    table = build_transport(project)
    waveform = Waveform.demo(400e3, 10000)
    states = []
    project.numerics.backend = "cpu"
    cpu = run_fields(project, table, waveform, checkpoint=lambda state: states.append(state))
    project.numerics.backend = "cuda"
    cuda = run_fields(project, table, waveform)
    assert cpu.electron_energy_density_j_m3 is not None
    assert cuda.electron_energy_density_j_m3 is not None
    diagnostics = cuda.diagnostics
    assert diagnostics["backend"]["actual"] == "cuda"
    assert diagnostics["backend"]["maximum_linear_relative_residual"] <= 1e-11
    assert diagnostics["ion_balance_relative_error"] < 1e-10
    assert diagnostics["electron_balance_relative_error"] < 1e-6
    assert diagnostics["electron_energy_balance_relative_error"] < 1e-8
    checkpoint = tmp_path / "cpu-checkpoint.npz"
    save_checkpoint(checkpoint, states[-1], "same-physics")
    project.numerics.maximum_cycles = 2
    continued = run_fields(
        project, table, waveform, restart=load_checkpoint(checkpoint, "same-physics")
    )
    reference = run_fields(
        project, table, waveform, restart=load_checkpoint(checkpoint, "same-physics")
    )
    assert continued.diagnostics["completed_cycles"] == 2
    assert np.array_equal(continued.potential_v, reference.potential_v)


@pytest.mark.parametrize("voltage", [-20.0, -10000.0])
def test_cpu_cuda_coupled_step_agrees_at_same_time_and_energy(cuda_available, voltage):
    project = Project()
    project.geometry.radial_cells, project.geometry.axial_cells = 6, 8
    results = []
    for device in ("cpu", "cuda"):
        with LinearBackend(device) as backend:
            mesh = AxisymmetricMesh(project.geometry, backend)
            n = np.full(mesh.shape, 1e16)
            potential = mesh.solve_poisson(n, n, voltage)
            phi, ne, _, _, residual = solve_electron_poisson(
                mesh,
                n,
                n,
                potential,
                np.full(mesh.shape, 4.5),
                np.full(mesh.shape, 2e4),
                np.zeros(mesh.shape),
                1e-10,
                voltage,
                1e16,
                1e-10,
                100,
                1,
                True,
            )
            assert residual < 1e-10
            results.append((phi.copy(), ne.copy()))
    for a, b in zip(results[0], results[1], strict=True):
        assert np.linalg.norm(a - b) / np.linalg.norm(a) < 1e-8
