from __future__ import annotations

import hashlib
import json
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import Project
from .constants import E_CHARGE, M_E, TOWNSEND
from .inputs import file_hash

Progress = Callable[[float, str], None]


@dataclass
class TransportTable:
    en_td: np.ndarray
    mean_energy_ev: np.ndarray
    mobility_m2_vs: np.ndarray
    excitation_m3_s: np.ndarray
    ionization_m3_s: np.ndarray
    elastic_loss_ev_m3_s: np.ndarray
    eedf_energy_ev: np.ndarray
    eedf: np.ndarray
    metadata: dict

    def coefficients(self, energy_ev: np.ndarray) -> dict[str, np.ndarray]:
        order = np.argsort(self.mean_energy_ev)
        x, unique = np.unique(self.mean_energy_ev[order], return_index=True)
        order = order[unique]
        return {
            name: np.interp(energy_ev, x, getattr(self, name)[order])
            for name in (
                "mobility_m2_vs",
                "excitation_m3_s",
                "ionization_m3_s",
                "elastic_loss_ev_m3_s",
            )
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            **{
                name: getattr(self, name)
                for name in self.__dataclass_fields__
                if name != "metadata"
            },
            metadata=json.dumps(self.metadata, ensure_ascii=False),
        )

    @classmethod
    def load(cls, path: Path) -> TransportTable:
        with np.load(path, allow_pickle=False) as data:
            return cls(
                **{
                    name: data[name].copy()
                    for name in cls.__dataclass_fields__
                    if name != "metadata"
                },
                metadata=json.loads(str(data["metadata"])),
            )

    @property
    def energy_range(self) -> tuple[float, float]:
        return float(self.mean_energy_ev.min()), float(self.mean_energy_ev.max())


def cache_key(project: Project, version: str) -> str:
    payload = {
        "version": version,
        "xsec": file_hash(project.path(project.electron_xsec)),
        "pressure": project.plasma.pressure_pa,
        "temperature": project.plasma.gas_temperature_k,
        "settings": project.to_dict()["transport"],
        "model": "DC-gas-heating-elastic-collision-moment-v2",
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def elastic_collision_solver(bp, solver):
    """Build the same elastic operator on the same velocity mesh, without inelastic terms.

    Centre AND face samples preserve Boltzpmp's momentum-transfer and thermal-gas
    coefficients. EFFECTIVE subtraction follows boltzpmp-core/processes.rs.
    """
    from dataclasses import replace

    energy = np.unique(np.r_[solver.mesh.eps_b, solver.mesh.eps_c])
    gases = []
    for gas in solver.mixture.gases:
        elastic_sections = []
        for section in gas.cross_sections:
            if section.kind not in {"ELASTIC", "EFFECTIVE"}:
                continue
            subtracted = np.zeros_like(energy)
            if section.kind == "EFFECTIVE":
                for other in gas.cross_sections:
                    if other.target == section.target and (
                        (other.kind == "EXCITATION" and other.threshold >= 0)
                        or other.kind in {"IONIZATION", "ATTACHMENT"}
                    ):
                        subtracted += other.sigma(energy)
            sigma = np.maximum(section.sigma(energy) - subtracted, 0)
            momentum = section.momentum_transfer(energy)
            if momentum is not None:
                momentum = np.column_stack((energy, np.maximum(momentum - subtracted, 0)))
            elastic_sections.append(
                replace(
                    section,
                    kind="ELASTIC",
                    data=np.column_stack((energy, sigma)),
                    mt_data=momentum,
                )
            )
        if elastic_sections:
            gases.append(bp.Gas(gas.name, gas.fraction, elastic_sections, mass_amu=gas.mass_amu))
    if not gases:
        raise ValueError("電子の弾性衝突過程がありません。")
    mixture = bp.Mixture(gases, N=solver.mixture.N, T_K=solver.mixture.T_K)
    return bp.PMSolver(
        mixture,
        energy_grid=solver.mesh.eps_b,
        n_theta=solver.mesh.n_theta,
        superelastic=False,
        gas_heating=solver.gas_heating,
    )


def elastic_energy_loss(result, elastic_solver) -> float:
    """Net elastic eV m^3/s from the Boltzpmp collision moment, without power subtraction.

    Elastic energy exchange with an isotropic neutral gas depends only on f0. Using
    an isotropic state avoids cancellation of large angular-redistribution terms.
    Negative net loss (gas heating) is clipped; the fluid's thermal floor supplies it.
    """
    mesh = result.mesh
    numbers = result.n.reshape(mesh.shape).sum(axis=1)
    state = (numbers[:, None] * mesh.w_theta[None, :]).ravel()
    # The exposed native API returns dn/dt, as verified against the ionization moment.
    rate = np.asarray(elastic_solver._core_solver.collision_apply(state.tolist())).reshape(
        mesh.shape
    )
    loss = -np.sum(rate * mesh.eps_c[:, None]) / (state.sum() * elastic_solver.mixture.N)
    if not np.isfinite(loss):
        raise ArithmeticError("弾性衝突のエネルギー移送が有限値ではありません。")
    return max(float(loss), 0.0)


def build_transport(
    project: Project,
    progress: Progress | None = None,
    cancelled: Callable[[], bool] | None = None,
    use_cache: bool = True,
) -> TransportTable:
    import boltzpmp as bp

    progress = progress or (lambda *_: None)
    cancelled = cancelled or (lambda: False)
    project.validate()
    xs_path = project.path(project.electron_xsec)
    key = cache_key(project, bp.__version__)
    cache = Path.cwd() / ".cache" / f"transport-{key}.npz"
    if use_cache and cache.exists():
        table = TransportTable.load(cache)
        progress(1.0, "Boltzpmp DC物性テーブルをキャッシュから読み込みました。")
        return table
    sections = bp.parse_lxcat(xs_path)
    maximum_data_energy = min(section.data[-1, 0] for section in sections)
    gas = bp.Gas("Ar", 1.0, sections, mass_amu=39.948)
    mixture = bp.Mixture(
        [gas], p_Pa=project.plasma.pressure_pa, T_K=project.plasma.gas_temperature_k
    )
    settings = project.transport
    fields_td = np.geomspace(settings.en_min_td, settings.en_max_td, settings.en_points)
    results = []
    tails = []
    truncation = []
    elastic_operators = {}
    elastic_losses = []
    for index, en in enumerate(fields_td):
        if cancelled():
            raise InterruptedError("電子物性計算を中断しました。")
        energy_max = settings.energy_max_ev
        result = None
        for _ in range(5):
            if energy_max > maximum_data_energy:
                raise ValueError("電子エネルギー格子が断面積の上限を超えます。")
            solver = bp.PMSolver(
                mixture,
                energy_grid=bp.graded_energy_grid(
                    energy_max,
                    settings.energy_step_min_ev,
                    2.0,
                ),
                n_theta=settings.angle_cells,
                superelastic=False,
                gas_heating=True,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                result = solver.solve_dc(float(en), scheme="limiter", max_steps=10000)
            if not result.converged:
                raise RuntimeError(f"Boltzpmp DC {en:g} Tdが収束しませんでした。")
            tail = float(result.extra.get("eepf_tail_ratio", float("inf")))
            if tail <= settings.tail_tolerance:
                break
            energy_max *= 2
        if result is None or tail > settings.tail_tolerance:
            raise RuntimeError(f"{en:g} TdのEEDF高エネルギー端が収束しませんでした。")
        results.append(result)
        if energy_max not in elastic_operators:
            elastic_operators[energy_max] = elastic_collision_solver(bp, solver)
        elastic_losses.append(elastic_energy_loss(result, elastic_operators[energy_max]))
        tails.append(tail)
        truncation.append(energy_max)
        progress(
            (index + 1) / len(fields_td),
            f"DC {en:.3g} Td / 平均 {result.mean_energy:.3g} eV ({index + 1}/{len(fields_td)})",
        )
    excitation = np.zeros(len(results))
    ionization = np.zeros(len(results))
    elastic = np.asarray(elastic_losses)
    warnings_list = []
    for i, result in enumerate(results):
        for section in sections:
            # The parser names are retained verbatim; processes() determines result keys.
            matches = [
                value
                for name, value in result.rate_coefficients.items()
                if name.startswith("Ar:") and section.name in name
            ]
            rate = sum(matches)
            if section.kind == "EXCITATION":
                excitation[i] += rate
            elif section.kind == "IONIZATION":
                ionization[i] += rate
        # Fall back to process labels if a backend changes the LXCat process name formatting.
        if not excitation[i]:
            excitation[i] = sum(
                v
                for k, v in result.rate_coefficients.items()
                if "excitation" in k.lower() and "superelastic" not in k.lower()
            )
        ionization[i] = float(result.reduced_ionization_frequency)
    sample_energy = np.unique(
        np.concatenate(
            (
                np.linspace(0.0, 2.0, 100),
                np.geomspace(2.0, max(truncation), 500),
            )
        )
    )
    distributions = np.array(
        [
            np.interp(sample_energy, result.energy_grid, result.eedf, left=0, right=0)
            for result in results
        ]
    )
    norm = np.trapezoid(distributions, sample_energy, axis=1)
    distributions /= norm[:, None]
    table = TransportTable(
        fields_td,
        np.array([r.mean_energy for r in results]),
        np.array(
            [
                r.drift_velocity / (en * TOWNSEND * mixture.N)
                for en, r in zip(fields_td, results, strict=True)
            ]
        ),
        excitation,
        ionization,
        elastic,
        sample_energy,
        distributions,
        {
            "boltzpmp_version": bp.__version__,
            "calculation": "DC",
            "source": str(xs_path),
            "source_sha256": file_hash(xs_path),
            "cache_key": key,
            "pressure_pa": project.plasma.pressure_pa,
            "gas_temperature_k": project.plasma.gas_temperature_k,
            "superelastic": False,
            "gas_heating": True,
            "tail_ratios": tails,
            "energy_cutoffs_ev": truncation,
            "electron_diffusion_closure": "Einstein with energy-equivalent Te (Maxwell closure)",
            "energy_transport_closure": "5/3 particle transport coefficients (Maxwell closure)",
            "elastic_loss_closure": (
                "Boltzpmp elastic-only collision energy moment; isotropic f0; gas heating enabled"
            ),
            "elastic_gas_heating_closure": "negative net loss clipped; Maxwell fluid thermal floor",
            "warnings": warnings_list,
        },
    )
    table.save(cache)
    progress(1.0, "Boltzpmp DC物性テーブルを保存しました。")
    return table


def effective_momentum_frequency(mobility_m2_vs):
    return E_CHARGE / (M_E * np.asarray(mobility_m2_vs))
