from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import bmat, coo_matrix, csr_matrix, eye
from scipy.sparse.linalg import spsolve

from .config import Geometry, Plasma
from .constants import E_CHARGE, EPSILON_0
from .inputs import density_expression, density_from_csv


@dataclass
class AxisymmetricMesh:
    geometry: Geometry
    linear_backend: object | None = None

    def __post_init__(self) -> None:
        self.geometry.validate()
        g = self.geometry
        self.r_edges = np.linspace(0, g.chamber_radius_m, g.radial_cells + 1)
        # Align the cathode/insulator junction with a face; avoid area errors on coarse grids.
        if g.cathode_radius_m < g.chamber_radius_m:
            n_inner = max(
                1,
                min(
                    g.radial_cells - 1,
                    round(
                        g.radial_cells * g.cathode_radius_m / g.chamber_radius_m,
                    ),
                ),
            )
            self.r_edges = np.concatenate(
                (
                    np.linspace(0, g.cathode_radius_m, n_inner + 1),
                    np.linspace(
                        g.cathode_radius_m, g.chamber_radius_m, g.radial_cells - n_inner + 1
                    )[1:],
                )
            )
        self.z_edges = np.linspace(0, g.gap_m, g.axial_cells + 1)
        self.r = 0.5 * (self.r_edges[:-1] + self.r_edges[1:])
        self.z = 0.5 * (self.z_edges[:-1] + self.z_edges[1:])
        self.dr = np.diff(self.r_edges)
        self.dz = np.diff(self.z_edges)
        self.R, self.Z = np.meshgrid(self.r, self.z, indexing="ij")
        self.shape = self.R.shape
        self.size = self.R.size
        self.ring_area = np.pi * np.diff(self.r_edges**2)
        self.volume = self.ring_area[:, None] * self.dz[None, :]
        self.radial_area = 2 * np.pi * self.r_edges[:, None] * self.dz[None, :]
        self.cathode_rows = self.r < g.cathode_radius_m
        ids = np.arange(self.size).reshape(self.shape)
        ri, rj = ids[:-1, :].ravel(), ids[1:, :].ravel()
        zi, zj = ids[:, :-1].ravel(), ids[:, 1:].ravel()
        self.face_i = np.concatenate((ri, zi))
        self.face_j = np.concatenate((rj, zj))
        self.face_area = np.concatenate(
            (
                self.radial_area[1:-1].ravel(),
                np.broadcast_to(
                    self.ring_area[:, None], (g.radial_cells, g.axial_cells - 1)
                ).ravel(),
            )
        )
        self.face_distance = np.concatenate(
            (
                np.broadcast_to(
                    np.diff(self.r)[:, None], (g.radial_cells - 1, g.axial_cells)
                ).ravel(),
                np.broadcast_to(
                    np.diff(self.z)[None, :], (g.radial_cells, g.axial_cells - 1)
                ).ravel(),
            )
        )
        self.face_axis = np.concatenate((np.zeros(len(ri), dtype=int), np.ones(len(zi), dtype=int)))
        bottom_ids = ids[self.cathode_rows, 0]
        top_ids = ids[:, -1]
        side_ids = ids[-1, :]
        self.boundary_cell = np.concatenate((bottom_ids, top_ids, side_ids))
        self.boundary_area = np.concatenate(
            (
                self.ring_area[self.cathode_rows],
                self.ring_area,
                self.radial_area[-1],
            )
        )
        self.boundary_distance = np.concatenate(
            (
                np.full(len(bottom_ids), self.dz[0] / 2),
                np.full(len(top_ids), self.dz[-1] / 2),
                np.full(len(side_ids), self.dr[-1] / 2),
            )
        )
        self.boundary_cathode = np.concatenate(
            (
                np.ones(len(bottom_ids), dtype=bool),
                np.zeros(len(top_ids) + len(side_ids), dtype=bool),
            )
        )
        self.boundary_axis = np.concatenate(
            (
                np.ones(len(bottom_ids) + len(top_ids), dtype=int),
                np.zeros(len(side_ids), dtype=int),
            )
        )
        self.boundary_sign = np.concatenate(
            (
                -np.ones(len(bottom_ids)),
                np.ones(len(top_ids) + len(side_ids)),
            )
        )
        self._assemble_poisson()
        self._prepare_sparse_layout()
        if self.linear_backend is not None and self.linear_backend.name == "cuda":
            pattern = self.poisson.copy()
            pattern.data[:] = 1
            self.linear_backend.register("poisson", pattern)
            self.linear_backend.register("energy", pattern)
            self.linear_backend.register("electron", self.jacobian_pattern)

    @property
    def cathode_area(self) -> float:
        return float(self.boundary_area[self.boundary_cathode].sum())

    @property
    def ground_area(self) -> float:
        return float(self.boundary_area[~self.boundary_cathode].sum())

    def _assemble_poisson(self) -> None:
        i, j = self.face_i, self.face_j
        conductance = self.face_area / self.face_distance
        self.face_conductance = conductance
        b, bg = self.boundary_cell, self.boundary_area / self.boundary_distance
        self.boundary_conductance = bg
        rows = np.concatenate((i, j, i, j, b))
        cols = np.concatenate((i, j, j, i, b))
        data = np.concatenate((conductance, conductance, -conductance, -conductance, bg))
        self.poisson = coo_matrix((data, (rows, cols)), shape=(self.size, self.size)).tocsr()

    def _prepare_sparse_layout(self):
        self.poisson.sort_indices()
        self.stencil_rows = np.repeat(np.arange(self.size), np.diff(self.poisson.indptr))
        keys = self.stencil_rows.astype(np.int64) * self.size + self.poisson.indices
        self.diagonal_positions = np.searchsorted(keys, np.arange(self.size) * (self.size + 1))
        self.ij_positions = np.searchsorted(keys, self.face_i * self.size + self.face_j)
        self.ji_positions = np.searchsorted(keys, self.face_j * self.size + self.face_i)
        pattern = self.poisson.copy()
        pattern.data[:] = 1
        self.jacobian_pattern = bmat([[pattern, eye(self.size)], [pattern, pattern]], format="csr")
        self.jacobian_pattern.sort_indices()
        width = 2 * self.size
        rows = np.repeat(np.arange(width, dtype=np.int64), np.diff(self.jacobian_pattern.indptr))
        jacobian_keys = rows * width + self.jacobian_pattern.indices
        r, c = self.stencil_rows, self.poisson.indices
        self.pp_positions = np.searchsorted(jacobian_keys, r * width + c)
        self.pn_positions = np.searchsorted(
            jacobian_keys, np.arange(self.size) * (width + 1) + self.size
        )
        self.np_positions = np.searchsorted(jacobian_keys, (r + self.size) * width + c)
        self.nn_positions = np.searchsorted(jacobian_keys, (r + self.size) * width + c + self.size)

    def stencil_matrix(self, values):
        return csr_matrix(
            (values, self.poisson.indices, self.poisson.indptr),
            shape=self.poisson.shape,
            copy=False,
        )

    def jacobian_matrix(self, row_scale, charge_scale, density_phi_data, density_matrix):
        values = np.empty(self.jacobian_pattern.nnz)
        values[self.pp_positions] = row_scale[self.stencil_rows] * self.poisson.data
        values[self.pn_positions] = row_scale * charge_scale
        values[self.np_positions] = density_phi_data
        values[self.nn_positions] = density_matrix.data
        return csr_matrix(
            (values, self.jacobian_pattern.indices, self.jacobian_pattern.indptr),
            shape=self.jacobian_pattern.shape,
            copy=False,
        )

    def boundary_voltage(self, wafer_voltage: float) -> np.ndarray:
        return np.where(self.boundary_cathode, wafer_voltage, 0.0)

    def poisson_rhs(
        self, ion_density: np.ndarray, electron_density: np.ndarray, wafer_voltage: float
    ) -> np.ndarray:
        rhs = (
            E_CHARGE
            / EPSILON_0
            * self.volume.ravel()
            * (ion_density.ravel() - electron_density.ravel())
        )
        np.add.at(
            rhs,
            self.boundary_cell,
            self.boundary_conductance * self.boundary_voltage(wafer_voltage),
        )
        return rhs

    def solve_poisson(
        self, ion_density: np.ndarray, electron_density: np.ndarray, wafer_voltage: float
    ) -> np.ndarray:
        return self.solve_linear(
            "poisson",
            self.poisson,
            self.poisson_rhs(
                ion_density,
                electron_density,
                wafer_voltage,
            ),
        ).reshape(self.shape)

    def solve_linear(self, key, matrix, rhs):
        if self.linear_backend is None:
            return spsolve(matrix, rhs)
        return self.linear_backend.solve(key, matrix, rhs)

    def electric_field(self, potential: np.ndarray, wafer_voltage: float):
        er = -np.gradient(potential, self.r, axis=0, edge_order=2)
        ez = -np.gradient(potential, self.z, axis=1, edge_order=2)
        er[0] = -(potential[1] - potential[0]) / (self.r[1] - self.r[0])
        er[-1] = potential[-1] / (self.geometry.chamber_radius_m - self.r[-1])
        ez[:, -1] = potential[:, -1] / (self.geometry.gap_m - self.z[-1])
        ez[self.cathode_rows, 0] = -(potential[self.cathode_rows, 0] - wafer_voltage) / self.z[0]
        ez[~self.cathode_rows, 0] = 0.0
        return er, ez

    def initial_density(self, plasma: Plasma, csv_path=None) -> np.ndarray:
        if plasma.density_profile == "uniform":
            density = np.full(self.shape, plasma.density_m3)
        elif plasma.density_profile == "expression":
            density = density_expression(
                plasma.density_expression,
                self.R,
                self.Z,
                plasma.density_m3,
                self.geometry.chamber_radius_m,
                self.geometry.gap_m,
            )
        else:
            density = density_from_csv(csv_path, self.R, self.Z)
        if not np.all(np.isfinite(density)) or np.any(density < 0) or density.max() <= 0:
            raise ValueError("密度分布は計算領域全体で有限・非負で、正の密度を含む必要があります。")
        return density

    def bilinear(self, values: np.ndarray, radius, height) -> np.ndarray:
        radius = np.clip(np.asarray(radius), self.r[0], self.r[-1])
        height = np.clip(np.asarray(height), self.z[0], self.z[-1])
        ir = np.clip(np.searchsorted(self.r, radius) - 1, 0, len(self.r) - 2)
        iz = np.clip(np.searchsorted(self.z, height) - 1, 0, len(self.z) - 2)
        fr = (radius - self.r[ir]) / (self.r[ir + 1] - self.r[ir])
        fz = (height - self.z[iz]) / (self.z[iz + 1] - self.z[iz])
        return (
            (1 - fr) * (1 - fz) * values[ir, iz]
            + fr * (1 - fz) * values[ir + 1, iz]
            + (1 - fr) * fz * values[ir, iz + 1]
            + fr * fz * values[ir + 1, iz + 1]
        )
