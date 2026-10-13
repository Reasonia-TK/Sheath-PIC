from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .constants import E_CHARGE, K_B, M_AR
from .inputs import file_hash


@dataclass
class CollisionTable:
    name: str
    energy_ev: np.ndarray
    sigma_m2: np.ndarray

    def at(self, energy_ev) -> np.ndarray:
        energy = np.asarray(energy_ev, dtype=float)
        if np.any(energy < 0) or not np.all(np.isfinite(energy)):
            raise ValueError("衝突エネルギーは0以上の有限な値が必要です。")
        if np.any(energy > self.energy_ev[-1] * (1 + 1e-12)):
            raise ValueError(f"{self.name} の断面積上限 {self.energy_ev[-1]:g} eVを超えました。")
        # Zero-energy endpoints make direct logarithmic interpolation undefined.
        return np.interp(energy, self.energy_ev, self.sigma_m2)


@dataclass
class IonCrossSections:
    isotropic: CollisionTable
    backscatter: CollisionTable
    source: str
    source_hash: str
    energy_frame: str

    @property
    def max_energy_ev(self) -> float:
        return min(self.isotropic.energy_ev[-1], self.backscatter.energy_ev[-1])

    def collision_energy(self, relative_speed) -> np.ndarray:
        mass = M_AR / 2 if self.energy_frame == "center_of_mass" else M_AR
        return 0.5 * mass * np.square(relative_speed) / E_CHARGE

    def momentum_frequency(self, speed, neutral_density: float, gas_temperature: float):
        relative = np.sqrt(np.square(speed) + 8 * K_B * gas_temperature / (np.pi * M_AR))
        energy = self.collision_energy(relative)
        # Equal-mass lab momentum loss: 1/2 for isotropic CM scattering, 1 for reversal.
        return (
            neutral_density
            * relative
            * (0.5 * self.isotropic.at(energy) + self.backscatter.at(energy))
        )

    @classmethod
    def load(cls, path: Path, energy_frame: str = "center_of_mass") -> IonCrossSections:
        if energy_frame not in {"center_of_mass", "lab"}:
            raise ValueError("断面積のエネルギー軸が不明です。")
        text = path.read_text(encoding="utf-8-sig")
        pattern = re.compile(
            r"^PROCESS:\s*(?P<name>[^\r\n]+).*?^COLUMNS:[^\r\n]*\r?\n"
            r"-{5,}[^\r\n]*\r?\n(?P<data>.*?)(?=^-{5,})",
            re.M | re.S,
        )
        tables = {}
        for match in pattern.finditer(text):
            name = match["name"]
            kind = "backscatter" if "backscat" in name.lower() else "isotropic"
            if "backscat" not in name.lower() and "isotropic" not in name.lower():
                continue
            rows = np.array(
                [
                    [float(x) for x in line.split()[:2]]
                    for line in match["data"].splitlines()
                    if re.match(r"\s*[0-9]", line)
                ]
            )
            if rows.ndim != 2 or rows.shape[1] != 2 or len(rows) < 2:
                raise ValueError(f"イオン断面積 {name} の表が不正です。")
            if not np.all(np.isfinite(rows)) or np.any(rows < 0):
                raise ValueError(f"イオン断面積 {name} に不正な値があります。")
            if np.any(np.diff(rows[:, 0]) <= 0) or kind in tables:
                raise ValueError(f"イオン断面積 {name} の重複またはエネルギー順序が不正です。")
            tables[kind] = CollisionTable(name, rows[:, 0], rows[:, 1])
        if set(tables) != {"isotropic", "backscatter"}:
            raise ValueError("イオン断面積にはIsotropicとBackscatの両方が必要です。")
        return cls(
            tables["isotropic"], tables["backscatter"], str(path), file_hash(path), energy_frame
        )


def scatter_equal_mass(
    velocity: np.ndarray,
    neutral_velocity: np.ndarray,
    backscatter: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Elastic Ar+ / Ar scattering in CM; backscatter exchanges the velocities."""
    relative = velocity - neutral_velocity
    speed = np.linalg.norm(relative, axis=1)
    mu = rng.uniform(-1, 1, len(velocity))
    azimuth = rng.uniform(0, 2 * np.pi, len(velocity))
    transverse = np.sqrt(1 - mu * mu)
    direction = np.column_stack((transverse * np.cos(azimuth), transverse * np.sin(azimuth), mu))
    outgoing = 0.5 * (velocity + neutral_velocity) + 0.5 * speed[:, None] * direction
    outgoing[backscatter] = neutral_velocity[backscatter]
    return outgoing
