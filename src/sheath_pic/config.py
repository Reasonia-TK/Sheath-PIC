from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path


@dataclass
class Geometry:
    cathode_radius_m: float = 0.150
    chamber_radius_m: float = 0.2891164991562635
    gap_m: float = 0.050
    radial_cells: int = 24
    axial_cells: int = 40

    def validate(self) -> None:
        if not 0 < self.cathode_radius_m <= self.chamber_radius_m:
            raise ValueError("駆動電極半径は0より大きく、チャンバー半径以下にしてください。")
        if self.gap_m <= 0 or not 4 <= self.radial_cells <= 160:
            raise ValueError("正のギャップと4～160の半径方向セル数が必要です。")
        if not 4 <= self.axial_cells <= 240:
            raise ValueError("軸方向セル数は4～240にしてください。")


@dataclass
class Plasma:
    pressure_pa: float = 1.3332236842105263
    gas_temperature_k: float = 300.0
    density_m3: float = 3e16
    density_profile: str = "uniform"
    density_expression: str = "ne0 * (1 + 0.2*cos(pi*r/R))"
    density_csv: str = ""
    density_mode: str = "self_consistent"
    density_relaxation_periods: float = 10.0
    initial_mean_energy_ev: float = 4.5
    source_mean_energy_ev: float = 4.5
    cathode_secondary_yield: float = 0.0
    ground_secondary_yield: float = 0.0
    secondary_energy_ev: float = 2.0
    electron_sticking: float = 1.0
    ion_energy_frame: str = "center_of_mass"
    electron_flux_limit: bool = True

    def validate(self) -> None:
        if self.pressure_pa <= 0 or self.gas_temperature_k <= 0 or self.density_m3 <= 0:
            raise ValueError("圧力、ガス温度、基準密度は正の値が必要です。")
        if self.density_profile not in {"uniform", "expression", "csv"}:
            raise ValueError("未知の密度分布形式です。")
        if self.density_mode not in {"self_consistent", "maintained"}:
            raise ValueError("未知の密度モードです。")
        if self.initial_mean_energy_ev <= 0 or self.source_mean_energy_ev <= 0:
            raise ValueError("電子の平均エネルギーは正の値が必要です。")
        if self.density_relaxation_periods <= 0:
            raise ValueError("密度維持の緩和時間は正の値が必要です。")
        for value in (self.cathode_secondary_yield, self.ground_secondary_yield):
            if not 0 <= value <= 1:
                raise ValueError("二次電子放出係数は0～1にしてください。")
        if not 0 < self.electron_sticking <= 1:
            raise ValueError("電子吸収率は0より大きく1以下にしてください。")
        if self.ion_energy_frame not in {"center_of_mass", "lab"}:
            raise ValueError("イオン断面積のエネルギー軸を指定してください。")


@dataclass
class Numerics:
    backend: str = "auto"
    electron_density_floor_fraction: float = 1e-8
    frequency_hz: float = 400e3
    demo_negative_peak_v: float = 10000.0
    waveform_csv: str = ""
    waveform_time_column: str = ""
    waveform_voltage_column: str = ""
    waveform_time_scale: float = 1.0
    waveform_voltage_scale: float = 1.0
    minimum_cycles: int = 2
    maximum_cycles: int = 8
    steps_per_period: int = 128
    stored_phases: int = 64
    cfl: float = 0.4
    periodic_tolerance: float = 0.02
    newton_tolerance: float = 2e-6
    newton_iterations: int = 30
    seed: int = 20261013
    particles: int = 20000
    particle_max_periods: int = 20
    radial_bins: int = 12
    energy_bins: int = 160
    angle_bins: int = 90

    def validate(self) -> None:
        if self.backend not in {"auto", "cpu", "cuda"}:
            raise ValueError("計算バックエンドはauto、cpu、cudaから選択してください。")
        if not 0 < self.electron_density_floor_fraction <= 1e-4:
            raise ValueError("電子エネルギーの密度床は0より大きく1e-4以下にしてください。")
        if self.frequency_hz <= 0 or self.demo_negative_peak_v <= 0:
            raise ValueError("周波数と負側ピークの大きさは正の値が必要です。")
        if not 1 <= self.minimum_cycles <= self.maximum_cycles <= 200:
            raise ValueError("周期数は1～200、最小周期数以下にならない最大周期数が必要です。")
        if not 16 <= self.steps_per_period <= 4096:
            raise ValueError("1周期の最小分割数は16～4096にしてください。")
        if not 8 <= self.stored_phases <= self.steps_per_period:
            raise ValueError("保存位相数は8以上、周期分割数以下にしてください。")
        if not 0 < self.cfl <= 0.5 or self.periodic_tolerance <= 0:
            raise ValueError("CFLは0～0.5、収束許容値は正の値が必要です。")
        if self.newton_tolerance <= 0 or not 1 <= self.newton_iterations <= 200:
            raise ValueError("Newton許容値は正、反復回数は1～200にしてください。")
        if self.seed < 0:
            raise ValueError("乱数seedは0以上の整数にしてください。")
        if not 100 <= self.particles <= 2000000:
            raise ValueError("粒子数は100～2,000,000にしてください。")
        if min(self.radial_bins, self.energy_bins, self.angle_bins) < 2:
            raise ValueError("分布のビン数は2以上にしてください。")
        if self.particle_max_periods < 1:
            raise ValueError("粒子の最大追跡周期数は1以上にしてください。")


@dataclass
class TransportSettings:
    en_min_td: float = 0.001
    en_max_td: float = 3000.0
    en_points: int = 36
    energy_max_ev: float = 400.0
    energy_step_min_ev: float = 0.01
    angle_cells: int = 24
    tail_tolerance: float = 1e-5

    def validate(self) -> None:
        if not 0 < self.en_min_td < self.en_max_td or not 4 <= self.en_points <= 256:
            raise ValueError("E/N範囲と計算点数（4～256）を確認してください。")
        if self.energy_max_ev <= 20 or self.energy_step_min_ev <= 0:
            raise ValueError("電子エネルギー上限は20 eVより大きい値が必要です。")
        if not 8 <= self.angle_cells <= 180:
            raise ValueError("電子の角度分割数は8～180にしてください。")
        if not 0 < self.tail_tolerance < 1:
            raise ValueError("EEDF末尾の許容値は0より大きく1より小さくしてください。")


@dataclass
class Project:
    name: str = "Ar CCP / 400 kHz"
    geometry: Geometry = field(default_factory=Geometry)
    plasma: Plasma = field(default_factory=Plasma)
    numerics: Numerics = field(default_factory=Numerics)
    transport: TransportSettings = field(default_factory=TransportSettings)
    electron_xsec: str = "xsec/Argon_電子衝突断面積.txt"
    ion_xsec: str = "xsec/Argon_イオン衝突断面積.txt"
    base_dir: str = ""

    def validate(self) -> None:
        for group in (self.geometry, self.plasma, self.numerics, self.transport):
            for descriptor in fields(group):
                value = getattr(group, descriptor.name)
                if isinstance(value, (int, float)) and not math.isfinite(value):
                    raise ValueError(f"{descriptor.name}は有限の数値が必要です。")
        for part in (self.geometry, self.plasma, self.numerics, self.transport):
            part.validate()
        stored_cells = (
            self.geometry.radial_cells * self.geometry.axial_cells * self.numerics.stored_phases
        )
        if stored_cells > 4000000:
            raise ValueError(
                "保存する格子×位相数が大きすぎます。格子か保存位相数を減らしてください。"
            )

    def path(self, value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else Path(self.base_dir or Path.cwd()) / path

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict, base_dir: str | None = None) -> Project:
        allowed = {f.name for f in fields(cls)}
        unknown = set(data) - allowed
        if unknown:
            raise ValueError(f"未知のプロジェクト項目: {', '.join(sorted(unknown))}")
        values = dict(data)
        for name, kind in (
            ("geometry", Geometry),
            ("plasma", Plasma),
            ("numerics", Numerics),
            ("transport", TransportSettings),
        ):
            if name in values:
                values[name] = kind(**values[name])
        if base_dir is not None:
            values["base_dir"] = base_dir
        project = cls(**values)
        project.validate()
        return project

    def save(self, path: Path) -> None:
        data = self.to_dict()
        data["base_dir"] = str(Path(self.base_dir or Path.cwd()).resolve())
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> Project:
        path = path.resolve()
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        base = Path(data.get("base_dir") or path.parent)
        if not base.is_absolute():
            base = path.parent / base
        return cls.from_dict(data, base_dir=str(base.resolve()))
