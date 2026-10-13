from __future__ import annotations

import ast
import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.interpolate import LinearNDInterpolator, RegularGridInterpolator


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_numeric_csv(path: Path) -> tuple[list[str], np.ndarray]:
    text = path.read_text(encoding="utf-8-sig")
    lines = [
        line for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#")
    ]
    if not lines:
        raise ValueError("CSVにデータがありません。")
    try:
        dialect = csv.Sniffer().sniff("\n".join(lines[:10]), delimiters=",;\t")
    except csv.Error:
        dialect = csv.excel
    rows = list(csv.reader(lines, dialect))
    try:
        [float(value) for value in rows[0]]
        headers = [f"column_{i + 1}" for i in range(len(rows[0]))]
    except ValueError:
        headers = [value.strip() for value in rows.pop(0)]
    if len(set(headers)) != len(headers):
        raise ValueError("CSV列名が重複しています。")
    values = []
    for i, row in enumerate(rows, start=2):
        if len(row) != len(headers):
            raise ValueError(f"CSVの{i}行目の列数が一致しません。")
        try:
            values.append([float(item.strip()) for item in row])
        except ValueError as exc:
            raise ValueError(f"CSVの{i}行目に数値以外があります。") from exc
    result = np.asarray(values, dtype=float)
    if result.ndim != 2 or len(result) < 2 or not np.all(np.isfinite(result)):
        raise ValueError("CSVには2行以上の有限な数値データが必要です。")
    return headers, result


@dataclass
class Waveform:
    time_s: np.ndarray
    voltage_v: np.ndarray
    period_s: float
    source: str
    source_hash: str = ""

    def __post_init__(self) -> None:
        self.time_s = np.asarray(self.time_s, dtype=float)
        self.voltage_v = np.asarray(self.voltage_v, dtype=float)
        if self.time_s.shape != self.voltage_v.shape or len(self.time_s) < 3:
            raise ValueError("波形には同じ長さの時刻・電圧配列が必要です。")
        if not np.all(np.isfinite(self.time_s)) or not np.all(np.isfinite(self.voltage_v)):
            raise ValueError("波形にNaNまたは無限大が含まれています。")
        if np.any(np.diff(self.time_s) <= 0):
            raise ValueError("時刻は重複なしの昇順にしてください。")
        if self.period_s <= 0 or self.time_s[-1] - self.time_s[0] < self.period_s * 0.999:
            raise ValueError("設定周波数の1周期以上の波形が必要です。")

    def at(self, time_s):
        phase = np.remainder(np.asarray(time_s), self.period_s)
        return np.interp(phase + self.time_s[0], self.time_s, self.voltage_v)

    @property
    def seam_error_v(self) -> float:
        end = np.interp(self.time_s[0] + self.period_s, self.time_s, self.voltage_v)
        return float(abs(end - self.voltage_v[0]))

    @classmethod
    def demo(cls, frequency_hz: float, negative_peak_v: float, points: int = 513) -> Waveform:
        period = 1 / frequency_hz
        time = np.linspace(0, period, points)
        voltage = -0.5 * negative_peak_v * (1 - np.cos(2 * np.pi * time / period))
        return cls(time, voltage, period, "合成デモ波形（実験値ではありません）")

    @classmethod
    def from_csv(
        cls,
        path: Path,
        frequency_hz: float,
        time_column: str = "",
        voltage_column: str = "",
        time_scale: float = 1.0,
        voltage_scale: float = 1.0,
    ) -> Waveform:
        headers, values = read_numeric_csv(path)
        ti = headers.index(time_column) if time_column else 0
        vi = headers.index(voltage_column) if voltage_column else 1
        if ti == vi:
            raise ValueError("時刻列と電圧列は別の列を選択してください。")
        return cls(
            values[:, ti] * time_scale,
            values[:, vi] * voltage_scale,
            1 / frequency_hz,
            str(path),
            file_hash(path),
        )


_FUNCTIONS = {
    "sin": np.sin,
    "cos": np.cos,
    "exp": np.exp,
    "sqrt": np.sqrt,
    "abs": np.abs,
    "minimum": np.minimum,
    "maximum": np.maximum,
}


def density_expression(
    expression: str, r: np.ndarray, z: np.ndarray, ne0: float, radius: float, height: float
) -> np.ndarray:
    tree = ast.parse(expression, mode="eval")
    nodes = list(ast.walk(tree))
    allowed = (
        ast.Expression,
        ast.BinOp,
        ast.UnaryOp,
        ast.Call,
        ast.Name,
        ast.Load,
        ast.Constant,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.Pow,
        ast.USub,
        ast.UAdd,
    )
    if len(nodes) > 100 or any(not isinstance(node, allowed) for node in nodes):
        raise ValueError("密度式は数値、r、z、R、H、ne0と許可された関数だけを使用できます。")
    names = {"r", "z", "R", "H", "ne0", "pi"} | set(_FUNCTIONS)
    for node in nodes:
        if isinstance(node, ast.Name) and node.id not in names:
            raise ValueError(f"密度式の未知の名前: {node.id}")
        if isinstance(node, ast.Call) and (
            not isinstance(node.func, ast.Name) or node.func.id not in _FUNCTIONS or node.keywords
        ):
            raise ValueError("密度式では許可された関数の位置引数だけを使用できます。")
        if isinstance(node, ast.Constant) and (
            isinstance(node.value, bool)
            or not isinstance(node.value, (int, float))
            or abs(node.value) > 1e100
        ):
            raise ValueError("密度式の定数は有限な数値にしてください。")
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
            if not isinstance(node.right, ast.Constant) or abs(node.right.value) > 12:
                raise ValueError("密度式の指数は絶対値12以下の定数にしてください。")
    namespace = dict(_FUNCTIONS, r=r, z=z, R=radius, H=height, ne0=ne0, pi=np.pi)
    with np.errstate(all="ignore"):
        result = eval(
            compile(tree, "<density expression>", "eval"), {"__builtins__": {}}, namespace
        )
    return np.broadcast_to(np.asarray(result, dtype=float), r.shape).copy()


def density_from_csv(path: Path, r_m: np.ndarray, z_m: np.ndarray) -> np.ndarray:
    headers, values = read_numeric_csv(path)
    required = ["r_mm", "z_mm", "ne_m3"]
    if not all(name in headers for name in required):
        raise ValueError("密度CSVには r_mm, z_mm, ne_m3 列が必要です。")
    points = values[:, [headers.index(name) for name in required]]
    xy = points[:, :2] * 1e-3
    if len(np.unique(xy, axis=0)) != len(xy):
        raise ValueError("密度CSVの座標が重複しています。")
    rs, zs = np.unique(xy[:, 0]), np.unique(xy[:, 1])
    query = np.column_stack((r_m.ravel(), z_m.ravel()))
    if len(rs) * len(zs) == len(points):
        table = np.empty((len(rs), len(zs)))
        table[np.searchsorted(rs, xy[:, 0]), np.searchsorted(zs, xy[:, 1])] = points[:, 2]
        interp = RegularGridInterpolator((rs, zs), table, bounds_error=False, fill_value=np.nan)
    else:
        interp = LinearNDInterpolator(xy, points[:, 2], fill_value=np.nan)
    return np.asarray(interp(query)).reshape(r_m.shape)
