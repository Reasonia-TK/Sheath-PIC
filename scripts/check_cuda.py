"""Check the active environment and an actual cuDSS FP64 solve without running a discharge."""

from __future__ import annotations

import importlib.metadata
import json
import platform
import sys

import numpy as np
from scipy.sparse import diags

from sheath_pic.linear import LinearBackend


def main():
    print(f"OS: {platform.system()} {platform.release()} / {platform.machine()}", flush=True)
    print(f"Python: {sys.version.split()[0]} / {sys.executable}", flush=True)
    for name in (
        "sheath-pic",
        "cupy-cuda13x",
        "nvidia-cudss-cu13",
        "nvidia-cublas",
        "nvidia-cuda-nvrtc",
    ):
        try:
            version = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            version = "未導入"
        print(f"{name}: {version}", flush=True)
    stage = "CuPy / CUDAデバイス / cuDSSの初期化"
    try:
        with LinearBackend("cuda") as backend:
            print(f"GPU: {backend.device}", flush=True)
            print(f"CUDA driver API: {backend.cp.cuda.runtime.driverGetVersion()}", flush=True)
            print(f"Compute capability: {backend.cp.cuda.Device().compute_capability}", flush=True)
            stage = "cuDSSのFP64行列計算"
            matrix = diags([-np.ones(19), np.full(20, 4.0), -np.ones(19)], [-1, 0, 1], format="csr")
            rhs = np.arange(20, dtype=np.float64)
            backend.register("check", matrix)
            solution = backend.solve("check", matrix, rhs)
            relative_residual = float(np.linalg.norm(matrix @ solution - rhs) / np.linalg.norm(rhs))
            if not np.isfinite(relative_residual) or relative_residual > 1e-11:
                raise ArithmeticError(f"解の相対残差が許容値を超えました: {relative_residual:g}")
            print(f"PASS: CUDA / FP64 / 相対残差 {relative_residual:.3g}", flush=True)
            print(json.dumps(backend.metadata(), ensure_ascii=False, indent=2), flush=True)
    except Exception as exc:
        print(f"FAIL: {stage}\n{type(exc).__name__}: {exc}", flush=True)
        print("CUDA依存の導入: uv sync --locked --extra cuda", flush=True)
        print("診断結果とGPU型番・ドライバー版を確認してください。", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
