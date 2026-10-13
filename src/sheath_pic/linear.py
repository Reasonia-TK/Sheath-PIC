"""FP64 CPU/CUDA sparse solves; native cuDSS analysis is reused per fixed pattern.

CUDA arrays own all pointers passed to cuDSS and share its stream. Only numerical
values and right-hand sides cross PCIe on subsequent solves. The physical model
and nonlinear stopping criteria are identical for both backends.
"""

# NVIDIA cuDSS dependency notice; see THIRD_PARTY_NOTICES.md.
# NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
# LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
# SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
# PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
# NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
# DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
# NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
# NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
# LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
# SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
# DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
# WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
# ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
# OF THESE LICENSED DELIVERABLES.
#
# U.S. Government End Users.  These Licensed Deliverables are a
# "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
# 1995), consisting of "commercial computer software" and "commercial
# computer software documentation" as such terms are used in 48
# C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
# only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
# 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
# U.S. Government End Users acquire the Licensed Deliverables with
# only those rights set forth herein.

from __future__ import annotations

import ctypes as ct
import importlib.metadata
import os
import time
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

# cuDSS 0.8 public headers: cudss.h / cudss_data_types.h and CUDA library_types.h.
ANALYSIS = 3
FACTORIZATION = 4
SOLVE = 1008
CUDA_R_64F = 1
CUDA_R_32I = 10


def _load_cudss():
    distribution = importlib.metadata.distribution("nvidia-cudss-cu13")
    candidates = [
        Path(distribution.locate_file(item))
        for item in distribution.files or []
        if str(item).endswith(("cudss64_0.dll", "libcudss.so.0"))
    ]
    if not candidates:
        raise RuntimeError("cuDSS共有ライブラリが見つかりません。cuda追加依存を同期してください。")
    path = candidates[0]
    directories = []
    if os.name == "nt":
        for directory in (path.parent, path.parent / "x86_64"):
            if directory.is_dir():
                directories.append(os.add_dll_directory(str(directory)))
    try:
        loader = ct.WinDLL if os.name == "nt" else ct.CDLL
        library = loader(str(path))
    except BaseException:
        for directory in directories:
            directory.close()
        raise
    pointer, pointer_out = ct.c_void_p, ct.POINTER(ct.c_void_p)
    signatures = {
        "cudssCreate": [pointer_out],
        "cudssDestroy": [pointer],
        "cudssConfigCreate": [pointer_out],
        "cudssConfigDestroy": [pointer],
        "cudssConfigSet": [pointer, ct.c_int, pointer, ct.c_size_t],
        "cudssDataCreate": [pointer, pointer_out],
        "cudssDataDestroy": [pointer, pointer],
        "cudssSetStream": [pointer, pointer],
        "cudssMatrixCreateCsr": [
            pointer_out,
            ct.c_int64,
            ct.c_int64,
            ct.c_int64,
            pointer,
            pointer,
            pointer,
            pointer,
            ct.c_int,
            ct.c_int,
            ct.c_int,
            ct.c_int,
            ct.c_int,
            ct.c_int,
        ],
        "cudssMatrixCreateDn": [
            pointer_out,
            ct.c_int64,
            ct.c_int64,
            ct.c_int64,
            pointer,
            ct.c_int,
            ct.c_int,
        ],
        "cudssMatrixDestroy": [pointer],
        "cudssExecute": [pointer, ct.c_int, pointer, pointer, pointer, pointer, pointer],
        "cudssDataGet": [
            pointer,
            pointer,
            ct.c_int,
            pointer,
            ct.c_size_t,
            ct.POINTER(ct.c_size_t),
        ],
    }
    for name, arguments in signatures.items():
        function = getattr(library, name)
        function.argtypes, function.restype = arguments, ct.c_int
    return library, directories


def _check(status, operation):
    if status:
        raise RuntimeError(f"cuDSS {operation}が失敗しました（status={status}）。")


class CudaSystem:
    """A fixed CSR system with reusable device buffers and symbolic factorization."""

    def __init__(self, cp, library, pattern: csr_matrix):
        self.cp, self.library = cp, library
        self.stream = cp.cuda.get_current_stream()
        self.n = pattern.shape[0]
        self.pattern = pattern.copy().astype(np.float64)
        self.pattern.sort_indices()
        rows = np.repeat(np.arange(self.n, dtype=np.int64), np.diff(self.pattern.indptr))
        self.rows = rows
        self.keys = rows * self.n + self.pattern.indices
        self.host_values = np.zeros(self.pattern.nnz, dtype=np.float64)
        self.host_scaled_values = np.empty(self.pattern.nnz, dtype=np.float64)
        self.host_rhs = np.empty(self.n, dtype=np.float64)
        self.rowptr = cp.asarray(self.pattern.indptr, dtype=cp.int32)
        self.columns = cp.asarray(self.pattern.indices, dtype=cp.int32)
        self.values = cp.empty(self.pattern.nnz, dtype=cp.float64)
        self.rhs = cp.empty(self.n, dtype=cp.float64)
        self.solution = cp.empty(self.n, dtype=cp.float64)
        self.handle, self.config, self.data = ct.c_void_p(), ct.c_void_p(), ct.c_void_p()
        self.a, self.b, self.x = ct.c_void_p(), ct.c_void_p(), ct.c_void_p()
        self.analyzed = False
        self.solves = 0
        self.maximum_relative_residual = 0.0
        self.refinements = 0
        try:
            _check(library.cudssCreate(ct.byref(self.handle)), "create")
            _check(library.cudssSetStream(self.handle, self.stream.ptr), "set stream")
            _check(library.cudssConfigCreate(ct.byref(self.config)), "config create")
            # GPU numerical factorization; no hybrid CPU execution. Deterministic mode.
            for parameter, value in ((15, 0), (24, 1)):
                scalar = ct.c_int(value)
                _check(
                    library.cudssConfigSet(
                        self.config, parameter, ct.byref(scalar), ct.sizeof(scalar)
                    ),
                    "config set",
                )
            _check(library.cudssDataCreate(self.handle, ct.byref(self.data)), "data create")
            _check(
                library.cudssMatrixCreateCsr(
                    ct.byref(self.a),
                    self.n,
                    self.n,
                    self.pattern.nnz,
                    self.rowptr.data.ptr,
                    None,
                    self.columns.data.ptr,
                    self.values.data.ptr,
                    CUDA_R_32I,
                    CUDA_R_32I,
                    CUDA_R_64F,
                    0,
                    0,
                    0,
                ),
                "CSR create",
            )
            for descriptor, array in ((self.b, self.rhs), (self.x, self.solution)):
                _check(
                    library.cudssMatrixCreateDn(
                        ct.byref(descriptor),
                        self.n,
                        1,
                        self.n,
                        array.data.ptr,
                        CUDA_R_64F,
                        0,
                    ),
                    "dense create",
                )
        except BaseException:
            self.close()
            raise

    def execute(self, phase):
        _check(
            self.library.cudssExecute(
                self.handle,
                phase,
                self.config,
                self.data,
                self.a,
                self.x,
                self.b,
            ),
            f"execute phase {phase}",
        )

    def solve(self, matrix, rhs):
        matrix = matrix.tocsr()
        matrix.sort_indices()
        if matrix.shape != (self.n, self.n) or np.shape(rhs) != (self.n,):
            raise ValueError("cuDSSの行列・右辺の形状が一致しません。")
        if np.array_equal(matrix.indptr, self.pattern.indptr) and np.array_equal(
            matrix.indices, self.pattern.indices
        ):
            np.copyto(self.host_values, matrix.data)
        else:
            rows = np.repeat(np.arange(self.n, dtype=np.int64), np.diff(matrix.indptr))
            keys = rows * self.n + matrix.indices
            positions = np.searchsorted(self.keys, keys)
            if np.any(positions >= len(self.keys)) or not np.array_equal(
                self.keys[positions], keys
            ):
                raise ValueError("行列が固定疎行列パターンの範囲外です。")
            self.host_values.fill(0)
            self.host_values[positions] = matrix.data
        # Potential and normalized density columns have very different magnitudes.
        # Equilibrate on the host before GPU factorization; undo column scaling in x.
        column_maximum = np.zeros(self.n)
        np.maximum.at(column_maximum, self.pattern.indices, abs(self.host_values))
        if np.any(column_maximum == 0):
            raise ArithmeticError("線形系にゼロの列があります。")
        column_scale = 1 / column_maximum
        scaled_values = self.host_scaled_values
        np.multiply(self.host_values, column_scale[self.pattern.indices], out=scaled_values)
        row_maximum = np.maximum.reduceat(abs(scaled_values), self.pattern.indptr[:-1])
        if np.any(row_maximum == 0):
            raise ArithmeticError("線形系にゼロの行があります。")
        row_scale = 1 / row_maximum
        scaled_values *= row_scale[self.rows]
        self.values.set(scaled_values, stream=self.stream)
        np.multiply(rhs, row_scale, out=self.host_rhs)
        self.rhs.set(self.host_rhs, stream=self.stream)
        if not self.analyzed:
            self.execute(ANALYSIS)
            self.analyzed = True
        self.execute(FACTORIZATION | SOLVE)
        result = self.solution.get(stream=self.stream) * column_scale
        info, written = ct.c_int(), ct.c_size_t()
        _check(
            self.library.cudssDataGet(
                self.handle,
                self.data,
                0,
                ct.byref(info),
                ct.sizeof(info),
                ct.byref(written),
            ),
            "data info",
        )
        if info.value or not np.isfinite(result).all():
            raise ArithmeticError(f"cuDSSの数値解が不正です（info={info.value}）。")
        denominator = max(np.linalg.norm(rhs, ord=np.inf), 1e-30)
        for attempt in range(7):
            residual = rhs - matrix @ result
            relative = float(np.linalg.norm(residual, ord=np.inf) / denominator)
            if relative <= 1e-11:
                break
            if attempt == 6:
                raise ArithmeticError(f"CUDA線形残差が収束しません（{relative:.3g}）。")
            np.multiply(residual, row_scale, out=self.host_rhs)
            self.rhs.set(self.host_rhs, stream=self.stream)
            self.execute(SOLVE)
            result += self.solution.get(stream=self.stream) * column_scale
            self.refinements += 1
        self.maximum_relative_residual = max(self.maximum_relative_residual, relative)
        self.solves += 1
        return result

    def close(self):
        self.stream.synchronize()
        library = self.library
        for descriptor in (self.a, self.b, self.x):
            if descriptor.value:
                library.cudssMatrixDestroy(descriptor)
                descriptor.value = None
        if self.data.value:
            library.cudssDataDestroy(self.handle, self.data)
            self.data.value = None
        if self.config.value:
            library.cudssConfigDestroy(self.config)
            self.config.value = None
        if self.handle.value:
            library.cudssDestroy(self.handle)
            self.handle.value = None


class LinearBackend:
    def __init__(self, requested="cpu"):
        if requested not in {"cpu", "cuda", "auto"}:
            raise ValueError("計算バックエンドはcpu、cuda、autoから選択してください。")
        self.requested, self.name = requested, "cpu"
        self.systems, self.patterns = {}, {}
        self.directories = []
        self.solve_seconds, self.solve_count = 0.0, 0
        self.unavailable_reason = ""
        self.device = "CPU"
        if requested != "cpu":
            os.environ.setdefault("CUPY_CACHE_DIR", str(Path.cwd() / ".cache" / "cupy"))
            try:
                import cupy as cp

                if cp.cuda.runtime.getDeviceCount() < 1:
                    raise RuntimeError("CUDA GPUが見つかりません。")
                self.library, self.directories = _load_cudss()
                self.cp = cp
                self.device = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)["name"]
                if isinstance(self.device, bytes):
                    self.device = self.device.decode()
                self.name = "cuda"
            except (
                ImportError,
                OSError,
                RuntimeError,
                importlib.metadata.PackageNotFoundError,
            ) as exc:
                self.unavailable_reason = str(exc)
                if requested == "cuda":
                    raise RuntimeError(
                        "CUDAバックエンドを初期化できません。"
                        "uv sync --locked --extra cudaとドライバーを確認してください。\n" + str(exc)
                    ) from exc

    def register(self, key, pattern):
        self.patterns[key] = pattern

    def solve(self, key, matrix, rhs):
        started = time.perf_counter()
        if self.name == "cpu":
            result = spsolve(matrix, rhs)
        else:
            if key not in self.systems:
                self.systems[key] = CudaSystem(self.cp, self.library, self.patterns[key])
            try:
                result = self.systems[key].solve(matrix, rhs)
            except ArithmeticError:
                self.systems.pop(key).close()
                raise
        self.solve_seconds += time.perf_counter() - started
        self.solve_count += 1
        return result

    def metadata(self):
        result = {
            "requested": self.requested,
            "actual": self.name,
            "device": self.device,
            "precision": "float64",
            "linear_solve_seconds": self.solve_seconds,
            "linear_solve_count": self.solve_count,
            "symbolic_analyses": len(self.systems) if self.name == "cuda" else self.solve_count,
            "maximum_linear_relative_residual": (
                max((item.maximum_relative_residual for item in self.systems.values()), default=0)
                if self.name == "cuda"
                else None
            ),
            "iterative_refinements": sum(item.refinements for item in self.systems.values()),
            "unavailable_reason": self.unavailable_reason,
        }
        if self.name == "cuda":
            result.update(
                cupy_version=self.cp.__version__,
                cudss_version=importlib.metadata.version("nvidia-cudss-cu13"),
                driver_version=self.cp.cuda.runtime.driverGetVersion(),
            )
        return result

    def close(self):
        for system in self.systems.values():
            system.close()
        self.systems.clear()
        for directory in self.directories:
            directory.close()
        self.directories.clear()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
