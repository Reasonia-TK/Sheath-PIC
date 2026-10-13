@echo off
cd /d "%~dp0"
set "UV_CACHE_DIR=%CD%\.uv-cache"
set "MPLCONFIGDIR=%CD%\.mpl-cache"
set "PYTHONIOENCODING=utf-8"
set "CUPY_CACHE_DIR=%CD%\.cache\cupy"
set "SHEATH_CUDA_EXTRA="
where nvidia-smi >nul 2>nul
if not errorlevel 1 set "SHEATH_CUDA_EXTRA=--extra cuda"
uv run --locked %SHEATH_CUDA_EXTRA% python -m sheath_pic
if errorlevel 1 pause
