@echo off
cd /d "%~dp0"
set "UV_CACHE_DIR=%CD%\.uv-cache"
set "MPLCONFIGDIR=%CD%\.mpl-cache"
set "PYTHONIOENCODING=utf-8"
uv run --locked python -m sheath_pic
if errorlevel 1 pause
