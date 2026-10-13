$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$env:UV_CACHE_DIR = Join-Path $PSScriptRoot '.uv-cache'
$env:MPLCONFIGDIR = Join-Path $PSScriptRoot '.mpl-cache'
$env:PYTHONIOENCODING = 'utf-8'
uv run --locked python -m sheath_pic
exit $LASTEXITCODE
