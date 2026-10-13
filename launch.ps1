$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$env:UV_CACHE_DIR = Join-Path $PSScriptRoot '.uv-cache'
$env:MPLCONFIGDIR = Join-Path $PSScriptRoot '.mpl-cache'
$env:PYTHONIOENCODING = 'utf-8'
$env:CUPY_CACHE_DIR = Join-Path $PSScriptRoot '.cache/cupy'
$sheathRunArguments = @('run', '--locked')
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    $sheathRunArguments += @('--extra', 'cuda')
}
$sheathRunArguments += @('python', '-m', 'sheath_pic')
uv @sheathRunArguments
exit $LASTEXITCODE
