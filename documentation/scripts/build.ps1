# Regenerate widget screenshots and rebuild the documentation site.
#
# Usage:
#     documentation/scripts/build.ps1                # screenshots + build
#     documentation/scripts/build.ps1 -SkipShots     # build only
#     documentation/scripts/build.ps1 -Serve         # screenshots + live-serve
#
# Expects a Python virtual environment at <repo>/.venv with mkdocs,
# mkdocs-material, pyyaml, napari, and the plugin itself installed.
[CmdletBinding()]
param(
    [switch]$SkipShots,
    [switch]$Serve
)

$ErrorActionPreference = "Stop"

# Resolve repo root from script location (scripts/ -> documentation/ -> repo)
$DocRoot  = Split-Path -Parent $PSScriptRoot
$RepoRoot = Split-Path -Parent $DocRoot
$Python   = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$MkdocsYml = Join-Path $DocRoot "mkdocs.yml"

if (-not (Test-Path $Python)) {
    Write-Error "Python not found at $Python. Create a .venv at the repo root first."
}

if (-not $SkipShots) {
    Write-Host "==> Capturing widget screenshots..." -ForegroundColor Cyan
    & $Python (Join-Path $PSScriptRoot "capture_widgets.py")
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

if ($Serve) {
    Write-Host "==> Serving site at http://127.0.0.1:8000 (Ctrl+C to stop)" -ForegroundColor Cyan
    & $Python -m mkdocs serve -f $MkdocsYml
} else {
    Write-Host "==> Building static site..." -ForegroundColor Cyan
    & $Python -m mkdocs build -f $MkdocsYml
    if ($LASTEXITCODE -eq 0) {
        $site = Join-Path $DocRoot "site\index.html"
        Write-Host "Built. Open: $site" -ForegroundColor Green
    }
}
