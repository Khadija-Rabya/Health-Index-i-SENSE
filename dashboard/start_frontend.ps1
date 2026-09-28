# Lance l'interface React du tableau de bord.
#   .\dashboard\start_frontend.ps1        -> mode preview (compile puis sert)
#   .\dashboard\start_frontend.ps1 -Dev   -> mode developpement (rechargement a chaud)
# Le script se place lui-meme dans dashboard\frontend : peu importe d'ou on l'appelle.
# ASCII pur volontairement : PowerShell 5.1 lit l'UTF-8 sans BOM comme du Windows-1252,
# ce qui casserait les caracteres accentues et les tirets longs.

param([switch]$Dev)

$ErrorActionPreference = "Stop"
$here  = Split-Path -Parent $MyInvocation.MyCommand.Path
$front = Join-Path $here "frontend"

if (-not (Test-Path (Join-Path $front "package.json"))) {
    Write-Host "Projet frontend introuvable dans $front" -ForegroundColor Red
    exit 1
}

Set-Location $front

if (-not (Test-Path "node_modules")) {
    Write-Host "Dependances absentes, installation en cours (npm install)..." -ForegroundColor Yellow
    npm install
}

if ($Dev) {
    Write-Host "Interface React, mode developpement" -ForegroundColor Cyan
    Write-Host "  URL : http://localhost:5173"
    Write-Host "  L'API doit tourner en parallele sur le port 8000."
    Write-Host ""
    npm run dev
}
else {
    Write-Host "Interface React, mode preview" -ForegroundColor Cyan
    Write-Host "  Compilation puis service sur http://localhost:4173"
    Write-Host "  L'API doit tourner en parallele sur le port 8000."
    Write-Host ""
    npm run build
    npm run preview -- --port 4173
}
