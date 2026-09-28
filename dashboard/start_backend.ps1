# Lance l'API FastAPI du tableau de bord.
#   .\dashboard\start_backend.ps1              mode rejeu (donnees locales)
#   .\dashboard\start_backend.ps1 -Live        ingestion directe depuis l'API i-SENSE
#
# Le script se place lui-meme dans dashboard\backend : peu importe d'ou on l'appelle.
# ASCII pur volontairement : PowerShell 5.1 lit l'UTF-8 sans BOM comme du Windows-1252,
# ce qui casserait les caracteres accentues et les tirets longs.

param([switch]$Live)

$ErrorActionPreference = "Stop"
$here    = Split-Path -Parent $MyInvocation.MyCommand.Path
$backend = Join-Path $here "backend"

if (-not (Test-Path (Join-Path $backend "main.py"))) {
    Write-Host "main.py introuvable dans $backend" -ForegroundColor Red
    exit 1
}

# On ne SUPPOSE plus quel interpreteur convient : on le TESTE.
# L'ancienne version codait en dur C:\isense_venv, dont il manquait fastapi ;
# l'erreur ne se voyait qu'au demarrage d'uvicorn, loin de sa cause.
$candidats = @("C:\isense_venv\Scripts\python.exe", "python")
$py = $null
foreach ($c in $candidats) {
    $ok = $false
    try {
        & $c -c "import fastapi, uvicorn, sklearn, pandas" 2>$null
        $ok = ($LASTEXITCODE -eq 0)
    } catch { $ok = $false }
    if ($ok) { $py = $c; break }
    Write-Host "  ecarte : $c (dependances manquantes)" -ForegroundColor DarkGray
}
if (-not $py) {
    Write-Host "Aucun interpreteur Python ne dispose de fastapi + uvicorn + scikit-learn." -ForegroundColor Red
    Write-Host "Installer : python -m pip install fastapi uvicorn scikit-learn pandas lightgbm" -ForegroundColor Yellow
    exit 1
}

# Mode d'alimentation. En direct, les identifiants sont lus dans l'environnement
# et ne transitent jamais par ce fichier : c'est vous qui les saisissez.
if ($Live) {
    if (-not $env:ISENSE_EMAIL -or -not $env:ISENSE_PASSWORD) {
        Write-Host "Mode direct demande, mais ISENSE_EMAIL / ISENSE_PASSWORD ne sont pas definis." -ForegroundColor Red
        Write-Host "Dans CE terminal, avant de relancer :" -ForegroundColor Yellow
        Write-Host "    `$env:ISENSE_EMAIL    = Read-Host 'Adresse i-SENSE'"
        Write-Host "    `$env:ISENSE_PASSWORD = Read-Host 'Mot de passe i-SENSE'"
        Write-Host ""
        Write-Host "Read-Host plutot qu'une affectation directe, pour deux raisons :" -ForegroundColor DarkGray
        Write-Host "  - entre guillemets DOUBLES, un `$ dans le mot de passe est pris pour" -ForegroundColor DarkGray
        Write-Host "    une variable et la valeur part tronquee (l'API repond alors 422) ;" -ForegroundColor DarkGray
        Write-Host "  - ce qui est saisi par Read-Host n'entre pas dans l'historique" -ForegroundColor DarkGray
        Write-Host "    PowerShell, contrairement a une ligne tapee au clavier." -ForegroundColor DarkGray
        exit 1
    }
    # Un espace colle en fin de valeur — copier-coller — fait echouer la
    # validation cote serveur avec un 422 sans rapport apparent.
    $env:ISENSE_EMAIL    = $env:ISENSE_EMAIL.Trim()
    $env:ISENSE_PASSWORD = $env:ISENSE_PASSWORD.Trim()
    $env:ISENSE_LIVE = "1"
    $alim = "API i-SENSE en direct (compte $($env:ISENSE_EMAIL))"
} else {
    $env:ISENSE_LIVE = ""
    $alim = "rejeu du jeu de donnees local"
}

Write-Host "API Health Index i-SENSE" -ForegroundColor Cyan
Write-Host "  python       : $py"
Write-Host "  dossier      : $backend"
Write-Host "  alimentation : $alim"
Write-Host "  URL          : http://127.0.0.1:8000   (documentation interactive : /docs)"
Write-Host ""
Write-Host "Premier demarrage : environ 60 secondes." -ForegroundColor Yellow
Write-Host "Le service reconstruit les variables sans fuite (ACP, Isolation Forest,"
Write-Host "z-scores calcules sur la seule fenetre d'entrainement) avant de repondre."
Write-Host ""

Set-Location $backend
& $py -m uvicorn main:app --host 127.0.0.1 --port 8000
