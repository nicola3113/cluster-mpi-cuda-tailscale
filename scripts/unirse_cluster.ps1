# unirse_cluster.ps1 -- prepara esta PC (Windows + GPU NVIDIA) como MAESTRO o NODO del cluster
# MPI + CUDA sobre la VPN Tailscale. Ejecutar en PowerShell COMO ADMINISTRADOR desde la carpeta scripts\:
#
#   Set-ExecutionPolicy -Scope Process Bypass -Force
#   .\unirse_cluster.ps1 -Rol maestro -Nombre ana  -AuthKey tskey-auth-XXXX
#   .\unirse_cluster.ps1 -Rol nodo    -Nombre juan -AuthKey tskey-auth-XXXX [-ClaveMaestro "ssh-ed25519 AAAA..."]
#
# Para -Rol nodo la clave publica del maestro se toma de -ClaveMaestro (texto o ruta a archivo)
# o, si no se pasa, de scripts\clave_maestro.pub.
param(
  [Parameter(Mandatory=$true)][ValidateSet('maestro','nodo')][string]$Rol,
  [Parameter(Mandatory=$true)][ValidatePattern('^[a-z0-9]+$')][string]$Nombre,
  [Parameter(Mandatory=$true)][string]$AuthKey,
  [string]$ClaveMaestro = '',
  [string]$Distro = 'Ubuntu-22.04'
)
$ErrorActionPreference = 'Stop'
function Paso($t) { Write-Host "`n=== $t ===" -ForegroundColor Cyan }
function RutaWsl($p) { (wsl -d $Distro -u root -- wslpath -a ($p -replace '\\','/')).Trim() }

if (-not ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
  throw 'Abre PowerShell como Administrador.'
}
$here = $PSScriptRoot
$repo = Split-Path $here -Parent

# Clave del maestro (solo nodos) -- se valida ANTES de tocar nada
if ($Rol -eq 'nodo') {
  if ($ClaveMaestro -eq '') { $ClaveMaestro = Join-Path $here 'clave_maestro.pub' }
  if (Test-Path $ClaveMaestro) { $ClaveMaestro = (Get-Content $ClaveMaestro -Raw).Trim() }
  if ($ClaveMaestro -notmatch '^(ssh-|ecdsa-)') {
    throw 'Falta la clave publica del maestro: pasala con -ClaveMaestro "ssh-ed25519 AAAA..." o ponla en scripts\clave_maestro.pub'
  }
}

Paso '1. GPU NVIDIA'
nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader
if ($LASTEXITCODE -ne 0) { throw 'No se detecta GPU NVIDIA / driver. Instala el driver desde nvidia.com.' }

Paso "2. WSL2 y $Distro"
$lista = (wsl -l -q) -replace "`0", '' | ForEach-Object { $_.Trim() } | Where-Object { $_ -ne '' }
if ($lista -notcontains $Distro) {
  Write-Host "Falta $Distro. Instalala (en D:) con:" -ForegroundColor Yellow
  Write-Host "  wsl --install -d $Distro --location D:\WSL\$Distro"
  Write-Host 'Crea tu usuario/contrasena cuando lo pida, escribe exit y vuelve a correr este script.'
  exit 1
}

Paso '3. Red espejo (mirrored) en %UserProfile%\.wslconfig'
$cfg = Join-Path $env:USERPROFILE '.wslconfig'
$txt = if (Test-Path $cfg) { Get-Content $cfg -Raw } else { '' }
if ($txt -match '(?m)^\s*networkingMode\s*=') {
  $txt = $txt -replace '(?m)^\s*networkingMode\s*=.*$', 'networkingMode=mirrored'
} elseif ($txt -match '(?m)^\[wsl2\]') {
  $txt = $txt -replace '(?m)^\[wsl2\]\s*$', "[wsl2]`r`nnetworkingMode=mirrored"
} else {
  $txt = "[wsl2]`r`nnetworkingMode=mirrored`r`n" + $txt
}
if (Test-Path $cfg) { Copy-Item $cfg "$cfg.bak" -Force }
Set-Content -Path $cfg -Value $txt -Encoding ASCII
Get-Content $cfg

Paso '4. Firewall (SSH 22 y OpenMPI 10000-10400)'
$wslId = '{40E0AC32-46A5-438A-A0B2-2B479E8F2E90}'
if (Get-Command New-NetFirewallHyperVRule -ErrorAction SilentlyContinue) {
  Get-NetFirewallHyperVRule -Name 'Cluster-MPI-TCP' -ErrorAction SilentlyContinue | Remove-NetFirewallHyperVRule
  New-NetFirewallHyperVRule -Name 'Cluster-MPI-TCP' -DisplayName 'Cluster MPI (WSL)' -Direction Inbound `
    -VMCreatorId $wslId -Protocol TCP -LocalPorts 22,'10000-10400' -Action Allow | Out-Null
}
Get-NetFirewallRule -DisplayName 'Cluster MPI (WSL)' -ErrorAction SilentlyContinue | Remove-NetFirewallRule
New-NetFirewallRule -DisplayName 'Cluster MPI (WSL)' -Direction Inbound -Protocol TCP -LocalPort 22,'10000-10400' -Action Allow -Profile Any | Out-Null
$sshWin = Get-Service sshd -ErrorAction SilentlyContinue
if ($sshWin -and $sshWin.Status -eq 'Running') {
  Write-Host 'AVISO: OpenSSH Server de Windows usa el puerto 22 y choca con el de WSL. Detenlo: Stop-Service sshd' -ForegroundColor Yellow
}

Paso '5. Tailscale'
$ts = 'C:\Program Files\Tailscale\tailscale.exe'
if (-not (Test-Path $ts)) {
  winget install -e --id Tailscale.Tailscale --accept-package-agreements --accept-source-agreements
  Start-Sleep 5
}
$tsName = if ($Rol -eq 'maestro') { "master-$Nombre" } else { "nodo-$Nombre" }
# 'login' agrega la cuenta del cluster sin cerrar la tuya (vuelves con: tailscale switch)
& $ts login --auth-key=$AuthKey --hostname=$tsName --unattended
Start-Sleep 3
$ip = (& $ts ip -4).Trim()
Write-Host "IP Tailscale: $ip"

Paso "6. Configurar $Distro como $Rol"
if ($Rol -eq 'maestro') {
  Write-Host 'Instala CUDA Toolkit (~3 GB), compila y genera matrices: puede tardar 10-20 min.' -ForegroundColor Yellow
  $sh = Join-Path $here 'setup_maestro.sh'
  $tmp = Join-Path $env:TEMP 'setup_maestro.sh'
  [IO.File]::WriteAllText($tmp, ((Get-Content $sh -Raw) -replace "`r`n", "`n"))
  wsl -d $Distro -u root -- bash (RutaWsl $tmp) (RutaWsl $repo)
} else {
  $sh = Join-Path $here 'setup_nodo.sh'
  $tmp = Join-Path $env:TEMP 'setup_nodo.sh'
  $key = Join-Path $env:TEMP 'clave_maestro.pub'
  [IO.File]::WriteAllText($tmp, ((Get-Content $sh -Raw) -replace "`r`n", "`n"))
  [IO.File]::WriteAllText($key, $ClaveMaestro + "`n")
  wsl -d $Distro -u root -- bash (RutaWsl $tmp) $Nombre (RutaWsl $key)
}
if ($LASTEXITCODE -ne 0) { throw "Fallo la configuracion de $Distro (ver mensajes arriba)." }

Paso '7. Reiniciar WSL y verificar'
wsl --shutdown
Start-Sleep 3
wsl -d $Distro -u root -- bash -c 'hostname; systemctl is-active ssh; ip -4 addr | grep "inet 100\."'

if ($Rol -eq 'maestro') {
  Write-Host "`nLISTO: esta PC es el MAESTRO ($tsName, $ip)." -ForegroundColor Green
  Write-Host "Comparte con los nodos el contenido de scripts\clave_maestro.pub (o subelo al repo)." -ForegroundColor Green
  Write-Host "Luego, para operar el cluster:  wsl -d $Distro -u mpiuser   y   cd ~/llm_cluster" -ForegroundColor Green
} else {
  Write-Host "`nLISTO. Envia al maestro:  nodo-$Nombre  $ip" -ForegroundColor Green
  Write-Host "Durante la practica deja una ventana de Ubuntu abierta (si WSL se apaga, el nodo desaparece)." -ForegroundColor Green
}
