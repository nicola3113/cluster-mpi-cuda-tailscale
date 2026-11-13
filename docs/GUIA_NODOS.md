# Guía para unirte al cluster (NODOS)

Tu laptop será un **nodo**: el maestro lanzará el programa en tu GPU NVIDIA a través de la VPN Tailscale. **No necesitas instalar CUDA Toolkit ni compilar nada.** Toma unos 10 minutos.

## Antes de empezar

El maestro te tiene que pasar:
- la **auth key** de Tailscale (`tskey-auth-…`), por privado;
- su **clave pública** (`ssh-ed25519 AAAA… mpiuser@master`). Si ya está en el repo como `scripts/clave_maestro.pub`, no hace falta que te la mande aparte.

Tu laptop necesita:
- Windows 10/11 actualizado.
- GPU NVIDIA con driver reciente. Revísalo con `nvidia-smi` en PowerShell.
- Unos 2 GB libres, de preferencia en D:.
- Estar enchufada durante la práctica.

## Paso 1 — Descargar el repo

Si tienes git:

```powershell
git clone https://github.com/nicola3113/cluster-mpi-cuda-tailscale.git D:\cluster-mpi-cuda-tailscale
```

Si no, en GitHub usa **Code → Download ZIP** y descomprímelo en `D:\cluster-mpi-cuda-tailscale`.

## Paso 2 — Instalar Ubuntu-22.04 (solo si no lo tienes)

Comprueba con `wsl -l -v`. Si no aparece `Ubuntu-22.04`, abre PowerShell como **Administrador**:

```powershell
wsl --install -d Ubuntu-22.04 --location D:\WSL\Ubuntu-22.04
```

- Si no tienes disco D:, quita `--location …`.
- Reinicia si te lo pide.
- Cuando Ubuntu abra, crea tu usuario y contraseña y escribe `exit`.

Tiene que ser **22.04**: todo el cluster debe tener la misma versión de OpenMPI.

## Paso 3 — Ejecutar el script

En PowerShell como **Administrador** (clic derecho → *Ejecutar como administrador*):

```powershell
cd D:\cluster-mpi-cuda-tailscale\scripts
Set-ExecutionPolicy -Scope Process Bypass -Force
.\unirse_cluster.ps1 -Rol nodo -Nombre tunombre -AuthKey tskey-auth-XXXXXXXX
```

- `-Nombre` va en minúsculas, sin espacios ni tildes, y debe ser **distinto** para cada compañero (por ejemplo `juan`).
- Si `scripts\clave_maestro.pub` no existe, agrega la clave al final del comando:
  `-ClaveMaestro "ssh-ed25519 AAAA... mpiuser@master"`

El script hace lo siguiente:

1. Comprueba tu GPU.
2. Activa la red espejo de WSL (y guarda una copia de tu `.wslconfig` como `.bak`).
3. Abre los puertos 22 y 10000-10400 solo para WSL.
4. Instala Tailscale y te une a la red como `nodo-tunombre`. Si ya usabas Tailscale, tu cuenta no se borra: vuelves a ella con `tailscale switch`.
5. Dentro de Ubuntu: instala OpenMPI y SSH, crea el usuario `mpiuser` y autoriza **solo** la clave del maestro.
6. Reinicia WSL y verifica.

Al final verás algo así:

```
LISTO. Envia al maestro:  nodo-juan  100.x.y.z
```

## Paso 4 — Avisar al maestro

Mándale esa línea por el chat del grupo.

## Paso 5 — Durante la práctica

1. **Deja una ventana de Ubuntu abierta**:
   ```powershell
   wsl -d Ubuntu-22.04
   ```
   Si WSL se apaga, desapareces del cluster.
2. Que Tailscale esté conectado (ícono en la bandeja, o `tailscale status`).
3. Comprobación rápida dentro de Ubuntu:
   ```bash
   hostname                 # nodo-tunombre
   nvidia-smi -L            # tu GPU
   systemctl is-active ssh  # active
   ```

Si algo falla, revisa [SOLUCION_PROBLEMAS.md](SOLUCION_PROBLEMAS.md).

## Al terminar el curso (opcional)

```powershell
& "C:\Program Files\Tailscale\tailscale.exe" logout
```

Y dentro de Ubuntu, para borrar el usuario del cluster:

```bash
sudo deluser --remove-home mpiuser
```
