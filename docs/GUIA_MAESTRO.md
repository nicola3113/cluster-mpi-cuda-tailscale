# Guía del MAESTRO

El maestro es **una sola PC**. Compila el programa, guarda las matrices, lanza `mpirun` y ejecuta el rank 0. Necesita una GPU NVIDIA, como los nodos.

## 0. Descargar el repo

Si tienes git:

```powershell
git clone https://github.com/nicola3113/cluster-mpi-cuda-tailscale.git D:\cluster-mpi-cuda-tailscale
```

Si no, entra al repo en GitHub, usa **Code → Download ZIP** y descomprímelo en `D:\cluster-mpi-cuda-tailscale`. Usa una ruta corta, sin tildes ni espacios.

## 1. Red Tailscale y auth key

Todas las PCs deben estar en **la misma tailnet** (la red privada de una cuenta de Tailscale).

1. Crea una cuenta en https://login.tailscale.com (por ejemplo, con tu Gmail). Esa cuenta será la dueña de la red.
2. Ve a **Settings → Keys → Generate auth key…** y configúrala así:
   - **Reusable:** activado (la usarán varias laptops).
   - **Expiration:** 7 días.
   - **Ephemeral:** desactivado.
   - **Pre-approved:** activado (si aparece).
3. Copia la clave `tskey-auth-…`, que solo se muestra una vez. **No la subas al repo.**

Si se va a usar una tailnet que ya existe, por ejemplo la de otro compañero, el dueño de esa red genera la clave y te la pasa.

## 2. Ubuntu-22.04 en WSL2

Si `wsl -l -v` no muestra `Ubuntu-22.04`, abre PowerShell como **Administrador**:

```powershell
wsl --install -d Ubuntu-22.04 --location D:\WSL\Ubuntu-22.04
```

Reinicia si te lo pide, crea tu usuario y contraseña, y escribe `exit`.

## 3. Ejecutar el script como maestro

En PowerShell como **Administrador**:

```powershell
cd D:\cluster-mpi-cuda-tailscale\scripts
Set-ExecutionPolicy -Scope Process Bypass -Force
.\unirse_cluster.ps1 -Rol maestro -Nombre tunombre -AuthKey tskey-auth-XXXXXXXX
```

Tarda entre 10 y 20 minutos (descarga CUDA Toolkit, unos 3 GB). El script hace lo siguiente:

| Paso | Acción |
|---|---|
| 1 | Comprueba la GPU con `nvidia-smi` |
| 2 | Comprueba que exista `Ubuntu-22.04` |
| 3 | Pone `networkingMode=mirrored` en `%UserProfile%\.wslconfig` (guarda copia `.bak`) |
| 4 | Abre los puertos TCP 22 y 10000-10400 en el firewall de Windows y de Hyper-V (solo para WSL) |
| 5 | Instala Tailscale (si falta) y lo une a la red como `master-tunombre` |
| 6 | Dentro de Ubuntu: OpenMPI, SSH, CUDA 12.6, usuario `mpiuser` con clave SSH; compila y genera las matrices en `/home/mpiuser/llm_cluster`, y hace una prueba con 1 GPU |
| 7 | Reinicia WSL y deja el nombre de host en `master` |

Al final, la clave pública queda en `scripts\clave_maestro.pub`.

## 4. Repartir a los compañeros

Cada compañero necesita tres cosas:

1. **El repo**, o al menos la carpeta `scripts/`.
2. **La auth key**, por privado.
3. **Tu clave pública**. Hay dos formas de dársela:
   - Súbela al repo (`git add scripts/clave_maestro.pub && git commit -m "clave maestro" && git push`). Si después descargan el repo actualizado, el script la toma solo.
   - O pásales su contenido (una línea `ssh-ed25519 AAAA… mpiuser@master`) para que la usen con `-ClaveMaestro "…"`.

Luego les mandas [GUIA_NODOS.md](GUIA_NODOS.md).

## 5. Operar el cluster

Abre Ubuntu como `mpiuser`:

```powershell
wsl -d Ubuntu-22.04 -u mpiuser
```

```bash
cd ~/llm_cluster
./cluster.sh descubrir     # agrega a nodos.txt los equipos "nodo-*" que estén en línea en Tailscale
./cluster.sh estado        # SSH + GPU + versión de MPI de cada nodo (deben salir OK)
./cluster.sh sync          # copia el binario y las matrices a cada nodo
./cluster.sh hostfile      # genera hosts.txt (maestro + nodos que responden)
./cluster.sh correr        # mpirun con todas las GPUs (o: ./cluster.sh correr 2)
./cluster.sh benchmark     # -np 1..N válidos, 3 corridas x 5 repeticiones, más la tabla
```

Otros comandos útiles:

```bash
./cluster.sh agregar juan 100.101.102.103   # agregar un nodo a mano
./cluster.sh quitar nodo-juan
LAYERS=4 REPEATS=10 ./cluster.sh correr 4    # más capas para que el cálculo pese más que la comunicación
```

Resultados del benchmark:
- `benchmark_results.csv`: datos crudos.
- `benchmark_summary.csv`: T medio, desviación, speedup y eficiencia.
- `benchmark_speedup.png`: gráfica.

Para pasarlos a Windows:

```bash
cp benchmark_* /mnt/d/
```

## 6. Qué esperar

Con una VPN (latencia de 5 a 50 ms) y N_seq = 128, la comunicación (`MPI_Allgather`) pesa más que el cálculo, así que **el speedup puede salir menor que 1**. Es un resultado válido y se explica con la Ley de Amdahl (manual, sección 8.8). Para mejorar la relación entre cálculo y comunicación:

```bash
python3 generate_matrices.py --out-dir matrices --n-seq 512   # regenerar las matrices
./cluster.sh sync
LAYERS=8 ./cluster.sh benchmark
```

Compara también la columna `comm(Allgather)` con `compute` en la salida de cada repetición.
