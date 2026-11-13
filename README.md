# Cluster MPI + CUDA sobre Tailscale (Semana 12 — PCyP)

Inferencia de una capa Transformer a escala **GPT-2 Small** (d_model = 768, d_ff = 3072, vocab = 50 257) con **paralelismo de tensores**. La capa FFN se reparte por columnas entre varias laptops con GPU NVIDIA y los resultados se juntan con `MPI_Allgather`. El código se mide para obtener **speedup** y **eficiencia** según la cantidad de GPUs.

Las laptops no tienen que estar en la misma red: se conectan por la VPN **Tailscale**. Cada equipo corre Ubuntu-22.04 en WSL2 con red espejo (*mirrored*), así que Ubuntu usa directamente la IP 100.x de Tailscale.

```
             Tailscale (VPN, 100.64.0.0/10)
 ┌────────────────────┐   SSH :22 + MPI :10000-10400   ┌──────────────────┐
 │ MAESTRO (rank 0)   │ ─────────────────────────────▶ │ nodo-juan (r1)   │
 │ Windows + WSL2     │ ─────────────────────────────▶ │ nodo-ana  (r2)   │
 │ CUDA Toolkit, nvcc │ ◀──── MPI_Allgather (FFN) ───▶ │ nodo-...  (rN)   │
 │ ~/llm_cluster      │                                │ solo driver + MPI│
 └────────────────────┘                                └──────────────────┘
```

## Contenido

| Carpeta | Qué hay |
|---|---|
| `codigo/` | Programa CUDA+MPI (`llm_mpi_cuda_gpt2small.cu`), arnés de validación en CPU, generador de matrices y scripts de benchmark del curso |
| `scripts/unirse_cluster.ps1` | **Script único** para Windows: deja la PC como `maestro` o como `nodo` |
| `scripts/setup_maestro.sh` | Lo llama el script anterior. Instala CUDA 12.6 + OpenMPI, compila y genera las matrices |
| `scripts/setup_nodo.sh` | Lo llama el script anterior. Instala OpenMPI + SSH y autoriza la clave del maestro |
| `scripts/cluster.sh` | Herramienta del maestro: descubrir nodos, sincronizar, ejecutar, benchmark |
| `docs/GUIA_MAESTRO.md` | Paso a paso para quien será el **maestro** |
| `docs/GUIA_NODOS.md` | Paso a paso para los **compañeros que se conectan** (el texto que les mandas) |
| `docs/SOLUCION_PROBLEMAS.md` | Errores comunes y cómo arreglarlos |
| `docs/*.docx, *.pptx` | Manual técnico y presentación de la clase |

## Resumen rápido

1. **El maestro** (una sola PC) sigue [docs/GUIA_MAESTRO.md](docs/GUIA_MAESTRO.md):
   - genera una *auth key* de Tailscale;
   - ejecuta `unirse_cluster.ps1 -Rol maestro`;
   - comparte `scripts/clave_maestro.pub` y la auth key.
2. **Cada compañero** sigue [docs/GUIA_NODOS.md](docs/GUIA_NODOS.md):
   - ejecuta `unirse_cluster.ps1 -Rol nodo`;
   - le manda al maestro la línea `nodo-nombre 100.x.y.z`.
3. **El maestro** abre Ubuntu (`wsl -d Ubuntu-22.04 -u mpiuser`) y ejecuta:
   ```bash
   cd ~/llm_cluster
   ./cluster.sh descubrir && ./cluster.sh estado && ./cluster.sh sync
   ./cluster.sh correr        # una ejecución con todas las GPUs
   ./cluster.sh benchmark     # speedup / eficiencia -> benchmark_summary.csv y benchmark_speedup.png
   ```

## Requisitos (todas las PCs)

- Windows 10 22H2+ o Windows 11, con WSL ≥ 2.0. Para la red *mirrored* se recomienda Windows 11 23H2+.
- GPU NVIDIA con driver de Windows reciente (≥ 560). **No** instales drivers NVIDIA dentro de Ubuntu.
- **Ubuntu-22.04** en WSL2, la misma en todos los equipos, para tener OpenMPI 4.1.2 igual en todo el cluster.
- Espacio: maestro ≈ 6 GB (CUDA Toolkit) y nodo ≈ 1 GB. Se recomienda instalar la distro en `D:\WSL\`.

## Decisiones de diseño

- **Los nodos no necesitan CUDA Toolkit.** El maestro compila con `-cudart static` y genera código para sm_61, sm_75, sm_86 y sm_89, más PTX para compute_89. Así funciona en GTX 10xx, GTX 16xx, RTX 20/30/40, y las RTX 50 compilan el PTX al vuelo (JIT). El maestro copia el binario a cada nodo con `rsync`.
- **Solo el maestro abre conexiones SSH** (`plm_rsh_no_tree_spawn = 1`). Por eso basta con autorizar la clave del maestro en cada nodo y los nodos no necesitan claves entre sí.
- **OpenMPI usa solo la interfaz Tailscale** (`btl/oob_tcp_if_include = 100.64.0.0/10`) y un rango de puertos fijo (10000-10400), que es el que se abre en el firewall de Windows y en el de Hyper-V (WSL).
- **Las matrices no se envían por MPI.** Cada nodo lee su copia local. `Wvocab.bin` (147 MB) no se copia a los nodos porque solo la usa el rank 0.
- **`d_ff = 3072` debe ser divisible entre la cantidad de procesos:** sirven 1, 2, 3, 4, 6, 8, 12, 16… `cluster.sh` lo valida.

## Seguridad

- **Nunca subas la auth key de Tailscale (`tskey-auth-…`) al repo.** Pásala por privado y revócala al terminar (admin console → Settings → Keys → Revoke).
- `clave_maestro.pub` es una clave **pública**, así que se puede compartir sin problema. La privada nunca sale de la PC del maestro.
- Los nodos solo aceptan SSH con clave (`PasswordAuthentication no`) y con el usuario `mpiuser`.
