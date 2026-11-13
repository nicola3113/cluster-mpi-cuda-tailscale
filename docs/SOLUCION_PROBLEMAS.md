# Solución de problemas

## En el nodo (compañero)

| Síntoma | Causa | Solución |
|---|---|---|
| `nvidia-smi` falla dentro de Ubuntu | Driver de Windows viejo, o se instaló un driver dentro de Linux | Actualiza el driver desde nvidia.com **en Windows** y ejecuta `wsl --shutdown`. No instales `nvidia-driver-*` en Ubuntu. |
| `Falta Ubuntu-22.04` | Tienes otra versión (24.04, etc.) | Instala 22.04 aparte: `wsl --install -d Ubuntu-22.04 --location D:\WSL\Ubuntu-22.04` |
| Aviso de OpenSSH Server de Windows | Ocupa el puerto 22 que necesita Ubuntu | `Stop-Service sshd; Set-Service sshd -StartupType Manual` |
| `No se puede cargar ... la ejecución de scripts está deshabilitada` | Política de ejecución | `Set-ExecutionPolicy -Scope Process Bypass -Force` en la misma ventana |
| `ip -4 addr` en Ubuntu no muestra ninguna `100.x` | La red espejo no se aplicó, o Tailscale está desconectado | `wsl --shutdown`, abre Tailscale, vuelve a abrir Ubuntu. Revisa `%UserProfile%\.wslconfig` |
| `New-NetFirewallHyperVRule` no existe | Windows 10 o WSL antiguo | Actualiza WSL con `wsl --update`. En Windows 10 la red *mirrored* no existe: usa Windows 11 |
| El nodo "desaparece" | WSL se apagó por inactividad | Deja una ventana de Ubuntu abierta |

Puedes volver a ejecutar `unirse_cluster.ps1` todas las veces que necesites; no rompe nada.

## En el maestro

| Síntoma | Solución |
|---|---|
| `./cluster.sh descubrir` no encuentra nada | El compañero tiene que aparecer en `tailscale status` como `nodo-…` y en línea. Si no, agrégalo a mano con `./cluster.sh agregar juan 100.x.y.z` |
| `estado` → `FALLA: Connection timed out` | Su Ubuntu está cerrado, su firewall no se abrió (que vuelva a ejecutar el script) o Tailscale está desconectado. Prueba desde Windows con `tailscale ping nodo-juan` |
| `estado` → `Permission denied (publickey)` | El nodo no tiene tu clave: debe volver a ejecutar el script con la `-ClaveMaestro` correcta (la de `scripts/clave_maestro.pub`) |
| `mpirun` se queda colgado al iniciar | Puertos 10000-10400 bloqueados en algún equipo, o mezcla de interfaces. Revisa que `~/.openmpi/mca-params.conf` tenga `100.64.0.0/10` en el maestro y en los nodos |
| `ERROR: no se pudo abrir 'matrices/E.bin'` en un rank | Falta ejecutar `./cluster.sh sync` |
| `no kernel image is available for execution on the device` | GPU muy nueva o muy antigua: en el maestro, agrega su `-gencode arch=compute_XX,code=sm_XX` en `setup_maestro.sh` y recompila |
| `d_ff no divisible` | Usa un número de procesos que divida a 3072: 1, 2, 3, 4, 6, 8, 12, 16 |
| Versiones distintas de OpenMPI en `estado` | Todos deben usar Ubuntu-22.04 (OpenMPI 4.1.2) |

## Pruebas útiles

```bash
# Desde el maestro (como mpiuser):
ssh mpiuser@100.x.y.z hostname                                   # SSH sin contraseña
mpirun --hostfile hosts.txt -np 2 hostname                       # MPI básico entre equipos
mpirun --hostfile hosts.txt -np 2 ./validate_cpu_mpi_gpt2small   # arnés CPU (no usa GPU)
```

```powershell
# Desde Windows:
tailscale status
tailscale ping nodo-juan     # dice si la conexión es directa o por relay (DERP, más lento)
```
