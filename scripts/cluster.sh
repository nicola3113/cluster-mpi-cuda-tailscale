#!/usr/bin/env bash
# cluster.sh -- administra el cluster MPI+CUDA sobre Tailscale DESDE EL MAESTRO.
# Vive en ~/llm_cluster (WSL Ubuntu-22.04, usuario mpiuser).
#
#   ./cluster.sh descubrir          agrega los equipos "nodo-*" que esten online en Tailscale
#   ./cluster.sh agregar NOMBRE IP  agrega un nodo a mano (IP 100.x de Tailscale)
#   ./cluster.sh quitar NOMBRE      lo saca de la lista
#   ./cluster.sh estado             prueba SSH + GPU de cada nodo
#   ./cluster.sh sync               copia binario + matrices a todos los nodos
#   ./cluster.sh hostfile           genera hosts.txt (maestro + nodos que responden)
#   ./cluster.sh correr [N]         mpirun con N procesos (1 por GPU/equipo)
#   ./cluster.sh benchmark          run_benchmark.sh + parse_benchmark.py con todos los N validos
set -uo pipefail
cd "$(dirname "$0")"

TS="/mnt/c/Program Files/Tailscale/tailscale.exe"
NODOS=nodos.txt
BIN=llm_mpi_cuda_gpt2small
D_FF=3072
touch "$NODOS"

ssh_ok() { ssh -o BatchMode=yes -o ConnectTimeout=6 "mpiuser@$1" true 2>/dev/null; }

case "${1:-ayuda}" in
descubrir)
  "$TS" status --json | python3 -c '
import json,sys
d=json.load(sys.stdin)
for p in (d.get("Peer") or {}).values():
    name=(p.get("DNSName") or p.get("HostName","")).split(".")[0].lower()
    if name.startswith("nodo-") and p.get("Online") and p.get("TailscaleIPs"):
        ip=[i for i in p["TailscaleIPs"] if "." in i][0]
        print(name, ip)
' | while read -r n ip; do
      sed -i "/^$n /d" "$NODOS"; echo "$n $ip" >> "$NODOS"; echo "+ $n $ip"
    done
  echo "Lista actual:"; cat "$NODOS" ;;
agregar)
  sed -i "/^$2 /d" "$NODOS"; echo "$2 $3" >> "$NODOS"; cat "$NODOS" ;;
quitar)
  sed -i "/^$2 /d" "$NODOS"; cat "$NODOS" ;;
estado)
  printf "%-22s %-16s %s\n" master "$(hostname -I | tr ' ' '\n' | grep '^100\.' | head -1)" \
    "$(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader)"
  while read -r n ip; do
    [ -z "$n" ] && continue
    if out=$(ssh -o BatchMode=yes -o ConnectTimeout=6 "mpiuser@$ip" \
        'echo "$(hostname) | $(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader 2>&1) | $(mpirun --version | head -1)"' 2>&1); then
      printf "%-22s %-16s OK  %s\n" "$n" "$ip" "$out"
    else
      printf "%-22s %-16s FALLA: %s\n" "$n" "$ip" "$(echo "$out" | tail -1)"
    fi
  done < "$NODOS" ;;
sync)
  while read -r n ip; do
    [ -z "$n" ] && continue
    echo "== $n ($ip)"
    rsync -az --exclude Wvocab.bin "$BIN" validate_cpu_mpi_gpt2small matrices "mpiuser@$ip:llm_cluster/" \
      && echo "   ok" || echo "   FALLA (revisar ./cluster.sh estado)"
  done < "$NODOS" ;;
hostfile)
  echo "master slots=1" > hosts.txt
  while read -r n ip; do
    [ -z "$n" ] && continue
    if ssh_ok "$ip"; then echo "$ip slots=1   # $n" >> hosts.txt; else echo "omitido $n ($ip): no responde"; fi
  done < "$NODOS"
  cat hosts.txt
  echo "Total GPUs/equipos: $(grep -c slots hosts.txt)" ;;
correr)
  [ -f hosts.txt ] || "$0" hostfile
  TOTAL=$(grep -c slots hosts.txt); N="${2:-$TOTAL}"
  if (( D_FF % N != 0 )); then echo "N=$N no divide a d_ff=$D_FF. Usa 1,2,3,4,6,8,12,16..."; exit 1; fi
  if (( N > TOTAL )); then echo "Solo hay $TOTAL equipos en hosts.txt"; exit 1; fi
  mpirun --hostfile hosts.txt -np "$N" --map-by node "./$BIN" \
    --matrices-dir matrices --layers "${LAYERS:-1}" --repeats "${REPEATS:-5}" --results-csv benchmark_results.csv ;;
benchmark)
  "$0" hostfile >/dev/null
  TOTAL=$(grep -c slots hosts.txt); NPS=""
  for ((k=1; k<=TOTAL; k++)); do (( D_FF % k == 0 )) && NPS="$NPS $k"; done
  echo "Probando -np:$NPS"
  rm -f benchmark_results.csv
  EXECUTABLE=./$BIN MATRICES_DIR=matrices HOSTFILE=hosts.txt NP_VALUES="$NPS" ./run_benchmark.sh
  python3 parse_benchmark.py --csv benchmark_results.csv ;;
*)
  sed -n '2,13p' "$0" ;;
esac
