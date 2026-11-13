#!/usr/bin/env bash
#
# run_benchmark.sh
#
# Ejecuta llm_mpi_cuda_gpt2small en el CLUSTER REAL del usuario (1 o mas
# computadoras WSL2, cada una con su propia GPU -- ver manual, seccion 5,
# para la configuracion del cluster y seccion 8 para la metodologia de
# benchmarking completa), variando el numero de procesos MPI (-np), y
# acumula los resultados en un unico CSV para que parse_benchmark.py
# calcule speedup y eficiencia.
#
# IMPORTANTE: este script se ejecuta DESDE EL NODO 0 (nodo1) del cluster
# real. El directorio de matrices/ y el ejecutable deben existir, IDENTICOS,
# en todos los nodos listados en hosts.txt (ver manual, seccion 5.6 y 8.4).
#
# Uso:
#   ./run_benchmark.sh
#   NP_VALUES="1 2" EXTERNAL_RUNS=5 ./run_benchmark.sh
#
# Variables de entorno configurables (todas opcionales, con valores por
# defecto razonables para un cluster de 2 nodos):
#   EXECUTABLE      Ruta al binario compilado (default: ../code/llm_mpi_cuda_gpt2small)
#   HOSTFILE        Archivo de hosts de MPI (default: hosts.txt, ver manual 5.8)
#   MATRICES_DIR    Directorio con los .bin generados (default: ../matrices)
#   NP_VALUES       Lista de valores -np a probar, separados por espacio
#                    (default: "1 2" -- 1 computadora vs. 2 computadoras;
#                    amplie a "1 2 4" etc. si su cluster tiene mas nodos/GPUs
#                    y d_ff es divisible entre esos valores)
#   LAYERS          Numero de capas simuladas por corrida (default: 1)
#   INTERNAL_REPEATS Repeticiones internas dentro de cada mpirun (default: 5,
#                    ver --repeats del programa)
#   EXTERNAL_RUNS   Veces que se vuelve a invocar mpirun DESDE CERO por cada
#                    valor de -np, para capturar tambien el overhead real de
#                    lanzamiento/red entre nodos (default: 3)
#   RESULTS_CSV     Archivo CSV de salida (default: benchmark_results.csv)
#
set -euo pipefail

EXECUTABLE="${EXECUTABLE:-../code/llm_mpi_cuda_gpt2small}"
HOSTFILE="${HOSTFILE:-hosts.txt}"
MATRICES_DIR="${MATRICES_DIR:-../matrices}"
NP_VALUES="${NP_VALUES:-1 2}"
LAYERS="${LAYERS:-1}"
INTERNAL_REPEATS="${INTERNAL_REPEATS:-5}"
EXTERNAL_RUNS="${EXTERNAL_RUNS:-3}"
RESULTS_CSV="${RESULTS_CSV:-benchmark_results.csv}"

if [ ! -x "$EXECUTABLE" ]; then
  echo "ERROR: no se encontro el ejecutable '$EXECUTABLE' (compilelo primero, ver manual seccion 5.7)." >&2
  exit 1
fi
if [ ! -d "$MATRICES_DIR" ]; then
  echo "ERROR: no se encontro el directorio de matrices '$MATRICES_DIR' (ejecute generate_matrices.py primero, ver manual 8.4)." >&2
  exit 1
fi

echo "======================================================================="
echo " BENCHMARKING: llm_mpi_cuda_gpt2small"
echo " -np a probar       : $NP_VALUES"
echo " capas simuladas    : $LAYERS"
echo " repeticiones internas (por mpirun): $INTERNAL_REPEATS"
echo " repeticiones externas (relanzando mpirun): $EXTERNAL_RUNS"
echo " CSV de resultados  : $RESULTS_CSV"
echo "======================================================================="

for NP in $NP_VALUES; do
  for EXT in $(seq 1 "$EXTERNAL_RUNS"); do
    echo ""
    echo "--- -np=$NP   corrida externa $EXT/$EXTERNAL_RUNS ---"
    if [ "$NP" -eq 1 ]; then
      # Con -np 1 no es necesario el hostfile (una sola computadora/GPU).
      mpirun -np 1 "$EXECUTABLE" \
        --matrices-dir "$MATRICES_DIR" --layers "$LAYERS" \
        --repeats "$INTERNAL_REPEATS" --results-csv "$RESULTS_CSV"
    else
      mpirun --hostfile "$HOSTFILE" -np "$NP" "$EXECUTABLE" \
        --matrices-dir "$MATRICES_DIR" --layers "$LAYERS" \
        --repeats "$INTERNAL_REPEATS" --results-csv "$RESULTS_CSV"
    fi
  done
done

echo ""
echo "======================================================================="
echo " Benchmarking completo. Resultados acumulados en: $RESULTS_CSV"
echo " Siguiente paso: python3 parse_benchmark.py --csv $RESULTS_CSV"
echo "======================================================================="
