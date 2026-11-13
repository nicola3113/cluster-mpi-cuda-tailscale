#!/usr/bin/env python3
"""
parse_benchmark.py

Lee el benchmark_results.csv producido por llm_mpi_cuda_gpt2small (via
run_benchmark.sh) en el cluster real del usuario, calcula tiempo medio,
Speedup y Eficiencia por numero de procesos MPI (= numero de computadoras
con GPU usadas), y genera una tabla resumen (texto + CSV) y un grafico PNG.
Ver manual, seccion 8.7, para la formula y la interpretacion de estos
numeros.

  Speedup(N)   = T(1) / T(N)
  Eficiencia(N) = Speedup(N) / N

Uso:
    python3 parse_benchmark.py --csv benchmark_results.csv
    python3 parse_benchmark.py --csv benchmark_results.csv --out resumen.csv --plot speedup.png
"""
import argparse
import csv
import statistics
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="benchmark_results.csv", help="CSV de entrada escrito por el programa (default: benchmark_results.csv)")
    ap.add_argument("--out", default="benchmark_summary.csv", help="CSV resumen de salida (default: benchmark_summary.csv)")
    ap.add_argument("--plot", default="benchmark_speedup.png", help="Archivo PNG del grafico (default: benchmark_speedup.png)")
    ap.add_argument("--no-plot", action="store_true", help="No generar el grafico (por si matplotlib no esta instalado)")
    args = ap.parse_args()

    by_ranks = defaultdict(list)
    meta = {}
    with open(args.csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            n = int(row["num_ranks"])
            by_ranks[n].append(float(row["total_time_s"]))
            meta = row  # nos quedamos con la ultima fila para imprimir la config

    if not by_ranks:
        raise SystemExit(f"'{args.csv}' no tiene filas de datos. Ejecute run_benchmark.sh primero.")

    if 1 not in by_ranks:
        print("ADVERTENCIA: no hay datos para -np=1 (1 computadora); no se puede calcular Speedup/Eficiencia "
              "sin una linea base. Ejecute tambien con -np 1 (ver manual 8.7).")

    ns_sorted = sorted(by_ranks.keys())
    t1_mean = statistics.mean(by_ranks[1]) if 1 in by_ranks else None

    print("=======================================================================")
    print(f" RESUMEN DE DESEMPENO -- {meta.get('n_seq','?')} tokens, d_model={meta.get('d_model','?')}, "
          f"d_ff={meta.get('d_ff','?')}, vocab={meta.get('vocab_size','?')}, capas={meta.get('num_layers','?')}")
    print("=======================================================================")
    header = f"{'N (procesos/GPUs)':>18s} | {'muestras':>8s} | {'T medio (s)':>12s} | {'desv.est (s)':>12s} | {'Speedup':>8s} | {'Eficiencia':>10s}"
    print(header)
    print("-" * len(header))

    rows_out = []
    for n in ns_sorted:
        times = by_ranks[n]
        t_mean = statistics.mean(times)
        t_std = statistics.pstdev(times) if len(times) > 1 else 0.0
        speedup = (t1_mean / t_mean) if t1_mean else float("nan")
        efficiency = (speedup / n) if t1_mean else float("nan")
        print(f"{n:>18d} | {len(times):>8d} | {t_mean:>12.4f} | {t_std:>12.4f} | {speedup:>8.3f} | {efficiency*100:>9.1f}%")
        rows_out.append({
            "num_ranks": n, "samples": len(times), "mean_time_s": round(t_mean, 6),
            "stdev_time_s": round(t_std, 6), "speedup": round(speedup, 4) if t1_mean else "",
            "efficiency": round(efficiency, 4) if t1_mean else "",
        })

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["num_ranks", "samples", "mean_time_s", "stdev_time_s", "speedup", "efficiency"])
        w.writeheader()
        w.writerows(rows_out)
    print(f"\nTabla resumen escrita en: {args.out}")

    if not args.no_plot and t1_mean:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
            speedups = [r["speedup"] for r in rows_out]
            efficiencies = [r["efficiency"] * 100 for r in rows_out]

            ax1.plot(ns_sorted, speedups, "o-", color="#2E74B5", label="Speedup medido")
            ax1.plot(ns_sorted, ns_sorted, "--", color="#999999", label="Speedup ideal (lineal)")
            ax1.set_xlabel("N (procesos MPI / GPUs)")
            ax1.set_ylabel("Speedup")
            ax1.set_title("Speedup vs. N")
            ax1.legend()
            ax1.grid(alpha=0.3)

            ax2.plot(ns_sorted, efficiencies, "o-", color="#2E7D32")
            ax2.axhline(100, linestyle="--", color="#999999")
            ax2.set_xlabel("N (procesos MPI / GPUs)")
            ax2.set_ylabel("Eficiencia (%)")
            ax2.set_title("Eficiencia vs. N")
            ax2.set_ylim(0, 115)
            ax2.grid(alpha=0.3)

            fig.suptitle("Paralelismo de Tensores MPI+CUDA -- resultados medidos en el cluster real")
            fig.tight_layout()
            fig.savefig(args.plot, dpi=150)
            print(f"Grafico escrito en: {args.plot}")
        except ImportError:
            print("(matplotlib no esta instalado -- omitido el grafico; la tabla CSV sigue siendo valida. "
                  "Instalar con: pip install matplotlib)")


if __name__ == "__main__":
    main()
