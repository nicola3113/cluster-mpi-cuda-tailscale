#!/usr/bin/env python3
"""
generate_matrices.py

Genera, de forma reproducible, las matrices numericas de UNA capa Transformer
con dimensiones realistas tipo "GPT-2 Small" (d_model=768, d_ff=3072,
vocab_size=50257) y las guarda como archivos binarios independientes que el
programa llm_mpi_cuda_gpt2small.cu lee directamente (no hardcodeadas en el
codigo fuente, a diferencia del caso ilustrativo anterior con d_model=4).

IMPORTANTE -- alcance de este generador:
  Los pesos (WQ, WK, WV, W_up, W_down, W_vocab) se inicializan con la MISMA
  distribucion normal(0, 0.02) que usa la inicializacion original de GPT-2
  (Radford et al., 2019) -- esto reproduce el TAMANO y la ESCALA NUMERICA
  real de un modelo GPT-2 Small recien inicializado, pero el modelo NO esta
  entrenado: el token "ganador" que prediga el programa es, por diseno,
  linguisticamente arbitrario. El objetivo de este caso de estudio NO es
  precision linguistica (eso ya se demostro con el caso ilustrativo de la
  seccion 3 del manual) sino medir, a la escala real de computo y
  comunicacion de un modelo de produccion, el SPEEDUP y la EFICIENCIA del
  paralelismo de tensores al distribuir la capa FFN entre 1, 2 o mas
  computadoras con GPU. Ver manual, seccion 8.

Formato binario de cada archivo .bin (little-endian, el nativo de x86/ARM):
    int32  rows
    int32  cols
    float32[rows*cols]   datos en orden row-major (C order)

Uso:
    python3 generate_matrices.py --out-dir ../matrices --n-seq 128 --seed 42

    (valores por defecto: d_model=768 d_ff=3072 vocab=50257 n_seq=128 seed=42,
     equivalentes a GPT-2 Small; se pueden cambiar via argumentos para
     experimentar con otras longitudes de secuencia)
"""
import argparse
import json
import os
import struct
import time

import numpy as np


def save_bin(path, mat):
    """Guarda una matriz 2D float32 en el formato binario descrito arriba."""
    mat = np.ascontiguousarray(mat, dtype=np.float32)
    rows, cols = mat.shape
    with open(path, "wb") as f:
        f.write(struct.pack("<ii", rows, cols))
        f.write(mat.tobytes(order="C"))
    return os.path.getsize(path)


def load_bin(path):
    """Lee una matriz guardada por save_bin (usado por design_gpt2small.py)."""
    with open(path, "rb") as f:
        rows, cols = struct.unpack("<ii", f.read(8))
        data = np.frombuffer(f.read(rows * cols * 4), dtype=np.float32)
        return data.reshape(rows, cols).copy()


def sinusoidal_positional_encoding(n_seq, d_model):
    """Formula sinusoidal real de Vaswani et al. (2017), identica en forma a
    la usada en el caso ilustrativo del manual (seccion 2.2), generalizada a
    d_model par arbitrario (768 es par, por lo que no requiere ajuste)."""
    pe = np.zeros((n_seq, d_model), dtype=np.float32)
    position = np.arange(n_seq, dtype=np.float64)[:, None]
    div_term = np.exp(np.arange(0, d_model, 2, dtype=np.float64) * -(np.log(10000.0) / d_model))
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    return pe.astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default="../matrices", help="Directorio de salida para los .bin (default: ../matrices)")
    ap.add_argument("--n-seq", type=int, default=128, help="Longitud de la secuencia de entrada (default: 128)")
    ap.add_argument("--d-model", type=int, default=768, help="Dimension oculta / embedding (default: 768, GPT-2 Small)")
    ap.add_argument("--d-ff", type=int, default=3072, help="Dimension intermedia de la FFN (default: 3072, GPT-2 Small = 4x d_model)")
    ap.add_argument("--vocab-size", type=int, default=50257, help="Tamano del vocabulario (default: 50257, tokenizer BPE de GPT-2)")
    ap.add_argument("--seed", type=int, default=42, help="Semilla para reproducibilidad (default: 42)")
    ap.add_argument("--init-std", type=float, default=0.02, help="Desviacion estandar de la inicializacion normal de los pesos (default: 0.02, igual que GPT-2 original)")
    args = ap.parse_args()

    if args.d_model % 2 != 0:
        raise SystemExit("ERROR: d_model debe ser par (requerido por la formula de Positional Encoding seno/coseno).")

    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    def W(shape):
        return rng.normal(0.0, args.init_std, size=shape).astype(np.float32)

    t0 = time.time()
    N, d, d_ff, V = args.n_seq, args.d_model, args.d_ff, args.vocab_size

    # a) Embedding semantico E: en un modelo real provendria de una tabla de
    #    embedding (vocab_size x d_model) indexada por los Token IDs del
    #    prompt; aqui se genera directamente como si ya fuera el resultado de
    #    ese lookup (igual simplificacion que uso design.py en el caso toy).
    E = W((N, d))

    # b) Positional Encoding -- determinista, NO aleatorio.
    PE = sinusoidal_positional_encoding(N, d)

    # c) Pesos de proyeccion de atencion (una sola cabeza "ancha" de d
    #    dimensiones completas -- ver manual 8.3 sobre la equivalencia de
    #    FLOPs con multi-head real).
    WQ = W((d, d))
    WK = W((d, d))
    WV = W((d, d))

    # d) Pesos de la FFN -- estas son las matrices que se fragmentan por
    #    columnas entre los nodos del cluster (Paralelismo de Tensores).
    W_up = W((d, d_ff))
    W_down = W((d_ff, d))

    # e) Un-embedding al vocabulario completo.
    W_vocab = W((d, V))

    sizes = {}
    sizes["E.bin"] = save_bin(os.path.join(args.out_dir, "E.bin"), E)
    sizes["PE.bin"] = save_bin(os.path.join(args.out_dir, "PE.bin"), PE)
    sizes["WQ.bin"] = save_bin(os.path.join(args.out_dir, "WQ.bin"), WQ)
    sizes["WK.bin"] = save_bin(os.path.join(args.out_dir, "WK.bin"), WK)
    sizes["WV.bin"] = save_bin(os.path.join(args.out_dir, "WV.bin"), WV)
    sizes["Wup.bin"] = save_bin(os.path.join(args.out_dir, "Wup.bin"), W_up)
    sizes["Wdown.bin"] = save_bin(os.path.join(args.out_dir, "Wdown.bin"), W_down)
    sizes["Wvocab.bin"] = save_bin(os.path.join(args.out_dir, "Wvocab.bin"), W_vocab)

    manifest = {
        "description": "Matrices de una capa Transformer a escala GPT-2 Small para benchmarking de paralelismo de tensores MPI+CUDA",
        "seed": args.seed,
        "init_std": args.init_std,
        "trained": False,
        "shapes": {
            "E": [N, d], "PE": [N, d], "WQ": [d, d], "WK": [d, d], "WV": [d, d],
            "W_up": [d, d_ff], "W_down": [d_ff, d], "W_vocab": [d, V],
        },
        "hyperparams": {"n_seq": N, "d_model": d, "d_ff": d_ff, "vocab_size": V},
        "file_sizes_bytes": sizes,
        "total_size_mb": round(sum(sizes.values()) / (1024 * 1024), 2),
        "format": "little-endian: int32 rows, int32 cols, then rows*cols float32 (row-major)",
    }
    with open(os.path.join(args.out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    dt = time.time() - t0
    print(f"Matrices generadas en {args.out_dir}/ en {dt:.2f}s")
    print(f"  N_seq={N}  d_model={d}  d_ff={d_ff}  vocab_size={V}  (seed={args.seed})")
    for name, sz in sizes.items():
        print(f"  {name:12s} {sz/1024/1024:8.2f} MB")
    print(f"  TOTAL        {manifest['total_size_mb']:8.2f} MB")
    print(f"  manifest.json escrito")


if __name__ == "__main__":
    main()
