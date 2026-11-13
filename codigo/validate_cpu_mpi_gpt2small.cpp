/*
 * validate_cpu_mpi_gpt2small.cpp
 *
 * Arnes de validacion EMPIRICA (no simulada) de la logica MPI del programa
 * llm_mpi_cuda_gpt2small.cu, a la escala real GPT-2 Small (d_model=768,
 * d_ff=3072, vocab_size=50257), generalizando el mismo arnes que ya se uso
 * para el caso ilustrativo (validate_cpu_mpi.cpp, manual Anexo B).
 *
 * Reimplementa en CPU, con la MISMA estructura algoritmica y el mismo orden
 * de operaciones que los kernels CUDA del .cu real (incluyendo la
 * fragmentacion de columnas de W_up, el MPI_Allgather, y la reconstruccion
 * EXPLICITA de la matriz global), leyendo las matrices de los MISMOS
 * archivos .bin que usa el programa CUDA real (generate_matrices.py).
 *
 * Objetivo: demostrar, con MPI real compilado y ejecutado (no un simulador),
 * que el resultado final NO depende del numero de procesos MPI usado
 * (siempre que -np divida a d_ff), tambien a esta escala realista -- y que
 * valores invalidos de -np se rechazan de forma segura. Ver manual,
 * seccion 8.6.
 *
 * Compilar:  mpicxx -O2 -o validate_cpu_mpi_gpt2small validate_cpu_mpi_gpt2small.cpp
 * Ejecutar:  mpirun -np <N> --oversubscribe ./validate_cpu_mpi_gpt2small --matrices-dir ../matrices_validate --layers 1
 *
 * NOTA: se recomienda usar un directorio de matrices con N_seq pequeno
 * (p.ej. 16, ver matrices_validate/ generado con --n-seq 16) para que esta
 * validacion de CORRECCION (no de desempeno) se ejecute en segundos en CPU;
 * el benchmarking de DESEMPENO real se hace con el programa CUDA en el
 * cluster del usuario (manual, seccion 8.7), no con este arnes.
 */
#include <mpi.h>
#include <vector>
#include <string>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <fstream>
#include <algorithm>

struct HostMatrix {
    int rows = 0, cols = 0;
    std::vector<float> data;
};

static int g_rank = -1;

HostMatrix read_bin_matrix(const std::string& path) {
    HostMatrix m;
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        std::fprintf(stderr, "[Rank %d] ERROR: no se pudo abrir '%s'. Ejecute generate_matrices.py primero.\n", g_rank, path.c_str());
        MPI_Abort(MPI_COMM_WORLD, 4);
    }
    int32_t rows = 0, cols = 0;
    f.read(reinterpret_cast<char*>(&rows), sizeof(int32_t));
    f.read(reinterpret_cast<char*>(&cols), sizeof(int32_t));
    m.rows = rows; m.cols = cols;
    m.data.resize((size_t)rows * (size_t)cols);
    f.read(reinterpret_cast<char*>(m.data.data()), (std::streamsize)(m.data.size() * sizeof(float)));
    if (!f) {
        std::fprintf(stderr, "[Rank %d] ERROR: '%s' truncado.\n", g_rank, path.c_str());
        MPI_Abort(MPI_COMM_WORLD, 4);
    }
    return m;
}

std::string get_arg(int argc, char** argv, const std::string& flag, const std::string& def) {
    for (int i = 1; i < argc - 1; ++i) if (flag == argv[i]) return std::string(argv[i + 1]);
    return def;
}

std::vector<float> matmul(const std::vector<float>& A, const float* B, int M, int N, int K) {
    std::vector<float> C((size_t)M * N, 0.0f);
    for (int r = 0; r < M; ++r)
        for (int c = 0; c < N; ++c) {
            float s = 0.0f;
            for (int k = 0; k < K; ++k) s += A[(size_t)r * K + k] * B[(size_t)k * N + c];
            C[(size_t)r * N + c] = s;
        }
    return C;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, num_ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);
    g_rank = rank;

    std::string matrices_dir = get_arg(argc, argv, "--matrices-dir", "../matrices_validate");
    int num_layers = std::atoi(get_arg(argc, argv, "--layers", "1").c_str());
    if (num_layers < 1) num_layers = 1;

    HostMatrix E  = read_bin_matrix(matrices_dir + "/E.bin");
    HostMatrix PE = read_bin_matrix(matrices_dir + "/PE.bin");
    HostMatrix WQ = read_bin_matrix(matrices_dir + "/WQ.bin");
    HostMatrix WK = read_bin_matrix(matrices_dir + "/WK.bin");
    HostMatrix WV = read_bin_matrix(matrices_dir + "/WV.bin");
    HostMatrix W_up = read_bin_matrix(matrices_dir + "/Wup.bin");
    HostMatrix W_down = read_bin_matrix(matrices_dir + "/Wdown.bin");

    const int N_seq = E.rows, d_model = E.cols, d_ff = W_up.cols;

    if (d_ff % num_ranks != 0) {
        if (rank == 0) std::fprintf(stderr, "ERROR: d_ff(%d) no divisible entre -np %d -- ABORTANDO (comportamiento esperado)\n", d_ff, num_ranks);
        MPI_Abort(MPI_COMM_WORLD, 3);
    }

    HostMatrix W_vocab;
    int vocab_size = 0;
    if (rank == 0) {
        W_vocab = read_bin_matrix(matrices_dir + "/Wvocab.bin");
        vocab_size = W_vocab.cols;
    }
    MPI_Bcast(&vocab_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // X = E + PE
    std::vector<float> X((size_t)N_seq * d_model);
    for (size_t i = 0; i < X.size(); ++i) X[i] = E.data[i] + PE.data[i];

    const int d_ff_local = d_ff / num_ranks;
    std::vector<float> Wup_local((size_t)d_model * d_ff_local);
    for (int r = 0; r < d_model; ++r)
        for (int c = 0; c < d_ff_local; ++c) {
            int gc = rank * d_ff_local + c;
            Wup_local[(size_t)r * d_ff_local + c] = W_up.data[(size_t)r * d_ff + gc];
        }

    for (int layer = 0; layer < num_layers; ++layer) {
        std::vector<float> Q = matmul(X, WQ.data.data(), N_seq, d_model, d_model);
        std::vector<float> K = matmul(X, WK.data.data(), N_seq, d_model, d_model);
        std::vector<float> V = matmul(X, WV.data.data(), N_seq, d_model, d_model);

        std::vector<float> S((size_t)N_seq * N_seq), P((size_t)N_seq * N_seq, 0.0f);
        for (int r = 0; r < N_seq; ++r)
            for (int c = 0; c < N_seq; ++c) {
                if (c > r) { S[(size_t)r * N_seq + c] = -1e20f; continue; }
                float dot = 0.0f;
                for (int k = 0; k < d_model; ++k) dot += Q[(size_t)r * d_model + k] * K[(size_t)c * d_model + k];
                S[(size_t)r * N_seq + c] = dot / std::sqrt((float)d_model);
            }
        for (int r = 0; r < N_seq; ++r) {
            float maxv = -1e20f;
            for (int c = 0; c <= r; ++c) maxv = std::max(maxv, S[(size_t)r * N_seq + c]);
            float sum = 0.0f;
            for (int c = 0; c <= r; ++c) sum += std::exp(S[(size_t)r * N_seq + c] - maxv);
            for (int c = 0; c < N_seq; ++c) P[(size_t)r * N_seq + c] = (c <= r) ? std::exp(S[(size_t)r * N_seq + c] - maxv) / sum : 0.0f;
        }
        std::vector<float> O((size_t)N_seq * d_model, 0.0f);
        for (int r = 0; r < N_seq; ++r)
            for (int c = 0; c < d_model; ++c) {
                float s = 0.0f;
                for (int k = 0; k < N_seq; ++k) s += P[(size_t)r * N_seq + k] * V[(size_t)k * d_model + c];
                O[(size_t)r * d_model + c] = s;
            }

        // FFN tensor-parallel: logica IDENTICA a llm_mpi_cuda_gpt2small.cu
        std::vector<float> Z_local = matmul(O, Wup_local.data(), N_seq, d_ff_local, d_model);
        for (auto& v : Z_local) v = std::max(0.0f, v);

        std::vector<float> H_flat((size_t)num_ranks * N_seq * d_ff_local);
        if (num_ranks > 1) {
            MPI_Allgather(Z_local.data(), N_seq * d_ff_local, MPI_FLOAT, H_flat.data(), N_seq * d_ff_local, MPI_FLOAT, MPI_COMM_WORLD);
        } else {
            H_flat = Z_local;
        }

        std::vector<float> H_global((size_t)N_seq * d_ff);
        for (int r = 0; r < num_ranks; ++r)
            for (int i = 0; i < N_seq; ++i)
                for (int j = 0; j < d_ff_local; ++j) {
                    size_t flat_idx = (size_t)r * (N_seq * d_ff_local) + (size_t)i * d_ff_local + j;
                    int gc = r * d_ff_local + j;
                    H_global[(size_t)i * d_ff + gc] = H_flat[flat_idx];
                }

        X = matmul(H_global, W_down.data.data(), N_seq, d_model, d_ff); // = H_out, entrada de la siguiente capa
    }

    std::vector<float> h_last(X.end() - d_model, X.end());

    if (rank == 0) {
        std::vector<float> logits(vocab_size);
        for (int v = 0; v < vocab_size; ++v) {
            float s = 0.0f;
            for (int k = 0; k < d_model; ++k) s += h_last[k] * W_vocab.data[(size_t)k * vocab_size + v];
            logits[v] = s;
        }
        float maxl = logits[0];
        for (int v = 1; v < vocab_size; ++v) maxl = std::max(maxl, logits[v]);
        double sume = 0.0;
        std::vector<float> probs(vocab_size);
        for (int v = 0; v < vocab_size; ++v) { probs[v] = std::exp(logits[v] - maxl); sume += probs[v]; }
        for (int v = 0; v < vocab_size; ++v) probs[v] = (float)(probs[v] / sume);
        int best = 0;
        for (int v = 1; v < vocab_size; ++v) if (probs[v] > probs[best]) best = v;

        std::printf("[-np %d, layers=%d] N_seq=%d d_model=%d d_ff=%d vocab=%d  "
                    "h_last[:4]=[%.6f %.6f %.6f %.6f]  logits[:4]=[%.6f %.6f %.6f %.6f]  "
                    "argmax=%d prob=%.6f%%\n",
                    num_ranks, num_layers, N_seq, d_model, d_ff, vocab_size,
                    h_last[0], h_last[1], h_last[2], h_last[3],
                    logits[0], logits[1], logits[2], logits[3],
                    best, probs[best] * 100.0);
    }

    MPI_Finalize();
    return 0;
}
