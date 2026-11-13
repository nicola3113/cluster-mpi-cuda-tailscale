/*
 * llm_mpi_cuda_gpt2small.cu
 *
 * EXTENSION A ESCALA REAL (GPT-2 Small) del programa hibrido MPI + CUDA de
 * llm_mpi_cuda_red_trafico.cu -- ver manual, SECCION 8, para la motivacion,
 * las simplificaciones mantenidas y la metodologia de benchmarking completa.
 *
 * Diferencias respecto al programa original (d_model=4, caso ilustrativo):
 *   1. Las matrices NO estan hardcodeadas en el codigo fuente: se leen de
 *      archivos binarios externos (ver generate_matrices.py y el formato
 *      descrito en el manual, seccion 8.4), lo que permite usar dimensiones
 *      realistas (d_model=768, d_ff=3072, vocab_size=50257 -- GPT-2 Small)
 *      sin inflar el codigo con miles de literales.
 *   2. El programa queda instrumentado con medicion de tiempo (MPI_Wtime)
 *      para poder comparar el desempeno real -- speedup y eficiencia -- al
 *      ejecutar en 1 computadora con su GPU frente a 2 o mas computadoras
 *      cada una con su propia GPU (ver manual, seccion 8.7).
 *   3. Soporta un numero configurable de "capas" (--layers), repitiendo el
 *      bloque Atencion+FFN con LOS MISMOS pesos para aproximar el costo
 *      total de un modelo de L capas sin necesitar L juegos de pesos
 *      independientes (simplificacion explicita, ver manual 8.3).
 *   4. La logica DE PARALELISMO DE TENSORES (fragmentacion de columnas de
 *      W_up, MPI_Allgather, reconstruccion explicita de la matriz global)
 *      es EXACTAMENTE LA MISMA, sin cambios algoritmicos, que la ya
 *      validada en llm_mpi_cuda_red_trafico.cu (manual, seccion 4.9 y
 *      Anexo B) -- solo cambian las dimensiones, que ahora se leen de los
 *      archivos en vez de ser constantes de compilacion.
 *   5. Solo el rank 0 carga W_vocab (~147 MB a esta escala): es la unica
 *      matriz que no necesita cada nodo, porque la proyeccion final al
 *      vocabulario se calcula una sola vez, solo para el ultimo token, y
 *      solo se imprime desde rank 0.
 *
 * Requisito de ejecucion (igual que antes, generalizado): el numero de
 * procesos MPI (-np) DEBE dividir exactamente a d_ff (=3072 para GPT-2
 * Small -> valores validos de -np: 1,2,3,4,6,8,12,16,24,32,48,64,96,...).
 *
 * Compilacion (ver manual, seccion 5.4/5.7, para la guia completa en WSL2):
 *   nvcc -O3 -arch=sm_80 -ccbin=mpicxx -c llm_mpi_cuda_gpt2small.cu -o llm_mpi_cuda_gpt2small.o
 *   mpicxx llm_mpi_cuda_gpt2small.o -L${CUDA_HOME}/lib64 -lcudart -o llm_mpi_cuda_gpt2small
 *
 * Generacion previa de las matrices (una vez, luego copiar matrices/ a
 * TODOS los nodos del cluster -- ver manual 8.4):
 *   python3 generate_matrices.py --out-dir matrices --n-seq 128
 *
 * Ejecucion en un cluster de N computadoras (1 proceso MPI por nodo, cada
 * uno con su propia GPU), requiere el hostfile del manual, seccion 5.8:
 *   mpirun --hostfile hosts.txt -np 2 ./llm_mpi_cuda_gpt2small \
 *       --matrices-dir matrices --layers 1 --repeats 5 \
 *       --results-csv benchmark_results.csv
 *
 * Argumentos de linea de comandos (todos opcionales):
 *   --matrices-dir <path>   Directorio con los .bin generados (default: ./matrices)
 *   --layers <N>            Numero de capas a simular, reusando los mismos
 *                            pesos (default: 1; GPT-2 Small real tiene 12)
 *   --repeats <N>            Repeticiones internas del pipeline para promediar
 *                            tiempos dentro de la misma ejecucion de mpirun
 *                            (default: 5)
 *   --results-csv <path>    Archivo CSV donde se acumulan los resultados de
 *                            tiempo (default: ./benchmark_results.csv)
 */

#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <string>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <ctime>
#include <mpi.h>
#include <cuda_runtime.h>

#define TILE_SIZE 16
#define MASK_VALUE (-1e20f)

#define CUDA_CHECK(call) do { \
    cudaError_t e = (call); \
    if (e != cudaSuccess) { \
        std::fprintf(stderr, "[Rank %d] Error CUDA %s en %s:%d\n", g_rank, cudaGetErrorString(e), __FILE__, __LINE__); \
        MPI_Abort(MPI_COMM_WORLD, 1); \
    } \
} while (0)

static int g_rank = -1;

// ============================================================================
// KERNELS CUDA -- identicos en logica a llm_mpi_cuda_red_trafico.cu; ya
// estaban parametrizados por dimension (N, d, K, etc.), por lo que no
// requieren ningun cambio para operar a escala GPT-2 Small.
// ============================================================================
__global__ void add_positional_encoding_kernel(const float* d_E, const float* d_PE, float* d_X, int N, int d) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < N && col < d) {
        int idx = row * d + col;
        d_X[idx] = d_E[idx] + d_PE[idx];
    }
}

__global__ void matmul_cuda_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int i = 0; i < K; ++i) {
            sum += A[row * K + i] * B[i * N + col];
        }
        C[row * N + col] = sum;
    }
}

__global__ void attention_scores_causal_kernel(const float* Q, const float* K, float* S, int N_seq, int d_head) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N_seq && col < N_seq) {
        if (col > row) {
            S[row * N_seq + col] = MASK_VALUE;
        } else {
            float dot = 0.0f;
            for (int k = 0; k < d_head; ++k) {
                dot += Q[row * d_head + k] * K[col * d_head + k];
            }
            S[row * N_seq + col] = dot / sqrtf((float)d_head);
        }
    }
}

__global__ void softmax_causal_kernel(const float* S, float* P, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N) {
        const float* S_row = &S[row * N];
        float* P_row = &P[row * N];
        float max_val = MASK_VALUE;
        for (int c = 0; c <= row; ++c) {
            if (S_row[c] > max_val) max_val = S_row[c];
        }
        float sum_exp = 0.0f;
        for (int c = 0; c <= row; ++c) {
            sum_exp += expf(S_row[c] - max_val);
        }
        for (int c = 0; c < N; ++c) {
            P_row[c] = (c <= row) ? (expf(S_row[c] - max_val) / sum_exp) : 0.0f;
        }
    }
}

__global__ void relu_kernel(float* Z, int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        Z[idx] = fmaxf(0.0f, Z[idx]);
    }
}

// ============================================================================
// Lectura de matrices desde archivo binario (formato: int32 rows, int32
// cols, luego rows*cols float32 row-major -- ver generate_matrices.py)
// ============================================================================
struct HostMatrix {
    int rows = 0, cols = 0;
    std::vector<float> data;
};

HostMatrix read_bin_matrix(const std::string& path) {
    HostMatrix m;
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        std::fprintf(stderr,
            "[Rank %d] ERROR: no se pudo abrir '%s'.\n"
            "  Verifique que ejecuto generate_matrices.py y que el directorio\n"
            "  de matrices esta presente (y es identico) en TODOS los nodos\n"
            "  del cluster -- ver manual, seccion 8.4.\n", g_rank, path.c_str());
        MPI_Abort(MPI_COMM_WORLD, 4);
    }
    int32_t rows = 0, cols = 0;
    f.read(reinterpret_cast<char*>(&rows), sizeof(int32_t));
    f.read(reinterpret_cast<char*>(&cols), sizeof(int32_t));
    if (!f || rows <= 0 || cols <= 0) {
        std::fprintf(stderr, "[Rank %d] ERROR: cabecera invalida en '%s'.\n", g_rank, path.c_str());
        MPI_Abort(MPI_COMM_WORLD, 4);
    }
    m.rows = rows; m.cols = cols;
    m.data.resize((size_t)rows * (size_t)cols);
    f.read(reinterpret_cast<char*>(m.data.data()), (std::streamsize)(m.data.size() * sizeof(float)));
    if (!f) {
        std::fprintf(stderr, "[Rank %d] ERROR: '%s' truncado (se esperaban %dx%d floats).\n", g_rank, path.c_str(), rows, cols);
        MPI_Abort(MPI_COMM_WORLD, 4);
    }
    return m;
}

// ============================================================================
// Parseo minimo de argumentos --flag valor (sin dependencias externas)
// ============================================================================
std::string get_arg(int argc, char** argv, const std::string& flag, const std::string& def) {
    for (int i = 1; i < argc - 1; ++i) {
        if (flag == argv[i]) return std::string(argv[i + 1]);
    }
    return def;
}

// ============================================================================
// Registro de resultados de tiempo en CSV (una fila por repeticion interna)
// ============================================================================
void append_csv_row(const std::string& path, const std::string& hostname, int num_ranks,
                     int n_seq, int d_model, int d_ff, int vocab_size, int num_layers,
                     int repeat_idx, double load_s, double compute_s, double comm_s, double total_s) {
    bool exists = std::ifstream(path).good();
    std::ofstream f(path, std::ios::app);
    if (!exists) {
        f << "timestamp,hostname_rank0,num_ranks,n_seq,d_model,d_ff,vocab_size,num_layers,"
             "repeat_idx,load_time_s,compute_time_s,comm_time_s,total_time_s\n";
    }
    std::time_t now = std::time(nullptr);
    char ts[32];
    std::strftime(ts, sizeof(ts), "%Y-%m-%dT%H:%M:%S", std::localtime(&now));
    f << ts << "," << hostname << "," << num_ranks << "," << n_seq << "," << d_model << ","
      << d_ff << "," << vocab_size << "," << num_layers << "," << repeat_idx << ","
      << std::fixed << std::setprecision(6) << load_s << "," << compute_s << "," << comm_s << "," << total_s << "\n";
}

// ============================================================================
// PROGRAMA PRINCIPAL MPI + CUDA
// ============================================================================
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, num_ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);
    g_rank = rank;

    char host_name[MPI_MAX_PROCESSOR_NAME];
    int host_name_len = 0;
    MPI_Get_processor_name(host_name, &host_name_len);

    // Asignacion de GPU local al nodo -- identica al programa original
    // (manual, seccion 4.5): MPI_Comm_split_type + rango LOCAL al nodo.
    MPI_Comm node_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node_comm);
    int local_rank = 0, local_size = 1;
    MPI_Comm_rank(node_comm, &local_rank);
    MPI_Comm_size(node_comm, &local_size);

    int num_gpus = 0;
    cudaGetDeviceCount(&num_gpus);
    if (num_gpus > 0) {
        CUDA_CHECK(cudaSetDevice(local_rank % num_gpus));
    } else {
        std::fprintf(stderr, "[Rank %d, %s] ERROR: no se detecto ninguna GPU CUDA en este nodo.\n", rank, host_name);
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
    int my_device = -1;
    cudaGetDevice(&my_device);

    // --------------------------------------------------------------
    // Argumentos de linea de comandos
    // --------------------------------------------------------------
    std::string matrices_dir = get_arg(argc, argv, "--matrices-dir", "./matrices");
    int num_layers = std::atoi(get_arg(argc, argv, "--layers", "1").c_str());
    int num_repeats = std::atoi(get_arg(argc, argv, "--repeats", "5").c_str());
    std::string results_csv = get_arg(argc, argv, "--results-csv", "./benchmark_results.csv");
    if (num_layers < 1) num_layers = 1;
    if (num_repeats < 1) num_repeats = 1;

    // --------------------------------------------------------------
    // Carga de matrices desde archivo (cada rank lee su propia copia
    // local del directorio de matrices -- ver manual 8.4; no requiere
    // transmision por MPI porque los datos no dependen de aleatoriedad
    // de ejecucion, solo de la semilla fija usada por generate_matrices.py).
    // --------------------------------------------------------------
    double t_load0 = MPI_Wtime();

    HostMatrix E  = read_bin_matrix(matrices_dir + "/E.bin");
    HostMatrix PE = read_bin_matrix(matrices_dir + "/PE.bin");
    HostMatrix WQ = read_bin_matrix(matrices_dir + "/WQ.bin");
    HostMatrix WK = read_bin_matrix(matrices_dir + "/WK.bin");
    HostMatrix WV = read_bin_matrix(matrices_dir + "/WV.bin");
    HostMatrix W_up = read_bin_matrix(matrices_dir + "/Wup.bin");
    HostMatrix W_down = read_bin_matrix(matrices_dir + "/Wdown.bin");

    const int N_seq = E.rows;
    const int d_model = E.cols;
    const int d_ff = W_up.cols;

    // --------------------------------------------------------------
    // Validaciones de consistencia dimensional entre archivos
    // --------------------------------------------------------------
    bool dims_ok = (PE.rows == N_seq && PE.cols == d_model &&
                    WQ.rows == d_model && WQ.cols == d_model &&
                    WK.rows == d_model && WK.cols == d_model &&
                    WV.rows == d_model && WV.cols == d_model &&
                    W_up.rows == d_model &&
                    W_down.rows == d_ff && W_down.cols == d_model);
    if (!dims_ok) {
        if (rank == 0) {
            std::fprintf(stderr, "\nERROR: las dimensiones de los archivos .bin en '%s' son inconsistentes "
                                  "entre si. Vuelva a generarlos con generate_matrices.py (no mezcle archivos "
                                  "de corridas con distintos --n-seq/--d-model/--d-ff).\n\n", matrices_dir.c_str());
        }
        MPI_Abort(MPI_COMM_WORLD, 5);
    }

    // d_ff debe ser divisible exactamente entre el numero de procesos MPI
    if (d_ff % num_ranks != 0) {
        if (rank == 0) {
            std::fprintf(stderr,
                "\nERROR DE CONFIGURACION: d_ff (%d) no es divisible entre el numero de\n"
                "procesos MPI solicitado (-np %d).\n"
                "Ejemplo valido para un cluster de 2 computadoras:\n"
                "  mpirun --hostfile hosts.txt -np 2 ./llm_mpi_cuda_gpt2small --matrices-dir %s\n\n",
                d_ff, num_ranks, matrices_dir.c_str());
        }
        MPI_Abort(MPI_COMM_WORLD, 3);
    }

    // Solo rank 0 carga W_vocab (147 MB a N_seq=128): es la unica matriz
    // que no necesita el resto de los nodos (la proyeccion final solo se
    // calcula, para el ultimo token, en rank 0).
    HostMatrix W_vocab;
    int vocab_size = 0;
    if (rank == 0) {
        W_vocab = read_bin_matrix(matrices_dir + "/Wvocab.bin");
        vocab_size = W_vocab.cols;
        if (W_vocab.rows != d_model) {
            std::fprintf(stderr, "\nERROR: Wvocab.bin tiene %d filas, se esperaban %d (d_model).\n\n", W_vocab.rows, d_model);
            MPI_Abort(MPI_COMM_WORLD, 5);
        }
    }
    MPI_Bcast(&vocab_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::cout << "=======================================================================\n";
        std::cout << " INFERENCIA DE LLM HIBRIDA (MPI + CUDA) -- ESCALA GPT-2 SMALL\n";
        std::cout << " (pesos sin entrenar -- benchmarking de paralelismo de tensores,\n";
        std::cout << "  ver manual seccion 8; NO es una prediccion linguistica real)\n";
        std::cout << "=======================================================================\n";
        std::cout << " Procesos MPI (nodos del cluster): " << num_ranks << "\n";
        std::cout << " Configuracion de tensores: N_seq=" << N_seq << ", d_model=" << d_model
                  << ", d_ff=" << d_ff << ", vocab_size=" << vocab_size
                  << ", capas simuladas=" << num_layers << "\n";
        std::cout << " Repeticiones internas para promediar tiempo: " << num_repeats << "\n";
        std::cout << "-----------------------------------------------------------------------\n";
    }
    std::fprintf(stdout, "[Rank %d/%d] nodo=%s  rank_local_al_nodo=%d/%d  GPU_asignada=%d\n",
                 rank, num_ranks, host_name, local_rank, local_size, my_device);
    std::fflush(stdout);
    MPI_Barrier(MPI_COMM_WORLD);

    // --------------------------------------------------------------
    // Fragmentacion de columnas de W_up (identico al programa original)
    // --------------------------------------------------------------
    const int d_ff_local = d_ff / num_ranks;
    std::vector<float> h_W_up_local((size_t)d_model * d_ff_local);
    for (int r = 0; r < d_model; ++r) {
        for (int c = 0; c < d_ff_local; ++c) {
            int global_col = rank * d_ff_local + c;
            h_W_up_local[(size_t)r * d_ff_local + c] = W_up.data[(size_t)r * d_ff + global_col];
        }
    }

    // --------------------------------------------------------------
    // Reserva de memoria en GPU y copia Host -> Device
    // --------------------------------------------------------------
    float *d_E, *d_PE, *d_X, *d_WQ, *d_WK, *d_WV;
    float *d_Q, *d_K, *d_V, *d_S, *d_P, *d_O;
    float *d_W_up_local, *d_Z_local;
    float *d_H_ffn_global, *d_W_down_global, *d_FFN_out;
    float *d_W_vocab = nullptr, *d_Logits = nullptr;

    CUDA_CHECK(cudaMalloc(&d_E,  (size_t)N_seq * d_model * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_PE, (size_t)N_seq * d_model * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_X,  (size_t)N_seq * d_model * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_WQ, (size_t)d_model * d_model * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_WK, (size_t)d_model * d_model * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_WV, (size_t)d_model * d_model * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Q,  (size_t)N_seq * d_model * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_K,  (size_t)N_seq * d_model * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_V,  (size_t)N_seq * d_model * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_S,  (size_t)N_seq * N_seq * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_P,  (size_t)N_seq * N_seq * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_O,  (size_t)N_seq * d_model * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_W_up_local, (size_t)d_model * d_ff_local * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Z_local,    (size_t)N_seq * d_ff_local * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_H_ffn_global,  (size_t)N_seq * d_ff * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_W_down_global, (size_t)d_ff * d_model * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_FFN_out, (size_t)N_seq * d_model * sizeof(float)));
    if (rank == 0) {
        CUDA_CHECK(cudaMalloc(&d_W_vocab, (size_t)d_model * vocab_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_Logits,  (size_t)vocab_size * sizeof(float)));
    }

    CUDA_CHECK(cudaMemcpy(d_E,  E.data.data(),  (size_t)N_seq * d_model * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_PE, PE.data.data(), (size_t)N_seq * d_model * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_WQ, WQ.data.data(), (size_t)d_model * d_model * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_WK, WK.data.data(), (size_t)d_model * d_model * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_WV, WV.data.data(), (size_t)d_model * d_model * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_W_up_local, h_W_up_local.data(), (size_t)d_model * d_ff_local * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_W_down_global, W_down.data.data(), (size_t)d_ff * d_model * sizeof(float), cudaMemcpyHostToDevice));
    if (rank == 0) {
        CUDA_CHECK(cudaMemcpy(d_W_vocab, W_vocab.data.data(), (size_t)d_model * vocab_size * sizeof(float), cudaMemcpyHostToDevice));
    }

    double t_load1 = MPI_Wtime();
    double load_time_local = t_load1 - t_load0;
    double load_time_max = 0.0;
    MPI_Reduce(&load_time_local, &load_time_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid_pe((d_model + TILE_SIZE - 1) / TILE_SIZE, (N_seq + TILE_SIZE - 1) / TILE_SIZE);
    dim3 grid_proj((d_model + TILE_SIZE - 1) / TILE_SIZE, (N_seq + TILE_SIZE - 1) / TILE_SIZE);
    dim3 grid_att((N_seq + TILE_SIZE - 1) / TILE_SIZE, (N_seq + TILE_SIZE - 1) / TILE_SIZE);
    dim3 grid_ff_local((d_ff_local + TILE_SIZE - 1) / TILE_SIZE, (N_seq + TILE_SIZE - 1) / TILE_SIZE);
    int softmax_threads = 256;
    int softmax_blocks = (N_seq + softmax_threads - 1) / softmax_threads;

    std::vector<float> h_H_local((size_t)N_seq * d_ff_local);
    std::vector<float> h_H_flat((size_t)num_ranks * N_seq * d_ff_local);
    std::vector<float> h_H_ffn_global((size_t)N_seq * d_ff);

    float h_last_result[8]; // solo para imprimir una vista previa (primeros valores)
    float best_prob = 0.0f;
    int best_idx = -1;

    // ================================================================
    // BUCLE DE REPETICIONES (promedio de tiempo dentro de la misma
    // ejecucion de mpirun -- ver manual 8.7 sobre por que se combina con
    // repeticiones EXTERNAS del propio mpirun en run_benchmark.sh)
    // ================================================================
    for (int rep = 0; rep < num_repeats; ++rep) {
        MPI_Barrier(MPI_COMM_WORLD);
        double t_total0 = MPI_Wtime();
        double comm_time_local = 0.0;

        // X = E + PE (recalculado cada repeticion para que el tiempo medido
        // incluya el pipeline completo, igual que en una inferencia real)
        add_positional_encoding_kernel<<<grid_pe, block>>>(d_E, d_PE, d_X, N_seq, d_model);
        CUDA_CHECK(cudaDeviceSynchronize());

        float* d_layer_in = d_X;

        for (int layer = 0; layer < num_layers; ++layer) {
            // Proyecciones Q, K, V
            matmul_cuda_kernel<<<grid_proj, block>>>(d_layer_in, d_WQ, d_Q, N_seq, d_model, d_model);
            matmul_cuda_kernel<<<grid_proj, block>>>(d_layer_in, d_WK, d_K, N_seq, d_model, d_model);
            matmul_cuda_kernel<<<grid_proj, block>>>(d_layer_in, d_WV, d_V, N_seq, d_model, d_model);
            CUDA_CHECK(cudaDeviceSynchronize());

            // Atencion causal S y Softmax P
            attention_scores_causal_kernel<<<grid_att, block>>>(d_Q, d_K, d_S, N_seq, d_model);
            CUDA_CHECK(cudaDeviceSynchronize());
            softmax_causal_kernel<<<softmax_blocks, softmax_threads>>>(d_S, d_P, N_seq);
            CUDA_CHECK(cudaDeviceSynchronize());

            // Contextualizacion O = P . V
            matmul_cuda_kernel<<<grid_proj, block>>>(d_P, d_V, d_O, N_seq, d_model, N_seq);
            CUDA_CHECK(cudaDeviceSynchronize());

            // FFN DISTRIBUIDA (Paralelismo de Tensores) -- logica identica
            // a la ya validada en llm_mpi_cuda_red_trafico.cu.
            matmul_cuda_kernel<<<grid_ff_local, block>>>(d_O, d_W_up_local, d_Z_local, N_seq, d_ff_local, d_model);
            CUDA_CHECK(cudaDeviceSynchronize());
            relu_kernel<<<(N_seq * d_ff_local + 255) / 256, 256>>>(d_Z_local, N_seq * d_ff_local);
            CUDA_CHECK(cudaDeviceSynchronize());

            CUDA_CHECK(cudaMemcpy(h_H_local.data(), d_Z_local, (size_t)N_seq * d_ff_local * sizeof(float), cudaMemcpyDeviceToHost));

            double t_comm0 = MPI_Wtime();
            if (num_ranks > 1) {
                MPI_Allgather(h_H_local.data(), N_seq * d_ff_local, MPI_FLOAT,
                              h_H_flat.data(), N_seq * d_ff_local, MPI_FLOAT,
                              MPI_COMM_WORLD);
            } else {
                h_H_flat = h_H_local;
            }
            comm_time_local += (MPI_Wtime() - t_comm0);

            // Reconstruccion EXPLICITA de la matriz global (ver manual 4.9
            // para la explicacion detallada de por que NO es valido
            // reinterpretar el buffer directamente).
            for (int r = 0; r < num_ranks; ++r) {
                for (int i = 0; i < N_seq; ++i) {
                    for (int j = 0; j < d_ff_local; ++j) {
                        size_t flat_idx = (size_t)r * (N_seq * d_ff_local) + (size_t)i * d_ff_local + j;
                        int global_col = r * d_ff_local + j;
                        h_H_ffn_global[(size_t)i * d_ff + global_col] = h_H_flat[flat_idx];
                    }
                }
            }

            CUDA_CHECK(cudaMemcpy(d_H_ffn_global, h_H_ffn_global.data(), (size_t)N_seq * d_ff * sizeof(float), cudaMemcpyHostToDevice));
            matmul_cuda_kernel<<<grid_proj, block>>>(d_H_ffn_global, d_W_down_global, d_FFN_out, N_seq, d_model, d_ff);
            CUDA_CHECK(cudaDeviceSynchronize());

            // La salida de esta capa es la entrada de la siguiente (sin
            // LayerNorm ni conexion residual -- simplificacion documentada
            // en el manual, seccion 8.3, identica a la del caso ilustrativo).
            d_layer_in = d_FFN_out;
        }

        // Extraer h_ultimo y proyectar al vocabulario (solo rank 0)
        if (rank == 0) {
            const float* d_h_ultimo = d_FFN_out + (size_t)(N_seq - 1) * d_model;
            dim3 grid_vocab((vocab_size + TILE_SIZE - 1) / TILE_SIZE, 1);
            dim3 block_vocab(TILE_SIZE, 1);
            matmul_cuda_kernel<<<grid_vocab, block_vocab>>>(d_h_ultimo, d_W_vocab, d_Logits, 1, vocab_size, d_model);
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        double total_time_local = MPI_Wtime() - t_total0;
        double total_time_max = 0.0, comm_time_max = 0.0;
        MPI_Reduce(&total_time_local, &total_time_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&comm_time_local, &comm_time_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            double compute_time_max = total_time_max - comm_time_max;
            std::cout << " [rep " << (rep + 1) << "/" << num_repeats << "] "
                      << "total=" << std::fixed << std::setprecision(4) << total_time_max << "s  "
                      << "comm(Allgather)=" << comm_time_max << "s  "
                      << "compute=" << compute_time_max << "s\n";
            append_csv_row(results_csv, host_name, num_ranks, N_seq, d_model, d_ff, vocab_size,
                           num_layers, rep, load_time_max, compute_time_max, comm_time_max, total_time_max);

            if (rep == num_repeats - 1) {
                std::vector<float> h_Logits(vocab_size);
                CUDA_CHECK(cudaMemcpy(h_Logits.data(), d_Logits, (size_t)vocab_size * sizeof(float), cudaMemcpyDeviceToHost));
                float max_l = h_Logits[0];
                for (int i = 1; i < vocab_size; ++i) if (h_Logits[i] > max_l) max_l = h_Logits[i];
                double sum_e = 0.0;
                std::vector<float> h_Probs(vocab_size);
                for (int i = 0; i < vocab_size; ++i) { h_Probs[i] = expf(h_Logits[i] - max_l); sum_e += h_Probs[i]; }
                for (int i = 0; i < vocab_size; ++i) h_Probs[i] = (float)(h_Probs[i] / sum_e);
                best_idx = 0; best_prob = h_Probs[0];
                for (int i = 1; i < vocab_size; ++i) if (h_Probs[i] > best_prob) { best_prob = h_Probs[i]; best_idx = i; }
                for (int i = 0; i < 8 && i < vocab_size; ++i) h_last_result[i] = h_Logits[i];
            }
        }
    }

    if (rank == 0) {
        std::cout << "\n=======================================================================\n";
        std::cout << " RESUMEN -- " << num_ranks << " proceso(s) MPI, " << num_layers << " capa(s), N_seq=" << N_seq << "\n";
        std::cout << "=======================================================================\n";
        std::cout << " Indice de vocabulario ganador: " << best_idx << "  (prob=" << std::setprecision(4) << (best_prob * 100.0f) << "%)\n";
        std::cout << " logits[0:8] = [";
        for (int i = 0; i < 8; ++i) std::cout << std::setprecision(4) << h_last_result[i] << (i < 7 ? ", " : "");
        std::cout << "]  (comparar con design_gpt2small.py para la misma semilla)\n";
        std::cout << " NOTA: pesos sin entrenar (ver manual 8.1) -- este indice no tiene\n";
        std::cout << " significado linguistico; el proposito de esta corrida es medir tiempo,\n";
        std::cout << " speedup y eficiencia a escala real (manual, seccion 8.7).\n";
        std::cout << " Tiempo de carga de matrices (maximo entre ranks): " << std::setprecision(4) << load_time_max << "s\n";
        std::cout << " Resultados acumulados en: " << results_csv << "\n";
        std::cout << "=======================================================================\n";
    }

    cudaFree(d_E); cudaFree(d_PE); cudaFree(d_X);
    cudaFree(d_WQ); cudaFree(d_WK); cudaFree(d_WV);
    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_S); cudaFree(d_P); cudaFree(d_O);
    cudaFree(d_W_up_local); cudaFree(d_Z_local);
    cudaFree(d_H_ffn_global); cudaFree(d_W_down_global); cudaFree(d_FFN_out);
    if (rank == 0) { cudaFree(d_W_vocab); cudaFree(d_Logits); }

    MPI_Comm_free(&node_comm);
    MPI_Finalize();
    return 0;
}
