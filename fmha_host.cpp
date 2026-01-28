#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <chrono>
#include "fmha_kernel.hpp"

// Type alias for bfloat16 (must match kernel definition)
using bhalf_t = __bf16;
using half_t  = _Float16;

#define CEIL_DIV(a, b) (a + b - 1) / b
#define HIP_CHECK_ERROR(cmd)                \
    do {                                    \
        hipError_t status = cmd;            \
        if (status != hipSuccess) {         \
            std::ostringstream ostr;                                                    \
            ostr << "HIP Function Failed (" << __FILE__ << "," << __LINE__ <<") "       \
                 << hipGetErrorString(status);                                          \
            throw std::runtime_error(ostr.str());                                       \
        }                                                                               \
    } while(0)                              \

class FMHAConfig {
public:
    int batch;          // Batch size (number of independent sequences)
    int num_heads_q;    // Number of query heads
    int num_heads_kv;   // Number of key/value heads (for GQA)
    int seqlen_q;       // Query sequence length
    int seqlen_kv;      // Key/value sequence length
    int head_dim_q;     // Dimension per query head
    int head_dim_kv;    // Dimension per key/value head

    /**
     * Constructor: Initialize FMHA configuration with all dimensions
     */
    FMHAConfig(int b, int nhq, int nhkv, int sq, int skv, int hdq, int hdkv)
        : batch(b), num_heads_q(nhq), num_heads_kv(nhkv),
          seqlen_q(sq), seqlen_kv(skv), head_dim_q(hdq), head_dim_kv(hdkv) {}

    size_t q_size() const { return batch * num_heads_q * seqlen_q * head_dim_q; }
    size_t kv_size() const { return batch * num_heads_kv * seqlen_kv * head_dim_kv; }
    size_t o_size() const { return batch * num_heads_q * seqlen_q * head_dim_q; }

    /**
     * Print configuration details and memory requirements
     */
    void print() const {
        std::cout << "FMHA Configuration:\n"
                  << "  Batch size: " << batch << "\n"
                  << "  Num heads Q: " << num_heads_q << "\n"
                  << "  Num heads KV: " << num_heads_kv << "\n"
                  << "  Sequence length Q: " << seqlen_q << "\n"
                  << "  Sequence length KV: " << seqlen_kv << "\n"
                  << "  Head dimension Q: " << head_dim_q << "\n"
                  << "  Head dimension KV: " << head_dim_kv << "\n"
                  << "  Data types: Q/K/V=bf16, O=fp16\n"
                  << "  Q tensor size: " << q_size() * sizeof(bhalf_t) / (1024.0 * 1024.0) << " MB (bf16)\n"
                  << "  K tensor size: " << kv_size() * sizeof(bhalf_t) / (1024.0 * 1024.0) << " MB (bf16)\n"
                  << "  V tensor size: " << kv_size() * sizeof(bhalf_t) / (1024.0 * 1024.0) << " MB (bf16)\n"
                  << "  O tensor size: " << o_size() * sizeof(half) / (1024.0 * 1024.0) << " MB (fp16)\n";
    }
};

// ============================================================================
// Tensor Initialization Functions
// ============================================================================

/**
 * Initialize a float16 (fp16) vector with random values
 *
 * Fills the vector with uniformly distributed random values in [-1.0, 1.0].
 * Used for initializing output tensors or fp16 test data.
 *
 * @param vec Reference to std::vector<half> to be initialized
 */
void initialize_random_half(std::vector<half>& vec) {
    std::random_device rd;                          // Obtain random seed
    std::mt19937 gen(rd());                         // Mersenne Twister RNG
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);  // Uniform in [-1, 1]

    for (size_t i = 0; i < vec.size(); i++) {
        vec[i] = __float2half(dis(gen));            // Convert fp32 -> fp16
    }
}

void initialize_random_bfloat16(bhalf_t* mat, int N) {
    std::random_device rd;                          // Obtain random seed
    std::mt19937 gen(rd());                         // Mersenne Twister RNG
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);  // Uniform in [-1, 1]

    for (size_t i = 0; i < N; i++) {
        mat[i] = static_cast<bhalf_t>(dis(gen));        // Convert fp32 -> bf16
    }
}

void run_fmha_benchmark(const FMHAConfig& config, int num_iterations = 100) {
    std::cout << "\n=== Running FMHA Benchmark ===\n";
    config.print();

    hipEvent_t start, end;
    // ========================================================================
    // Step 1: Allocate Host Memory
    // ========================================================================
    // Using std::vector for automatic memory management and exception safety
    // Q, K, V use bf16 (2 bytes per element)
    // O uses fp16 (2 bytes per element)
    bhalf_t *h_Q, *h_K, *h_V;
    half_t *h_O;

    h_Q = (bhalf_t*)malloc(config.q_size() * sizeof(bhalf_t));
    h_K = (bhalf_t*)malloc(config.kv_size() * sizeof(bhalf_t));
    h_V = (bhalf_t*)malloc(config.kv_size() * sizeof(bhalf_t));
    h_O = (half_t*)malloc(config.o_size() * sizeof(half_t));

    // ========================================================================
    // Step 2: Initialize Input Tensors with Random Data
    // ========================================================================
    std::cout << "\nInitializing tensors...\n";
    initialize_random_bfloat16(h_Q, config.q_size());  // Random values in [-1, 1]
    initialize_random_bfloat16(h_K, config.kv_size());
    initialize_random_bfloat16(h_V, config.kv_size());

    // ========================================================================
    // Step 3: Allocate Device (GPU) Memory
    // ========================================================================
    bhalf_t *d_Q, *d_K, *d_V;  // Device pointers for bf16 input tensors
    half_t *d_O;                  // Device pointer for fp16 output tensor

    HIP_CHECK(hipMalloc((void**)&d_Q, config.q_size() * sizeof(bhalf_t)));
    HIP_CHECK(hipMalloc((void**)&d_K, config.kv_size() * sizeof(bhalf_t)));
    HIP_CHECK(hipMalloc((void**)&d_V, config.kv_size() * sizeof(bhalf_t)));
    HIP_CHECK(hipMalloc((void**)&d_O, config.o_size() * sizeof(half_t)));
    //HIP_CHECK(hipMalloc(&d_O, config.o_size() * sizeof(half)));

    // ========================================================================
    // Step 4: Copy Input Data from Host to Device
    // ========================================================================
    std::cout << "Preparing data on GPU...\n";
    HIP_CHECK(hipMemset(d_O, 0, config.o_size() * sizeof(half_t)));
    HIP_CHECK(hipMemcpy(d_Q, h_Q.data(), config.q_size() * sizeof(bhalf_t), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_K, h_K.data(), config.kv_size() * sizeof(bhalf_t), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_V, h_V.data(), config.kv_size() * sizeof(bhalf_t), hipMemcpyHostToDevice));

    dim3 gridDim(CEIL_DIV(config.seqlen_q, 256), config.num_heads_q, config.batch);
    dim3 blockDim(256, 1, 1);
    // ========================================================================
    // Step 5: Create HIP Stream for Asynchronous Execution
    // ========================================================================
    // HIP streams allow kernels to execute asynchronously with respect to the host
    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));
    hipEvent_t start, end;

    // ========================================================================
    // Step 6: Warm-up Run
    // ========================================================================
    // First kernel launch is often slower due to:
    // - JIT compilation and code caching
    // - GPU frequency scaling (boost clocks)
    // - Cache warming
    // We run one iteration and wait for completion before benchmarking
    std::cout << "Warming up...\n";
    launch_fmha_forward(d_Q, d_K, d_V, d_O,
                       config.batch, config.num_heads_q, config.num_heads_kv,
                       config.seqlen_q, config.seqlen_kv,
                       config.head_dim_q, config.head_dim_kv,
                       stream);
    HIP_CHECK(hipStreamSynchronize(stream));  // Wait for warm-up to complete

    // ========================================================================
    // Step 7: Timed Benchmark Iterations
    // ========================================================================
    std::cout << "Running " << num_iterations << " iterations...\n";
    auto start = std::chrono::high_resolution_clock::now();

    // Launch kernel multiple times for averaging
    for (int i = 0; i < num_iterations; i++) {
        launch_fmha_forward(d_Q, d_K, d_V, d_O,
                           config.batch, config.num_heads_q, config.num_heads_kv,
                           config.seqlen_q, config.seqlen_kv,
                           config.head_dim_q, config.head_dim_kv,
                           stream);
    }

    // Wait for all kernels to complete before stopping timer
    HIP_CHECK(hipStreamSynchronize(stream));
    auto end = std::chrono::high_resolution_clock::now();

    // ========================================================================
    // Step 8: Calculate Performance Metrics
    // ========================================================================
    std::chrono::duration<double, std::milli> elapsed = end - start;
    double avg_time_ms = elapsed.count() / num_iterations;

    // Estimate FLOPs (Floating Point Operations)
    // Attention computation involves two main matrix multiplications:
    // 1. Q @ K^T: (batch * num_heads_q * seqlen_q * seqlen_kv * head_dim_q) MACs
    // 2. P @ V:   (batch * num_heads_q * seqlen_q * seqlen_kv * head_dim_q) MACs
    // Each MAC (multiply-accumulate) counts as 2 FLOPs
    // Note: This doesn't include softmax operations (exp, div)
    long long flops_per_iter = 2LL * config.batch * config.num_heads_q *
                               config.seqlen_q * config.seqlen_kv * config.head_dim_q;
    double tflops = (flops_per_iter / (avg_time_ms / 1000.0)) / 1e12;

    // ========================================================================
    // Step 9: Display Results
    // ========================================================================
    std::cout << "\n=== Results ===\n";
    std::cout << "Average time per iteration: " << avg_time_ms << " ms\n";
    std::cout << "Throughput: " << 1000.0 / avg_time_ms << " iterations/second\n";
    std::cout << "Estimated performance: " << tflops << " TFLOPS\n";

    // ========================================================================
    // Step 10: Copy Results Back to Host
    // ========================================================================
    HIP_CHECK(hipMemcpy(h_O.data(), d_O, config.o_size() * sizeof(half), hipMemcpyDeviceToHost));

    // Display sample output values for sanity check
    std::cout << "\nSample output values (first 5):\n";
    for (int i = 0; i < std::min(5, (int)h_O.size()); i++) {
        std::cout << "  O[" << i << "] = " << __half2float(h_O[i]) << "\n";
    }

    // ========================================================================
    // Step 11: Cleanup Device Resources
    // ========================================================================
    HIP_CHECK(hipFree(d_Q));
    HIP_CHECK(hipFree(d_K));
    HIP_CHECK(hipFree(d_V));
    HIP_CHECK(hipFree(d_O));
    HIP_CHECK(hipStreamDestroy(stream));
    // Note: Host memory (h_Q, h_K, h_V, h_O) is automatically freed when
    // std::vectors go out of scope
}  // End of run_fmha_benchmark

int main(int argc, char* argv[]) {
    std::cout << "Flash Multi-Head Attention (FMHA) HIP Implementation\n";
    std::cout << "====================================================\n\n";

    int config_index;
    if (argc >= 2) {
        config_index = atoi(argv[1]);
    }
    // ========================================================================
    // Device Detection and Information
    // ========================================================================
    int device_count;
    HIP_CHECK(hipGetDeviceCount(&device_count));
    std::cout << "Found " << device_count << " HIP device(s)\n";

    // Verify at least one GPU is available
    if (device_count == 0) {
        std::cerr << "No HIP devices found!\n";
        return EXIT_FAILURE;
    }

    // Query and display device properties
    // Device 0 is used by default; could be modified to support device selection
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));
    std::cout << "Using device: " << prop.name << "\n";
    std::cout << "Compute capability: " << prop.major << "." << prop.minor << "\n";
    std::cout << "Total memory: " << prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0) << " GB\n\n";
    // ========================================================================
    // Define Test Configurations
    // ========================================================================
    // Format: FMHAConfig(batch, num_heads_q, num_heads_kv, seqlen_q, seqlen_kv, head_dim_q, head_dim_kv)
    //
    // Test scenarios:
    // 1. High-dimensional heads (256) with moderate head count (16)
    // 2. Lower-dimensional heads (128) with moderate head count (16)
    // 3. High-dimensional heads (256) with high head count (32)
    // 4. Lower-dimensional heads (128) with high head count (32)
    //
    // These configurations test:
    // - Memory bandwidth (larger head_dim)
    // - Parallelism (more heads)
    // - Register pressure (head_dim affects accumulator size)
    std::vector<FMHAConfig> configs = {
        FMHAConfig(30720, 16, 16, 1, 2, 256, 256),  // Config 1
        FMHAConfig(30720, 16, 16, 1, 2, 128, 128),  // Config 2
        FMHAConfig(30720, 32, 32, 1, 2, 256, 256),  // Config 3
        FMHAConfig(30720, 32, 32, 1, 2, 128, 128),  // Config 4
    };

    // ========================================================================
    // Run Benchmarks
    // ========================================================================
    run_fmha_benchmark(configs[config_index], 100);

    // ========================================================================
    // Completion
    // ========================================================================
    std::cout << "\n\nAll benchmarks completed successfully!\n";
    return EXIT_SUCCESS;
}  // End of main
