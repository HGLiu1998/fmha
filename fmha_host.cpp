/**
 * Flash Multi-Head Attention (FMHA) - Host Implementation
 * 
 * Benchmarks FMHA kernel with various configurations on AMD GPUs.
 * Supports mixed precision: bf16 inputs (Q/K/V), fp16 output (O).
 */

#include <hip/hip_runtime.h>
#include <iostream>
#include <sstream>
#include <vector>
#include <cmath>
#include <random>
#include <chrono>
#include <fstream>
#include "fmha_mfma_kernel.hpp"

// Type aliases for numeric types
using bhalf_t = __bf16;      // bfloat16 for Q, K, V
using half_t  = _Float16;    // float16 for O

// Utility macros
#define CEIL_DIV(a, b) (a + b - 1) / b
#define HIP_CHECK(cmd)                      \
    do {                                    \
        hipError_t status = cmd;            \
        if (status != hipSuccess) {         \
            std::ostringstream ostr;                                                    \
            ostr << "HIP Function Failed (" << __FILE__ << "," << __LINE__ <<") "       \
                 << hipGetErrorString(status);                                          \
            throw std::runtime_error(ostr.str());                                       \
        }                                                                               \
    } while(0)

/**
 * FMHA Configuration
 * 
 * Encapsulates all parameters for multi-head attention computation.
 * Supports Grouped Query Attention (GQA) when num_heads_q != num_heads_kv.
 */
class FMHAConfig {
public:
    int batch;          // Number of independent sequences
    int num_heads_q;    // Query heads
    int num_heads_kv;   // Key/Value heads (GQA: can be < num_heads_q)
    int seqlen_q;       // Query sequence length
    int seqlen_kv;      // Key/Value sequence length
    int head_dim_q;     // Dimension per query head
    int head_dim_kv;    // Dimension per key/value head

    FMHAConfig(int b, int nhq, int nhkv, int sq, int skv, int hdq, int hdkv)
        : batch(b), num_heads_q(nhq), num_heads_kv(nhkv),
          seqlen_q(sq), seqlen_kv(skv), head_dim_q(hdq), head_dim_kv(hdkv) {}

    // Tensor sizes in number of elements
    size_t q_size() const { return batch * num_heads_q * seqlen_q * head_dim_q; }
    size_t kv_size() const { return batch * num_heads_kv * seqlen_kv * head_dim_kv; }
    size_t o_size() const { return batch * num_heads_q * seqlen_q * head_dim_q; }

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
                  << "  O tensor size: " << o_size() * sizeof(half_t) / (1024.0 * 1024.0) << " MB (fp16)\n";
    }
};

// ============================================================================
// Initialization Utilities
// ============================================================================

/** Initialize fp16 vector with random uniform values in [-1, 1] */
void initialize_random_half(std::vector<half_t>& vec) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

    for (size_t i = 0; i < vec.size(); i++) {
        vec[i] = static_cast<half_t>(dis(gen));
    }
}

/** Initialize bf16 array with random uniform values in [-1, 1] */
void initialize_random_bfloat16(bhalf_t* mat, int N) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

    for (size_t i = 0; i < N; i++) {
        //mat[i] = static_cast<bhalf_t>(dis(gen));
        mat[i] = static_cast<bhalf_t>(0.1f);
    }
}

/**
 * FMHA Benchmark Runner
 * 
 * Measures kernel performance with the following workflow:
 * 1. Allocate and initialize host/device memory
 * 2. Warm-up iterations to stabilize GPU clocks
 * 3. Timed benchmark iterations
 * 4. Calculate and display performance metrics
 * 
 * @param config FMHA configuration parameters
 * @param num_iterations Number of timed iterations (default: 100)
 * @param warm_ups Number of warm-up iterations (default: 50)
 */
void run_fmha_benchmark(const FMHAConfig& config, int num_iterations = 1, int warm_ups = 0) {
    std::cout << "\n=== Running FMHA Benchmark ===\n";
    config.print();

    // Allocate host memory (bf16 for inputs, fp16 for output)
    bhalf_t *h_Q = (bhalf_t*)malloc(config.q_size() * sizeof(bhalf_t));
    bhalf_t *h_K = (bhalf_t*)malloc(config.kv_size() * sizeof(bhalf_t));
    bhalf_t *h_V = (bhalf_t*)malloc(config.kv_size() * sizeof(bhalf_t));
    half_t  *h_O = (half_t*)malloc(config.o_size() * sizeof(half_t));

    // Initialize with random data
    std::cout << "\nInitializing tensors...\n";
    initialize_random_bfloat16(h_Q, config.q_size());
    initialize_random_bfloat16(h_K, config.kv_size());
    initialize_random_bfloat16(h_V, config.kv_size());

    // Allocate device memory
    bhalf_t *d_Q, *d_K, *d_V;
    half_t  *d_O;
    HIP_CHECK(hipMalloc((void**)&d_Q, config.q_size() * sizeof(bhalf_t)));
    HIP_CHECK(hipMalloc((void**)&d_K, config.kv_size() * sizeof(bhalf_t)));
    HIP_CHECK(hipMalloc((void**)&d_V, config.kv_size() * sizeof(bhalf_t)));
    HIP_CHECK(hipMalloc((void**)&d_O, config.o_size() * sizeof(half_t)));

    // Copy input data to device
    std::cout << "Preparing data on GPU...\n";
    HIP_CHECK(hipMemset(d_O, 0, config.o_size() * sizeof(half_t)));
    HIP_CHECK(hipMemcpy(d_Q, h_Q, config.q_size() * sizeof(bhalf_t), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_K, h_K, config.kv_size() * sizeof(bhalf_t), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_V, h_V, config.kv_size() * sizeof(bhalf_t), hipMemcpyHostToDevice));

    // Configure kernel launch parameters
    // Grid: (1, num_heads_q, batch) - One block per head per batch element
    dim3 gridDim(1, config.num_heads_q, config.batch);
    dim3 blockDim(256, 1, 1);

    // Create HIP events for timing
    hipEvent_t start, end;

    // Warm-up phase: stabilize GPU clocks and caches
    std::cout << "Running " << warm_ups << " warm-up iterations...\n";
    for (int i = 0; i < warm_ups; ++i) {
        fmha_mfma<<<gridDim, blockDim>>>(d_Q, d_K, d_V, d_O,
                           config.batch, config.num_heads_q, config.num_heads_kv,
                           config.seqlen_q, config.seqlen_kv,
                           config.head_dim_q, config.head_dim_kv,
                           1.0f / sqrt(config.head_dim_q));
    }
    HIP_CHECK(hipDeviceSynchronize());

    // Benchmark phase: timed iterations
    std::cout << "Running " << num_iterations << " timed iterations...\n";
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&end));
    
    HIP_CHECK(hipEventRecord(start, NULL));
    for (int i = 0; i < num_iterations; i++) {
        fmha_mfma<<<gridDim, blockDim>>>(d_Q, d_K, d_V, d_O,
                           config.batch, config.num_heads_q, config.num_heads_kv,
                           config.seqlen_q, config.seqlen_kv,
                           config.head_dim_q, config.head_dim_kv,
                           1.0f / sqrt(config.head_dim_q));
    }
    HIP_CHECK(hipEventRecord(end, NULL));
    HIP_CHECK(hipEventSynchronize(end));

    // Calculate performance metrics
    float elapsed_ms;
    HIP_CHECK(hipEventElapsedTime(&elapsed_ms, start, end));
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(end));

    double avg_time_ms = elapsed_ms / num_iterations;
    
    // Estimate FLOPs: 2 matmuls (Q@K^T and P@V), each MAC = 2 FLOPs
    // Note: Doesn't include softmax ops (exp, div)
    long long flops_per_iter = 2LL * config.batch * config.num_heads_q *
                               config.seqlen_q * config.seqlen_kv * config.head_dim_q;
    double tflops = (flops_per_iter / (avg_time_ms / 1000.0)) / 1e12;

    // Display results
    std::cout << "\n=== Performance Results ===\n";
    std::cout << "Average time:  " << avg_time_ms << " ms\n";
    std::cout << "Throughput:    " << 1000.0 / avg_time_ms << " iter/s\n";
    std::cout << "Performance:   " << tflops << " TFLOPS\n";

    // Verify correctness: run one more iteration and check output
    HIP_CHECK(hipMemset(d_O, 0, config.o_size() * sizeof(half_t)));
    fmha_mfma<<<gridDim, blockDim>>>(d_Q, d_K, d_V, d_O,
        config.batch, config.num_heads_q, config.num_heads_kv,
        config.seqlen_q, config.seqlen_kv,
        config.head_dim_q, config.head_dim_kv,
        1.0f / sqrt(config.head_dim_q));
    HIP_CHECK(hipMemcpy(h_O, d_O, config.o_size() * sizeof(half_t), hipMemcpyDeviceToHost));

    std::cout << "\nSample outputs (sanity check):\n";
    for (int i = 0; i < std::min(5, (int)config.o_size()); i++) {
        std::cout << "  O[" << i << "] = " << static_cast<float>(h_O[i]) << "\n";
    }

    // #region agent log — Host-side debug logging
    {
        std::ofstream dbg("/home/arliu/workspace/fmha/.cursor/debug.log", std::ios::app);
        if (dbg.is_open()) {
            auto ts = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count();
            dbg << "{\"timestamp\":" << ts
                << ",\"location\":\"fmha_host.cpp:post_kernel\""
                << ",\"message\":\"Host verification results\""
                << ",\"hypothesisId\":\"all\""
                << ",\"data\":{\"batch\":" << config.batch
                << ",\"heads_q\":" << config.num_heads_q
                << ",\"heads_kv\":" << config.num_heads_kv
                << ",\"seqlen_q\":" << config.seqlen_q
                << ",\"seqlen_kv\":" << config.seqlen_kv
                << ",\"head_dim_q\":" << config.head_dim_q
                << ",\"head_dim_kv\":" << config.head_dim_kv
                << ",\"expected_score\":\"" << config.head_dim_q << " * 0.01 = " << config.head_dim_q * 0.01 << "\""
                << ",\"O_samples\":[";
            for (int i = 0; i < std::min(5, (int)config.o_size()); i++) {
                if (i > 0) dbg << ",";
                dbg << static_cast<float>(h_O[i]);
            }
            dbg << "]}}\n";
            dbg.close();
        }
    }
    // #endregion

    // Cleanup
    free(h_Q); free(h_K); free(h_V); free(h_O);
    HIP_CHECK(hipFree(d_Q));
    HIP_CHECK(hipFree(d_K));
    HIP_CHECK(hipFree(d_V));
    HIP_CHECK(hipFree(d_O));
} 

/**
 * Main Entry Point
 * 
 * Usage: ./fmha_benchmark [config_index]
 *   config_index: 0-3 (selects one of four test configurations)
 *                 If not provided, runs all configurations
 */
int main(int argc, char* argv[]) {
    std::cout << "Flash Multi-Head Attention (FMHA) - HIP Benchmark\n";
    std::cout << "==================================================\n\n";

    // Parse command-line arguments
    int config_index = -1;  // -1 means run all configs
    if (argc >= 2) {
        config_index = atoi(argv[1]);
        if (config_index < 0 || config_index > 17) {
            std::cerr << "Error: config_index must be 0-17\n";
            std::cerr << "Usage: " << argv[0] << " [config_index]\n";
            return EXIT_FAILURE;
        }
    }

    // Initialize GPU
    int device_count;
    HIP_CHECK(hipGetDeviceCount(&device_count));
    std::cout << "Found " << device_count << " HIP device(s)\n";

    if (device_count == 0) {
        std::cerr << "Error: No HIP devices found!\n";
        return EXIT_FAILURE;
    }

    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));
    std::cout << "Using: " << prop.name << "\n";
    std::cout << "Compute: " << prop.major << "." << prop.minor << "\n";
    std::cout << "Memory: " << prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0) << " GB\n\n";

    // Define test configurations
    // Format: FMHAConfig(batch, num_heads_q, num_heads_kv, seqlen_q, seqlen_kv, head_dim_q, head_dim_kv)
    std::vector<FMHAConfig> configs = {
        // Test various seqlen_kv values (1-16) with 16 heads, head_dim=128
        FMHAConfig(1, 16, 16, 1, 1, 128, 128),   // Config 0
        FMHAConfig(1, 16, 16, 1, 2, 128, 128),   // Config 1
        FMHAConfig(1, 16, 16, 1, 4, 128, 128),   // Config 2
        FMHAConfig(1, 16, 16, 1, 8, 128, 128),   // Config 3
        FMHAConfig(1, 16, 16, 1, 16, 128, 128),  // Config 4
        
        // Test various seqlen_kv values with 16 heads, head_dim=256
        FMHAConfig(30720, 16, 16, 1, 1, 256, 256),   // Config 5
        FMHAConfig(30720, 16, 16, 1, 2, 256, 256),   // Config 6
        FMHAConfig(30720, 16, 16, 1, 4, 256, 256),   // Config 7
        FMHAConfig(30720, 16, 16, 1, 8, 256, 256),   // Config 8
        FMHAConfig(30720, 16, 16, 1, 16, 256, 256),  // Config 9
        
        // Test 32 heads with various seqlen_kv, head_dim=128
        FMHAConfig(30720, 32, 32, 1, 1, 128, 128),   // Config 10
        FMHAConfig(30720, 32, 32, 1, 2, 128, 128),   // Config 11
        FMHAConfig(30720, 32, 32, 1, 8, 128, 128),   // Config 12
        FMHAConfig(30720, 32, 32, 1, 16, 128, 128),  // Config 13
        
        // Test 32 heads with various seqlen_kv, head_dim=256
        FMHAConfig(30720, 32, 32, 1, 1, 256, 256),   // Config 14
        FMHAConfig(30720, 32, 32, 1, 2, 256, 256),   // Config 15
        FMHAConfig(30720, 32, 32, 1, 8, 256, 256),   // Config 16
        FMHAConfig(30720, 32, 32, 1, 16, 256, 256),  // Config 17
    };

    // Run benchmark(s)
    if (config_index == -1) {
        // Run all configurations
        for (size_t i = 0; i < configs.size(); i++) {
            std::cout << "\n=== Configuration " << i << " ===\n";
            run_fmha_benchmark(configs[i], 1);
        }
    } else {
        // Run single configuration
        run_fmha_benchmark(configs[config_index], 1);
    }

    std::cout << "\n\nBenchmark completed successfully!\n";
    return EXIT_SUCCESS;
}
