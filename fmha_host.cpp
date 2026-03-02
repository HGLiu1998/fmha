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
    int seqlen_q;       // Query sequence length (always 1 for decode)
    int max_seqlen_kv;  // Maximum seqlen_kv (for Poisson range: 2-16)
    int head_dim_q;     // Dimension per query head
    int head_dim_kv;    // Dimension per key/value head

    FMHAConfig(int b, int nhq, int nhkv, int sq, int max_skv, int hdq, int hdkv)
        : batch(b), num_heads_q(nhq), num_heads_kv(nhkv),
          seqlen_q(sq), max_seqlen_kv(max_skv), head_dim_q(hdq), head_dim_kv(hdkv) {}

    // Generate variable sequence lengths using Poisson distribution (λ=4, range 2-16)
    std::vector<int> generate_seqlens_kv() const {
        std::vector<int> seqlens(batch);
        // Poisson distribution with λ=4, clamped to [2, max_seqlen_kv]
        std::mt19937 gen(42);  // Fixed seed for reproducibility
        std::poisson_distribution<int> dist(4);
        for (int i = 0; i < batch; i++) {
            int len = dist(gen);
            seqlens[i] = std::max(2, std::min(len, max_seqlen_kv));
        }
        return seqlens;
    }
    
    // Compute cumulative sequence lengths (offsets for packed representation)
    std::vector<int> compute_cu_seqlens_kv(const std::vector<int>& seqlens_kv) const {
        std::vector<int> cu_seqlens(batch + 1);
        cu_seqlens[0] = 0;
        for (int i = 0; i < batch; i++) {
            cu_seqlens[i + 1] = cu_seqlens[i] + seqlens_kv[i];
        }
        return cu_seqlens;
    }
    
    // Tensor sizes in number of elements
    size_t q_size() const { return batch * num_heads_q * seqlen_q * head_dim_q; }
    size_t o_size() const { return batch * num_heads_q * seqlen_q * head_dim_q; }
    
    // KV size for packed representation
    size_t kv_size_packed(const std::vector<int>& seqlens_kv) const {
        int total_seqlen = 0;
        for (int len : seqlens_kv) total_seqlen += len;
        return num_heads_kv * total_seqlen * head_dim_kv;
    }

    void print(const std::vector<int>& seqlens_kv) const {
        std::cout << "FMHA Configuration:\n"
                  << "  Batch size: " << batch << " (decode mode: " << batch << " independent requests)\n"
                  << "  Num heads Q: " << num_heads_q << "\n"
                  << "  Num heads KV: " << num_heads_kv << (num_heads_q == num_heads_kv ? " (MHA)" : " (GQA)") << "\n"
                  << "  Sequence length Q: " << seqlen_q << " (decode: generating 1 token)\n"
                  << "  Sequence length KV: Variable (Poisson λ=4, range [2, " << max_seqlen_kv << "])\n"
                  << "  Head dimension Q: " << head_dim_q << "\n"
                  << "  Head dimension KV: " << head_dim_kv << "\n"
                  << "  Data types: Q/K/V=bf16, O=fp16, compute=fp32\n"
                  << "  Kernel: MFMA 16x16x16 (bf16→fp32 accumulation)\n"
                  << "  Memory layout: Packed (no padding between batches)\n";
        
        size_t kv_size = kv_size_packed(seqlens_kv);
        std::cout << "  Q tensor size: " << q_size() * sizeof(bhalf_t) / (1024.0 * 1024.0) << " MB (bf16)\n";
        std::cout << "  K tensor size: " << kv_size * sizeof(bhalf_t) / (1024.0 * 1024.0) << " MB (bf16, packed)\n";
        std::cout << "  V tensor size: " << kv_size * sizeof(bhalf_t) / (1024.0 * 1024.0) << " MB (bf16, packed)\n";
        std::cout << "  O tensor size: " << o_size() * sizeof(half_t) / (1024.0 * 1024.0) << " MB (fp16)\n";
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
        mat[i] = static_cast<bhalf_t>(dis(gen));
        //mat[i] = static_cast<bhalf_t>(0.1f);
    }
}

bool do_validation(const FMHAConfig& config, bhalf_t *h_Q, bhalf_t *h_K, bhalf_t *h_V, half_t *ref_O, half_t *h_O)
{
   // std::cout << "start validation" << std::endl;

    float scores[16] = {0};
    bool res = true;

    for (int b = 0; b < config.batch; b++) {
        for (int h = 0; h < config.num_heads_q; h++) {
            // Step 1: Compute Q @ K^T scores for this head
            for (int s = 0; s < config.seqlen_kv; s++) {
                float sum = 0;
                for (int d = 0; d < config.head_dim_q; d++) {
                    bhalf_t q = h_Q[b * config.seqlen_q * config.head_dim_q * config.num_heads_q + h * config.seqlen_q * config.head_dim_q + d];
                    bhalf_t k = h_K[b * config.seqlen_kv * config.head_dim_kv * config.num_heads_kv + h * config.seqlen_kv * config.head_dim_kv + s * config.head_dim_kv + d];  
                    sum += (float)q * (float)k;                 
                }
                scores[s] = sum;
            }
            if (h == 0 && b == 0)  {
                printf("Scores: ");
                for (int i = 0; i < config.seqlen_kv; ++i) {
                    printf("%f ", scores[i]);
                }
                printf("\n");
            }
            for (int s = 0; s < config.seqlen_kv; s++) {
                scores[s] *= (1.0f / sqrtf(config.head_dim_q));
            }

            // Step 2: Softmax + P @ V for this head
            bhalf_t softmax_scores[16] = {0};
            float maxVal = -INFINITY;
            float sumExp = 0.0f;
            for (int s = 0; s < config.seqlen_kv; s++) {
                maxVal = fmaxf(maxVal, scores[s]);
            }
            for (int s = 0; s < config.seqlen_kv; s++) {
                sumExp += expf(scores[s] - maxVal);
            }
            for (int s = 0; s < config.seqlen_kv; s++) {
                softmax_scores[s] = static_cast<bhalf_t>(expf(scores[s] - maxVal) / sumExp);
            }
            for (int d = 0; d < config.head_dim_q; d++) {
                float sum = 0;
                for (int s = 0; s < config.seqlen_kv; s++) {
                    sum += (float)(softmax_scores[s] * (float)h_V[b * config.seqlen_kv * config.head_dim_kv * config.num_heads_kv + h * config.seqlen_kv * config.head_dim_kv + s + d * config.seqlen_kv]);
                }
                ref_O[b * config.num_heads_q * config.seqlen_q * config.head_dim_q + h * config.seqlen_q * config.head_dim_q + d] = static_cast<half_t>(sum);
            }
        }

        // Step 3: Validate this batch element
        for (int h = 0; h < config.num_heads_q; h++) {
            for (int s = 0; s < config.seqlen_q; s++) {
                for (int d = 0; d < config.head_dim_q; d++) {
                    double o = (float)h_O[b * config.num_heads_q * config.seqlen_q * config.head_dim_q + h * config.seqlen_q * config.head_dim_q + s * config.head_dim_q + d];
                    double r = (float)ref_O[b * config.num_heads_q * config.seqlen_q * config.head_dim_q + h * config.seqlen_q * config.head_dim_q + s * config.head_dim_q + d];
                    double err = std::abs(o - r);
                    if (err > 1e-2 + 1e-2 * std::abs(r)) {
                        //std::cout << "Error! out " << o << " != ref " << r << std::endl;
                        res = false;
                    }
                }
            }
        }
    }
    if (res) std::cout << "Validation success!" << std::endl;
    return res;

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
void run_fmha_benchmark(const FMHAConfig& config, int num_iterations = 20, int warm_ups = 5) {
    std::cout << "\n=== Running FMHA Benchmark ===\n";
    
    // Generate per-batch sequence lengths
    std::vector<int> seqlens_kv = config.generate_seqlens_kv();
    std::vector<int> cu_seqlens_kv = config.compute_cu_seqlens_kv(seqlens_kv);
    
    config.print(seqlens_kv);
    
    std::cout << "\nVariable sequence lengths (first 10): ";
    for (int i = 0; i < std::min(10, (int)seqlens_kv.size()); i++) {
        std::cout << seqlens_kv[i] << " ";
    }
    std::cout << "...\n";
    
    // Calculate statistics
    double avg = 0;
    int total_seqlen = cu_seqlens_kv[config.batch];
    for (int len : seqlens_kv) avg += len;
    avg /= seqlens_kv.size();
    std::cout << "Average seqlen_kv: " << avg << "\n";
    std::cout << "Total seqlen_kv: " << total_seqlen << " (packed, no padding)\n";

    // Allocate host memory (bf16 for inputs, fp16 for output)
    size_t kv_size = config.kv_size_packed(seqlens_kv);
    
    bhalf_t *h_Q = (bhalf_t*)malloc(config.q_size() * sizeof(bhalf_t));
    bhalf_t *h_K = (bhalf_t*)malloc(kv_size * sizeof(bhalf_t));
    bhalf_t *h_V = (bhalf_t*)malloc(kv_size * sizeof(bhalf_t));
    half_t  *h_O = (half_t*)malloc(config.o_size() * sizeof(half_t));
    half_t  *ref_O = (half_t*)malloc(config.o_size() * sizeof(half_t));
    int     *h_cu_seqlens_kv = (int*)malloc((config.batch + 1) * sizeof(int));
    std::copy(cu_seqlens_kv.begin(), cu_seqlens_kv.end(), h_cu_seqlens_kv);

    // Initialize with random data
    std::cout << "\nInitializing tensors...\n";
    initialize_random_bfloat16(h_Q, config.q_size());
    initialize_random_bfloat16(h_K, kv_size);
    initialize_random_bfloat16(h_V, kv_size);

    // Allocate device memory
    bhalf_t *d_Q, *d_K, *d_V;
    half_t  *d_O;
    int     *d_cu_seqlens_kv = nullptr;
    HIP_CHECK(hipMalloc((void**)&d_Q, config.q_size() * sizeof(bhalf_t)));
    HIP_CHECK(hipMalloc((void**)&d_K, kv_size * sizeof(bhalf_t)));
    HIP_CHECK(hipMalloc((void**)&d_V, kv_size * sizeof(bhalf_t)));
    HIP_CHECK(hipMalloc((void**)&d_O, config.o_size() * sizeof(half_t)));
    HIP_CHECK(hipMalloc((void**)&d_cu_seqlens_kv, (config.batch + 1) * sizeof(int)));

    // Copy input data to device
    std::cout << "Preparing data on GPU...\n";
    HIP_CHECK(hipMemset(d_O, 0, config.o_size() * sizeof(half_t)));
    HIP_CHECK(hipMemcpy(d_Q, h_Q, config.q_size() * sizeof(bhalf_t), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_K, h_K, kv_size * sizeof(bhalf_t), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_V, h_V, kv_size * sizeof(bhalf_t), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_cu_seqlens_kv, h_cu_seqlens_kv, (config.batch + 1) * sizeof(int), hipMemcpyHostToDevice));

    // Configure kernel launch parameters
    // Grid: (1, num_heads_q, batch) - One block per head per batch element
    dim3 gridDim(1, config.num_heads_q, config.batch);
    dim3 blockDim(256, 1, 1);

    // Create HIP events for timing
    hipEvent_t start, end;

    // Warm-up phase: stabilize GPU clocks and caches
    std::cout << "Running " << warm_ups << " warm-up iterations...\n";
    for (int i = 0; i < warm_ups; ++i) {
        fmha_mfma<<<gridDim, blockDim>>>(d_Q, d_K, d_V, d_O, d_cu_seqlens_kv,
                           config.batch, config.num_heads_q, config.num_heads_kv,
                           config.seqlen_q, config.max_seqlen_kv,
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
        fmha_mfma<<<gridDim, blockDim>>>(d_Q, d_K, d_V, d_O, d_cu_seqlens_kv,
                           config.batch, config.num_heads_q, config.num_heads_kv,
                           config.seqlen_q, config.max_seqlen_kv,
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
    long long flops_per_iter = 4LL * config.batch * config.num_heads_q *
                               config.seqlen_q * config.seqlen_kv * config.head_dim_q;
    double tflops = (flops_per_iter / (avg_time_ms / 1000.0)) / 1e12;

    // Display results
    std::cout << "\n=== Performance Results ===\n";
    std::cout << "Average time:  " << avg_time_ms << " ms per iteration\n";
    std::cout << "Throughput:    " << 1000.0 / avg_time_ms << " iter/s\n";
    std::cout << "Performance:   " << tflops << " TFLOPS (compute only, excludes softmax)\n";
    std::cout << "Per-head latency: " << (avg_time_ms * 1000.0) / (config.batch * config.num_heads_q) << " μs\n";
    
    // Memory bandwidth estimate
    double bytes_per_iter = (config.q_size() + 2 * config.kv_size()) * sizeof(bhalf_t) + 
                            config.o_size() * sizeof(half_t);
    double bandwidth_gbs = (bytes_per_iter / (avg_time_ms / 1000.0)) / 1e9;
    std::cout << "Memory bandwidth: " << bandwidth_gbs << " GB/s\n";

    // Verify correctness: run one more iteration and check output
    HIP_CHECK(hipMemset(d_O, 0, config.o_size() * sizeof(half_t)));
    fmha_mfma<<<gridDim, blockDim>>>(d_Q, d_K, d_V, d_O, d_cu_seqlens_kv,
        config.batch, config.num_heads_q, config.num_heads_kv,
        config.seqlen_q, config.max_seqlen_kv,
        config.head_dim_q, config.head_dim_kv,
        1.0f / sqrt(config.head_dim_q));
    HIP_CHECK(hipMemcpy(h_O, d_O, config.o_size() * sizeof(half_t), hipMemcpyDeviceToHost));

    // Note: Validation needs implementation for variable sequence lengths
    std::cout << "Validation: TODO - implement per-batch varlen validation\n";

    // Cleanup
    free(h_Q); free(h_K); free(h_V); free(h_O); free(ref_O); free(h_cu_seqlens_kv);
    HIP_CHECK(hipFree(d_Q));
    HIP_CHECK(hipFree(d_K));
    HIP_CHECK(hipFree(d_V));
    HIP_CHECK(hipFree(d_O));
    HIP_CHECK(hipFree(d_cu_seqlens_kv));
} 

/**
 * Main Entry Point
 * 
 * Usage: ./fmha_benchmark [config_index]
 *   config_index: 0-3 (selects one of 4 test configurations)
 *                 If not provided, runs all configurations
 * 
 * Test Matrix:
 *   - Batch: 30720 (fixed, decode mode for LLM inference)
 *   - Heads: 16, 32
 *   - Head dimension: 128, 256
 *   - Sequence length Q: 1 (decode mode - generating 1 token at a time)
 *   - Sequence length KV: Variable (Poisson λ=4, range [2, 16])
 * 
 * Memory Layout (Packed Representation):
 *   - Q, O: [B, H, S_q, D] (standard BHSD layout)
 *   - K, V: [1, total_seqlen_kv, H, D] (packed, no padding between batches)
 *     where total_seqlen_kv = sum of all per-batch seqlen_kv values
 *   - cu_seqlens_kv: [B+1] cumulative offsets to locate each batch's KV data
 *     Kernel uses cu_seqlens_kv[batch_idx] to find start offset for each batch
 */
int main(int argc, char* argv[]) {
    std::cout << "Flash Multi-Head Attention (FMHA) - HIP Benchmark\n";
    std::cout << "==================================================\n\n";

    // Parse command-line arguments
    int config_index = -1;  // -1 means run all configs
    if (argc >= 2) {
        config_index = atoi(argv[1]);
        if (config_index < 0 || config_index > 18) {
            std::cerr << "Error: config_index must be 0-18\n";
            std::cerr << "  Configs 0-15: Uniform seqlen_kv (2, 4, 8, 16)\n";
            std::cerr << "  Configs 16-18: Variable seqlen_kv (Poisson λ=4, range 2-16)\n";
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

    // Define test configurations: 4 core configs with variable seqlen_kv
    // Target spec: Decode mode inference (seqlen_q=1) with large batch (30720)
    // Variable KV cache length (seqlen_kv: Poisson λ=4, range 2-16)
    // Format: FMHAConfig(batch, num_heads_q, num_heads_kv, seqlen_q, max_seqlen_kv, head_dim_q, head_dim_kv)
    std::vector<FMHAConfig> configs = {
        FMHAConfig(30720, 16, 16, 1, 16, 128, 128),  // Config 0: 16 heads, dim 128
        FMHAConfig(30720, 16, 16, 1, 16, 256, 256),  // Config 1: 16 heads, dim 256
        FMHAConfig(30720, 32, 32, 1, 16, 128, 128),  // Config 2: 32 heads, dim 128
        FMHAConfig(30720, 32, 32, 1, 16, 256, 256),  // Config 3: 32 heads, dim 256
    };

    // Run benchmark(s)
    if (config_index == -1) {
        // Run all configurations
        for (size_t i = 0; i < configs.size(); i++) {
            std::cout << "\n=== Configuration " << i << " ===\n";
            run_fmha_benchmark(configs[i]);
        }
    } else {
        // Run single configuration
        run_fmha_benchmark(configs[config_index]);
    }

    std::cout << "\n\nBenchmark completed successfully!\n";
    return EXIT_SUCCESS;
}
