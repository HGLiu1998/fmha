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
#include "fmha_mfma_coalesced_kernel.hpp"
#include "fmha_mfma_allwarp_kernel.hpp"
#include "fmha_mfma_4x4x4_kernel.hpp"

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
            int len;
            do {
                len = dist(gen);
            } while (len < 2 || len > max_seqlen_kv);
            seqlens[i] = len;
            //seqlens[i] = 16;
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

/**
 * CPU Reference Implementation for Variable-Length FMHA
 * 
 * Validates GPU output against CPU computation using packed K/V layout.
 * K/V layout: [1, total_seqlen_kv, H, D] where sequences are packed without padding
 * 
 * @param config Configuration parameters
 * @param seqlens_kv Per-batch actual sequence lengths [B]
 * @param cu_seqlens_kv Cumulative sequence offsets [B+1]
 * @param h_Q Query tensor (host): [B, H_q, S_q, D]
 * @param h_K Key tensor (host, packed): [1, total_seqlen_kv, H_kv, D]
 * @param h_V Value tensor (host, packed): [1, total_seqlen_kv, H_kv, D]
 * @param ref_O Reference output (computed): [B, H_q, S_q, D]
 * @param h_O GPU output (to validate): [B, H_q, S_q, D]
 */
bool do_validation(const FMHAConfig& config, 
                   const std::vector<int>& seqlens_kv,
                   const std::vector<int>& cu_seqlens_kv,
                   bhalf_t *h_Q, bhalf_t *h_K, bhalf_t *h_V, 
                   half_t *ref_O, half_t *h_O)
{
    float scores[16] = {0};
    bool res = true;
    int num_errors = 0;
    const int max_errors_to_print = 5;

    for (int b = 0; b < config.batch; b++) {
        // Get this batch's actual seqlen_kv and KV offset
        const int batch_seqlen_kv = seqlens_kv[b];
        const int kv_offset = cu_seqlens_kv[b];  // Start position in packed KV
        const int kv_stride = config.num_heads_kv * config.head_dim_kv;  // Stride for packed layout
        
        for (int h = 0; h < config.num_heads_q; h++) {
            // Step 1: Compute Q @ K^T scores for this head
            // Q layout: [B, H_q, S_q, D] - standard BHSD
            // K layout (packed): [1, total_seqlen_kv, H_kv, D]
            for (int s = 0; s < batch_seqlen_kv; s++) {
                float sum = 0;
                for (int d = 0; d < config.head_dim_q; d++) {
                    // Q index: [b, h, 0, d] (seqlen_q=1 for decode)
                    size_t q_idx = (size_t)b * config.num_heads_q * config.seqlen_q * config.head_dim_q
                                 + (size_t)h * config.seqlen_q * config.head_dim_q
                                 + d;
                    bhalf_t q = h_Q[q_idx];
                    
                    // K index (packed): [kv_offset + s, h, d]
                    // Layout: [seqlen, head, dim] with stride = num_heads_kv * head_dim_kv
                    size_t k_idx = (size_t)(kv_offset + s) * kv_stride
                                 + (size_t)h * config.head_dim_kv
                                 + d;
                    bhalf_t k = h_K[k_idx];
                    
                    sum += (float)q * (float)k;
                }
                scores[s] = sum;
            }
            
            // Debug: Print scores for first batch/head
            if (h == 0 && b == 0) {
                printf("CPU Validation - Scores (batch 0, head 0, seqlen_kv=%d): ", batch_seqlen_kv);
                for (int i = 0; i < batch_seqlen_kv; ++i) {
                    printf("%f ", scores[i]);
                }
                printf("\n");
            }
            
            // Apply softmax scale
            for (int s = 0; s < batch_seqlen_kv; s++) {
                scores[s] *= (1.0f / sqrtf(config.head_dim_q));
            }

            // Step 2: Softmax
            bhalf_t softmax_scores[16] = {0};
            float maxVal = -INFINITY;
            for (int s = 0; s < batch_seqlen_kv; s++) {
                maxVal = fmaxf(maxVal, scores[s]);
            }
            float sumExp = 0.0f;
            for (int s = 0; s < batch_seqlen_kv; s++) {
                sumExp += expf(scores[s] - maxVal);
            }
            for (int s = 0; s < batch_seqlen_kv; s++) {
                softmax_scores[s] = static_cast<bhalf_t>(expf(scores[s] - maxVal) / sumExp);
            }
            
            // Step 3: P @ V (attention-weighted sum)
            // V layout (packed): [1, total_seqlen_kv, H_kv, D]
            for (int d = 0; d < config.head_dim_q; d++) {
                float sum = 0;
                for (int s = 0; s < batch_seqlen_kv; s++) {
                    // V index (packed): [kv_offset + s, h, d]
                    size_t v_idx = (size_t)(kv_offset + s) * kv_stride
                                 + (size_t)h * config.head_dim_kv
                                 + d;
                    sum += (float)softmax_scores[s] * (float)h_V[v_idx];
                }
                // O index: [b, h, 0, d] (seqlen_q=1)
                size_t o_idx = (size_t)b * config.num_heads_q * config.seqlen_q * config.head_dim_q
                             + (size_t)h * config.seqlen_q * config.head_dim_q
                             + d;
                ref_O[o_idx] = static_cast<half_t>(sum);
            }
        }

        // Step 4: Validate this batch element
        for (int h = 0; h < config.num_heads_q; h++) {
            for (int s = 0; s < config.seqlen_q; s++) {  // Always 1 for decode
                for (int d = 0; d < config.head_dim_q; d++) {
                    size_t idx = (size_t)b * config.num_heads_q * config.seqlen_q * config.head_dim_q
                               + (size_t)h * config.seqlen_q * config.head_dim_q
                               + (size_t)s * config.head_dim_q
                               + d;
                    double o = (float)h_O[idx];
                    double r = (float)ref_O[idx];
                    double err = std::abs(o - r);
                    
                    // Relative error tolerance: 1e-2 (1%)
                    if (err > 1e-2 + 1e-2 * std::abs(r)) {
                        if (num_errors < max_errors_to_print) {
                            std::cout << "Error at [b=" << b << ",h=" << h << ",s=" << s << ",d=" << d 
                                      << "]: GPU=" << o << " CPU=" << r << " err=" << err << std::endl;
                        }
                        num_errors++;
                        res = false;
                    }
                }
            }
        }
    }
    
    if (res) {
        std::cout << "✓ Validation passed!" << std::endl;
    } else {
        std::cout << "✗ Validation failed: " << num_errors << " errors found";
        if (num_errors > max_errors_to_print) {
            std::cout << " (showing first " << max_errors_to_print << ")";
        }
        std::cout << std::endl;
    }
    
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
    //dim3 gridDim(1, config.num_heads_q, config.batch);
    dim3 gridDim(1, CEIL_DIV(config.num_heads_q, 16), config.batch);
    dim3 blockDim(256, 1, 1);

    // Create HIP events for timing
    hipEvent_t start, end;

    // Warm-up phase: stabilize GPU clocks and caches
    std::cout << "Running " << warm_ups << " warm-up iterations...\n";
    for (int i = 0; i < warm_ups; ++i) {
        fmha_mfma_4x4x4<<<gridDim, blockDim>>>(d_Q, d_K, d_V, d_O, d_cu_seqlens_kv,
                           config.batch, config.num_heads_q, config.num_heads_kv,
                           config.seqlen_q, config.head_dim_q, config.head_dim_kv,
                           1.0f / sqrt(config.head_dim_q));
    }
    HIP_CHECK(hipDeviceSynchronize());

    // Benchmark phase: timed iterations
    std::cout << "Running " << num_iterations << " timed iterations...\n";
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&end));
    
    HIP_CHECK(hipEventRecord(start, NULL));
    for (int i = 0; i < num_iterations; i++) {
        fmha_mfma_4x4x4<<<gridDim, blockDim>>>(d_Q, d_K, d_V, d_O, d_cu_seqlens_kv,
                           config.batch, config.num_heads_q, config.num_heads_kv,
                           config.seqlen_q, config.head_dim_q, config.head_dim_kv,
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
    
    // Calculate actual compute work for variable-length sequences
    // Total seqlen_kv across all batches
    double avg_seqlen_kv = avg;  // Already computed above
    long long total_seqlen_kv = total_seqlen;  // Already computed above
    
    // Estimate FLOPs for variable-length FMHA:
    // - Q@K^T: For each (batch, head, query_token), compute dot product with seqlen_kv keys
    //   FLOPs = batch * num_heads * seqlen_q * seqlen_kv * head_dim * 2 (MAC = 2 FLOPs)
    // - P@V: For each (batch, head, query_token, output_dim), weighted sum over seqlen_kv values
    //   FLOPs = batch * num_heads * seqlen_q * head_dim * seqlen_kv * 2
    // Total: 4 * batch * num_heads * seqlen_q * seqlen_kv * head_dim
    //
    // For variable lengths, we use the actual total work across all batches:
    // Sum over batches: 4 * num_heads_q * seqlen_q * seqlens_kv[b] * head_dim_q
    long long flops_per_iter = 4LL * config.num_heads_q * config.seqlen_q * 
                               total_seqlen_kv * config.head_dim_q;
    double tflops = (flops_per_iter / (avg_time_ms / 1000.0)) / 1e12;

    // Display results
    std::cout << "\n=== Performance Results ===\n";
    std::cout << "Average time:  " << avg_time_ms << " ms per iteration\n";
    std::cout << "Throughput:    " << 1000.0 / avg_time_ms << " iter/s\n";
    std::cout << "Performance:   " << tflops << " TFLOPS (compute only, excludes softmax)\n";
    std::cout << "Avg seqlen_kv: " << avg_seqlen_kv << " (used for FLOPs: " 
              << flops_per_iter << ")\n";
    std::cout << "Per-head latency: " << (avg_time_ms * 1000.0) / (config.batch * config.num_heads_q) 
              << " μs\n";
    
    // Memory bandwidth estimate (uses actual packed KV size)
    double bytes_per_iter = (config.q_size() * sizeof(bhalf_t)) +
                            (2 * kv_size * sizeof(bhalf_t)) +  // K + V (packed)
                            (config.o_size() * sizeof(half_t));
    double bandwidth_gbs = (bytes_per_iter / (avg_time_ms / 1000.0)) / 1e9;
    std::cout << "Memory bandwidth: " << bandwidth_gbs << " GB/s (actual data transferred)\n";

    // Verify correctness: run one more iteration and check output
    std::cout << "\n=== Validation ===\n";
    HIP_CHECK(hipMemset(d_O, 0, config.o_size() * sizeof(half_t)));
    fmha_mfma_4x4x4<<<gridDim, blockDim>>>(d_Q, d_K, d_V, d_O, d_cu_seqlens_kv,
        config.batch, config.num_heads_q, config.num_heads_kv,
        config.seqlen_q, config.head_dim_q, config.head_dim_kv,
        1.0f / sqrt(config.head_dim_q));
    HIP_CHECK(hipMemcpy(h_O, d_O, config.o_size() * sizeof(half_t), hipMemcpyDeviceToHost));

    // Validate GPU output against CPU reference
    do_validation(config, seqlens_kv, cu_seqlens_kv, h_Q, h_K, h_V, ref_O, h_O);

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
        if (config_index < 0 || config_index > 3) {
            std::cerr << "Error: config_index must be 0-3\n";
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
        FMHAConfig(10, 16, 16, 1, 16, 128, 128), // Testing
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
