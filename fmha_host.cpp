#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <chrono>

// Kernel launch function declaration
extern "C" void launch_fmha_forward(
    const half* d_Q,
    const half* d_K,
    const half* d_V,
    half* d_O,
    const int batch,
    const int num_heads_q,
    const int num_heads_kv,
    const int seqlen_q,
    const int seqlen_kv,
    const int head_dim_q,
    const int head_dim_kv,
    hipStream_t stream
);

#define HIP_CHECK(call) \
    do { \
        hipError_t err = call; \
        if (err != hipSuccess) { \
            std::cerr << "HIP Error: " << hipGetErrorString(err) \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

class FMHAConfig {
public:
    int batch;
    int num_heads_q;
    int num_heads_kv;
    int seqlen_q;
    int seqlen_kv;
    int head_dim_q;
    int head_dim_kv;
    
    FMHAConfig(int b, int nhq, int nhkv, int sq, int skv, int hdq, int hdkv)
        : batch(b), num_heads_q(nhq), num_heads_kv(nhkv),
          seqlen_q(sq), seqlen_kv(skv), head_dim_q(hdq), head_dim_kv(hdkv) {}
    
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
                  << "  Q tensor size: " << q_size() * sizeof(half) / (1024.0 * 1024.0) << " MB\n"
                  << "  K tensor size: " << kv_size() * sizeof(half) / (1024.0 * 1024.0) << " MB\n"
                  << "  V tensor size: " << kv_size() * sizeof(half) / (1024.0 * 1024.0) << " MB\n"
                  << "  O tensor size: " << o_size() * sizeof(half) / (1024.0 * 1024.0) << " MB\n";
    }
};

void initialize_random_half(std::vector<half>& vec) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    
    for (size_t i = 0; i < vec.size(); i++) {
        vec[i] = __float2half(dis(gen));
    }
}

void run_fmha_benchmark(const FMHAConfig& config, int num_iterations = 100) {
    std::cout << "\n=== Running FMHA Benchmark ===\n";
    config.print();
    
    // Allocate host memory
    std::vector<half> h_Q(config.q_size());
    std::vector<half> h_K(config.kv_size());
    std::vector<half> h_V(config.kv_size());
    std::vector<half> h_O(config.o_size());
    
    // Initialize with random values
    std::cout << "\nInitializing tensors...\n";
    initialize_random_half(h_Q);
    initialize_random_half(h_K);
    initialize_random_half(h_V);
    
    // Allocate device memory
    half *d_Q, *d_K, *d_V, *d_O;
    HIP_CHECK(hipMalloc(&d_Q, config.q_size() * sizeof(half)));
    HIP_CHECK(hipMalloc(&d_K, config.kv_size() * sizeof(half)));
    HIP_CHECK(hipMalloc(&d_V, config.kv_size() * sizeof(half)));
    HIP_CHECK(hipMalloc(&d_O, config.o_size() * sizeof(half)));
    
    // Copy data to device
    std::cout << "Copying data to device...\n";
    HIP_CHECK(hipMemcpy(d_Q, h_Q.data(), config.q_size() * sizeof(half), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_K, h_K.data(), config.kv_size() * sizeof(half), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_V, h_V.data(), config.kv_size() * sizeof(half), hipMemcpyHostToDevice));
    
    // Create stream
    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));
    
    // Warm-up run
    std::cout << "Warming up...\n";
    launch_fmha_forward(d_Q, d_K, d_V, d_O,
                       config.batch, config.num_heads_q, config.num_heads_kv,
                       config.seqlen_q, config.seqlen_kv,
                       config.head_dim_q, config.head_dim_kv,
                       stream);
    HIP_CHECK(hipStreamSynchronize(stream));
    
    // Benchmark
    std::cout << "Running " << num_iterations << " iterations...\n";
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < num_iterations; i++) {
        launch_fmha_forward(d_Q, d_K, d_V, d_O,
                           config.batch, config.num_heads_q, config.num_heads_kv,
                           config.seqlen_q, config.seqlen_kv,
                           config.head_dim_q, config.head_dim_kv,
                           stream);
    }
    
    HIP_CHECK(hipStreamSynchronize(stream));
    auto end = std::chrono::high_resolution_clock::now();
    
    // Calculate performance metrics
    std::chrono::duration<double, std::milli> elapsed = end - start;
    double avg_time_ms = elapsed.count() / num_iterations;
    
    // Calculate FLOPs (approximate)
    // Attention: Q@K^T (matmul) + softmax + P@V (matmul)
    long long flops_per_iter = 2LL * config.batch * config.num_heads_q * 
                               config.seqlen_q * config.seqlen_kv * config.head_dim_q;
    double tflops = (flops_per_iter / (avg_time_ms / 1000.0)) / 1e12;
    
    std::cout << "\n=== Results ===\n";
    std::cout << "Average time per iteration: " << avg_time_ms << " ms\n";
    std::cout << "Throughput: " << 1000.0 / avg_time_ms << " iterations/second\n";
    std::cout << "Estimated performance: " << tflops << " TFLOPS\n";
    
    // Copy result back to host
    HIP_CHECK(hipMemcpy(h_O.data(), d_O, config.o_size() * sizeof(half), hipMemcpyDeviceToHost));
    
    // Print sample output
    std::cout << "\nSample output values (first 5):\n";
    for (int i = 0; i < std::min(5, (int)h_O.size()); i++) {
        std::cout << "  O[" << i << "] = " << __half2float(h_O[i]) << "\n";
    }
    
    // Cleanup
    HIP_CHECK(hipFree(d_Q));
    HIP_CHECK(hipFree(d_K));
    HIP_CHECK(hipFree(d_V));
    HIP_CHECK(hipFree(d_O));
    HIP_CHECK(hipStreamDestroy(stream));
}

int main(int argc, char** argv) {
    std::cout << "Flash Multi-Head Attention (FMHA) HIP Implementation\n";
    std::cout << "====================================================\n\n";
    
    // Check HIP device
    int device_count;
    HIP_CHECK(hipGetDeviceCount(&device_count));
    std::cout << "Found " << device_count << " HIP device(s)\n";
    
    if (device_count == 0) {
        std::cerr << "No HIP devices found!\n";
        return EXIT_FAILURE;
    }
    
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));
    std::cout << "Using device: " << prop.name << "\n";
    std::cout << "Compute capability: " << prop.major << "." << prop.minor << "\n";
    std::cout << "Total memory: " << prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0) << " GB\n\n";
    
    // Test configurations based on specifications
    std::vector<FMHAConfig> configs = {
        // batch=30720, seqlen_q=1, seqlen_kv=2, head_dim=256, num_heads=16
        FMHAConfig(30720, 16, 16, 1, 2, 256, 256),
        
        // batch=30720, seqlen_q=1, seqlen_kv=2, head_dim=128, num_heads=16
        FMHAConfig(30720, 16, 16, 1, 2, 128, 128),
        
        // batch=30720, seqlen_q=1, seqlen_kv=2, head_dim=256, num_heads=32
        FMHAConfig(30720, 32, 32, 1, 2, 256, 256),
        
        // batch=30720, seqlen_q=1, seqlen_kv=2, head_dim=128, num_heads=32
        FMHAConfig(30720, 32, 32, 1, 2, 128, 128),
    };
    
    // Run benchmarks for each configuration
    for (size_t i = 0; i < configs.size(); i++) {
        std::cout << "\n\n========================================\n";
        std::cout << "Configuration " << (i + 1) << " of " << configs.size() << "\n";
        std::cout << "========================================\n";
        run_fmha_benchmark(configs[i], 100);
    }
    
    std::cout << "\n\nAll benchmarks completed successfully!\n";
    return EXIT_SUCCESS;
}
