#include <hip/hip_runtime.h>
#include <stdio.h>
#include <math.h>

using bhalf_t = __bf16;
using half_t  = _Float16;
using bf16x2 = __bf16 __attribute__((ext_vector_type(2)));
using bf16x4 = __bf16 __attribute__((ext_vector_type(4)));
using floatx4 = float __attribute__((ext_vector_type(4)));

#define WARP_SIZE 64
#define BLOCK_SIZE 256

__global__
__launch_bounds__(256, 1)
void fmha_forward_kernel(
    const bhalf_t* Q,        // [B, H_q, S_q, D] - bf16
    const bhalf_t* K,        // [B, H_kv, S_kv, D] - bf16
    const bhalf_t* V,        // [B, H_kv, S_kv, D] - bf16
    half_t* O,               // [B, H_q, S_q, D] - fp16
    const int batch,
    const int num_heads_q,
    const int num_heads_kv,
    const int seqlen_q,
    const int seqlen_kv,
    const int head_dim_q,
    const int head_dim_kv,
    const float softmax_scale
) {
    // Thread mapping: (batch, head, seq_position)
    const int batchIdx = blockIdx.z;
    //const int head_idx = blockIdx.y; No head. we reduce head dimension to seqlen.
    // move pointer to the start of the batch
    Q += batchIdx * num_heads_q * seqlen_q * head_dim_q;
    K += batchIdx * num_heads_kv * seqlen_kv * head_dim_kv;
    V += batchIdx * num_heads_kv * seqlen_kv * head_dim_kv;
    O += batchIdx * num_heads_q * seqlen_q * head_dim_q;
    
    const int tid = threadIdx.x;
    const int warpIdx = tid / 64; 
    const int laneIdx = tid % 64;

    const int headIdx = laneIdx / 4; //[0-15];
    const int laneInHead = laneIdx % 4; //[0-3];

    Q += headIdx * seqlen_q * head_dim_q;
    // Need to handle K transpose or not
    K += headIdx * seqlen_kv; 
    V += headIdx * seqlen_kv * head_dim_kv;
    O += headIdx * seqlen_q * head_dim_q;

    const int kv_head_idx = (head_idx * num_heads_kv) / num_heads_q;
    const int M_iter = CEIL_DIV(seqlen_kv, 4); // seqlen_k might be 1 to 16.
    const int N_iter = CEIL_DIV(seqlen_q, 4); // seqlen_q fix to 1. Don't need to iterate now.

    // Q will be 1 * 16 * 128 * 2B = 4KB to 1 * 32 * 256 * 2B = 16KB.
    // K will be 4KB to 64KB. If tile size = 4, then need 4 * 16/32 * 128/256 * 2B = 16KB to 32KB.
    // scores will be 16/32 * 1 * [1 - 16] * 4B = 64B to 2KB.
    // 256 threads load seqlen_q * head_dim_q * num_of_heads 
    __shared__ __attribute__((aligned(128))) bhalf_t Qs[num_heads_q * seqlen_q * (head_dim_q + 4)];
    __shared__ __attribute__((aligned(128))) bhalf_t Ks[num_heads_kv * 4 * (head_dim_kv + 4)]; // V reuse it?
    __shared__ __attribute__((aligned(128))) float scores[num_heads_kv * seqlqn_q * seqlen_kv];

    const int total_q_elems = head_dim_q * seqlen_q * num_heads_q;
    const int total_k_elems = head_dim_kv * seqlen_kv * num_heads_kv;
    constexpr int elems_q_per_thread = total_q_elems / 256;
    constexpr int elems_k_per_thread = total_k_elems / 256;
    constexpr int vec_size = 4; // one thread need 4 elements to do instructions.
    constexpr int vec_q_per_thread = CEIL_DIV(elems_q_per_thread, vec_size);
    constexpr int vec_k_per_thread = CEIL_DIV(elems_k_per_thread, vec_size);

    
    if (warpIdx == 0) {
        #pragma unroll
        for (int i = 0; i < vec_q_per_thread; ++ i) {
            int flat_idx = (tid * vec_q_per_thread + i) * vec_size;

            int h = flat_idx / (head_dim_q * seqlen_q);
            int d = flat_idx % head_dim_q;

            int qLDSOffset = h * (head_dim_q + 4) * seqlen_q + d;
            int qGlobalOffset = h * head_dim_q * seqlen_q + d;

            *(bf16x4*)(&Qs[qLDSOffset]) = Q[qGlobalOffset];
        }
        
        for (int i = 0; i < M_iter; ++i) {
            #pragma unroll
            for (int i = 0; i < vec_k_per_thread; ++ i) {
                int flat_idx = (tid * vec_k_per_thread + i) * vec_size;
                int h - flat_idx / (4 * 128);
                
            }
        }
    
    }

    

    bf16x4 q, k, v;
    floatx4
    
}