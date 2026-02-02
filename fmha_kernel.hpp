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

    /**
    Q += headIdx * seqlen_q * head_dim_q;
    // Need to handle K transpose or not
    K += headIdx * seqlen_kv * head_dim_kv; 
    V += headIdx * seqlen_kv * head_dim_kv;
    O += headIdx * seqlen_q * head_dim_q;
    **/

    const int kv_head_idx = (head_idx * num_heads_kv) / num_heads_q;
    const int M_iter = CEIL_DIV(seqlen_kv, 4); // seqlen_k might be 1 to 16.
    const int N_iter = CEIL_DIV(seqlen_q, 4); // seqlen_q fix to 1. Don't need to iterate now.

    // Q will be 1 * 16 * 128 * 2B = 4KB to 1 * 32 * 256 * 2B = 16KB.
    // K will be 4KB to 64KB. If tile size = 4, then need 4 * 16/32 * 128/256 * 2B = 16KB to 32KB.
    // scores will be 16/32 * 1 * [1 - 16] * 4B = 64B to 2KB.
    // 256 threads load seqlen_q * head_dim_q * num_of_heads 
    __shared__ __attribute__((aligned(128))) bhalf_t Qs[num_heads_q * seqlen_q * (head_dim_q + 4)];
    __shared__ __attribute__((aligned(128))) bhalf_t Ks[num_heads_kv * 4 * (head_dim_kv + 4)]; // tile size 4
    __shared__ __attribute__((aligned(128))) float scores[num_heads_kv * seqlen_q * seqlen_kv];

    const int total_q_elems = head_dim_q * seqlen_q * num_heads_q;
    constexpr int vec_size = 4;
    constexpr int elems_q_per_thread = total_q_elems / 256;
    constexpr int vec_q_per_thread = CEIL_DIV(elems_q_per_thread, vec_size);

    // K load: per-head, per-tile so each head's data is loaded with correct global/LDS mapping.
    // Tile = 4 rows; last tile has rows_this_tile = min(4, seqlen_kv - 4*m) when seqlen_kv % 4 != 0.
    const int vecs_per_head_row = head_dim_kv / vec_size;  // vectors per row (same head)

    if (warpIdx == 0) {
        #pragma unroll
        for (int i = 0; i < vec_q_per_thread; ++ i) {
            int flat_idx = (tid * vec_q_per_thread + i) * vec_size;
            int h = flat_idx / head_dim_q;
            int d = flat_idx % head_dim_q;
            int qLDSOffset = h * (head_dim_q + 4) * seqlen_q + d;
            int qGlobalOffset = h * head_dim_q * seqlen_q + d;
            *(bf16x4*)(&Qs[qLDSOffset]) = Q[qGlobalOffset];
        }

        for (int m = 0; m < M_iter; ++m) {
            const int row_start = 4 * m;
            const int rows_this_tile = (seqlen_kv - row_start < 4) ? (seqlen_kv - row_start) : 4;  // 1..4, handles seqlen_kv % 4 != 0
            const int vecs_this_tile = num_heads_kv * rows_this_tile * vecs_per_head_row;
            const int vecs_per_thread_k = CEIL_DIV(vecs_this_tile, 256);

            for (int i = 0; i < vecs_per_thread_k; ++i) {
                int vec_idx = tid * vecs_per_thread_k + i;
                if (vec_idx >= vecs_this_tile)
                    continue;
                // Decode: vec_idx = h * (rows_this_tile * vecs_per_head_row) + s_local * vecs_per_head_row + d4/4
                int vec_in_head_tile = vec_idx % (rows_this_tile * vecs_per_head_row);
                int h = vec_idx / (rows_this_tile * vecs_per_head_row);
                int s_local = vec_in_head_tile / vecs_per_head_row;
                int d4 = (vec_in_head_tile % vecs_per_head_row) * vec_size;
                int s_global = row_start + s_local;

                int kGlobalOffset = h * seqlen_kv * head_dim_kv + s_global * head_dim_kv + d4;
                int kLDSOffset = h * 4 * (head_dim_kv + 4) + s_local * (head_dim_kv + 4) + d4;

                // Always 4 consecutive elements in the same row; padding rows (s_local >= rows_this_tile) not accessed here
                *(bf16x4*)(&Ks[kLDSOffset]) = K[kGlobalOffset];
            }
        }
    }

    

    bf16x4 q, k, v;
    floatx4
    
}