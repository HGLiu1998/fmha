#pragma once

#include <hip/hip_runtime.h>
#include <stdio.h>
#include <math.h>

using bhalf_t = __bf16;
using half_t  = _Float16;
using bf16x4 = __bf16 __attribute__((ext_vector_type(4)));
using bf16x8 = __bf16 __attribute__((ext_vector_type(8)));
using floatx4 = float __attribute__((ext_vector_type(4)));

// MFMA input type for gfx90a (builtin uses i16 vector)
//using mfma_bf16x4 = short __attribute__((ext_vector_type(4)));

#define WARP_SIZE 64
#define BLOCK_SIZE 256
#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))
__global__
__launch_bounds__(256, 1)
void fmha_mfma(
    const bhalf_t* __restrict__ Q,     // [B, H_q,  S_q,  D] bf16
    const bhalf_t* __restrict__ K,     // [B, H_kv, S_kv, D] bf16
    const bhalf_t* __restrict__ V,     // [B, H_kv, S_kv, D] bf16
    half_t*        __restrict__ O,     // [B, H_q,  S_q,  D] fp16
    const int batch,
    const int num_heads_q,
    const int num_heads_kv,
    const int seqlen_q,                // always 1 (decode)
    const int seqlen_kv,               // 1-16
    const int head_dim_q,              // 128 or 256
    const int head_dim_kv,             // 128 or 256
    const float softmax_scale)
{
    // ========================================================================
    // Block / Thread Mapping
    // ========================================================================
    const int batch_idx = blockIdx.z;
    const int head_idx  = blockIdx.y;   // one block per head
    const int tid       = threadIdx.x;  // [0, 256)
    const int warp_id   = tid / 64;// [0, 4)
    const int lane_id   = tid % 64;   // [0, 64)

    const int lane_row   = lane_id / 16;  // row in 16x16 result
    const int lane_col = lane_id % 16;  // which 4-column block (0-3)

    // ========================================================================
    // Adjust Pointers to This Batch + Head
    // ========================================================================
    const bhalf_t* Q_ptr = Q + (size_t)batch_idx * num_heads_q  * seqlen_q  * head_dim_q
                             + (size_t)head_idx  * seqlen_q  * head_dim_q;
    const bhalf_t* K_ptr = K + (size_t)batch_idx * num_heads_kv * seqlen_kv * head_dim_kv
                             + (size_t)head_idx  * seqlen_kv * head_dim_kv;
    const bhalf_t* V_ptr = V + (size_t)batch_idx * num_heads_kv * seqlen_kv * head_dim_kv
                             + (size_t)head_idx  * seqlen_kv * head_dim_kv;
    half_t* O_ptr = O + (size_t)batch_idx * num_heads_q * seqlen_q * head_dim_q
                      + (size_t)head_idx  * seqlen_q * head_dim_q;
    const uint BK = 64; 
    
    const uint mem = seqlen_kv * seqlen_q;
    __shared__ __attribute__((aligned(128))) float scores[260] = {0};

    floatx4 acc = {0};
    bf16x4 a = {0}, b = {0};
    for (int k = 0; k < head_dim_q; k += BK) {
        const uint warp_idx = k * BK + warp_id * 16;
        const uint aRegLoc = lane_row * 4 + lane_col * head_dim_q;
        const uint bRegLoc = lane_row * 4 + lane_col * head_dim_kv;
        if (warp_idx + aRegLoc < head_dim_q) {
            a = *(bf16x4*)(&Q_ptr[warp_idx + aRegLoc]);
        }
        if (warp_idx + bRegLoc < seqlen_kv * head_dim_kv) {
            b = *(bf16x4*)(&K_ptr[warp_idx + bRegLoc]);
        }
        __syncthreads();

        acc = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a, b, acc, 0, 0, 0);
        if (tid == 0 || tid == 1 || tid == 16) {
            printf("%d, %f, %f, %f, %f\n", tid, (float)(a[0]), (float)(a[1]), (float)(a[2]), (float)(a[3]));
            printf("%d, %f, %f, %f, %f\n", tid, (float)(b[0]), (float)(b[1]), (float)(b[2]), (float)(b[3]));
            printf("%d, %f, %f, %f, %f\n", tid, (float)(acc[0]), (float)(acc[1]), (float)(acc[2]), (float)(acc[3]));
        }
    }

    for (int i = 0; i < 4; ++i) {
        const uint row = i / 4;
        int idx = (lane_row * 4 + row) * 16 + lane_col;
        scores[idx] += acc[i];
    }

    __syncthreads();
    if (tid == 0) {
        for (int i = 0; i < seqlen_kv; ++i) {
            printf("%f ", scores[i]);
        }
    }
    
}   
