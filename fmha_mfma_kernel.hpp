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
    const int head_dim_kv,
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
    
    __shared__ __attribute__((aligned(128))) float scores[256];
    __shared__ __attribute__((aligned(128))) bhalf_t softmax_scores[256];

    scores[tid] = 0.0f;
    __syncthreads();

    floatx4 acc = {0};
    bf16x4 a = {0}, b = {0};
    if (warp_id == 0) {
        for (int k = 0; k < CEIL_DIV(head_dim_q, 16); k += 1) {
            const uint warp_idx = k * 16;
            const uint aRegLoc = lane_row * 4 + lane_col * head_dim_q;
            const uint bRegLoc = lane_row * 4 + lane_col * head_dim_kv;
            if (warp_idx + aRegLoc < head_dim_q) {
                a = *(bf16x4*)(&Q_ptr[warp_idx + aRegLoc]);
            }
            if (warp_idx + bRegLoc < seqlen_kv * head_dim_kv) {
                b = *(bf16x4*)(&K_ptr[warp_idx + bRegLoc]);
            }

            acc = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a, b, acc, 0, 0, 0);

        }
        
        for (int i = 0; i < 4; ++i) {
            int idx = (lane_row * 4 + i) * 16 + lane_col;
            scores[idx] = acc[i];
            //scores[idx] += acc[i];
        }
    }

    __syncthreads();

    if (tid < seqlen_kv) {
        scores[tid] *= softmax_scale;
    }
    // Step 1: Find global max
    float maxVal = -INFINITY;
    if (tid < seqlen_kv) {
        maxVal = scores[tid];
    }
    // Reduce max across threads
    for (int i = 8; i > 0; i /= 2) {
        maxVal = fmaxf(maxVal, __shfl_xor(maxVal, i));
    }
    // Step 2: Compute exp and sum
    float sumExp = 0.0f;
    if (tid < seqlen_kv) {
        scores[tid] = expf(scores[tid] - maxVal);
        sumExp = scores[tid];
    }
    for (int i = 8; i > 0; i /= 2) {
        sumExp += __shfl_xor(sumExp, i);
    }
    if (tid < seqlen_kv) {
        scores[tid] /= sumExp;
        softmax_scores[tid] = static_cast<__bf16>(scores[tid]); 
    }
    if (tid >= seqlen_kv && tid < 16) {
        softmax_scores[tid] = static_cast<__bf16>(0.0f);
    }
    __syncthreads();

    if (tid == 0 && head_idx == 0 && batch_idx == 0)  {
        printf("\nSoftmax %d:\n", tid);
        for (int i = 0 ; i < seqlen_kv; i++) {
            printf("%f, ", (float)softmax_scores[i]);
        }
        printf("\n");
    }


    for (int d = 0; d < CEIL_DIV(head_dim_q, BK); d += 1) {
        a = {0};
        b = {0};
        acc = {0};
        const uint dim_idx = d * BK + warp_id * 16;
        const uint aRegLoc = lane_row * 4 + lane_col * head_dim_q;
        const uint bRegLoc = lane_row * 4 + lane_col * seqlen_kv;
        if (lane_col == 0) {
            a = *(bf16x4*)(&softmax_scores[aRegLoc]);
        }
        if (dim_idx + lane_col < head_dim_kv) {
            if (lane_row * 4 + 4 <= seqlen_kv) {
                b = *(bf16x4*)(&V_ptr[(dim_idx + lane_col) * seqlen_kv + lane_row * 4]);
            } else {
                for (int i = 0; i < 4; i++) {
                    int s = lane_row * 4 + i;
                    if (s < seqlen_kv) {
                        b[i] = V_ptr[(dim_idx + lane_col) * seqlen_kv + s];
                    }
                }
            }
        }
        acc = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a, b, acc, 0, 0, 0);
        for (int i = 0; i < 4; ++i) {
            const uint cRegLoc = (lane_row * 4 + i) * head_dim_q + lane_col + dim_idx; 
            if (cRegLoc < head_dim_q) {
                O_ptr[cRegLoc] = static_cast<half_t>(acc[i]);
            }
        }
        if ((tid == 64 || tid == 0) && head_idx == 0 && batch_idx == 0)  {
            printf("\nAcc %d:\n", tid);
            printf("%f, %f, %f, %f\n", (float)a[0], (float)a[1], (float)a[2], (float)a[3]);
            printf("%f, %f, %f, %f\n", (float)b[0], (float)b[1], (float)b[2], (float)b[3]);
            printf("%f, %f, %f, %f\n", (float)acc[0], (float)acc[1], (float)acc[2], (float)acc[3]);
        }

    }
}   
