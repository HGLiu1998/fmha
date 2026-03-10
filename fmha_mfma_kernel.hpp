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

#define ASM_DEBUG(maker) \
    __builtin_amdgcn_sched_barrier(0); \
    asm volatile(maker);               \
    __builtin_amdgcn_sched_barrier(0); \

__global__
__launch_bounds__(256, 2)
void fmha_mfma(
    const bhalf_t* __restrict__ Q,         // [B, H_q,  S_q,  D] bf16
    const bhalf_t* __restrict__ K,         // Packed: [1, total_S_kv, H_kv, D]
    const bhalf_t* __restrict__ V,         // Packed: [1, total_S_kv, H_kv, D]
    half_t*        __restrict__ O,         // [B, H_q,  S_q,  D] fp16
    const int* __restrict__ cu_seqlens_kv, // [B+1] cumulative seqlens (cu_seqlens_kv[i] = start offset for batch i)
    const int batch,
    const int num_heads_q,
    const int num_heads_kv,
    const int seqlen_q,                    // always 1 (decode)
    const int head_dim_q,                  // 128 or 256
    const int head_dim_kv,
    const float softmax_scale)
{
    // ========================================================================
    // Block / Thread Mapping
    // ========================================================================
    const int batch_idx = blockIdx.z;
    const int head_idx  = blockIdx.y;   // one block per head
    const int tid       = threadIdx.x;  // [0, 256)
    const int warp_id   = tid / 64;     // [0, 4)
    const int lane_id   = tid % 64;     // [0, 64)

    const int lane_row   = lane_id / 16;  // row in 16x16 result
    const int lane_col = lane_id % 16;    // which 4-column block (0-3)

    // Get actual seqlen_kv and offset for this batch
    // Packed layout: K/V are [1, total_seqlen_kv, H, D]
    const int seqlen_start = cu_seqlens_kv[batch_idx];
    const int seqlen_end = cu_seqlens_kv[batch_idx + 1];
    const int seqlen_kv = seqlen_end - seqlen_start;
    
    // Offset: skip to this batch's sequences, then to this head
    // Layout: [seqlen, head, dim] so stride is head_dim_kv, then num_heads_kv * head_dim_kv
    const size_t kv_offset = (size_t)seqlen_start * num_heads_kv * head_dim_kv  // skip previous batches
                            + head_idx * head_dim_kv;

    // ========================================================================
    // Adjust Pointers to This Batch + Head
    // ========================================================================
    const bhalf_t* Q_ptr = Q + (size_t)batch_idx * num_heads_q * seqlen_q * head_dim_q
                             + (size_t)head_idx * seqlen_q * head_dim_q;
    const bhalf_t* K_ptr = K + kv_offset;
    const bhalf_t* V_ptr = V + kv_offset;
    half_t* O_ptr = O + (size_t)batch_idx * num_heads_q * seqlen_q * head_dim_q
                      + (size_t)head_idx * seqlen_q * head_dim_q;
    const uint BK = 64;
    
    // K/V stride for packed layout: [seqlen, head, dim]
    const int kv_stride = num_heads_kv * head_dim_kv;
    
    constexpr int max_seqlen_kv = 16;
    constexpr int max_hd_pad = 256 + 4;
    __shared__ __attribute__((aligned(128))) bhalf_t Q_lds[max_hd_pad];
    __shared__ __attribute__((aligned(128))) bhalf_t KV_lds[max_seqlen_kv * max_hd_pad];
    __shared__ __attribute__((aligned(128))) float scores[256];
    __shared__ __attribute__((aligned(128))) bhalf_t softmax_scores[256];

    scores[tid] = 0.0f;
    softmax_scores[tid] = 0.0f;

    const int elements_per_thread = 8;
    const int threads_per_row = head_dim_q / elements_per_thread;
    const int rounds = 256 / threads_per_row;
    const int kv_row = tid / threads_per_row;
    const int kv_col = (tid % threads_per_row) * elements_per_thread;

    ASM_DEBUG("; Load K to LDS and preload V");
    for (int r = kv_row; r < seqlen_kv; r += rounds) {
        *(bf16x8*)(&KV_lds[r * max_hd_pad + kv_col]) = *(const bf16x8*)(&K_ptr[r * kv_stride + kv_col]);
    }

    bf16x8 v_reg_0 = {0}, v_reg_1 = {0};
    if (kv_row < seqlen_kv) {
        v_reg_0 = *(const bf16x8*)(&V_ptr[kv_row * kv_stride + kv_col]);
    }
    const int kv_row_1 = kv_row + rounds;
    if (kv_row_1 < seqlen_kv) {
        v_reg_1 = *(const bf16x8*)(&V_ptr[kv_row_1 * kv_stride + kv_col]);
    }
    
    __syncthreads();

    floatx4 acc = {0};
    bf16x4 a = {0}, b = {0};

    ASM_DEBUG("; Q @ K^T");

    if (warp_id == 0) {
        const uint aRegLoc = lane_row * 4 + lane_col * head_dim_q;
        const uint bRegLoc = lane_row * 4 + lane_col * max_hd_pad;
        for (int k = 0; k < CEIL_DIV(head_dim_q, 16); ++k) {
            const uint dim_idx = k * 16;
            if (dim_idx + aRegLoc < head_dim_q) {
                a = *(bf16x4*)(&Q_ptr[dim_idx + aRegLoc]);
            }
            if (dim_idx + bRegLoc < seqlen_kv * kv_stride) {
                b = *(bf16x4*)(&KV_lds[dim_idx + bRegLoc]);
            }

            acc = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a, b, acc, 0, 0, 0);
        }

        if (lane_row == 0 && lane_col < seqlen_kv) {
            scores[lane_col] = acc[0];  // Only first element holds row 0
        }
    }   


    ASM_DEBUG("; Softmax");

    if (tid < seqlen_kv) {
        scores[tid] *= softmax_scale;
    }

    __syncthreads();

    // Step 1: Find global max across all threads
    float maxVal = -INFINITY;
    if (tid < seqlen_kv) {
        maxVal = scores[tid];
    }

    // Reduce max within warp (64 threads on AMD)
    for (int offset = 8; offset > 0; offset /= 2) {
        maxVal = fmaxf(maxVal, __shfl_xor(maxVal, offset, 16));
    }

    // Step 2: Compute exp and sum
    float sumExp = 0.0f;
    if (tid < seqlen_kv) {
        scores[tid] = expf(scores[tid] - maxVal);
        sumExp = scores[tid];
    }

    // Reduce sum within warp
    for (int offset = 8; offset > 0; offset /= 2) {
        sumExp += __shfl_xor(sumExp, offset, 16);
    }
    // Step 3: Normalize
    if (tid < seqlen_kv) {
        scores[tid] /= sumExp;
        softmax_scores[tid] = static_cast<__bf16>(scores[tid]);
    }
    if (tid >= seqlen_kv && tid < 16) {
        softmax_scores[tid] = static_cast<__bf16>(0.0f);
    }
    __syncthreads();

    ASM_DEBUG("; load V to LDS");


    if (kv_row < seqlen_kv) {
        *(bf16x8*)(&KV_lds[kv_row * max_hd_pad + kv_col]) = v_reg_0;
    }
    if (kv_row_1 < seqlen_kv) {
        *(bf16x8*)(&KV_lds[kv_row_1 * max_hd_pad + kv_col]) = v_reg_1;
    }
    __syncthreads();


    const uint aRegLoc = lane_row * 4 + lane_col * 16;
    const uint bRegLoc = lane_row * 4 * max_hd_pad + lane_col;

    
    ASM_DEBUG("; S @ V");

    a = *(bf16x4*)(&softmax_scores[aRegLoc]);

    for (int d = 0; d < CEIL_DIV(head_dim_q, BK); d += 1) {
        b = {0};
        acc = {0};
        const uint dim_idx = d * BK + warp_id * 16;

        
        for (int i = 0; i < 4; ++i) {
            if (lane_row * 4 + i < seqlen_kv) {
                b[i] = KV_lds[bRegLoc + dim_idx + i * max_hd_pad];
            }
        }
        acc = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a, b, acc, 0, 0, 0);


        if (lane_row == 0) {
            O_ptr[lane_col + dim_idx] = static_cast<half_t>(acc[0]);
        }

    }
}   
