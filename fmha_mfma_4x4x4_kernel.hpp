#pragma once

#include <hip/hip_runtime.h>
#include <stdio.h>
#include <math.h>

using bhalf_t = __bf16;
using half_t  = _Float16;
using bf16x4 = __bf16 __attribute__((ext_vector_type(4)));
using bf16x8 = __bf16 __attribute__((ext_vector_type(8)));
using floatx4 = float __attribute__((ext_vector_type(4)));

#ifndef WARP_SIZE
#define WARP_SIZE 64
#endif
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256
#endif
#ifndef CEIL_DIV
#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))
#endif

// ============================================================================
// MFMA 4x4x4 FMHA Decode Kernel
//
// Uses __builtin_amdgcn_mfma_f32_4x4x4bf16_1k (v_mfma_f32_4x4x4_16b_bf16)
//
// MFMA 4x4x4 with 16 blocks:
//   C[4 × 4]  +=  A[4 × 4]  ×  B[4 × 4]
//   M=4(seq)   N=4(dim/kv)   K=4(reduction)
//   16 blocks = 16 heads (independent)
//
// Thread layout:
//   block_id     = lane_id / 4  → head [0, 16)
//   tid_in_block = lane_id % 4  → N dimension worker [0, 4)
//
//   Thread t provides:  A[t, 0..3] (row t)  and  B[0..3, t] (col t)
//   Thread t receives:  C[0..3, t] (col t)
//
// For decode (seqlen_q = 1):
//   M=4 but only row 0 is populated (seqlen_q=1, rows 1-3 = 0)
//   C[0, t] = useful result for all t
//
// QK^T phase:  C[4(seq) × 4(kv)] += A[4(seq) × 4(dim_k)] × B[4(dim_k) × 4(kv)]
//   A[0,:] = Q[k:k+4]             (tid_in_block==0 only, rows 1-3 = 0)
//   B[:,t] = K[kv_group+t, k:k+4] (each thread handles one kv position)
//   C[0,t] = Q · K[kv_group+t]    → score for kv position kv_group+t
//   Iterate: 4 kv_groups × head_dim/4 k-tiles
//
// V phase:    C[4(seq) × 4(dim)] += A[4(seq) × 4(kv_k)] × B[4(kv_k) × 4(dim)]
//   A[0,:] = weights[kv_base:kv_base+4]  (tid_in_block==0 only)
//   B[:,t] = V[kv_base+0..3, dim+t]      (each thread handles one dim)
//   C[0,t] = Σ_kv w[kv] × V[kv, dim+t]  → output for dim+t
//   Iterate: head_dim/4 dim_groups × seqlen_kv/4 kv_chunks
//
// Warp usage:
//   4 warps all compute the same 16 heads (QK^T redundant, L2-cached)
//   V phase: each warp writes head_dim/4 output dims (no overlap)
//
// Grid: (1, CEIL_DIV(num_heads_q, 16), batch)
// Shared memory: 0 bytes
// Barriers: 0
// ============================================================================

__global__
__launch_bounds__(256, 2)
void fmha_mfma_4x4x4(
    const bhalf_t* __restrict__ Q,         // [B, H_q,  S_q,  D] bf16
    const bhalf_t* __restrict__ K,         // Packed: [1, total_S_kv, H_kv, D]
    const bhalf_t* __restrict__ V,         // Packed: [1, total_S_kv, H_kv, D]
    half_t*        __restrict__ O,         // [B, H_q,  S_q,  D] fp16
    const int* __restrict__ cu_seqlens_kv, // [B+1] cumulative seqlens
    const int batch,
    const int num_heads_q,
    const int num_heads_kv,
    const int seqlen_q,                    // always 1 (decode)
    const int head_dim_q,                  // 128 or 256
    const int head_dim_kv,
    const float softmax_scale)
{
    // ========================================================================
    // Thread Mapping
    //
    // block_id     = lane_id / 4   → head [0, 16)
    // tid_in_block = lane_id % 4   → N-dimension worker [0, 4)
    //   QK^T: t → kv position within group
    //   V:    t → dim position within group
    // ========================================================================
    const int batch_idx    = blockIdx.z;
    const int head_group   = blockIdx.y;      // which group of 16 heads
    const int tid          = threadIdx.x;     // [0, 256)
    const int warp_id      = tid / 64;        // [0, 4)
    const int lane_id      = tid % 64;        // [0, 64)
    const int block_id     = lane_id / 4;     // [0, 16) = head
    const int tid_in_block = lane_id % 4;     // [0, 4)  = N-dim worker

    const int head_idx = head_group * 16 + block_id;
    if (head_idx >= num_heads_q) return;

    // Batch-specific KV info
    const int seqlen_start = cu_seqlens_kv[batch_idx];
    const int seqlen_end   = cu_seqlens_kv[batch_idx + 1];
    const int seqlen_kv    = seqlen_end - seqlen_start;
    const int kv_stride    = num_heads_kv * head_dim_kv;

    // Per-head pointers
    const bhalf_t* Q_ptr = Q + (size_t)batch_idx * num_heads_q * seqlen_q * head_dim_q
                             + (size_t)head_idx * seqlen_q * head_dim_q;
    const bhalf_t* K_base = K + (size_t)seqlen_start * kv_stride
                              + (size_t)head_idx * head_dim_kv;
    const bhalf_t* V_base = V + (size_t)seqlen_start * kv_stride
                              + (size_t)head_idx * head_dim_kv;
    half_t* O_ptr = O + (size_t)batch_idx * num_heads_q * seqlen_q * head_dim_q
                      + (size_t)head_idx * seqlen_q * head_dim_q;

    // ========================================================================
    // Phase 1: QK^T
    //
    // C[4(seq) × 4(kv)] += A[4(seq) × 4(dim_k)] × B[4(dim_k) × 4(kv)]
    //
    //   A[0, i] = Q[k+i]            (row 0 = query, rows 1-3 = 0)
    //   B[i, t] = K[kv_group+t, k+i] (col t = one kv position)
    //   C[0, t] = score[kv_group+t]
    //
    // tid_in_block → kv position within kv_group (N dimension)
    // Iterate: 4 kv_groups × head_dim/4 k-tiles
    // Gather 4 scores per kv_group via __shfl(width=4)
    // ========================================================================
    float scores[16];

    #pragma unroll
    for (int kv_group = 0; kv_group < 16; kv_group += 4) {
        floatx4 acc = {0, 0, 0, 0};
        const int kv_pos = kv_group + tid_in_block;
        const bool kv_valid = (kv_pos < seqlen_kv);

        for (int k = 0; k < head_dim_q; k += 4) {
            bf16x4 a_val, b_val;

            // A: row 0 = Q, rows 1-3 = 0 (seqlen_q = 1)
            if (tid_in_block == 0) {
                a_val = *(const bf16x4*)(&Q_ptr[k]);
            } else {
                a_val = bf16x4{0, 0, 0, 0};
            }

            // B: col t = K at kv position kv_group + t
            if (kv_valid) {
                b_val = *(const bf16x4*)(&K_base[kv_pos * kv_stride + k]);
            } else {
                b_val = bf16x4{0, 0, 0, 0};
            }

            acc = __builtin_amdgcn_mfma_f32_4x4x4bf16_1k(a_val, b_val, acc, 0, 0, 0);
        }

        // C[0, t] = score for kv_group + t
        // Gather within 4-thread sub-group: all threads get all 4 scores
        #pragma unroll
        for (int s = 0; s < 4; s++) {
            scores[kv_group + s] = __shfl(acc[0], s, 4);
        }
    }

    // ========================================================================
    // Phase 2: Softmax (serial in registers, no LDS, no barrier)
    // ========================================================================
    float weights[16];
    {
        float maxVal = -INFINITY;
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            if (i < seqlen_kv)
                maxVal = fmaxf(maxVal, scores[i] * softmax_scale);
        }

        float sumExp = 0.0f;
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            if (i < seqlen_kv) {
                weights[i] = expf(scores[i] * softmax_scale - maxVal);
                sumExp += weights[i];
            } else {
                weights[i] = 0.0f;
            }
        }

        float inv_sum = (sumExp > 0.0f) ? (1.0f / sumExp) : 0.0f;
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            weights[i] *= inv_sum;
        }
    }

    // ========================================================================
    // Phase 3: Attn × V
    //
    // C[4(seq) × 4(dim)] += A[4(seq) × 4(kv_k)] × B[4(kv_k) × 4(dim)]
    //
    //   A[0, i] = weights[kv_base+i]          (row 0 = weights, rows 1-3 = 0)
    //   B[i, t] = V[kv_base+i, dim_group+t]   (col t = one dim position)
    //   C[0, t] = output[dim_group+t]
    //
    // block_id = head (same as QK^T, each block is one head)
    // tid_in_block → dim position within 4-dim group (N dimension)
    // All 16 blocks (heads) process the SAME dim_group simultaneously,
    // each with its own head's weights and V data.
    //
    // Each warp handles head_dim/4 dims to avoid redundant writes.
    // ========================================================================
    const int dims_per_warp = head_dim_q / 4;
    const int warp_dim_start = warp_id * dims_per_warp;

    for (int dg = 0; dg < dims_per_warp; dg += 4) {
        const int dim_group = warp_dim_start + dg;
        const int out_d     = dim_group + tid_in_block;

        floatx4 out_acc = {0, 0, 0, 0};

        for (int kv_base = 0; kv_base < seqlen_kv; kv_base += 4) {
            bf16x4 a_val, b_val;

            // A: row 0 = softmax weights for this kv chunk
            if (tid_in_block == 0) {
                #pragma unroll
                for (int i = 0; i < 4; i++) {
                    int kv = kv_base + i;
                    a_val[i] = (kv < seqlen_kv)
                        ? static_cast<bhalf_t>(weights[kv])
                        : static_cast<bhalf_t>(0.0f);
                }
            } else {
                a_val = bf16x4{0, 0, 0, 0};
            }

            // B: col t = V values at dim out_d across kv positions
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                int kv = kv_base + i;
                if (kv < seqlen_kv) {
                    b_val[i] = V_base[kv * kv_stride + out_d];
                } else {
                    b_val[i] = static_cast<bhalf_t>(0.0f);
                }
            }

            out_acc = __builtin_amdgcn_mfma_f32_4x4x4bf16_1k(a_val, b_val, out_acc, 0, 0, 0);
        }

        // C[0, t] = output[dim_group + t]
        O_ptr[out_d] = static_cast<half_t>(out_acc[0]);
    }
}
