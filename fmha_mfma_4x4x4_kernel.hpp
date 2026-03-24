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
// MFMA 4x4x4 FMHA Decode Kernel — LDS-Staged Cooperative Loading
//
// Uses __builtin_amdgcn_mfma_f32_4x4x4bf16_1k (v_mfma_f32_4x4x4_16b_bf16)
//
// MFMA 4x4x4 with 16 blocks:
//   C[4 × 4]  +=  A[4 × 4]  ×  B[4 × 4]
//   16 blocks = 16 heads (independent)
//
// Thread layout:
//   block_id     = lane_id / 4  → head [0, 16)
//   tid_in_block = lane_id % 4  → N dimension worker [0, 4)
//
// LDS Layout:
//   Q_lds:  [16 heads × hd_pad]                  — loaded once
//   KV_lds: [4 kv_positions × 16 heads × hd_pad] — reloaded per kv_group
//   hd_pad = head_dim + 4 (bank conflict padding)
//
// Grid: (1, CEIL_DIV(num_heads_q, 16), batch)
// Block: 256 threads (4 warps)
// ============================================================================

__global__
__launch_bounds__(256, 1)
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
    // ========================================================================
    const int batch_idx    = blockIdx.z;
    const int head_group   = blockIdx.y;
    const int tid          = threadIdx.x;     // [0, 256)
    const int warp_id      = tid / 64;        // [0, 4)
    const int lane_id      = tid % 64;        // [0, 64)
    const int block_id     = lane_id / 4;     // [0, 16) = head within group
    const int tid_in_block = lane_id % 4;     // [0, 4)  = N-dim worker

    const int head_base = head_group * 16;
    const int head_idx  = head_base + block_id;
    if (head_idx >= num_heads_q) return;

    // Batch-specific KV info
    const int seqlen_start = cu_seqlens_kv[batch_idx];
    const int seqlen_end   = cu_seqlens_kv[batch_idx + 1];
    const int seqlen_kv    = seqlen_end - seqlen_start;
    const int kv_stride    = num_heads_kv * head_dim_kv;

    const int hd_pad = head_dim_q + 4;

    // ========================================================================
    // LDS Allocation
    // Q_lds:  16 heads × hd_pad
    // KV_lds: 4 kv_pos × 16 heads × hd_pad
    // ========================================================================
    constexpr int max_hd_pad = 256 + 4;
    __shared__ __attribute__((aligned(128))) bhalf_t Q_lds[16 * max_hd_pad];
    __shared__ __attribute__((aligned(128))) bhalf_t KV_lds[4 * 16 * max_hd_pad];

    // Pointers
    const bhalf_t* Q_batch = Q + (size_t)batch_idx * num_heads_q * seqlen_q * head_dim_q;
    half_t* O_ptr = O + (size_t)batch_idx * num_heads_q * seqlen_q * head_dim_q
                      + (size_t)head_idx * seqlen_q * head_dim_q;
    const bhalf_t* K_batch = K + (size_t)seqlen_start * kv_stride;
    const bhalf_t* V_batch = V + (size_t)seqlen_start * kv_stride;

    // ========================================================================
    // Phase 1: Load Q → Q_lds (cooperative, all 256 threads)
    // ========================================================================
    {
        const int total_q_vals = 16 * head_dim_q;
        const int vals_per_round = 256 * 8;

        for (int round_start = 0; round_start < total_q_vals; round_start += vals_per_round) {
            const int val_idx = round_start + tid * 8;
            if (val_idx < total_q_vals) {
                const int head_local = val_idx / head_dim_q;
                const int dim_offset = val_idx % head_dim_q;
                const int actual_head = head_base + head_local;

                bf16x8 q_data;
                if (actual_head < num_heads_q) {
                    q_data = *(const bf16x8*)(&Q_batch[actual_head * head_dim_q + dim_offset]);
                } else {
                    q_data = bf16x8{0, 0, 0, 0, 0, 0, 0, 0};
                }
                *(bf16x8*)(&Q_lds[head_local * hd_pad + dim_offset]) = q_data;
            }
        }
    }

    // ========================================================================
    // Phase 2: QK^T — process 4 kv positions at a time
    // ========================================================================
    float scores[16];
    #pragma unroll
    for (int i = 0; i < 16; i++) scores[i] = 0.0f;

    const int num_kv_groups = CEIL_DIV(seqlen_kv, 4);

    for (int g = 0; g < num_kv_groups; g++) {
        const int kv_group = g * 4;
        const int n_valid = min(4, seqlen_kv - kv_group);

        // --- Cooperative load K → KV_lds ---
        {
            const int vals_per_kv = 16 * head_dim_q;
            const int vals_per_round = 256 * 8;

            const int total_valid_vals = n_valid * vals_per_kv;
            for (int round_start = 0; round_start < total_valid_vals; round_start += vals_per_round) {
                const int val_idx = round_start + tid * 8;
                if (val_idx < total_valid_vals) {
                    const int kv_local   = val_idx / vals_per_kv;
                    const int within_kv  = val_idx % vals_per_kv;
                    const int head_local = within_kv / head_dim_q;
                    const int dim_offset = within_kv % head_dim_q;

                    const bhalf_t* src = K_batch
                        + (size_t)(kv_group + kv_local) * kv_stride
                        + (size_t)(head_base + head_local) * head_dim_kv
                        + dim_offset;
                    *(bf16x8*)(&KV_lds[kv_local * 16 * hd_pad + head_local * hd_pad + dim_offset])
                        = *(const bf16x8*)(src);
                }
            }

            if (n_valid < 4) {
                const int zero_vals = (4 - n_valid) * vals_per_kv;
                for (int round_start = 0; round_start < zero_vals; round_start += vals_per_round) {
                    const int val_idx = round_start + tid * 8;
                    if (val_idx < zero_vals) {
                        const int kv_local   = n_valid + val_idx / vals_per_kv;
                        const int within_kv  = val_idx % vals_per_kv;
                        const int head_local = within_kv / head_dim_q;
                        const int dim_offset = within_kv % head_dim_q;
                        *(bf16x8*)(&KV_lds[kv_local * 16 * hd_pad + head_local * hd_pad + dim_offset])
                            = bf16x8{0, 0, 0, 0, 0, 0, 0, 0};
                    }
                }
            }
        }

        __syncthreads();

        // --- MFMA loop: Q_lds × K_lds^T ---
        {
            floatx4 acc = {0, 0, 0, 0};

            for (int k = 0; k < head_dim_q; k += 4) {
                bf16x4 a_val, b_val;

                if (tid_in_block == 0) {
                    a_val = *(const bf16x4*)(&Q_lds[block_id * hd_pad + k]);
                } else {
                    a_val = bf16x4{0, 0, 0, 0};
                }

                b_val = *(const bf16x4*)(&KV_lds[tid_in_block * 16 * hd_pad + block_id * hd_pad + k]);

                acc = __builtin_amdgcn_mfma_f32_4x4x4bf16_1k(a_val, b_val, acc, 0, 0, 0);
            }

            #pragma unroll
            for (int s = 0; s < 4; s++) {
                scores[kv_group + s] = __shfl(acc[0], s, 4);
            }
        }

        __syncthreads();
    }

    // ========================================================================
    // Phase 3: Softmax (registers, no LDS, no barrier)
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
    // Phase 4: Attn × V
    //
    // Each warp handles head_dim/4 output dims.
    // v_acc[i] accumulates C[0,t] for dim_group i across all kv_chunks.
    // Max accumulators: D=128 → 32/4=8, D=256 → 64/4=16
    // ========================================================================
    const int dims_per_warp = head_dim_q / 4;
    const int warp_dim_start = warp_id * dims_per_warp;
    const int num_dim_groups = dims_per_warp / 4;  // 8 (D=128) or 16 (D=256)

    float v_acc[16];  // max 16 for D=256
    for (int i = 0; i < num_dim_groups; i++) v_acc[i] = 0.0f;

    const int num_kv_chunks = CEIL_DIV(seqlen_kv, 4);

    for (int c = 0; c < num_kv_chunks; c++) {
        const int kv_chunk = c * 4;
        const int n_valid = min(4, seqlen_kv - kv_chunk);

        // --- Cooperative load V → KV_lds ---
        {
            const int vals_per_kv = 16 * head_dim_q;
            const int vals_per_round = 256 * 8;

            const int total_valid_vals = n_valid * vals_per_kv;
            for (int round_start = 0; round_start < total_valid_vals; round_start += vals_per_round) {
                const int val_idx = round_start + tid * 8;
                if (val_idx < total_valid_vals) {
                    const int kv_local   = val_idx / vals_per_kv;
                    const int within_kv  = val_idx % vals_per_kv;
                    const int head_local = within_kv / head_dim_q;
                    const int dim_offset = within_kv % head_dim_q;

                    const bhalf_t* src = V_batch
                        + (size_t)(kv_chunk + kv_local) * kv_stride
                        + (size_t)(head_base + head_local) * head_dim_kv
                        + dim_offset;
                    *(bf16x8*)(&KV_lds[kv_local * 16 * hd_pad + head_local * hd_pad + dim_offset])
                        = *(const bf16x8*)(src);
                }
            }

            if (n_valid < 4) {
                const int zero_vals = (4 - n_valid) * vals_per_kv;
                for (int round_start = 0; round_start < zero_vals; round_start += vals_per_round) {
                    const int val_idx = round_start + tid * 8;
                    if (val_idx < zero_vals) {
                        const int kv_local   = n_valid + val_idx / vals_per_kv;
                        const int within_kv  = val_idx % vals_per_kv;
                        const int head_local = within_kv / head_dim_q;
                        const int dim_offset = within_kv % head_dim_q;
                        *(bf16x8*)(&KV_lds[kv_local * 16 * hd_pad + head_local * hd_pad + dim_offset])
                            = bf16x8{0, 0, 0, 0, 0, 0, 0, 0};
                    }
                }
            }
        }

        __syncthreads();

        // --- MFMA: weights × V ---
        for (int dg_idx = 0; dg_idx < num_dim_groups; dg_idx++) {
            const int dg = dg_idx * 4;
            const int dim_group = warp_dim_start + dg;
            const int out_d     = dim_group + tid_in_block;

            bf16x4 a_val, b_val;

            if (tid_in_block == 0) {
                #pragma unroll
                for (int i = 0; i < 4; i++) {
                    int kv = kv_chunk + i;
                    a_val[i] = (kv < seqlen_kv)
                        ? static_cast<bhalf_t>(weights[kv])
                        : static_cast<bhalf_t>(0.0f);
                }
            } else {
                a_val = bf16x4{0, 0, 0, 0};
            }

            #pragma unroll
            for (int i = 0; i < 4; i++) {
                b_val[i] = KV_lds[i * 16 * hd_pad + block_id * hd_pad + out_d];
            }

            floatx4 mfma_acc = {v_acc[dg_idx], 0, 0, 0};
            mfma_acc = __builtin_amdgcn_mfma_f32_4x4x4bf16_1k(a_val, b_val, mfma_acc, 0, 0, 0);
            v_acc[dg_idx] = mfma_acc[0];
        }

        __syncthreads();
    }

    // ========================================================================
    // Phase 5: Write output
    // ========================================================================
    for (int dg_idx = 0; dg_idx < num_dim_groups; dg_idx++) {
        const int dg = dg_idx * 4;
        const int dim_group = warp_dim_start + dg;
        const int out_d     = dim_group + tid_in_block;
        O_ptr[out_d] = static_cast<half_t>(v_acc[dg_idx]);
    }
}
