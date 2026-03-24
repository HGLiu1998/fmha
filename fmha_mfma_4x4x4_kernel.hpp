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
// MFMA 4x4x4 FMHA Decode Kernel — LDS-Staged, Templated on HEAD_DIM
//
// Template on HEAD_DIM eliminates all runtime divisions in cooperative loads,
// right-sizes LDS (higher occupancy for D=128), and enables full unrolling.
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
// LDS Layout (right-sized per HEAD_DIM):
//   Q_lds:  [16 heads × hd_pad]                  — loaded once
//   KV_lds: [4 kv_positions × 16 heads × hd_pad] — reloaded per kv_group
//   hd_pad = HEAD_DIM + 4 (bank conflict padding)
//
// D=128: Q=4224 + KV=16896 = 21120 bytes → occupancy 3
// D=256: Q=8320 + KV=33280 = 41600 bytes → occupancy 1
//
// Grid: (1, CEIL_DIV(num_heads_q, 16), batch)
// Block: 256 threads (4 warps)
// ============================================================================

template <int HEAD_DIM>
__global__
__launch_bounds__(256, (HEAD_DIM == 128) ? 3 : 1)
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
    const int head_dim_kv,
    const float softmax_scale)
{
    // ========================================================================
    // Compile-time constants
    // ========================================================================
    constexpr int hd_pad = HEAD_DIM + 4;
    // Per-kv-position: 16 heads × HEAD_DIM values
    constexpr int vals_per_kv = 16 * HEAD_DIM;
    // 256 threads × 8 values per bf16x8 = 2048 values per round
    constexpr int vals_per_round = 256 * 8;
    // Rounds needed per kv position: D=128 → 1, D=256 → 2
    constexpr int rounds_per_kv = vals_per_kv / vals_per_round;

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

    // ========================================================================
    // LDS Allocation (right-sized per HEAD_DIM)
    // ========================================================================
    __shared__ __attribute__((aligned(128))) bhalf_t Q_lds[16 * hd_pad];
    __shared__ __attribute__((aligned(128))) bhalf_t KV_lds[4 * 16 * hd_pad];

    // ========================================================================
    // Pre-compute per-thread cooperative load offsets (compile-time divisors)
    //
    // For each round r of a kv position load:
    //   global_idx = r * vals_per_round + tid * 8
    //   head_local = global_idx / HEAD_DIM    (compile-time divisor!)
    //   dim_offset = global_idx % HEAD_DIM    (compile-time mask!)
    //
    // Since HEAD_DIM is constexpr and power-of-2, compiler uses shifts.
    // For D=128 (1 round): head_local = tid/16, dim_offset = (tid%16)*8
    // For D=256 (2 rounds): round 0: tid/32, (tid%32)*8; round 1: 8+tid/32, (tid%32)*8
    // ========================================================================
    // Precompute for round 0 (always used)
    constexpr int r0_base = 0;
    const int r0_gidx = r0_base + tid * 8;
    const int r0_head = r0_gidx / HEAD_DIM;     // compile-time shift
    const int r0_dim  = r0_gidx % HEAD_DIM;     // compile-time mask
    const int r0_lds_off = r0_head * hd_pad + r0_dim;  // LDS offset within a kv slot

    // Precompute for round 1 (only used when HEAD_DIM=256)
    int r1_head = 0, r1_dim = 0, r1_lds_off = 0;
    if constexpr (rounds_per_kv > 1) {
        constexpr int r1_base = vals_per_round;
        const int r1_gidx = r1_base + tid * 8;
        r1_head = r1_gidx / HEAD_DIM;
        r1_dim  = r1_gidx % HEAD_DIM;
        r1_lds_off = r1_head * hd_pad + r1_dim;
    }

    // Pointers
    const bhalf_t* Q_batch = Q + (size_t)batch_idx * num_heads_q * seqlen_q * HEAD_DIM;
    half_t* O_ptr = O + (size_t)batch_idx * num_heads_q * seqlen_q * HEAD_DIM
                      + (size_t)head_idx * seqlen_q * HEAD_DIM;
    const bhalf_t* K_batch = K + (size_t)seqlen_start * kv_stride;
    const bhalf_t* V_batch = V + (size_t)seqlen_start * kv_stride;

    // ========================================================================
    // Phase 1: Load Q → Q_lds (cooperative, all 256 threads)
    //
    // Q layout: [B, H_q, S_q=1, D] — 16 heads contiguous at head_base
    // Q_lds: [head_local * hd_pad + dim]
    // D=128: 1 round, D=256: 2 rounds
    // ========================================================================
    {
        const bhalf_t* q_src = Q_batch + head_base * HEAD_DIM;

        // Round 0
        if (head_base + r0_head < num_heads_q) {
            *(bf16x8*)(&Q_lds[r0_lds_off]) = *(const bf16x8*)(q_src + r0_gidx);
        } else {
            *(bf16x8*)(&Q_lds[r0_lds_off]) = bf16x8{0, 0, 0, 0, 0, 0, 0, 0};
        }

        // Round 1 (D=256 only)
        if constexpr (rounds_per_kv > 1) {
            const int r1_gidx_val = vals_per_round + tid * 8;
            if (head_base + r1_head < num_heads_q) {
                *(bf16x8*)(&Q_lds[r1_lds_off]) = *(const bf16x8*)(q_src + r1_gidx_val);
            } else {
                *(bf16x8*)(&Q_lds[r1_lds_off]) = bf16x8{0, 0, 0, 0, 0, 0, 0, 0};
            }
        }
    }

    // Precompute MFMA LDS read bases (used in both QK^T and V phases)
    const int q_lds_base = block_id * hd_pad;                                // Q[my_head]
    const int k_lds_base = tid_in_block * 16 * hd_pad + block_id * hd_pad;   // K[my_kv, my_head]

    // ========================================================================
    // Phase 2: QK^T — process 4 kv positions at a time
    //
    // Only iterate ceil(seqlen_kv/4) groups.
    // No zero-fill needed: invalid K → garbage scores → softmax masking.
    // ========================================================================
    float scores[16];
    #pragma unroll
    for (int i = 0; i < 16; i++) scores[i] = 0.0f;

    const int num_kv_groups = CEIL_DIV(seqlen_kv, 4);

    for (int g = 0; g < num_kv_groups; g++) {
        const int kv_group = g * 4;

        // --- Cooperative load K[kv_group..kv_group+3] → KV_lds ---
        // No zero-fill: softmax masking handles invalid positions.
        #pragma unroll
        for (int kv = 0; kv < 4; kv++) {
            const int kv_pos = kv_group + kv;
            // Load even if kv_pos >= seqlen_kv (garbage is fine for K)
            const int clamped_kv = min(kv_pos, max(seqlen_kv - 1, 0));
            const bhalf_t* k_src = K_batch + (size_t)clamped_kv * kv_stride
                                   + (size_t)head_base * head_dim_kv;
            const int kv_lds_base = kv * 16 * hd_pad;

            // Round 0
            *(bf16x8*)(&KV_lds[kv_lds_base + r0_lds_off]) =
                *(const bf16x8*)(k_src + r0_head * head_dim_kv + r0_dim);

            // Round 1 (D=256 only)
            if constexpr (rounds_per_kv > 1) {
                *(bf16x8*)(&KV_lds[kv_lds_base + r1_lds_off]) =
                    *(const bf16x8*)(k_src + r1_head * head_dim_kv + r1_dim);
            }
        }

        __syncthreads();

        // --- MFMA loop: Q_lds × K_lds^T ---
        {
            floatx4 acc = {0, 0, 0, 0};

            #pragma unroll
            for (int k = 0; k < HEAD_DIM; k += 4) {
                bf16x4 a_val, b_val;

                if (tid_in_block == 0) {
                    a_val = *(const bf16x4*)(&Q_lds[q_lds_base + k]);
                } else {
                    a_val = bf16x4{0, 0, 0, 0};
                }

                b_val = *(const bf16x4*)(&KV_lds[k_lds_base + k]);

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
    // Each warp handles HEAD_DIM/4 output dims.
    // v_acc[i] accumulates C[0,t] for dim_group i across kv_chunks.
    // D=128 → 8 accumulators, D=256 → 16 accumulators.
    // ========================================================================
    constexpr int dims_per_warp = HEAD_DIM / 4;
    constexpr int num_dim_groups = dims_per_warp / 4;
    const int warp_dim_start = warp_id * dims_per_warp;

    float v_acc[num_dim_groups];
    #pragma unroll
    for (int i = 0; i < num_dim_groups; i++) v_acc[i] = 0.0f;

    const int num_kv_chunks = CEIL_DIV(seqlen_kv, 4);

    for (int c = 0; c < num_kv_chunks; c++) {
        const int kv_chunk = c * 4;
        const int n_valid = min(4, seqlen_kv - kv_chunk);

        // --- Cooperative load V[kv_chunk..kv_chunk+3] → KV_lds ---
        // V DOES need zeros for invalid positions (weight × garbage could NaN)
        #pragma unroll
        for (int kv = 0; kv < 4; kv++) {
            const int kv_pos = kv_chunk + kv;
            const int kv_lds_base = kv * 16 * hd_pad;

            if (kv_pos < seqlen_kv) {
                const bhalf_t* v_src = V_batch + (size_t)kv_pos * kv_stride
                                       + (size_t)head_base * head_dim_kv;

                // Round 0
                *(bf16x8*)(&KV_lds[kv_lds_base + r0_lds_off]) =
                    *(const bf16x8*)(v_src + r0_head * head_dim_kv + r0_dim);

                // Round 1 (D=256 only)
                if constexpr (rounds_per_kv > 1) {
                    *(bf16x8*)(&KV_lds[kv_lds_base + r1_lds_off]) =
                        *(const bf16x8*)(v_src + r1_head * head_dim_kv + r1_dim);
                }
            } else {
                // Zero invalid V positions
                *(bf16x8*)(&KV_lds[kv_lds_base + r0_lds_off]) =
                    bf16x8{0, 0, 0, 0, 0, 0, 0, 0};
                if constexpr (rounds_per_kv > 1) {
                    *(bf16x8*)(&KV_lds[kv_lds_base + r1_lds_off]) =
                        bf16x8{0, 0, 0, 0, 0, 0, 0, 0};
                }
            }
        }

        __syncthreads();

        // --- MFMA: weights × V ---
        // Pre-build a_val once per kv_chunk (same for all dim_groups)
        bf16x4 a_val;
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
        for (int dg_idx = 0; dg_idx < num_dim_groups; dg_idx++) {
            const int out_d = warp_dim_start + dg_idx * 4 + tid_in_block;

            bf16x4 b_val;
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
    #pragma unroll
    for (int dg_idx = 0; dg_idx < num_dim_groups; dg_idx++) {
        const int out_d = warp_dim_start + dg_idx * 4 + tid_in_block;
        O_ptr[out_d] = static_cast<half_t>(v_acc[dg_idx]);
    }
}

// ============================================================================
// Host-side launch helper (dispatches to correct HEAD_DIM specialization)
// ============================================================================
inline void launch_fmha_mfma_4x4x4(
    dim3 grid, dim3 block, hipStream_t stream,
    const bhalf_t* Q, const bhalf_t* K, const bhalf_t* V,
    half_t* O, const int* cu_seqlens_kv,
    int batch, int num_heads_q, int num_heads_kv,
    int seqlen_q, int head_dim_q, int head_dim_kv,
    float softmax_scale)
{
    if (head_dim_q == 128) {
        fmha_mfma_4x4x4<128><<<grid, block, 0, stream>>>(
            Q, K, V, O, cu_seqlens_kv,
            batch, num_heads_q, num_heads_kv, seqlen_q,
            head_dim_kv, softmax_scale);
    } else {
        fmha_mfma_4x4x4<256><<<grid, block, 0, stream>>>(
            Q, K, V, O, cu_seqlens_kv,
            batch, num_heads_q, num_heads_kv, seqlen_q,
            head_dim_kv, softmax_scale);
    }
}
