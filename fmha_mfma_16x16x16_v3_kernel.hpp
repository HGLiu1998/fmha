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

#define ASM_V3(maker) \
    __builtin_amdgcn_sched_barrier(0); \
    asm volatile(maker);               \
    __builtin_amdgcn_sched_barrier(0);

// ============================================================================
// MFMA 16x16x16 v3 — All-Warp QK^T, Register Softmax, Clamped V Loads
//
// Improvements over original (fmha_mfma_kernel.hpp):
//   1. All 4 warps compute QK^T redundantly (no idle warps during QK^T)
//   2. Softmax entirely in registers, broadcast via __shfl (no softmax LDS)
//   3. Clamped V loads (no zero-fill needed, invalid rows load last valid row)
//   4. One fewer barrier (no softmax LDS broadcast barrier)
//
// Flow:
//   Phase 1: ALL 256 threads load Q+K → LDS (cooperative)
//   Barrier 1: Q+K visible
//   Phase 2: ALL 4 warps compute QK^T via MFMA (redundant, same result)
//   Phase 3: Per-warp softmax in registers (no LDS, no barrier)
//   Barrier 2: All warps done reading K → safe to overwrite with V
//   Phase 4: ALL 256 threads load V → LDS (cooperative, clamped loads)
//   Barrier 3: V visible
//   Phase 5: ALL 4 warps compute Attn×V (softmax from registers via __shfl)
//
// Grid: (1, num_heads_q, batch)
// Block: 256 threads (4 warps)
// ============================================================================

__global__
__launch_bounds__(256, 1)
void fmha_mfma_v3(
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
    // Block / Thread Mapping
    // ========================================================================
    const int batch_idx = blockIdx.z;
    const int head_idx  = blockIdx.y;
    const int tid       = threadIdx.x;   // [0, 256)
    const int warp_id   = tid / 64;      // [0, 4)
    const int lane_id   = tid % 64;      // [0, 64)
    const int lane_row  = lane_id / 16;  // [0, 4)
    const int lane_col  = lane_id % 16;  // [0, 16)

    const int seqlen_start = cu_seqlens_kv[batch_idx];
    const int seqlen_end   = cu_seqlens_kv[batch_idx + 1];
    const int seqlen_kv    = seqlen_end - seqlen_start;

    const int kv_stride = num_heads_kv * head_dim_kv;
    const size_t kv_offset = (size_t)seqlen_start * kv_stride
                           + head_idx * head_dim_kv;

    const bhalf_t* Q_ptr = Q + (size_t)batch_idx * num_heads_q * seqlen_q * head_dim_q
                             + (size_t)head_idx * seqlen_q * head_dim_q;
    const bhalf_t* K_ptr = K + kv_offset;
    const bhalf_t* V_ptr = V + kv_offset;
    half_t* O_ptr = O + (size_t)batch_idx * num_heads_q * seqlen_q * head_dim_q
                      + (size_t)head_idx * seqlen_q * head_dim_q;

    const uint BK = 64;

    constexpr int max_seqlen_kv = 16;
    constexpr int max_hd_pad = 256 + 4;

    __shared__ __attribute__((aligned(128))) bhalf_t Q_lds[max_hd_pad];
    __shared__ __attribute__((aligned(128))) bhalf_t KV_lds[max_seqlen_kv * max_hd_pad];

    // ========================================================================
    // Phase 1: Load Q → LDS (cooperative, all 256 threads)
    // ========================================================================
    ASM_V3("; v3 Q load to LDS");
    {
        const int q_vecs = head_dim_q / 8;  // 16 (D=128) or 32 (D=256)
        if (tid < q_vecs) {
            *(bf16x8*)(&Q_lds[tid * 8]) = *(const bf16x8*)(&Q_ptr[tid * 8]);
        }
    }

    // ========================================================================
    // Phase 1b: Load K → LDS (cooperative, all 256 threads)
    // Only valid rows loaded; invalid rows don't matter because softmax
    // masks them to -inf → exp = 0.
    // ========================================================================
    ASM_V3("; v3 K load to LDS");
    {
        const int threads_per_row = head_dim_q / 8;             // 16 or 32
        const int k_rounds = BLOCK_SIZE / threads_per_row;      // 16 or 8
        const int k_row = tid / threads_per_row;
        const int k_col = (tid % threads_per_row) * 8;
        for (int r = k_row; r < seqlen_kv; r += k_rounds) {
            *(bf16x8*)(&KV_lds[r * max_hd_pad + k_col]) =
                *(const bf16x8*)(&K_ptr[r * kv_stride + k_col]);
        }
    }

    // ========================================================================
    // Barrier 1: Q + K in LDS visible to all warps
    // ========================================================================
    __syncthreads();

    // ========================================================================
    // Phase 2: ALL 4 warps compute QK^T via MFMA (redundant per warp)
    //
    // Each warp independently computes scores[1,16] = Q[1,D] × K^T[D,16].
    // Redundant computation is free: warps 1-3 were idle in the original.
    // Each warp uses its own SIMD unit's matrix core.
    //
    // MFMA mapping (same as original):
    //   A: a[i] = Q_lds[dim_tile*16 + lane_row*4 + i]  (Q vector)
    //   B: b[i] = KV_lds[lane_col * max_hd_pad + dim_tile*16 + lane_row*4 + i]
    //   C: acc[0] at lane_row=0 = score[lane_col]
    //
    // Prefetching: next tile's A/B loaded while current MFMA executes.
    // ========================================================================
    ASM_V3("; v3 QK^T all warps");
    float sm_val;  // Softmax value for this lane (valid at lane_row=0)
    {
        floatx4 acc = {0};
        const int total_tiles = CEIL_DIV(head_dim_q, 16);
        const int last_tile = total_tiles - 1;

        // Prefetch first tile
        bf16x4 a_pre = *(const bf16x4*)(&Q_lds[lane_row * 4]);
        bf16x4 b_pre = *(const bf16x4*)(&KV_lds[lane_col * max_hd_pad + lane_row * 4]);

        for (int k = 0; k < total_tiles; ++k) {
            bf16x4 a = a_pre;
            bf16x4 b = b_pre;

            // Prefetch next tile (clamped to last valid tile)
            const int next_k = min(k + 1, last_tile);
            const uint next_dim = next_k * 16;
            a_pre = *(const bf16x4*)(&Q_lds[next_dim + lane_row * 4]);
            b_pre = *(const bf16x4*)(&KV_lds[lane_col * max_hd_pad + next_dim + lane_row * 4]);

            acc = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a, b, acc, 0, 0, 0);
        }

        // ==================================================================
        // Phase 3: Softmax in registers (per-warp, no LDS)
        //
        // acc[0] at lane_row=0 holds score[lane_col].
        // __shfl_xor with width=16 reduces across lane_col dimension.
        // Result: each lane_row=0 thread has its own sm_val.
        // ==================================================================
        ASM_V3("; v3 softmax");
        float score = acc[0] * softmax_scale;

        float maxVal = (lane_col < seqlen_kv) ? score : -INFINITY;
        for (int off = 8; off > 0; off /= 2)
            maxVal = fmaxf(maxVal, __shfl_xor(maxVal, off, 16));

        float exp_val = (lane_col < seqlen_kv) ? expf(score - maxVal) : 0.0f;
        float sumExp = exp_val;
        for (int off = 8; off > 0; off /= 2)
            sumExp += __shfl_xor(sumExp, off, 16);

        sm_val = exp_val * __builtin_amdgcn_rcpf(sumExp);
    }

    // ========================================================================
    // Barrier 2: All warps done reading K from KV_lds → safe to overwrite
    // ========================================================================
    __syncthreads();

    // ========================================================================
    // Phase 4: Load V → LDS (cooperative, clamped loads — no zero-fill)
    //
    // Instead of zeroing KV_lds then loading valid V rows, we load ALL 16
    // rows but clamp invalid positions to the last valid row. Since
    // softmax_weight[invalid] = 0, the clamped V data contributes nothing
    // (0 × valid_bf16 = 0). Clamped loads hit L2 cache (warm from K loads).
    // ========================================================================
    ASM_V3("; v3 V clamped load to LDS");
    {
        const int threads_per_row = head_dim_q / 8;             // 16 or 32
        const int v_rounds = BLOCK_SIZE / threads_per_row;      // 16 or 8
        const int v_row = tid / threads_per_row;
        const int v_col = (tid % threads_per_row) * 8;
        const int clamped_max = max(seqlen_kv - 1, 0);
        for (int r = v_row; r < max_seqlen_kv; r += v_rounds) {
            const int clamped_r = min(r, clamped_max);
            *(bf16x8*)(&KV_lds[r * max_hd_pad + v_col]) =
                *(const bf16x8*)(&V_ptr[clamped_r * kv_stride + v_col]);
        }
    }

    // ========================================================================
    // Barrier 3: V in LDS visible to all warps
    // ========================================================================
    __syncthreads();

    // ========================================================================
    // Phase 5: Attn × V via MFMA (all warps, softmax from registers)
    //
    // Each warp already has sm_val in registers from Phase 3.
    // Use __shfl to broadcast softmax weights to the A register:
    //   For MFMA A: a[i] = A[lane_col, lane_row*4+i]
    //   Only lane_col==0 has real data: a[i] = weight[lane_row*4+i]
    //   Others: a = 0
    //
    // sm_val is valid at lane_row=0, lane_col=s → weight[s].
    // __shfl(sm_val, lane_row*4+i, 64) reads from lane_id=(lane_row*4+i),
    // which is in lane_row=0 group (since lane_row*4+i ∈ [0,15]).
    //
    // 4 warps split the D dimension: warp k handles dims [d*BK+k*16, +16).
    // ========================================================================
    ASM_V3("; v3 Attn x V");
    {
        const uint bRegLoc = lane_row * 4 * max_hd_pad + lane_col;

        const int total_d_tiles = CEIL_DIV(head_dim_q, BK);
        const int last_d = total_d_tiles - 1;

        // Build A register from softmax weights via __shfl
        bf16x4 a;
        if (lane_col == 0) {
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                float w = __shfl(sm_val, lane_row * 4 + i, 64);
                a[i] = static_cast<bhalf_t>(w);
            }
        } else {
            a = bf16x4{0, 0, 0, 0};
        }

        // Prefetch first B tile
        bf16x4 b_pre;
        {
            const uint dim0 = warp_id * 16;
            b_pre[0] = KV_lds[bRegLoc + dim0];
            b_pre[1] = KV_lds[bRegLoc + dim0 + max_hd_pad];
            b_pre[2] = KV_lds[bRegLoc + dim0 + 2 * max_hd_pad];
            b_pre[3] = KV_lds[bRegLoc + dim0 + 3 * max_hd_pad];
        }

        for (int d = 0; d < total_d_tiles; ++d) {
            floatx4 acc = {0};
            const uint dim_idx = d * BK + warp_id * 16;

            bf16x4 b = b_pre;

            // Prefetch next B tile (clamped to last valid tile)
            const int next_d = min(d + 1, last_d);
            const uint next_dim = next_d * BK + warp_id * 16;
            b_pre[0] = KV_lds[bRegLoc + next_dim];
            b_pre[1] = KV_lds[bRegLoc + next_dim + max_hd_pad];
            b_pre[2] = KV_lds[bRegLoc + next_dim + 2 * max_hd_pad];
            b_pre[3] = KV_lds[bRegLoc + next_dim + 3 * max_hd_pad];

            acc = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a, b, acc, 0, 0, 0);

            if (lane_row == 0) {
                O_ptr[lane_col + dim_idx] = static_cast<half_t>(acc[0]);
            }
        }
    }
}
