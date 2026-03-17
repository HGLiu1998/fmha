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

#define ASM_DEBUG_4x4(maker) \
    __builtin_amdgcn_sched_barrier(0); \
    asm volatile(maker);               \
    __builtin_amdgcn_sched_barrier(0); \

// ============================================================================
// Multi-Head 4x4x4 MFMA FMHA Kernel (Optimized)
//
// 16 heads per block via v_mfma_f32_4x4x4_bf16_1k (16 independent 4x4 blocks).
//
// Optimizations:
//   1. COOPERATIVE COALESCED K/V loads (bf16x8, 100% cache line utilization)
//   2. LDS reads in MFMA loops (~20 cy vs ~400 cy global)
//   3. SINGLE KV_lds buffer reused for K then V
//   4. Global fallback for kv_pos beyond LDS capacity (rare, <1% at Poisson l=4)
//
// LDS layout (runtime-sized KV_lds):
//   Q_lds[16][260]         -- Q for all heads, stride=260 (max D=256 + pad=4)
//   scores_lds[256]        -- QK^T scores / softmax weights
//   KV_lds[28096]          -- K then V (single buffer, 56,192 bytes)
//     Indexed: KV_lds[seq * kv_seq_stride + head * head_dim_padded + dim]
//     D=128: head_dim_padded=130, kv_seq_stride=2080, max 13 kv positions
//     D=256: head_dim_padded=258, kv_seq_stride=4128, max 6 kv positions
//
// Bank conflict analysis (KV_PAD=2, D=128):
//   Same head, diff seq: stride=4160B, 4160/4%32=16 -> NO conflict
//   Diff head, same seq: stride=260B,  260/4%32=1  -> NO conflict
//
// Grid: (1, CEIL_DIV(num_heads_q, 16), batch)
// Barriers: 3
// ============================================================================

__global__
__launch_bounds__(256, 1)
void fmha_mfma_4x4x4(
    const bhalf_t* __restrict__ Q,         // [B, H_q, S_q, D]
    const bhalf_t* __restrict__ K,         // [1, total_S_kv, H_kv, D]
    const bhalf_t* __restrict__ V,         // [1, total_S_kv, H_kv, D]
    half_t*        __restrict__ O,         // [B, H_q, S_q, D]
    const int* __restrict__ cu_seqlens_kv, // [B+1]
    const int batch,
    const int num_heads_q,
    const int num_heads_kv,
    const int seqlen_q,
    const int head_dim_q,
    const int head_dim_kv,
    const float softmax_scale)
{
    // ========================================================================
    // Constants
    // ========================================================================
    constexpr int HEADS_PER_BLOCK  = 16;           // 4x4x4 MFMA: 16 independent blocks
    constexpr int Q_STRIDE         = 260;          // max head_dim(256) + Q_PAD(4)
    constexpr int KV_PAD           = 2;            // padding to avoid bank conflicts
    constexpr int KV_LDS_ELEMS     = 28096;        // (65536 - 8320 - 1024) / 2
    constexpr int MAX_KV_POSITIONS = 16;           // max seqlen_kv supported in scores

    // ========================================================================
    // Thread Mapping
    //
    // 4x4x4 MFMA has 16 independent 4x4 blocks per wavefront.
    //   local_head   = lane_id / 4   -> which of the 16 heads this lane works on
    //   lane_in_head = lane_id % 4   -> position within the 4-lane MFMA block
    // ========================================================================
    const int batch_idx    = blockIdx.z;
    const int head_group   = blockIdx.y;            // which group of 16 heads
    const int tid          = threadIdx.x;            // [0, 256)
    const int warp_id      = tid / 64;               // [0, 4)
    const int lane_id      = tid % 64;               // [0, 64)
    const int local_head   = lane_id / 4;            // [0, 16) head index within block
    const int lane_in_head = lane_id % 4;            // [0, 4)  position in MFMA block
    const int global_head  = head_group * HEADS_PER_BLOCK + local_head;

    // ========================================================================
    // Batch / Sequence Info
    // ========================================================================
    const int seqlen_start     = cu_seqlens_kv[batch_idx];
    const int seqlen_kv        = cu_seqlens_kv[batch_idx + 1] - seqlen_start;
    const int kv_global_stride = num_heads_kv * head_dim_kv;

    // ========================================================================
    // Global Memory Pointers
    // ========================================================================

    // Per-head pointers (for global fallback reads + output writes)
    const size_t kv_head_offset = (size_t)seqlen_start * kv_global_stride
                                + global_head * head_dim_kv;
    const bhalf_t* K_this_head = K + kv_head_offset;
    const bhalf_t* V_this_head = V + kv_head_offset;

    half_t* O_this_head = O + (size_t)batch_idx * num_heads_q * seqlen_q * head_dim_q
                            + (size_t)global_head * seqlen_q * head_dim_q;

    // Base pointers for cooperative coalesced loads (all 16 heads of this group)
    const size_t kv_group_offset = (size_t)seqlen_start * kv_global_stride
                                 + (size_t)head_group * HEADS_PER_BLOCK * head_dim_kv;
    const bhalf_t* K_group_base = K + kv_group_offset;
    const bhalf_t* V_group_base = V + kv_group_offset;

    // ========================================================================
    // Shared Memory
    // ========================================================================
    __shared__ __attribute__((aligned(128))) bhalf_t Q_lds[HEADS_PER_BLOCK * Q_STRIDE];
    __shared__ __attribute__((aligned(128))) float   scores_lds[HEADS_PER_BLOCK * MAX_KV_POSITIONS];
    __shared__ __attribute__((aligned(128))) bhalf_t KV_lds[KV_LDS_ELEMS];

    scores_lds[tid] = 0.0f;

    // ========================================================================
    // Runtime KV LDS Layout
    //
    // KV_lds is a flat buffer indexed as:
    //   KV_lds[seq * kv_seq_stride + head * head_dim_padded + dim]
    //
    // head_dim_padded = head_dim_q + KV_PAD  (avoids bank conflicts)
    // kv_seq_stride   = HEADS_PER_BLOCK * head_dim_padded
    // num_kv_in_lds   = min(seqlen_kv, KV_LDS_ELEMS / kv_seq_stride)
    // ========================================================================
    const int head_dim_padded = head_dim_q + KV_PAD;
    const int kv_seq_stride   = HEADS_PER_BLOCK * head_dim_padded;
    const int max_kv_in_lds   = KV_LDS_ELEMS / kv_seq_stride;
    const int num_kv_in_lds   = (seqlen_kv < max_kv_in_lds) ? seqlen_kv : max_kv_in_lds;

    // ========================================================================
    // Cooperative Load Mapping
    //
    // All 256 threads cooperate to load bf16x8 (16 bytes) chunks.
    //   D=128: vec_chunks_per_head=16, heads_per_round=16 -> 1 round
    //   D=256: vec_chunks_per_head=32, heads_per_round=8  -> 2 rounds
    // ========================================================================
    const int vec_chunks_per_head = head_dim_q / 8;
    const int coop_head           = tid / vec_chunks_per_head;
    const int coop_dim            = (tid % vec_chunks_per_head) * 8;
    const int heads_per_round     = BLOCK_SIZE / vec_chunks_per_head;

    // ========================================================================
    // Phase 0: Cooperative Load  Q -> Q_lds,  K -> KV_lds
    // ========================================================================
    ASM_DEBUG_4x4("; Phase 0: Load Q + K to LDS");

    // --- Load Q -> Q_lds ---
    for (int round = 0; round < HEADS_PER_BLOCK; round += heads_per_round) {
        const int head = coop_head + round;
        if (head < HEADS_PER_BLOCK) {
            const int q_head_idx = head_group * HEADS_PER_BLOCK + head;
            if (q_head_idx < num_heads_q) {
                const bhalf_t* Q_src = Q + (size_t)batch_idx * num_heads_q * seqlen_q * head_dim_q
                                         + (size_t)q_head_idx * seqlen_q * head_dim_q;
                *(bf16x8*)(&Q_lds[head * Q_STRIDE + coop_dim]) =
                    *(const bf16x8*)(&Q_src[coop_dim]);
            }
        }
    }

    // --- Load K -> KV_lds (coalesced) ---
    for (int seq = 0; seq < num_kv_in_lds; seq++) {
        if (coop_head < HEADS_PER_BLOCK) {
            *(bf16x8*)(&KV_lds[seq * kv_seq_stride + coop_head * head_dim_padded + coop_dim]) =
                *(const bf16x8*)(&K_group_base[seq * kv_global_stride + coop_head * head_dim_kv + coop_dim]);
        }
        if (heads_per_round < HEADS_PER_BLOCK) {
            const int head_round2 = coop_head + heads_per_round;
            if (head_round2 < HEADS_PER_BLOCK) {
                *(bf16x8*)(&KV_lds[seq * kv_seq_stride + head_round2 * head_dim_padded + coop_dim]) =
                    *(const bf16x8*)(&K_group_base[seq * kv_global_stride + head_round2 * head_dim_kv + coop_dim]);
            }
        }
    }

    __syncthreads();  // -- Barrier 1: Q + K in LDS --

    // ========================================================================
    // Phase 1: QK^T via 4x4x4 MFMA
    //
    //   Each warp handles 4 kv positions: kv_pos = warp_id*4 + lane_in_head
    //   A operand = Q from Q_lds (broadcast across block)
    //   B operand = K from KV_lds (LDS path), or global memory (fallback)
    // ========================================================================
    ASM_DEBUG_4x4("; Phase 1: QK^T");

    floatx4 mfma_acc = {0};
    bf16x4 q_operand = {0}, k_operand = {0};
    const int kv_pos = warp_id * 4 + lane_in_head;

    #pragma unroll
    for (int dim_tile = 0; dim_tile < CEIL_DIV(head_dim_q, 4); dim_tile++) {
        const int dim_offset = dim_tile * 4;

        q_operand = *(const bf16x4*)(&Q_lds[local_head * Q_STRIDE + dim_offset]);

        k_operand = {0};
        if (kv_pos < num_kv_in_lds) {
            k_operand = *(const bf16x4*)(&KV_lds[kv_pos * kv_seq_stride + local_head * head_dim_padded + dim_offset]);
        } else if (kv_pos < seqlen_kv) {
            k_operand = *(const bf16x4*)(&K_this_head[kv_pos * kv_global_stride + dim_offset]);
        }

        mfma_acc = __builtin_amdgcn_mfma_f32_4x4x4bf16_1k(q_operand, k_operand, mfma_acc, 0, 0, 0);
    }

    // Extract diagonal: acc[lane_in_head] = dot(Q[local_head], K[kv_pos])
    if (kv_pos < MAX_KV_POSITIONS) {
        scores_lds[local_head * MAX_KV_POSITIONS + kv_pos] = mfma_acc[lane_in_head];
    }

    __syncthreads();  // -- Barrier 2: scores visible, safe to overwrite KV_lds --

    // ========================================================================
    // Phase 2a: Softmax (warp shuffles within 16-thread sub-groups)
    //   256 threads -> 16 sub-groups of 16 -> one per head
    //
    // Phase 2b: Cooperative Load V -> KV_lds (overwrites K, reuses buffer)
    //   Runs concurrently with softmax — separate LDS regions, no conflict.
    // ========================================================================
    ASM_DEBUG_4x4("; Phase 2: Softmax + Load V to LDS");

    // --- Softmax ---
    const int softmax_head = tid / MAX_KV_POSITIONS;
    const int softmax_kv   = tid % MAX_KV_POSITIONS;

    float raw_score = -INFINITY;
    if (softmax_kv < seqlen_kv) {
        raw_score = scores_lds[softmax_head * MAX_KV_POSITIONS + softmax_kv] * softmax_scale;
    }

    float max_score = raw_score;
    #pragma unroll
    for (int offset = 8; offset > 0; offset /= 2)
        max_score = fmaxf(max_score, __shfl_xor(max_score, offset, 16));

    float exp_score = (softmax_kv < seqlen_kv) ? expf(raw_score - max_score) : 0.0f;
    float sum_exp = exp_score;
    #pragma unroll
    for (int offset = 8; offset > 0; offset /= 2)
        sum_exp += __shfl_xor(sum_exp, offset, 16);

    scores_lds[softmax_head * MAX_KV_POSITIONS + softmax_kv] =
        (sum_exp > 0.0f) ? (exp_score / sum_exp) : 0.0f;

    // --- Load V -> KV_lds (coalesced, reuses K buffer) ---
    for (int seq = 0; seq < num_kv_in_lds; seq++) {
        if (coop_head < HEADS_PER_BLOCK) {
            *(bf16x8*)(&KV_lds[seq * kv_seq_stride + coop_head * head_dim_padded + coop_dim]) =
                *(const bf16x8*)(&V_group_base[seq * kv_global_stride + coop_head * head_dim_kv + coop_dim]);
        }
        if (heads_per_round < HEADS_PER_BLOCK) {
            const int head_round2 = coop_head + heads_per_round;
            if (head_round2 < HEADS_PER_BLOCK) {
                *(bf16x8*)(&KV_lds[seq * kv_seq_stride + head_round2 * head_dim_padded + coop_dim]) =
                    *(const bf16x8*)(&V_group_base[seq * kv_global_stride + head_round2 * head_dim_kv + coop_dim]);
            }
        }
    }

    __syncthreads();  // -- Barrier 3: softmax weights + V in LDS --

    // ========================================================================
    // Phase 3: Attn x V via 4x4x4 MFMA
    //
    //   Each warp covers head_dim_q/4 output dimensions.
    //   A operand = softmax weights from scores_lds
    //   B operand = V values from KV_lds (LDS path), or global (fallback)
    //
    //   Inner loop tiles seqlen_kv by 4 (MFMA K-dimension).
    //   Outer loop iterates over 4 output dims at a time (MFMA N-dimension).
    // ========================================================================
    ASM_DEBUG_4x4("; Phase 3: Attn x V");

    const int dims_per_warp  = head_dim_q / 4;
    const int warp_dim_start = warp_id * dims_per_warp;

    for (int dim_iter = 0; dim_iter < dims_per_warp; dim_iter += 4) {
        const int dim_offset = warp_dim_start + dim_iter;
        mfma_acc = {0};

        #pragma unroll
        for (int kv_tile = 0; kv_tile < CEIL_DIV(seqlen_kv, 4); kv_tile++) {
            const int kv_base = kv_tile * 4;

            // A operand: softmax attention weights (4 consecutive kv positions)
            bf16x4 attn_weights = {0};
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                const int kv_idx = kv_base + i;
                attn_weights[i] = (kv_idx < seqlen_kv)
                    ? static_cast<__bf16>(scores_lds[local_head * MAX_KV_POSITIONS + kv_idx])
                    : static_cast<__bf16>(0.0f);
            }

            // B operand: V values (4 kv positions, scalar reads)
            bf16x4 v_operand = {0};
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                const int kv_idx = kv_base + i;
                if (kv_idx < num_kv_in_lds) {
                    v_operand[i] = KV_lds[kv_idx * kv_seq_stride + local_head * head_dim_padded
                                         + dim_offset + lane_in_head];
                } else if (kv_idx < seqlen_kv) {
                    v_operand[i] = V_this_head[kv_idx * kv_global_stride
                                              + dim_offset + lane_in_head];
                }
            }

            mfma_acc = __builtin_amdgcn_mfma_f32_4x4x4bf16_1k(attn_weights, v_operand, mfma_acc, 0, 0, 0);
        }

        const int out_dim = dim_offset + lane_in_head;
        if (out_dim < head_dim_q) {
            O_this_head[out_dim] = static_cast<half_t>(mfma_acc[lane_in_head]);
        }
    }
}
