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
// Multi-Head 4×4×4 MFMA FMHA Kernel (Method B)
//
// Key idea: use v_mfma_f32_4x4x4_bf16_1k which produces 16 independent
// 4×4 blocks per wavefront. Each block handles a different HEAD (0-15).
//
// 4×4×4 MFMA register layout:
//   64 lanes → 16 blocks of 4 lanes each
//   block_id = lane_id / 4       (0-15, maps to head)
//   lane_in_block = lane_id % 4  (0-3, maps to kv_pos or dim)
//
//   A operand: a[0..3] packed bf16x4 (broadcast: all rows identical since seqlen_q=1)
//   B operand: b[0..3] packed bf16x4 (4 k-elements for this lane's column)
//   C output:  acc[0..3] = C[0..3, lane_in_block]  (4 rows × 1 col)
//
// Diagonal extraction: since seqlen_q=1, all rows of A are identical,
// so all rows of C = A×B are identical. acc[lane_in_block] = C[j][j]
// gives the correct scalar result for this lane's position.
//
// Method B mapping:
//   block_id (0-15) → HEAD (consistent across ALL phases)
//   lane_in_block   → kv position (QK^T) or dim within tile (Attn×V)
//   warp_id (0-3)   → kv chunk (QK^T) or dim range (Attn×V)
//
// Memory access pattern:
//   Q: loaded to Q_lds cooperatively (16 heads × head_dim), broadcast within blocks
//   K: read directly from global per-lane (each lane has its own head, L1-cached)
//   V: read directly from global per-lane (same pattern as K)
//   No KV_lds needed — saves LDS and eliminates 1 barrier
//
// Grid: (1, CEIL_DIV(num_heads_q, 16), batch)
//   Each block handles UP TO 16 heads.
//
// Barrier count: 2
//   Barrier 1: Q in LDS
//   Barrier 2: QK^T scores in scores_lds (for softmax + Attn×V)
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
    // Block / Thread Mapping
    // ========================================================================
    const int batch_idx    = blockIdx.z;
    const int head_group   = blockIdx.y;   // which group of 16 heads
    const int tid          = threadIdx.x;  // [0, 256)
    const int warp_id      = tid / 64;     // [0, 4)
    const int lane_id      = tid % 64;     // [0, 64)

    // 4×4×4 MFMA mapping
    const int block_id      = lane_id / 4;  // [0, 16) → head within group
    const int lane_in_block = lane_id % 4;  // [0, 4)  → kv_pos or dim

    // Actual head index for this lane
    const int head_idx = head_group * 16 + block_id;

    // Get actual seqlen_kv and offset for this batch
    const int seqlen_start = cu_seqlens_kv[batch_idx];
    const int seqlen_end   = cu_seqlens_kv[batch_idx + 1];
    const int seqlen_kv    = seqlen_end - seqlen_start;

    const int kv_stride = num_heads_kv * head_dim_kv;

    // ========================================================================
    // Pointers — each lane's block_id determines its head
    // ========================================================================
    const bhalf_t* Q_ptr = Q + (size_t)batch_idx * num_heads_q * seqlen_q * head_dim_q
                             + (size_t)head_idx * seqlen_q * head_dim_q;

    const size_t kv_offset = (size_t)seqlen_start * num_heads_kv * head_dim_kv
                           + head_idx * head_dim_kv;
    const bhalf_t* K_ptr = K + kv_offset;
    const bhalf_t* V_ptr = V + kv_offset;

    half_t* O_ptr = O + (size_t)batch_idx * num_heads_q * seqlen_q * head_dim_q
                      + (size_t)safe_head * seqlen_q * head_dim_q;

    // ========================================================================
    // Shared Memory
    //
    // Q_lds: [16 heads][head_dim + 4 pad] — one Q row per head
    // scores_lds: [16 heads × 16 kv_pos] — QK^T scores + softmax weights
    //
    // No KV_lds needed: K and V are read directly from global memory
    // per-lane (each lane accesses its own head's data, L1-cached).
    // ========================================================================
    constexpr int MAX_HD     = 256;
    constexpr int MAX_HD_PAD = MAX_HD + 4;   // 260, bank conflict padding
    constexpr int MAX_SKV    = 16;
    constexpr int NUM_HEADS_PER_BLOCK = 16;

    __shared__ __attribute__((aligned(128))) bhalf_t Q_lds[NUM_HEADS_PER_BLOCK * MAX_HD_PAD];
    __shared__ __attribute__((aligned(128))) float scores_lds[NUM_HEADS_PER_BLOCK * MAX_SKV];

    // Pre-zero scores_lds: 256 floats = 1024 bytes, 1 per thread
    scores_lds[tid] = 0.0f;

    // ========================================================================
    // Cooperative load parameters (256 threads, bf16x8 = 8 elements each)
    // ========================================================================
    const int epl = 8;                      // elements per bf16x8 load
    const int tpr = head_dim_q / epl;       // threads per row (16 for D=128, 32 for D=256)
    const int rpr = BLOCK_SIZE / tpr;       // rows per round (16 for D=128, 8 for D=256)
    const int my_row = tid / tpr;
    const int my_col = (tid % tpr) * epl;

    // ========================================================================
    // Phase 0: Load Q → Q_lds (all 16 heads cooperatively)
    //
    // Q layout: [B, H_q, S_q, D] — each head has head_dim bf16 values
    // Q_lds layout: [head][dim+pad]
    //
    // 256 threads load 16 heads × head_dim values.
    // For D=128: 16 rows × 16 bf16x8 = 256 loads = 1 round
    // For D=256: 16 rows × 32 bf16x8 = 512 loads = 2 rounds
    // ========================================================================
    ASM_DEBUG_4x4("; Phase 0: Load Q to Q_lds");
    {
        const int q_head = my_row;
        const int q_head_idx = head_group * 16 + q_head;

        if (q_head < NUM_HEADS_PER_BLOCK && q_head_idx < num_heads_q) {
            const bhalf_t* Q_src = Q + (size_t)batch_idx * num_heads_q * seqlen_q * head_dim_q
                                     + (size_t)q_head_idx * seqlen_q * head_dim_q;
            *(bf16x8*)(&Q_lds[q_head * MAX_HD_PAD + my_col]) =
                *(const bf16x8*)(&Q_src[my_col]);
        }
        // For D=256: need second round (rpr=8, only 8 heads per round)
        if (rpr < NUM_HEADS_PER_BLOCK) {
            const int q_head_2 = my_row + rpr;
            const int q_head_idx_2 = head_group * 16 + q_head_2;
            if (q_head_2 < NUM_HEADS_PER_BLOCK && q_head_idx_2 < num_heads_q) {
                const bhalf_t* Q_src_2 = Q + (size_t)batch_idx * num_heads_q * seqlen_q * head_dim_q
                                           + (size_t)q_head_idx_2 * seqlen_q * head_dim_q;
                *(bf16x8*)(&Q_lds[q_head_2 * MAX_HD_PAD + my_col]) =
                    *(const bf16x8*)(&Q_src_2[my_col]);
            }
        }
    }

    __syncthreads();  // ── Barrier 1: Q in LDS ──

    // ========================================================================
    // Phase 1: QK^T via 4×4×4 MFMA
    //
    // Each warp processes a different chunk of kv positions:
    //   warp 0 → kv [0..3], warp 1 → kv [4..7], etc.
    //
    // Within each warp, each 4×4 block handles one head:
    //   block_id (0-15) → head
    //   lane_in_block (0-3) → kv position within this warp's chunk
    //
    // For each dim tile (4 elements wide):
    //   A = Q[head, dim_tile]  (broadcast from Q_lds)
    //   B = K[head, kv_pos, dim_tile] (from global, L1-cached)
    //   acc += A × B
    //
    // K is read directly from global memory per-head. Each lane within
    // a block reads a different kv position but the SAME head's K data.
    // The small working set (seqlen_kv ≤ 16 × 4 bf16 per tile) fits in L1.
    //
    // After all dim tiles: extract diagonal acc[lane_in_block]
    // = dot(Q[head], K[kv_pos]) for this lane's kv position.
    // ========================================================================
    ASM_DEBUG_4x4("; Phase 1: QK^T via 4x4x4 MFMA");

    floatx4 acc = {0};
    bf16x4 a_reg = {0}, b_reg = {0};

    const int kv_base = warp_id * 4;  // this warp's starting kv position
    const int kv_pos_qkt = kv_base + lane_in_block;

    if (head_valid) {
        #pragma unroll
        for (int dt = 0; dt < CEIL_DIV(head_dim_q, 4); dt++) {
            const int dim_off = dt * 4;

            // A operand: Q[head, dim_off..dim_off+3] — BROADCAST within block
            // All 4 lanes in a block share the same head → same Q values
            // → LDS broadcast: 1 transaction, 0 bank conflicts
            a_reg = *(const bf16x4*)(&Q_lds[block_id * MAX_HD_PAD + dim_off]);

            // B operand: K[head, kv_pos, dim_off..dim_off+3] — from global
            // Each lane reads its own kv position for its own head.
            b_reg = {0};
            if (kv_pos_qkt < seqlen_kv) {
                b_reg = *(const bf16x4*)(&K_ptr[kv_pos_qkt * kv_stride + dim_off]);
            }

            acc = __builtin_amdgcn_mfma_f32_4x4x4bf16_1k(a_reg, b_reg, acc, 0, 0, 0);
        }

        // Diagonal extraction: acc[lane_in_block] = dot(Q[head], K[kv_pos])
        float my_score = acc[lane_in_block];

        // Write to scores_lds[head][kv_pos] for cross-warp visibility
        if (kv_pos_qkt < MAX_SKV) {
            scores_lds[block_id * MAX_SKV + kv_pos_qkt] = my_score;
        }
    }

    __syncthreads();  // ── Barrier 2: QK^T scores visible ──

    // ========================================================================
    // Phase 2: Softmax (per head, across all kv positions)
    //
    // scores_lds[head][0..seqlen_kv-1] contains the raw dot products.
    //
    // Thread mapping for softmax:
    //   tid 0..15   → head 0, each thread handles one kv position
    //   tid 16..31  → head 1, etc.
    //   tid 0..255  → all 16 heads covered
    //
    // Softmax via warp shuffles within 16-thread sub-groups.
    // ========================================================================
    ASM_DEBUG_4x4("; Phase 2: Softmax");

    const int sm_head = tid / 16;   // [0, 16) → head
    const int sm_kv   = tid % 16;   // [0, 16) → kv position
    const int sm_head_idx = head_group * 16 + sm_head;

    float raw_score = -INFINITY;
    if (sm_kv < seqlen_kv && sm_head_idx < num_heads_q) {
        raw_score = scores_lds[sm_head * MAX_SKV + sm_kv] * softmax_scale;
    }

    // Max reduction within 16-thread sub-group
    float maxVal = raw_score;
    #pragma unroll
    for (int off = 8; off > 0; off /= 2) {
        maxVal = fmaxf(maxVal, __shfl_xor(maxVal, off, 16));
    }

    // Exp + sum
    float exp_val = (sm_kv < seqlen_kv && sm_head_idx < num_heads_q) ? expf(raw_score - maxVal) : 0.0f;
    float sumExp = exp_val;
    #pragma unroll
    for (int off = 8; off > 0; off /= 2) {
        sumExp += __shfl_xor(sumExp, off, 16);
    }

    // Normalize and store back for Attn×V phase
    float norm_val = (sumExp > 0.0f) ? (exp_val / sumExp) : 0.0f;
    scores_lds[sm_head * MAX_SKV + sm_kv] = norm_val;

    __syncthreads();  // ── Barrier 3: softmax done ──

    // ========================================================================
    // Phase 3: Attn×V via 4×4×4 MFMA
    //
    // Each warp handles a range of output dimensions:
    //   warp 0 → dims [0..DW), warp 1 → dims [DW..2*DW), etc.
    //   where DW = head_dim / 4 (each warp covers head_dim/4 dims)
    //
    // Within each dim tile (4 elements):
    //   A = softmax_weights[head, kv_chunk]  (broadcast from scores_lds)
    //   B = V[head, kv_pos, dim_tile]        (from global, L1-cached)
    //   acc += A × B
    //
    // After all kv_tiles: extract diagonal acc[lane_in_block]
    //
    // Output: O[head, dim] = acc[lane_in_block]
    // ========================================================================
    ASM_DEBUG_4x4("; Phase 3: Attn x V via 4x4x4 MFMA");

    if (head_valid) {
        // Each warp covers head_dim/4 output dimensions, in chunks of 4
        const int dims_per_warp = head_dim_q / 4;  // 32 for D=128, 64 for D=256
        const int dim_start = warp_id * dims_per_warp;

        for (int d = 0; d < dims_per_warp; d += 4) {
            const int dim_off = dim_start + d;

            acc = {0};

            // Iterate over kv positions in chunks of 4
            #pragma unroll
            for (int kv_tile = 0; kv_tile < CEIL_DIV(seqlen_kv, 4); kv_tile++) {
                const int kv_start = kv_tile * 4;

                // A operand: softmax weights for this head — BROADCAST within block
                bf16x4 sw = {0};
                #pragma unroll
                for (int i = 0; i < 4; i++) {
                    const int kv_idx = kv_start + i;
                    sw[i] = (kv_idx < seqlen_kv)
                        ? static_cast<__bf16>(scores_lds[block_id * MAX_SKV + kv_idx])
                        : static_cast<__bf16>(0.0f);
                }

                // B operand: V[head, kv_pos, dim] — scalar gather from global
                // We need b_reg[k] = B[k][lane_in_block] = V[kv_start+k, dim_off+lane_in_block]
                // k varies across register indices (different kv positions),
                // lane_in_block selects the output dim column.
                // Cannot use vector load (that would read consecutive dims at ONE kv pos).
                bf16x4 v_val = {0};
                #pragma unroll
                for (int i = 0; i < 4; i++) {
                    const int kv_idx = kv_start + i;
                    if (kv_idx < seqlen_kv) {
                        v_val[i] = V_ptr[kv_idx * kv_stride + dim_off + lane_in_block];
                    }
                }

                acc = __builtin_amdgcn_mfma_f32_4x4x4bf16_1k(sw, v_val, acc, 0, 0, 0);
            }

            // Diagonal extraction: acc[lane_in_block] = output for this dim
            const int out_dim = dim_off + lane_in_block;
            if (out_dim < head_dim_q) {
                O_ptr[out_dim] = static_cast<half_t>(acc[lane_in_block]);
            }
        }
    }
}
