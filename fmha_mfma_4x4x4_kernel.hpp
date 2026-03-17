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
// Multi-Head 4x4x4 MFMA FMHA Kernel (LDS-based)
//
// Key idea: use v_mfma_f32_4x4x4_bf16_1k which produces 16 independent
// 4x4 blocks per wavefront. Each block handles a different HEAD (0-15).
//
// 4x4x4 MFMA register layout:
//   64 lanes -> 16 blocks of 4 lanes each
//   local_head = lane_id / 4       (0-15, maps to head)
//   lane_in_head = lane_id % 4     (0-3)
//
// QK^T phase (1 kv_pos per warp per iteration):
//   A = Q[head, dim_tile]  -- broadcast (all rows identical, seqlen_q=1)
//   B = K[kv, head, dim_tile] -- broadcast (all cols identical, 1 kv_pos)
//   Both broadcast -> C is scalar (all elements identical)
//   acc[0] = dot(Q[head], K[kv, head]) after all dim tiles
//
// Attn x V phase (warp distributes dim range):
//   A = softmax_weights[head, kv_chunk] -- broadcast within block
//   B = V[kv, head, dim] -- lane_in_head selects dim column
//   acc[lane_in_head] = weighted sum for dim = dim_off + lane_in_head
//
// Memory:
//   Q  -> Q_lds cooperatively (16 heads x head_dim)
//   K  -> KV_lds cooperatively, then overwritten by V
//   scores_lds for QK^T scores and softmax weights
//
// KV_lds layout: [kv_pos][head][dim + pad]
//   Each kv row stores all 16 heads' data for one kv position.
//   For D=128: row = 16 * 132 = 2112 bf16 = 4,224B -> max 13 kv rows
//   For D=256: row = 16 * 260 = 4160 bf16 = 8,320B -> max 6 kv rows
//
// Grid: (1, CEIL_DIV(num_heads_q, 16), batch)
//
// Barrier count: 3
//   Barrier 1: Q + K in LDS
//   Barrier 2: QK^T scores visible, safe to overwrite KV_lds with V
//   Barrier 3: softmax + V in LDS
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

    // 4x4x4 MFMA mapping
    const int local_head    = lane_id / 4;  // [0, 16) -> head within group
    const int lane_in_head  = lane_id % 4;  // [0, 4)  -> kv_pos or dim

    // Actual head index for this lane
    const int global_head = head_group * 16 + local_head;

    // Get actual seqlen_kv and offset for this batch
    const int seqlen_start = cu_seqlens_kv[batch_idx];
    const int seqlen_end   = cu_seqlens_kv[batch_idx + 1];
    const int seqlen_kv    = seqlen_end - seqlen_start;

    // KV packed layout: [1, total_S_kv, H_kv, D]
    // Stride to next kv position in global memory
    const int kv_global_stride = num_heads_kv * head_dim_kv;

    // ========================================================================
    // Output pointer
    // ========================================================================
    half_t* O_ptr = O + (size_t)batch_idx * num_heads_q * seqlen_q * head_dim_q
                      + (size_t)global_head * seqlen_q * head_dim_q;

    // ========================================================================
    // Shared Memory
    //
    // Q_lds:      [16 heads][dim + pad]
    // KV_lds:     [max_kv][16 heads][dim + pad]  -- K first, then V
    // scores_lds: [16 heads x 16 kv_pos]
    //
    // LDS budget (64KB max):
    //   Q_lds:      16 * 260 * 2B       =  8,320B
    //   scores_lds: 256 * 4B            =  1,024B
    //   KV_lds:     remaining           ~ 55,192B
    //     D=128: row = 16*132*2B = 4,224B -> 13 kv rows (54,912B)
    //     D=256: row = 16*260*2B = 8,320B ->  6 kv rows (49,920B)
    // ========================================================================
    constexpr int NUM_HEADS  = 16;
    constexpr int MAX_HD_PAD = 260;           // max head_dim(256) + 4 padding
    constexpr int MAX_SKV    = 16;
    constexpr int MAX_KV_IN_LDS = 13;        // fits D=128; D=256 uses min(seqlen_kv, 6)

    // Runtime LDS stride per head within a kv row
    const int head_lds_stride = head_dim_q + 4;  // 132 for D=128, 260 for D=256
    // Runtime LDS stride per kv row (all 16 heads)
    const int kv_lds_row_stride = NUM_HEADS * head_lds_stride;
    // How many kv positions fit in LDS
    const int max_kv_in_lds = (head_dim_q <= 128) ? 13 : 6;

    __shared__ __attribute__((aligned(128))) bhalf_t Q_lds[NUM_HEADS * MAX_HD_PAD];
    __shared__ __attribute__((aligned(128))) bhalf_t KV_lds[MAX_KV_IN_LDS * NUM_HEADS * MAX_HD_PAD];
    __shared__ __attribute__((aligned(128))) float scores_lds[NUM_HEADS * MAX_SKV];

    // Pre-zero scores_lds: 256 floats = 1024 bytes, 1 per thread
    scores_lds[tid] = 0.0f;

    // ========================================================================
    // Cooperative load parameters
    //
    // Q load: 16 rows (heads) x head_dim elements
    //   D=128: 16 rows x 16 bf16x8 = 256 loads -> 1 round (rpr=16)
    //   D=256: 16 rows x 32 bf16x8 = 512 loads -> 2 rounds (rpr=8)
    //
    // KV load per kv_pos: 16 heads x head_dim elements (same shape as Q)
    //   Each "sub-row" = one head's data for one kv position
    //   Same thread mapping as Q: tid/tpr -> head, (tid%tpr)*8 -> dim offset
    // ========================================================================
    const int epl = 8;                      // elements per bf16x8 load
    const int tpr = head_dim_q / epl;       // threads per head-row (16 for D=128, 32 for D=256)
    const int rpr = BLOCK_SIZE / tpr;       // head-rows per round (16 for D=128, 8 for D=256)
    const int my_subrow = tid / tpr;        // which head within the round
    const int my_col    = (tid % tpr) * epl; // dim offset within head

    // ========================================================================
    // Phase 0: Load Q -> Q_lds + K -> KV_lds (cooperative coalesced)
    //
    // Q: [B, H_q, S_q, D] -> Q_lds[head][dim+pad]
    // K: [1, total_S_kv, H_kv, D] -> KV_lds[kv][head][dim+pad]
    // ========================================================================
    ASM_DEBUG_4x4("; Phase 0: Load Q and K to LDS");

    // --- Load Q -> Q_lds ---
    {
        const int q_head_local  = my_subrow;
        const int q_head_global = head_group * NUM_HEADS + q_head_local;

        if (q_head_local < NUM_HEADS && q_head_global < num_heads_q) {
            const bhalf_t* Q_src = Q + (size_t)batch_idx * num_heads_q * head_dim_q
                                     + (size_t)q_head_global * head_dim_q;
            *(bf16x8*)(&Q_lds[q_head_local * head_lds_stride + my_col]) =
                *(const bf16x8*)(&Q_src[my_col]);
        }
        // For D=256: second round for heads 8-15
        if (rpr < NUM_HEADS) {
            const int q_head_local_2  = my_subrow + rpr;
            const int q_head_global_2 = head_group * NUM_HEADS + q_head_local_2;
            if (q_head_local_2 < NUM_HEADS && q_head_global_2 < num_heads_q) {
                const bhalf_t* Q_src_2 = Q + (size_t)batch_idx * num_heads_q * head_dim_q
                                           + (size_t)q_head_global_2 * head_dim_q;
                *(bf16x8*)(&Q_lds[q_head_local_2 * head_lds_stride + my_col]) =
                    *(const bf16x8*)(&Q_src_2[my_col]);
            }
        }
    }

    // --- Load K -> KV_lds ---
    // KV_lds[kv][head][dim+pad]
    // K global: K[(seqlen_start + kv) * H_kv * D + head * D + dim]
    {
        const int kv_to_load = (seqlen_kv < max_kv_in_lds) ? seqlen_kv : max_kv_in_lds;

        for (int kv = 0; kv < kv_to_load; kv++) {
            const bhalf_t* K_row = K + (size_t)(seqlen_start + kv) * kv_global_stride;

            // Round 1: load up to rpr heads (16 for D=128, 8 for D=256)
            const int h_local  = my_subrow;
            const int h_global = head_group * NUM_HEADS + h_local;
            if (h_local < NUM_HEADS && h_global < num_heads_kv) {
                *(bf16x8*)(&KV_lds[kv * kv_lds_row_stride + h_local * head_lds_stride + my_col]) =
                    *(const bf16x8*)(&K_row[h_global * head_dim_kv + my_col]);
            }

            // Round 2: for D=256 (rpr=8), load heads 8-15
            if (rpr < NUM_HEADS) {
                const int h_local_2  = my_subrow + rpr;
                const int h_global_2 = head_group * NUM_HEADS + h_local_2;
                if (h_local_2 < NUM_HEADS && h_global_2 < num_heads_kv) {
                    *(bf16x8*)(&KV_lds[kv * kv_lds_row_stride + h_local_2 * head_lds_stride + my_col]) =
                        *(const bf16x8*)(&K_row[h_global_2 * head_dim_kv + my_col]);
                }
            }
        }
    }

    __syncthreads();  // -- Barrier 1: Q + K in LDS --

    // ========================================================================
    // Phase 1: QK^T via 4x4x4 MFMA
    //
    // Each warp iterates over kv positions: kv = warp_id, warp_id+4, ...
    //
    // For each kv_pos, iterate over dim tiles (4 elements):
    //   A = Q[local_head, dim_tile]     -- broadcast (all rows identical)
    //   B = K[kv, local_head, dim_tile] -- broadcast (all cols identical)
    //   Both broadcast -> acc is scalar (all elements identical)
    //
    // After all dim tiles: acc[0] = dot(Q[head], K[kv, head])
    // Only lane_in_head==0 writes score (all lanes have same value)
    // ========================================================================
    ASM_DEBUG_4x4("; Phase 1: QK^T via 4x4x4 MFMA");

    floatx4 acc = {0};
    bf16x4 a_reg = {0}, b_reg = {0};

    const int kv_in_lds = (seqlen_kv < max_kv_in_lds) ? seqlen_kv : max_kv_in_lds;

    for (int kv = warp_id; kv < kv_in_lds; kv += 4) {
        acc = {0};

        #pragma unroll
        for (int dt = 0; dt < CEIL_DIV(head_dim_q, 4); dt++) {
            const int dim_off = dt * 4;

            // A operand: Q[local_head, dim_off..dim_off+3]
            // Broadcast: all 4 lanes in block read same head's Q
            a_reg = *(const bf16x4*)(&Q_lds[local_head * head_lds_stride + dim_off]);

            // B operand: K[kv, local_head, dim_off..dim_off+3]
            // Broadcast: all 4 lanes read same kv_pos, same head
            b_reg = *(const bf16x4*)(&KV_lds[kv * kv_lds_row_stride
                                             + local_head * head_lds_stride
                                             + dim_off]);

            acc = __builtin_amdgcn_mfma_f32_4x4x4bf16_1k(a_reg, b_reg, acc, 0, 0, 0);
        }

        // All elements of acc are identical (broadcast x broadcast = scalar)
        // Only lane_in_head==0 writes to avoid redundant writes
        if (lane_in_head == 0) {
            scores_lds[local_head * MAX_SKV + kv] = acc[0];
        }
    }

    __syncthreads();  // -- Barrier 2: QK^T scores visible, safe to overwrite KV_lds --

    // ========================================================================
    // Phase 2: Softmax + Load V -> KV_lds
    //
    // Softmax: 256 threads -> 16 heads x 16 kv positions
    //   tid / 16 -> head, tid % 16 -> kv position
    //   Warp shuffles within 16-thread sub-groups
    //
    // V load: cooperative coalesced (same pattern as K load)
    //   Reuses KV_lds buffer (K no longer needed after QK^T)
    // ========================================================================
    ASM_DEBUG_4x4("; Phase 2: Softmax + V to LDS");

    // --- Softmax ---
    const int sm_head     = tid / 16;   // [0, 16) -> head
    const int sm_kv       = tid % 16;   // [0, 16) -> kv position
    const int sm_head_idx = head_group * NUM_HEADS + sm_head;

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
    float exp_val = (sm_kv < seqlen_kv && sm_head_idx < num_heads_q)
                    ? expf(raw_score - maxVal) : 0.0f;
    float sumExp = exp_val;
    #pragma unroll
    for (int off = 8; off > 0; off /= 2) {
        sumExp += __shfl_xor(sumExp, off, 16);
    }

    // Normalize and store back
    float norm_val = (sumExp > 0.0f) ? (exp_val / sumExp) : 0.0f;
    scores_lds[sm_head * MAX_SKV + sm_kv] = norm_val;

    // --- Load V -> KV_lds (cooperative, reuses K buffer) ---
    {
        const int kv_to_load = (seqlen_kv < max_kv_in_lds) ? seqlen_kv : max_kv_in_lds;

        for (int kv = 0; kv < kv_to_load; kv++) {
            const bhalf_t* V_row = V + (size_t)(seqlen_start + kv) * kv_global_stride;

            const int h_local  = my_subrow;
            const int h_global = head_group * NUM_HEADS + h_local;
            if (h_local < NUM_HEADS && h_global < num_heads_kv) {
                *(bf16x8*)(&KV_lds[kv * kv_lds_row_stride + h_local * head_lds_stride + my_col]) =
                    *(const bf16x8*)(&V_row[h_global * head_dim_kv + my_col]);
            }

            if (rpr < NUM_HEADS) {
                const int h_local_2  = my_subrow + rpr;
                const int h_global_2 = head_group * NUM_HEADS + h_local_2;
                if (h_local_2 < NUM_HEADS && h_global_2 < num_heads_kv) {
                    *(bf16x8*)(&KV_lds[kv * kv_lds_row_stride + h_local_2 * head_lds_stride + my_col]) =
                        *(const bf16x8*)(&V_row[h_global_2 * head_dim_kv + my_col]);
                }
            }
        }
    }

    __syncthreads();  // -- Barrier 3: softmax + V in LDS --

    // ========================================================================
    // Phase 3: Attn x V via 4x4x4 MFMA
    //
    // Each warp handles a range of output dimensions:
    //   warp 0 -> dims [0..DW), warp 1 -> dims [DW..2*DW), etc.
    //   DW = head_dim / 4 (each warp covers head_dim/4 dims)
    //
    // Within each dim tile (4 elements):
    //   Iterate over kv positions in chunks of 4:
    //     A = softmax_weights[head, kv_start..kv_start+3] -- BROADCAST
    //         All 4 lanes in block share same head -> same weights
    //
    //     B = V[kv, head, dim_off + lane_in_head] -- from KV_lds
    //         b[k] = V[kv_start+k, local_head, dim_off + lane_in_head]
    //         lane_in_head selects output dim column
    //
    //   acc += A x B
    //
    // After all kv tiles:
    //   acc[lane_in_head] = weighted sum for dim = dim_off + lane_in_head
    //   Write to O[head, dim]
    // ========================================================================
    ASM_DEBUG_4x4("; Phase 3: Attn x V via 4x4x4 MFMA");

    // Each warp covers head_dim/4 output dimensions, in chunks of 4
    const int dims_per_warp = head_dim_q / 4;  // 32 for D=128, 64 for D=256
    const int dim_start = warp_id * dims_per_warp;

    for (int d = 0; d < dims_per_warp; d += 4) {
        const int dim_off = dim_start + d;

        acc = {0};

        // Iterate over kv positions in chunks of 4
        #pragma unroll
        for (int kv_tile = 0; kv_tile < CEIL_DIV(kv_in_lds, 4); kv_tile++) {
            const int kv_start = kv_tile * 4;

            // A operand: softmax weights -- BROADCAST within block
            bf16x4 sw = {0};
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                const int kv_idx = kv_start + i;
                sw[i] = (kv_idx < seqlen_kv)
                    ? static_cast<__bf16>(scores_lds[local_head * MAX_SKV + kv_idx])
                    : static_cast<__bf16>(0.0f);
            }

            // B operand: V[kv, head, dim] from KV_lds
            // b[k] = V[kv_start+k, local_head, dim_off + lane_in_head]
            // lane_in_head selects the output dim column (scalar reads)
            bf16x4 v_val = {0};
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                const int kv_idx = kv_start + i;
                if (kv_idx < kv_in_lds) {
                    v_val[i] = KV_lds[kv_idx * kv_lds_row_stride
                                      + local_head * head_lds_stride
                                      + dim_off + lane_in_head];
                }
            }

            acc = __builtin_amdgcn_mfma_f32_4x4x4bf16_1k(sw, v_val, acc, 0, 0, 0);
        }

        // Diagonal extraction: acc[lane_in_head] = output for this dim
        const int out_dim = dim_off + lane_in_head;
        if (out_dim < head_dim_q) {
            O_ptr[out_dim] = static_cast<half_t>(acc[lane_in_head]);
        }
    }
}
