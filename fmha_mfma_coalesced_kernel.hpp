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

// LDS padding (bf16 elements) to reduce bank conflicts on MFMA LDS reads.
constexpr int LDS_PAD = 4;

// ============================================================================
// Coalesced MFMA FMHA Kernel
//
// Optimizations over the original fmha_mfma kernel:
//
//   1. COALESCED K/V loading: all 256 threads cooperate with bf16x8 loads
//      → 100% cache line utilization (vs ~25% non-coalesced)
//
//   2. OVERLAPPED K+V HBM latency: V loaded to registers while K loads to LDS
//      → total HBM latency ≈ max(K,V) instead of K+V  (~900 vs ~1800 cycles)
//
//   3. REGISTER-ONLY softmax: no scores[] shared memory array
//      → eliminates shared memory read-modify-writes + 1 barrier
//
//   4. REGISTER-CACHED attention weights for Attn×V
//      → softmax bf16x4 loaded once, reused across all Attn×V iterations
//
//   5. MINIMAL shared memory: KV_lds (single buffer) + softmax_out[16]
//      → same LDS footprint as v1, no 2× LDS penalty
//
// Barrier count: 3 (vs 4-5 in previous versions)
// ============================================================================

__global__
__launch_bounds__(256, 2)
void fmha_mfma_coalesced(
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
    const int tid       = threadIdx.x;  // [0, 256)
    const int warp_id   = tid / 64;     // [0, 4)
    const int lane_id   = tid % 64;     // [0, 64)

    // MFMA 16×16×16 register mapping (bf16_1k):
    //   A[m][k]: m = lane_col, k = lane_row*4 + reg_idx
    //   B[k][n]: k = lane_row*4 + reg_idx, n = lane_col
    //   C[m][n]: m = lane_row*4 + reg_idx, n = lane_col
    const int lane_row = lane_id / 16;  // 0-3
    const int lane_col = lane_id % 16;  // 0-15

    // Packed KV layout
    const int seqlen_start = cu_seqlens_kv[batch_idx];
    const int seqlen_end   = cu_seqlens_kv[batch_idx + 1];
    const int seqlen_kv    = seqlen_end - seqlen_start;

    const size_t kv_offset = (size_t)seqlen_start * num_heads_kv * head_dim_kv
                           + head_idx * head_dim_kv;

    // ========================================================================
    // Pointers
    // ========================================================================
    const bhalf_t* Q_ptr = Q + (size_t)batch_idx * num_heads_q * seqlen_q * head_dim_q
                             + (size_t)head_idx * seqlen_q * head_dim_q;
    const bhalf_t* K_ptr = K + kv_offset;
    const bhalf_t* V_ptr = V + kv_offset;
    half_t* O_ptr = O + (size_t)batch_idx * num_heads_q * seqlen_q * head_dim_q
                      + (size_t)head_idx * seqlen_q * head_dim_q;

    const int kv_stride  = num_heads_kv * head_dim_kv;
    const int lds_stride = head_dim_q + LDS_PAD;
    const uint BK = 64;  // 4 warps × 16 output dims per Attn×V iteration

    // ========================================================================
    // Shared Memory — SINGLE KV_lds buffer (reused for K then V)
    // ========================================================================
    constexpr int MAX_HD   = 256;
    constexpr int MAX_LDSS = MAX_HD + LDS_PAD;  // 260
    constexpr int MAX_SKV  = 16;

    __shared__ __attribute__((aligned(128))) bhalf_t KV_lds[MAX_SKV * MAX_LDSS];
    __shared__ __attribute__((aligned(128))) bhalf_t softmax_out[16];

    // Zero-init softmax_out (only 16 bf16 = 32 bytes, vs 512+1024 before)
    if (tid < 16) {
        softmax_out[tid] = static_cast<__bf16>(0.0f);
    }

    // ========================================================================
    // Phase 0: Load K → LDS, Load V → REGISTERS (overlapped in HBM)
    // ========================================================================
    // Both K and V loads are issued back-to-back. The hardware pipelines
    // outstanding HBM requests, so total latency ≈ max(K,V) ≈ 1 HBM trip
    // instead of 2 sequential trips.
    //
    // K goes to LDS (needed immediately for QK^T).
    // V goes to registers (parked until after QK^T, then written to LDS).
    // This keeps a SINGLE KV_lds buffer — no 2× LDS overhead.

    const int epl  = 8;                    // elements per bf16x8 load
    const int tpr  = head_dim_q / epl;     // threads per row
    const int rpr  = BLOCK_SIZE / tpr;     // rows per round
    const int my_row = tid / tpr;
    const int my_col = (tid % tpr) * epl;

    // K → KV_lds (coalesced: consecutive threads → consecutive 16B chunks)
    for (int r = my_row; r < seqlen_kv; r += rpr) {
        *(bf16x8*)(&KV_lds[r * lds_stride + my_col]) =
            *(const bf16x8*)(&K_ptr[r * kv_stride + my_col]);
    }

    // V → registers (coalesced, pipelined with K loads)
    // For D=128: rpr=16, 1 round suffices (v_reg_1 unused)
    // For D=256: rpr=8,  2 rounds needed for S>8
    bf16x8 v_reg_0 = {0}, v_reg_1 = {0};
    if (my_row < seqlen_kv) {
        v_reg_0 = *(const bf16x8*)(&V_ptr[my_row * kv_stride + my_col]);
    }
    const int my_row_1 = my_row + rpr;
    if (my_row_1 < seqlen_kv) {
        v_reg_1 = *(const bf16x8*)(&V_ptr[my_row_1 * kv_stride + my_col]);
    }

    __syncthreads();  // ── Barrier 1: K in LDS ──

    // ========================================================================
    // Phase 1: QK^T via MFMA (warp 0)
    // ========================================================================
    floatx4 acc = {0};
    bf16x4 a = {0}, b = {0};

    if (warp_id == 0) {
        for (int k = 0; k < CEIL_DIV(head_dim_q, 16); k++) {
            const int dim_off = k * 16;

            // A operand: Q (from global, L1-cached after first access)
            if (lane_col == 0) {
                a = *(const bf16x4*)(&Q_ptr[dim_off + lane_row * 4]);
            }

            // B operand: K from KV_lds
            if (lane_col < seqlen_kv) {
                b = *(const bf16x4*)(&KV_lds[lane_col * lds_stride + dim_off + lane_row * 4]);
            }

            acc = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a, b, acc, 0, 0, 0);
        }

        // ====================================================================
        // Phase 2: Softmax — ENTIRELY IN REGISTERS (no scores[] array)
        // ====================================================================
        // acc[0] at (lane_row=0, lane_col) holds the dot product for seq lane_col.
        // Softmax computed via warp shuffles within the 16-thread sub-group.
        // No shared memory needed for intermediate scores.

        float my_score = -INFINITY;
        if (lane_row == 0 && lane_col < seqlen_kv) {
            my_score = acc[0] * softmax_scale;
        }

        // Max reduction (within 16-thread sub-group: __shfl_xor width=16)
        float maxVal = my_score;
        for (int off = 8; off > 0; off /= 2) {
            maxVal = fmaxf(maxVal, __shfl_xor(maxVal, off, 16));
        }

        // Exp + sum
        float exp_val = 0.0f;
        if (lane_row == 0 && lane_col < seqlen_kv) {
            exp_val = expf(my_score - maxVal);
        }
        float sumExp = exp_val;
        for (int off = 8; off > 0; off /= 2) {
            sumExp += __shfl_xor(sumExp, off, 16);
        }

        // Normalize → bf16, write to shared
        if (lane_row == 0 && lane_col < seqlen_kv) {
            softmax_out[lane_col] = static_cast<__bf16>(exp_val / sumExp);
        }
    }

    __syncthreads();  // ── Barrier 2: QK^T done + softmax written ──
                      //    Safe to overwrite KV_lds (warp 0 finished reading K)
                      //    softmax_out visible to all warps

    // ========================================================================
    // Phase 3: Write V from registers → KV_lds (fast LDS store, ~20 cycles)
    // ========================================================================
    if (my_row < seqlen_kv) {
        *(bf16x8*)(&KV_lds[my_row * lds_stride + my_col]) = v_reg_0;
    }
    if (my_row_1 < seqlen_kv) {
        *(bf16x8*)(&KV_lds[my_row_1 * lds_stride + my_col]) = v_reg_1;
    }

    __syncthreads();  // ── Barrier 3: V in LDS ──

    // ========================================================================
    // Phase 4: Attn×V via MFMA (all 4 warps)
    // ========================================================================
    // Attention weights → REGISTERS (loaded once, reused every iteration)
    bf16x4 attn_w = {0};
    if (lane_col == 0) {
        attn_w = *(const bf16x4*)(&softmax_out[lane_row * 4]);
    }

    for (int d = 0; d < CEIL_DIV(head_dim_q, BK); d++) {
        b = {0};
        acc = {0};
        const int dim_idx = d * BK + warp_id * 16;

        // B operand: V from KV_lds (4 scalar bf16 reads, different seq rows)
        for (int i = 0; i < 4; ++i) {
            int si = lane_row * 4 + i;
            if (si < seqlen_kv) {
                b[i] = KV_lds[si * lds_stride + lane_col + dim_idx];
            }
        }

        acc = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(attn_w, b, acc, 0, 0, 0);

        // Output: C[0][lane_col] → O[lane_col + dim_idx]
        if (lane_row == 0) {
            O_ptr[lane_col + dim_idx] = static_cast<half_t>(acc[0]);
        }
    }
}
