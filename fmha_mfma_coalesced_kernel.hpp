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
// stride = head_dim + 4.  For D=128: 132 bf16 = 264 bytes.
// lane_col increments bank by 2 → 16 unique bank pairs → ≤ 2-way conflicts.
constexpr int LDS_PAD = 4;

// ============================================================================
// Coalesced MFMA FMHA Kernel — V2
//
// Three major improvements over v1:
//
//   1. PRE-LOAD BOTH K AND V to LDS in one phase
//      → eliminates second HBM round-trip (~900 cycles) and one __syncthreads
//
//   2. ALL 4 WARPS compute QK^T (tiles distributed round-robin)
//      → 4× faster QK^T compute; warps write partial scores, then reduce
//
//   3. REGISTER-CACHED attention weights for Attn×V
//      → softmax bf16x4 loaded to registers ONCE, reused across all iterations
//
// Barrier count: 3 (down from 4 in v1)
//
// Estimated speedup (D=128, S=16 vs v1):
//   QK^T:   8 MFMAs on warp0 → 2 MFMAs/warp × 4 warps parallel = ~3× faster
//   V load: eliminated entirely                                  = ~900 cycle saving
//   Total:  ~1.5-2× faster per block
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
    // Shared Memory
    // ========================================================================
    // K_lds and V_lds: both resident simultaneously (no union)
    // Max: 2 × 16 × (256+4) × 2 bytes = 16,640 bytes
    constexpr int MAX_HD   = 256;
    constexpr int MAX_LDSS = MAX_HD + LDS_PAD;  // 260
    constexpr int MAX_SKV  = 16;

    __shared__ __attribute__((aligned(128))) bhalf_t K_lds[MAX_SKV * MAX_LDSS];
    __shared__ __attribute__((aligned(128))) bhalf_t V_lds[MAX_SKV * MAX_LDSS];
    __shared__ __attribute__((aligned(128))) float   scores[64];      // 4 warps × 16 partial scores
    __shared__ __attribute__((aligned(128))) bhalf_t softmax_out[16]; // final softmax (bf16)

    // Zero-init softmax_out (indices seqlen_kv..15 must be 0 for MFMA masking)
    if (tid < 16) {
        softmax_out[tid] = static_cast<__bf16>(0.0f);
    }

    // ========================================================================
    // Phase 0: Load K AND V to LDS — COALESCED, pipelined
    // ========================================================================
    // All 256 threads cooperate. bf16x8 loads (16 bytes each).
    // For D=128: 16 threads/row → 16 rows/round → 1 round for S≤16
    // For D=256: 32 threads/row → 8 rows/round  → 2 rounds for S≤16
    //
    // K and V loads issued back-to-back; hardware pipelines both in flight.
    {
        const int epl = 8;                        // elements per load
        const int tpr = head_dim_q / epl;         // threads per row
        const int rpr = BLOCK_SIZE / tpr;          // rows per round
        const int row0 = tid / tpr;
        const int col  = (tid % tpr) * epl;

        // K loads — perfectly coalesced within each row
        for (int r = row0; r < seqlen_kv; r += rpr) {
            *(bf16x8*)(&K_lds[r * lds_stride + col]) =
                *(const bf16x8*)(&K_ptr[r * kv_stride + col]);
        }
        // V loads — pipelined with K (hardware overlaps outstanding requests)
        for (int r = row0; r < seqlen_kv; r += rpr) {
            *(bf16x8*)(&V_lds[r * lds_stride + col]) =
                *(const bf16x8*)(&V_ptr[r * kv_stride + col]);
        }
    }
    __syncthreads();  // ── Barrier 1: K + V in LDS ──

    // ========================================================================
    // Phase 1: QK^T via MFMA — ALL 4 WARPS
    // ========================================================================
    // Tiles distributed round-robin: warp w handles tiles w, w+4, w+8, ...
    // For D=128: 8 tiles → 2 per warp.  For D=256: 16 tiles → 4 per warp.
    //
    // Each warp accumulates its partial dot products independently.
    // After the loop, partial scores are written to scores[warp_id*16 + lane_col].

    floatx4 acc = {0};
    bf16x4 a = {0}, b = {0};

    {
        const int total_tiles = CEIL_DIV(head_dim_q, 16);

        for (int t = warp_id; t < total_tiles; t += 4) {
            const int dim_off = t * 16;

            // A operand: Q (from global, L1-cached after first access)
            // Only lane_col==0 has valid Q data; others provide 0 → A[m≥1][k]=0
            if (lane_col == 0) {
                a = *(const bf16x4*)(&Q_ptr[dim_off + lane_row * 4]);
            }

            // B operand: K from K_lds
            if (lane_col < seqlen_kv) {
                b = *(const bf16x4*)(&K_lds[lane_col * lds_stride + dim_off + lane_row * 4]);
            }

            acc = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a, b, acc, 0, 0, 0);
        }

        // Write each warp's partial score
        if (lane_row == 0 && lane_col < seqlen_kv) {
            scores[warp_id * 16 + lane_col] = acc[0];
        }
    }
    __syncthreads();  // ── Barrier 2: partial scores written ──

    // ========================================================================
    // Phase 2: Reduce partial scores + Softmax
    // ========================================================================
    // Only tid < seqlen_kv (≤16, all within warp 0) participate.
    // Sum 4 warps' partial scores → apply scale → softmax → write bf16 result.

    if (tid < seqlen_kv) {
        // Sum partial scores from all 4 warps
        float s = scores[tid] + scores[16 + tid] + scores[32 + tid] + scores[48 + tid];
        scores[tid] = s * softmax_scale;
    }

    // Max reduction (warp-shuffle within warp 0, first 16 lanes)
    float maxVal = -INFINITY;
    if (tid < seqlen_kv) {
        maxVal = scores[tid];
    }
    for (int off = 8; off > 0; off /= 2) {
        maxVal = fmaxf(maxVal, __shfl_xor(maxVal, off, 16));
    }

    // Exp + sum
    float sumExp = 0.0f;
    if (tid < seqlen_kv) {
        scores[tid] = expf(scores[tid] - maxVal);
        sumExp = scores[tid];
    }
    for (int off = 8; off > 0; off /= 2) {
        sumExp += __shfl_xor(sumExp, off, 16);
    }

    // Normalize → bf16
    if (tid < seqlen_kv) {
        softmax_out[tid] = static_cast<__bf16>(scores[tid] / sumExp);
    }
    __syncthreads();  // ── Barrier 3: softmax ready ──

    // ========================================================================
    // Phase 3: Attn×V via MFMA — all 4 warps, V already in LDS
    // ========================================================================
    // Pre-load attention weights to REGISTERS (constant across all iterations).
    // Only lane_col==0 threads carry valid softmax data; others are 0.
    // This replaces per-iteration shared memory reads of softmax_scores[256].

    bf16x4 attn_w = {0};
    if (lane_col == 0) {
        attn_w = *(const bf16x4*)(&softmax_out[lane_row * 4]);
    }

    for (int d = 0; d < CEIL_DIV(head_dim_q, BK); d++) {
        b = {0};
        acc = {0};
        const int dim_idx = d * BK + warp_id * 16;

        // B operand: V from V_lds (4 scalar bf16 reads, different seq rows)
        for (int i = 0; i < 4; ++i) {
            int si = lane_row * 4 + i;
            if (si < seqlen_kv) {
                b[i] = V_lds[si * lds_stride + lane_col + dim_idx];
            }
        }

        acc = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(attn_w, b, acc, 0, 0, 0);

        // Output: C[0][lane_col] → O[lane_col + dim_idx]
        if (lane_row == 0) {
            O_ptr[lane_col + dim_idx] = static_cast<half_t>(acc[0]);
        }
    }
}
