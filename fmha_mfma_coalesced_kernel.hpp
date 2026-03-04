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

// LDS padding (in bf16 elements) to reduce bank conflicts.
// With pad=4, row stride = head_dim+4. For head_dim=128: stride=132 bf16 = 264 bytes.
// This makes lane_col shift bank by 2, spreading accesses across 16 banks.
constexpr int LDS_PAD = 4;

// ============================================================================
// Coalesced MFMA FMHA Kernel
//
// Key optimization over fmha_mfma:
//   K and V are loaded to LDS with COALESCED global memory access.
//   All 256 threads cooperate to load each K/V row contiguously (bf16x8).
//   MFMA then reads from LDS instead of global memory.
//
// Memory traffic comparison (D=128, S=16):
//   Old (non-coalesced K):  16 cache lines per wavefront, ~25% utilization
//   New (coalesced → LDS):  8 cache lines per wavefront, 100% utilization
//   → 4-8x better memory efficiency for K/V loads
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
    const int head_idx  = blockIdx.y;   // one block per head
    const int tid       = threadIdx.x;  // [0, 256)
    const int warp_id   = tid / 64;     // [0, 4)
    const int lane_id   = tid % 64;     // [0, 64)

    // MFMA 16x16x16 register mapping:
    //   A[row][col]: row = lane_col, col = lane_row*4 + reg_idx
    //   B[row][col]: row = lane_row*4 + reg_idx, col = lane_col
    //   C[row][col]: row = lane_row*4 + reg_idx, col = lane_col
    const int lane_row  = lane_id / 16;  // 0-3 (selects which 4-element group)
    const int lane_col  = lane_id % 16;  // 0-15 (selects column)

    // Get actual seqlen_kv and offset for this batch (packed layout)
    const int seqlen_start = cu_seqlens_kv[batch_idx];
    const int seqlen_end   = cu_seqlens_kv[batch_idx + 1];
    const int seqlen_kv    = seqlen_end - seqlen_start;

    // KV offset: packed layout [seqlen, head, dim]
    const size_t kv_offset = (size_t)seqlen_start * num_heads_kv * head_dim_kv
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

    const int kv_stride = num_heads_kv * head_dim_kv;  // stride between seq positions
    const int lds_stride = head_dim_q + LDS_PAD;       // padded stride in LDS
    const uint BK = 64;                                 // output dims per iteration (4 warps × 16)

    // ========================================================================
    // Shared Memory
    // ========================================================================
    // KV_lds: holds K during QK^T, then V during Attn×V (reused sequentially)
    // Max: 16 seq × (256 dim + 4 pad) = 4160 bf16 = 8320 bytes
    constexpr int MAX_HEAD_DIM   = 256;
    constexpr int MAX_LDS_STRIDE = MAX_HEAD_DIM + LDS_PAD;  // 260
    constexpr int MAX_SEQLEN_KV  = 16;

    __shared__ __attribute__((aligned(128))) bhalf_t KV_lds[MAX_SEQLEN_KV * MAX_LDS_STRIDE];
    __shared__ __attribute__((aligned(128))) float scores[256];
    __shared__ __attribute__((aligned(128))) bhalf_t softmax_scores[256];

    scores[tid] = 0.0f;
    softmax_scores[tid] = static_cast<__bf16>(0.0f);

    // ========================================================================
    // Phase 1: Load K to LDS — COALESCED
    // ========================================================================
    // K per seq position: head_dim_q contiguous elements, rows separated by kv_stride
    // All 256 threads cooperate. Each loads bf16x8 (16 bytes = 8 bf16 elements).
    //
    // Thread mapping:
    //   threads_per_row = head_dim / 8  (16 for D=128, 32 for D=256)
    //   rows_per_round  = 256 / threads_per_row  (16 for D=128, 8 for D=256)
    //   my_row = tid / threads_per_row  → which seq position
    //   my_col = (tid % threads_per_row) * 8  → which 8-element chunk
    //
    // Coalescing: within a wavefront, consecutive threads access consecutive
    //   16-byte chunks in the same row → perfectly coalesced.
    //   For D=128: 16 threads/row × 16 bytes = 256 bytes = 2 cache lines, 100% utilized.
    {
        const int elems_per_load  = 8;  // bf16x8
        const int threads_per_row = head_dim_q / elems_per_load;
        const int rows_per_round  = BLOCK_SIZE / threads_per_row;
        const int my_row_base     = tid / threads_per_row;
        const int my_col          = (tid % threads_per_row) * elems_per_load;

        for (int r = my_row_base; r < seqlen_kv; r += rows_per_round) {
            *(bf16x8*)(&KV_lds[r * lds_stride + my_col]) =
                *(const bf16x8*)(&K_ptr[r * kv_stride + my_col]);
        }
    }
    __syncthreads();

    // ========================================================================
    // Phase 2: QK^T via MFMA (warp 0 only)
    // ========================================================================
    // Computes scores[j] = Σ_k Q[k] × K[seq=j, k]  for j = 0..seqlen_kv-1
    //
    // MFMA 16×16×16 register mapping:
    //   A[m][k] = a[k%4] at lane (k/4, m)  →  A represents Q
    //     A[0][k] = Q[dim_offset + k], A[m≥1][k] = 0  (seqlen_q = 1)
    //   B[k][n] = b[k%4] at lane (k/4, n)  →  B represents K
    //     B[k][n] = K[seq=n, dim_offset + k]
    //   C[m][n] = Σ_k A[m][k] × B[k][n]
    //     C[0][n] = Σ_k Q[dim_offset+k] × K[n, dim_offset+k]  ← partial dot product
    //
    // Loop over dim_offset = 0, 16, 32, ... to accumulate full dot product.

    floatx4 acc = {0};
    bf16x4 a = {0}, b = {0};

    if (warp_id == 0) {
        for (int k = 0; k < CEIL_DIV(head_dim_q, 16); k++) {
            const int dim_offset = k * 16;

            // A operand: Q (from global, only 4 threads read, L1-cached after 1st access)
            if (lane_col == 0) {
                a = *(const bf16x4*)(&Q_ptr[dim_offset + lane_row * 4]);
            }
            // else: a stays {0} — A[m≥1][k] = 0

            // B operand: K from LDS (coalesced load already done in Phase 1)
            if (lane_col < seqlen_kv) {
                b = *(const bf16x4*)(&KV_lds[lane_col * lds_stride + dim_offset + lane_row * 4]);
            }
            // else: b stays {0} — masked out seq positions

            acc = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a, b, acc, 0, 0, 0);
        }

        // Extract scores: C[0][lane_col] holds the full dot product for seq position lane_col
        if (lane_row == 0 && lane_col < seqlen_kv) {
            scores[lane_col] = acc[0];
        }
    }

    // ========================================================================
    // Phase 3: Softmax
    // ========================================================================
    if (tid < seqlen_kv) {
        scores[tid] *= softmax_scale;
    }
    __syncthreads();

    // Step 1: Find max across valid scores
    float maxVal = -INFINITY;
    if (tid < seqlen_kv) {
        maxVal = scores[tid];
    }
    for (int offset = 8; offset > 0; offset /= 2) {
        maxVal = fmaxf(maxVal, __shfl_xor(maxVal, offset, 16));
    }

    // Step 2: Compute exp(score - max) and sum
    float sumExp = 0.0f;
    if (tid < seqlen_kv) {
        scores[tid] = expf(scores[tid] - maxVal);
        sumExp = scores[tid];
    }
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

    // ========================================================================
    // Phase 4: Load V to LDS — COALESCED (reusing KV_lds)
    // ========================================================================
    // Same coalesced loading strategy as K.
    {
        const int elems_per_load  = 8;
        const int threads_per_row = head_dim_q / elems_per_load;
        const int rows_per_round  = BLOCK_SIZE / threads_per_row;
        const int my_row_base     = tid / threads_per_row;
        const int my_col          = (tid % threads_per_row) * elems_per_load;

        for (int r = my_row_base; r < seqlen_kv; r += rows_per_round) {
            *(bf16x8*)(&KV_lds[r * lds_stride + my_col]) =
                *(const bf16x8*)(&V_ptr[r * kv_stride + my_col]);
        }
    }
    __syncthreads();

    // ========================================================================
    // Phase 5: Attn×V via MFMA (all 4 warps)
    // ========================================================================
    // Computes output[d] = Σ_s softmax[s] × V[s, d]  for d = 0..head_dim-1
    //
    // Each warp handles 16 output dimensions. 4 warps × 16 = 64 dims per iteration.
    //   dim_idx = d * 64 + warp_id * 16
    //
    // MFMA mapping:
    //   A[0][k] = softmax[k], A[m≥1][k] = 0
    //   B[k][n] = V[seq=k, dim=n+dim_idx]  ← read from LDS
    //   C[0][n] = Σ_k softmax[k] × V[k, n+dim_idx]

    for (int d = 0; d < CEIL_DIV(head_dim_q, BK); d++) {
        a = {0};
        b = {0};
        acc = {0};
        const int dim_idx = d * BK + warp_id * 16;

        // A: softmax scores from shared memory
        const uint aRegLoc = lane_row * 4 + lane_col * 16;
        a = *(const bf16x4*)(&softmax_scores[aRegLoc]);

        // B: V from LDS (4 individual bf16 reads from different seq positions)
        for (int i = 0; i < 4; ++i) {
            int seq_idx = lane_row * 4 + i;
            if (seq_idx < seqlen_kv) {
                b[i] = KV_lds[seq_idx * lds_stride + lane_col + dim_idx];
            }
        }

        acc = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a, b, acc, 0, 0, 0);

        // Write output: C[0][lane_col] → O[lane_col + dim_idx]
        if (lane_row == 0) {
            O_ptr[lane_col + dim_idx] = static_cast<half_t>(acc[0]);
        }
    }
}
