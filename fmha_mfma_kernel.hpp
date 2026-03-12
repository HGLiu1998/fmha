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
// MFMA FMHA Kernel — Cooperative K+Q Loading, 2 Barriers
//
// Flow:
//   Phase 1: ALL 256 threads load Q+K → LDS (cooperative, no pre-zero needed)
//   Barrier 1: K+Q visible to all warps
//   Phase 2: Warp 0 does QK^T + softmax (masking handles garbage at invalid K)
//   Barrier 2: softmax visible + warp 0 done reading K
//   Phase 3: Each warp independently zeros KV_lds + loads V (lgkmcnt, no barrier)
//   Phase 4: All warps do Attn×V
//
// Why no pre-zero for K?
//   Invalid K rows have garbage → QK^T produces garbage scores →
//   softmax masking (-INFINITY for invalid, 0.0f for exp) → correct result.
//
// Why per-warp V loading works without barrier?
//   Each warp zeros ALL KV_lds, then loads V to valid rows.
//   Within a wavefront, LDS stores are in-order → zeros before V writes.
//   After lgkmcnt(0), each warp sees: V at valid rows, zeros at invalid rows.
//   Invalid: softmax[0,s]=0 × V[s,d]=0 = 0 in MFMA (no NaN). ✓
//   Redundant HBM reads across warps hit L2 cache (warm from first access).
// ============================================================================

__global__
__attribute__((amdgpu_waves_per_eu(1, 1)))
__launch_bounds__(256, 1)
void fmha_mfma(
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

    const size_t kv_offset = (size_t)seqlen_start * num_heads_kv * head_dim_kv
                           + head_idx * head_dim_kv;

    const bhalf_t* Q_ptr = Q + (size_t)batch_idx * num_heads_q * seqlen_q * head_dim_q
                             + (size_t)head_idx * seqlen_q * head_dim_q;
    const bhalf_t* K_ptr = K + kv_offset;
    const bhalf_t* V_ptr = V + kv_offset;
    half_t* O_ptr = O + (size_t)batch_idx * num_heads_q * seqlen_q * head_dim_q
                      + (size_t)head_idx * seqlen_q * head_dim_q;
    const uint BK = 64;
    const int kv_stride = num_heads_kv * head_dim_kv;

    constexpr int max_seqlen_kv = 16;
    constexpr int max_hd_pad = 256 + 4;

    __shared__ __attribute__((aligned(128))) bhalf_t Q_lds[max_hd_pad];
    __shared__ __attribute__((aligned(128))) bhalf_t KV_lds[max_seqlen_kv * max_hd_pad];
    __shared__ __attribute__((aligned(128))) bhalf_t softmax_scores[256];

    // Zero softmax_scores (each thread writes its own position, no barrier)
    softmax_scores[tid] = static_cast<__bf16>(0.0f);

    // ========================================================================
    // Phase 1: Cooperative Q + K Loading (ALL 256 threads)
    //
    // No pre-zeroing of KV_lds:
    //   Invalid K rows have garbage → softmax masking handles it.
    //   K loading is 4x faster with 256 threads vs. 64.
    // ========================================================================

    // Load Q → Q_lds
    {
        const int q_vecs = head_dim_q / 8;  // 16 (D=128) or 32 (D=256)
        if (tid < q_vecs) {
            *(bf16x8*)(&Q_lds[tid * 8]) = *(const bf16x8*)(&Q_ptr[tid * 8]);
        }
    }

    // Load K → KV_lds (all 256 threads cooperate)
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
    // ── Barrier 1 ──
    // All warps' K+Q LDS writes visible to all warps.
    // ========================================================================
    __syncthreads();

    // ========================================================================
    // Phase 2: QK^T + Softmax (warp 0 only)
    //
    // Reads K+Q from LDS. Invalid K rows (≥ seqlen_kv) have garbage scores
    // but softmax masking produces correct weights:
    //   maxVal: invalid → -INFINITY (neutral for fmaxf)
    //   exp_val: invalid → 0.0f (neutral for sum)
    // ========================================================================
    floatx4 acc = {0};
    bf16x4 a = {0}, b = {0};

    if (warp_id == 0) {
        // ── QK^T (SW-pipelined) ──
        const int total_tiles = CEIL_DIV(head_dim_q, 16);
        const int last_tile = total_tiles - 1;

        bf16x4 a_pre = *(const bf16x4*)(&Q_lds[lane_row * 4]);
        bf16x4 b_pre = *(const bf16x4*)(&KV_lds[lane_col * max_hd_pad + lane_row * 4]);

        for (int k = 0; k < total_tiles; ++k) {
            //__builtin_amdgcn_sched_barrier(0);
            a = a_pre;
            b = b_pre;

            const int next_k = min(k + 1, last_tile);
            const uint next_dim = next_k * 16;
            a_pre = *(const bf16x4*)(&Q_lds[next_dim + lane_row * 4]);
            b_pre = *(const bf16x4*)(&KV_lds[lane_col * max_hd_pad + next_dim + lane_row * 4]);

            //asm volatile("s_waitcnt lgkmcnt(2)\n" ::: "memory");
            acc = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a, b, acc, 0, 0, 0);
            //__builtin_amdgcn_sched_barrier(0);
        }

        // ── Fused softmax (register-only, warp shuffles) ──
        float score = acc[0] * softmax_scale;

        float maxVal = (lane_col < seqlen_kv) ? score : -INFINITY;
        for (int off = 8; off > 0; off /= 2)
            maxVal = fmaxf(maxVal, __shfl_xor(maxVal, off, 16));

        float exp_val = (lane_col < seqlen_kv) ? expf(score - maxVal) : 0.0f;
        float sumExp = exp_val;
        for (int off = 8; off > 0; off /= 2)
            sumExp += __shfl_xor(sumExp, off, 16);

        float sm_val = exp_val * __builtin_amdgcn_rcpf(sumExp);

        if (lane_row == 0) {
            softmax_scores[lane_col] = static_cast<__bf16>(sm_val);
        }
    }

    // ========================================================================
    // ── Barrier 2 ──
    // a) softmax_scores written by warp 0 → visible to all warps
    // b) Warp 0 done reading K from KV_lds → safe to overwrite with V
    // ========================================================================
    __syncthreads();

    // ========================================================================
    // Phase 3: V → KV_lds (each warp independently, no barrier needed)
    //
    // Each warp: zero ALL KV_lds → load V to valid rows → lgkmcnt(0)
    //   Within-warp LDS store ordering: zeros issued before V writes.
    //   After lgkmcnt(0): valid rows = V data, invalid rows = zeros.
    //   4x redundant HBM V reads hit L2 cache (warm from first warp).
    // ========================================================================
    {
        // Zero KV_lds (within-warp, each lane handles ~8 vec8 stores)
        constexpr int total_lds_vec = max_seqlen_kv * max_hd_pad / 8;  // 520
        const bf16x8 zero8 = {0};
        for (int i = lane_id; i < total_lds_vec; i += 64) {
            *(bf16x8*)(&KV_lds[i * 8]) = zero8;
        }

        // Load V to valid rows (overwrites zeros, within-warp ordered after zeros)
        const int w_tpr = head_dim_q / 8;       // threads per row: 16 or 32
        const int w_rounds = 64 / w_tpr;         // rows per round: 4 or 2
        const int w_row = lane_id / w_tpr;
        const int w_col = (lane_id % w_tpr) * 8;
        for (int r = w_row; r < seqlen_kv; r += w_rounds) {
            *(bf16x8*)(&KV_lds[r * max_hd_pad + w_col]) =
                *(const bf16x8*)(&V_ptr[r * kv_stride + w_col]);
        }
    }
    // Wait for THIS wavefront's global loads + LDS stores to complete.
    // No s_barrier needed: each warp reads only its OWN LDS writes.
    asm volatile("s_waitcnt vmcnt(0) lgkmcnt(0)" ::: "memory");

    // ========================================================================
    // Phase 4: Attn×V via MFMA (all 4 warps, SW-pipelined)
    // ========================================================================
    const uint aRegLoc = lane_row * 4 + lane_col * 16;
    const uint bRegLoc = lane_row * 4 * max_hd_pad + lane_col;

    a = *(bf16x4*)(&softmax_scores[aRegLoc]);

    const int total_d_tiles = CEIL_DIV(head_dim_q, BK);
    const int last_d = total_d_tiles - 1;

    bf16x4 b_pre;
    {
        const uint dim0 = warp_id * 16;
        b_pre[0] = KV_lds[bRegLoc + dim0];
        b_pre[1] = KV_lds[bRegLoc + dim0 + max_hd_pad];
        b_pre[2] = KV_lds[bRegLoc + dim0 + 2 * max_hd_pad];
        b_pre[3] = KV_lds[bRegLoc + dim0 + 3 * max_hd_pad];
    }

    for (int d = 0; d < total_d_tiles; ++d) {
        acc = {0};
        const uint dim_idx = d * BK + warp_id * 16;

        b = b_pre;

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
