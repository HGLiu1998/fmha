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

#define ASM_DEBUG_AW(maker) \
    __builtin_amdgcn_sched_barrier(0); \
    asm volatile(maker);               \
    __builtin_amdgcn_sched_barrier(0); \

// ============================================================================
// All-Warp QK^T MFMA FMHA Kernel
//
// Optimization over the base fmha_mfma kernel:
//
//   ALL 4 WARPS participate in QK^T (instead of warp 0 only).
//   Each warp handles every 4th dim tile, then partial sums are
//   reduced across warps via shared memory.
//
//   This eliminates the biggest s_barrier stall: warps 1-3 no longer
//   idle for ~400 cycles while warp 0 does QK^T alone.
//
// Barrier count: 3 (vs 4 in base kernel)
//   Barrier 1: K in LDS (same)
//   Barrier A: QK^T partial sums visible (all warps arrive ~together)
//   Barrier B: softmax + V in LDS both done
// ============================================================================

__global__
__attribute__((amdgpu_waves_per_eu(1, 1)))
__launch_bounds__(256, 1)
void fmha_mfma_allwarp(
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

    const int lane_row  = lane_id / 16;  // row in 16x16 result (0-3)
    const int lane_col  = lane_id % 16;  // column (0-15)

    // Get actual seqlen_kv and offset for this batch
    const int seqlen_start = cu_seqlens_kv[batch_idx];
    const int seqlen_end   = cu_seqlens_kv[batch_idx + 1];
    const int seqlen_kv    = seqlen_end - seqlen_start;

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
    const uint BK = 64;

    // K/V stride for packed layout: [seqlen, head, dim]
    const int kv_stride = num_heads_kv * head_dim_kv;

    constexpr int max_seqlen_kv = 16;
    constexpr int max_hd_pad = 256 + 4;
    __shared__ __attribute__((aligned(128))) bhalf_t Q_lds[max_hd_pad];
    __shared__ __attribute__((aligned(128))) bhalf_t KV_lds[max_seqlen_kv * max_hd_pad];
    __shared__ __attribute__((aligned(128))) float scores[256];
    __shared__ __attribute__((aligned(128))) bhalf_t softmax_scores[256];

    scores[tid] = 0.0f;
    softmax_scores[tid] = 0.0f;

    // Pre-zero KV_lds: enables branchless MFMA operand reads
    // (invalid seq positions read 0 → zero contribution to matmul)
    // 16 * 260 / 8 = 520 bf16x8 stores, ~2 per thread
    {
        constexpr int total_lds_vec = max_seqlen_kv * max_hd_pad / 8;
        const bf16x8 zero8 = {0};
        for (int i = tid; i < total_lds_vec; i += 256) {
            *(bf16x8*)(&KV_lds[i * 8]) = zero8;
        }
    }

    const int elements_per_thread = 8;
    const int threads_per_row = head_dim_q / elements_per_thread;
    const int rounds = 256 / threads_per_row;
    const int kv_row = tid / threads_per_row;
    const int kv_col = (tid % threads_per_row) * elements_per_thread;

    // ========================================================================
    // Load K → LDS and preload V → registers (overlapped HBM latency)
    // ========================================================================
    ASM_DEBUG_AW("; Load K to LDS and preload V");
    for (int r = kv_row; r < seqlen_kv; r += rounds) {
        *(bf16x8*)(&KV_lds[r * max_hd_pad + kv_col]) = *(const bf16x8*)(&K_ptr[r * kv_stride + kv_col]);
    }

    bf16x8 v_reg_0 = {0}, v_reg_1 = {0};
    if (kv_row < seqlen_kv) {
        v_reg_0 = *(const bf16x8*)(&V_ptr[kv_row * kv_stride + kv_col]);
    }
    const int kv_row_1 = kv_row + rounds;
    if (kv_row_1 < seqlen_kv) {
        v_reg_1 = *(const bf16x8*)(&V_ptr[kv_row_1 * kv_stride + kv_col]);
    }

    __syncthreads();  // Barrier 1: K in LDS

    // ========================================================================
    // QK^T via MFMA — ALL 4 WARPS (each handles every 4th dim tile)
    //
    // D=128: 8 tiles → 2 per warp.  D=256: 16 tiles → 4 per warp.
    // All warps finish at ~the same time → minimal barrier stall.
    // ========================================================================
    floatx4 acc = {0};
    bf16x4 a = {0}, b = {0};

    ASM_DEBUG_AW("; Q @ K^T (all 4 warps)");

    const int total_tiles = CEIL_DIV(head_dim_q, 16);
    for (int k = warp_id; k < total_tiles; k += 4) {
        const uint dim_idx = k * 16;

        // A operand: Q — load for ALL lane_cols (branchless)
        // All lanes load the same Q data (only depends on lane_row).
        // MFMA produces identical results across all M rows; we only
        // extract row 0 (lane_row=0), so extra rows are harmless.
        a = *(const bf16x4*)(&Q_ptr[dim_idx + lane_row * 4]);

        // B operand: K from LDS — unconditional read (branchless)
        // Pre-zeroed KV_lds → invalid seq positions (lane_col >= seqlen_kv)
        // read 0 → zero contribution to MFMA accumulator.
        b = *(const bf16x4*)(&KV_lds[lane_col * max_hd_pad + dim_idx + lane_row * 4]);

        acc = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a, b, acc, 0, 0, 0);
    }

    // Write partial QK^T sums per warp (branchless for seqlen_kv)
    // Invalid positions have acc[0]=0 (from zero B), same as pre-zeroed scores.
    if (lane_row == 0) {
        scores[warp_id * 16 + lane_col] = acc[0];
    }

    __syncthreads();  // Barrier A: partial sums visible (all warps arrive ~together)

    // ========================================================================
    // Cross-warp reduce + Softmax + V→LDS (overlapped, independent memory)
    // ========================================================================
    ASM_DEBUG_AW("; Reduce + Softmax + V to LDS");

    // Cross-warp reduction — branchless with ternaries (v_cndmask, no exec mask)
    // scores[] was pre-zeroed, so invalid positions read 0 from all 4 warps.
    float total = (tid < 16)
        ? (scores[tid] + scores[16 + tid] + scores[32 + tid] + scores[48 + tid]) * softmax_scale
        : 0.0f;

    // Max reduction — invalid positions contribute -INF (neutral for fmax)
    float maxVal = (tid < seqlen_kv) ? total : -INFINITY;
    for (int off = 8; off > 0; off /= 2) {
        maxVal = fmaxf(maxVal, __shfl_xor(maxVal, off, 16));
    }

    // Exp + sum — invalid positions contribute 0 (neutral for +)
    float exp_val = (tid < seqlen_kv) ? expf(total - maxVal) : 0.0f;
    float sumExp = exp_val;
    for (int off = 8; off > 0; off /= 2) {
        sumExp += __shfl_xor(sumExp, off, 16);
    }

    // Normalize — single branch covers both valid and zero-padding
    // For tid >= seqlen_kv: exp_val=0, sumExp>0 → 0/sumExp = 0 (correct padding)
    if (tid < 16) {
        softmax_scores[tid] = static_cast<__bf16>(exp_val / sumExp);
    }

    // V regs → LDS (all threads, overlapped with softmax — separate memory regions)
    if (kv_row < seqlen_kv) {
        *(bf16x8*)(&KV_lds[kv_row * max_hd_pad + kv_col]) = v_reg_0;
    }
    if (kv_row_1 < seqlen_kv) {
        *(bf16x8*)(&KV_lds[kv_row_1 * max_hd_pad + kv_col]) = v_reg_1;
    }

    __syncthreads();  // Barrier B: softmax + V in LDS both done

    // ========================================================================
    // Attn×V via MFMA (all 4 warps)
    // ========================================================================
    const uint aRegLoc = lane_row * 4 + lane_col * 16;
    const uint bRegLoc = lane_row * 4 * max_hd_pad + lane_col;

    ASM_DEBUG_AW("; S @ V");

    a = *(bf16x4*)(&softmax_scores[aRegLoc]);

    for (int d = 0; d < CEIL_DIV(head_dim_q, BK); d += 1) {
        acc = {0};
        const uint dim_idx = d * BK + warp_id * 16;

        // V from LDS — unconditional reads (branchless)
        // Pre-zeroed KV_lds → invalid seq rows read 0 → zero contribution.
        b[0] = KV_lds[bRegLoc + dim_idx];
        b[1] = KV_lds[bRegLoc + dim_idx + max_hd_pad];
        b[2] = KV_lds[bRegLoc + dim_idx + 2 * max_hd_pad];
        b[3] = KV_lds[bRegLoc + dim_idx + 3 * max_hd_pad];

        acc = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a, b, acc, 0, 0, 0);

        if (lane_row == 0) {
            O_ptr[lane_col + dim_idx] = static_cast<half_t>(acc[0]);
        }
    }
}
