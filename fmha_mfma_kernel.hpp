#pragma once

#include <hip/hip_runtime.h>
#include <stdio.h>
#include <math.h>

using bhalf_t = __bf16;
using half_t  = _Float16;
using bf16x4 = __bf16 __attribute__((ext_vector_type(4)));
using bf16x8 = __bf16 __attribute__((ext_vector_type(8)));
using floatx4 = float __attribute__((ext_vector_type(4)));

// MFMA input type for gfx90a (builtin uses i16 vector)
//using mfma_bf16x4 = short __attribute__((ext_vector_type(4)));

#define WARP_SIZE 64
#define BLOCK_SIZE 256
#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))

#define ASM_DEBUG(maker) \
    __builtin_amdgcn_sched_barrier(0); \
    asm volatile(maker);               \
    __builtin_amdgcn_sched_barrier(0); \

// ============================================================================
// Pipeline-Optimized MFMA FMHA Kernel
//
// Pipeline fixes over the original version (based on ASM analysis):
//
//   1. Q preloaded → Q_lds: eliminates global_load_dwordx2 (~200 cycle stall)
//      from QK^T inner loop. Both MFMA operands now from LDS (~20 cycles).
//
//   2. Pre-zero KV_lds: removes ALL exec-mask branches from MFMA hot loops.
//      QK^T: 2 branches/iter → 0.  Attn×V: 4 branches/iter → 0.
//
//   3. Attn×V: 4 LDS reads issued back-to-back (pipelined) instead of
//      4 serial read→wait→branch→pack sequences (~80 cycles → ~24 cycles).
//
//   4. Software pipeline: next iteration's LDS reads prefetched during
//      current MFMA execution. Hides LDS latency behind MFMA compute.
//
//   5. Fast reciprocal in softmax: v_rcp_f32 + v_mul_f32 (2 instr)
//      replaces IEEE v_div_scale/v_div_fmas sequence (~20 instr).
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
    const int head_idx  = blockIdx.y;   // one block per head
    const int tid       = threadIdx.x;  // [0, 256)
    const int warp_id   = tid / 64;     // [0, 4)
    const int lane_id   = tid % 64;     // [0, 64)

    const int lane_row   = lane_id / 16;  // row in 16x16 result
    const int lane_col = lane_id % 16;    // which 4-column block (0-3)

    // Get actual seqlen_kv and offset for this batch
    const int seqlen_start = cu_seqlens_kv[batch_idx];
    const int seqlen_end = cu_seqlens_kv[batch_idx + 1];
    const int seqlen_kv = seqlen_end - seqlen_start;
    
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

    // ========================================================================
    // [Pipeline Fix 2] Pre-zero KV_lds — enables branchless MFMA operand reads
    // Invalid seq positions read 0 → zero contribution to matmul.
    // Cost: ~2 bf16x8 stores per thread. Saves: ~16 cycles/iter × N iters.
    // ========================================================================
    {
        constexpr int total_lds_vec = max_seqlen_kv * max_hd_pad / 8;  // 520
        const bf16x8 zero8 = {0};
        for (int i = tid; i < total_lds_vec; i += 256) {
            *(bf16x8*)(&KV_lds[i * 8]) = zero8;
        }
    }

    // ========================================================================
    // [Pipeline Fix 1] Load Q → Q_lds once (eliminates global_load from QK^T loop)
    // Before: global_load_dwordx2 per QK^T iteration → s_waitcnt vmcnt(0) ~200 cycles
    // After:  ds_read_b64 per iteration → s_waitcnt lgkmcnt(0) ~20 cycles
    // ========================================================================
    if (tid < head_dim_q / 8) {
        *(bf16x8*)(&Q_lds[tid * 8]) = *(const bf16x8*)(&Q_ptr[tid * 8]);
    }

    const int elements_per_thread = 8;
    const int threads_per_row = head_dim_q / elements_per_thread;
    const int rounds = 256 / threads_per_row;
    const int kv_row = tid / threads_per_row;
    const int kv_col = (tid % threads_per_row) * elements_per_thread;

    ASM_DEBUG("; Load K to LDS and preload V");
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
    
    __syncthreads();

    // ========================================================================
    // QK^T via MFMA — warp 0 only
    //
    // [Pipeline Fixes 1+2+4] All loads from LDS, branchless, software-pipelined:
    //   - Q from Q_lds (was: global_load ~200 cycles, now: ds_read ~20 cycles)
    //   - K from KV_lds (branchless: pre-zeroed invalid positions)
    //   - Q broadcast to all lane_cols (removes lane_col==0 branch)
    //   - Prefetch: next K tile loaded DURING current MFMA execution
    //
    // ASM before: global_load + 2 exec branches + s_waitcnt vmcnt(0) per iter
    // ASM after:  2 ds_reads + 0 branches + s_waitcnt lgkmcnt(0) per iter
    // ========================================================================
    floatx4 acc = {0};
    bf16x4 a = {0}, b = {0};

    ASM_DEBUG("; Q @ K^T");

    if (warp_id == 0) {
        const int total_tiles = CEIL_DIV(head_dim_q, 16);

        // --- Software Pipeline Prologue: prefetch tile 0 ---
        // A: Q from Q_lds — all lane_cols load same Q data (broadcast)
        // B: K from KV_lds — branchless (pre-zeroed for invalid positions)
        bf16x4 a_pre = *(const bf16x4*)(&Q_lds[lane_row * 4]);
        bf16x4 b_pre = *(const bf16x4*)(&KV_lds[lane_col * max_hd_pad + lane_row * 4]);

        for (int k = 0; k < total_tiles; ++k) {
            // Use prefetched data (already in registers, zero wait)
            a = a_pre;
            b = b_pre;

            // Prefetch NEXT tile's data — issues ds_reads that execute
            // in parallel with the MFMA below (~16 MFMA cycles hide
            // ~20 LDS cycles → only ~4 cycle residual stall)
            if (k + 1 < total_tiles) {
                const uint next_dim = (k + 1) * 16;
                a_pre = *(const bf16x4*)(&Q_lds[next_dim + lane_row * 4]);
                b_pre = *(const bf16x4*)(&KV_lds[lane_col * max_hd_pad + next_dim + lane_row * 4]);
            }

            acc = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a, b, acc, 0, 0, 0);
        }

        // Write scores — branchless for seqlen_kv (invalid positions = 0)
        if (lane_row == 0) {
            scores[lane_col] = acc[0];
        }
    }   


    ASM_DEBUG("; Softmax");

    // ========================================================================
    // Softmax — [Pipeline Fix 5] fast reciprocal replaces IEEE division
    //
    // Before: v_div_scale + v_rcp + v_fma × 4 + v_div_fmas + v_div_fixup (~20 instr)
    // After:  v_rcp_f32 + v_mul_f32 (2 instr) — sufficient for softmax precision
    // ========================================================================

    if (tid < seqlen_kv) {
        scores[tid] *= softmax_scale;
    }

    __syncthreads();

    // Max reduction (branchless ternaries → v_cndmask)
    float maxVal = (tid < seqlen_kv) ? scores[tid] : -INFINITY;
    for (int offset = 8; offset > 0; offset /= 2) {
        maxVal = fmaxf(maxVal, __shfl_xor(maxVal, offset, 16));
    }

    // Exp + sum reduction
    float exp_val = (tid < seqlen_kv) ? expf(scores[tid] - maxVal) : 0.0f;
    float sumExp = exp_val;
    for (int offset = 8; offset > 0; offset /= 2) {
        sumExp += __shfl_xor(sumExp, offset, 16);
    }

    // Normalize with fast reciprocal: __builtin_amdgcn_rcpf → v_rcp_f32 (1 instr)
    // then multiply. Replaces the 20-instruction IEEE division sequence.
    float inv_sum = __builtin_amdgcn_rcpf(sumExp);
    if (tid < 16) {
        softmax_scores[tid] = static_cast<__bf16>(exp_val * inv_sum);
    }
    __syncthreads();

    ASM_DEBUG("; load V to LDS");

    if (kv_row < seqlen_kv) {
        *(bf16x8*)(&KV_lds[kv_row * max_hd_pad + kv_col]) = v_reg_0;
    }
    if (kv_row_1 < seqlen_kv) {
        *(bf16x8*)(&KV_lds[kv_row_1 * max_hd_pad + kv_col]) = v_reg_1;
    }
    __syncthreads();

    // ========================================================================
    // Attn×V via MFMA — all 4 warps
    //
    // [Pipeline Fixes 2+3+4]:
    //   - 4 LDS reads issued BACK-TO-BACK (hardware pipelines them)
    //     Before: 4 × (branch + ds_read_u16 + s_waitcnt + v_perm) ~80 cycles
    //     After:  4 × ds_read_u16 + 1 × s_waitcnt ~24 cycles
    //   - Branchless: pre-zeroed KV_lds → invalid seq rows read 0
    //   - Software pipeline: prefetch next V tile during MFMA
    // ========================================================================
    const uint aRegLoc = lane_row * 4 + lane_col * 16;
    const uint bRegLoc = lane_row * 4 * max_hd_pad + lane_col;

    ASM_DEBUG("; S @ V");

    a = *(bf16x4*)(&softmax_scores[aRegLoc]);

    const int total_d_tiles = CEIL_DIV(head_dim_q, BK);

    // --- Software Pipeline Prologue: prefetch first V tile ---
    bf16x4 b_pre;
    {
        const uint dim0 = warp_id * 16;
        b_pre[0] = KV_lds[bRegLoc + dim0];
        b_pre[1] = KV_lds[bRegLoc + dim0 + max_hd_pad];
        b_pre[2] = KV_lds[bRegLoc + dim0 + 2 * max_hd_pad];
        b_pre[3] = KV_lds[bRegLoc + dim0 + 3 * max_hd_pad];
    }

    for (int d = 0; d < total_d_tiles; d += 1) {
        acc = {0};
        const uint dim_idx = d * BK + warp_id * 16;

        // Use prefetched V data (already in registers)
        b = b_pre;

        // Prefetch NEXT V tile — 4 ds_reads issued back-to-back,
        // pipelined in LDS unit while current MFMA executes
        if (d + 1 < total_d_tiles) {
            const uint next_dim = (d + 1) * BK + warp_id * 16;
            b_pre[0] = KV_lds[bRegLoc + next_dim];
            b_pre[1] = KV_lds[bRegLoc + next_dim + max_hd_pad];
            b_pre[2] = KV_lds[bRegLoc + next_dim + 2 * max_hd_pad];
            b_pre[3] = KV_lds[bRegLoc + next_dim + 3 * max_hd_pad];
        }

        acc = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a, b, acc, 0, 0, 0);

        if (lane_row == 0) {
            O_ptr[lane_col + dim_idx] = static_cast<half_t>(acc[0]);
        }
    }
}   
