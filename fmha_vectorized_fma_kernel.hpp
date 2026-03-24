#pragma once

#include <hip/hip_runtime.h>
#include <stdio.h>
#include <math.h>

using bhalf_t = __bf16;
using half_t  = _Float16;
using bf16x4 = __bf16 __attribute__((ext_vector_type(4)));
using bf16x8 = __bf16 __attribute__((ext_vector_type(8)));
using fp16x8 = _Float16 __attribute__((ext_vector_type(8)));
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
// Vectorized FMA FMHA Decode Kernel
//
// Uses vectorized bf16x8 loads + scalar FMA (no MFMA instructions).
// This kernel is deeply memory-bound (arithmetic intensity ~0.25 FLOPs/byte),
// so the bottleneck is memory bandwidth, not compute. Using coalesced 128-bit
// vector loads dramatically improves effective bandwidth.
//
// Architecture:
//   256 threads = 16 heads x 16 threads per head
//   Thread mapping: head = tid / 16, dim_chunk = tid % 16
//   Each thread owns head_dim/16 contiguous output dimensions
//
// Pipeline:
//   Stage 0: Cache Q in registers (no LDS, no barrier)
//   Stage 1: QK^T via bf16x8 vectorized dot product + warp shuffle reduction
//            -> scores to scores_lds
//            -- barrier 1 --
//   Stage 2: Softmax (16-thread sub-groups, warp shuffles)
//            -- barrier 2 --
//   Stage 3: Attn x V via bf16x8 vectorized FMA, weights cached in registers
//            -> output to registers
//   Stage 4: Write O with 128-bit vector stores
//
// Key advantages over 4x4x4 MFMA version:
//   - bf16x8 (128-bit) coalesced K/V loads vs scattered bf16x4/scalar loads
//   - Q cached in registers: eliminates Q_lds (8KB) + 1 barrier
//   - 16 threads/head: all threads do useful work (no wasted warps for small seqlen_kv)
//   - No MFMA diagonal waste: 100% of compute is used (vs 25% with 4x4x4)
//   - Higher occupancy target (2 blocks/CU vs 1)
//
// Memory coalescing analysis (within a warp of 64 threads):
//   - 4 heads per warp, 16 threads per head
//   - For K[kv_pos]: head0 threads read dims [0:127], head1 reads dims [0:127]
//     at next head offset. With layout [seqlen, head, dim], heads are contiguous:
//     -> 4 heads x 256 bytes = 1KB contiguous per warp. Perfect coalescing.
//   - Same for V loads
//
// Shared memory: scores_lds only = 16 x 17 x 4 = 1,088 bytes (vs ~9.4KB before)
// Barriers: 2 (down from 3)
//
// Grid: (1, CEIL_DIV(num_heads_q, 16), batch)
// Block: 256 threads (4 warps of 64)
// ============================================================================

__global__
__launch_bounds__(256, 2)
void fmha_vectorized_fma(
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
    // Thread Mapping: 16 threads per head, 16 heads per block
    //
    // tid / 16 -> head within block [0, 16)
    // tid % 16 -> dimension chunk index [0, 16)
    //
    // Each thread handles a contiguous chunk of head_dim/16 dimensions:
    //   D=128: 8 dims per thread (1 bf16x8 load per K/V access)
    //   D=256: 16 dims per thread (2 bf16x8 loads per K/V access)
    //
    // Within a warp (64 threads): 4 heads x 16 threads
    //   Threads 0-15: head 0, dims partitioned across threads
    //   Threads 16-31: head 1, etc.
    //   Memory for adjacent heads is contiguous in [seqlen, head, dim] layout
    //   -> 128-bit loads from consecutive threads hit consecutive cache lines
    // ========================================================================
    const int batch_idx  = blockIdx.z;
    const int head_group = blockIdx.y;    // which group of 16 heads
    const int tid        = threadIdx.x;   // [0, 256)

    const int head_local  = tid / 16;     // [0, 16) head within this block
    const int dim_thread  = tid % 16;     // [0, 16) dim chunk index
    const int head_idx    = head_group * 16 + head_local;

    // Each thread handles a contiguous chunk of dims
    const int dims_per_thread = head_dim_q / 16;  // 8 for D=128, 16 for D=256
    const int dim_start = dim_thread * dims_per_thread;

    // Batch-specific KV info
    const int seqlen_start = cu_seqlens_kv[batch_idx];
    const int seqlen_end   = cu_seqlens_kv[batch_idx + 1];
    const int seqlen_kv    = seqlen_end - seqlen_start;
    const int kv_stride    = num_heads_kv * head_dim_kv;

    // Is this thread's head valid?
    const bool active = (head_idx < num_heads_q);

    // ========================================================================
    // Pointers
    //
    // Q: [B, H_q, S_q, D] -- standard BHSD layout
    // K/V: [1, total_S_kv, H_kv, D] -- packed, kv_stride = H_kv * D
    // O: [B, H_q, S_q, D]
    // ========================================================================
    const bhalf_t* Q_ptr = Q + (size_t)batch_idx * num_heads_q * seqlen_q * head_dim_q
                             + (size_t)head_idx * seqlen_q * head_dim_q;

    const bhalf_t* K_base = K + (size_t)seqlen_start * kv_stride
                              + (size_t)head_idx * head_dim_kv;
    const bhalf_t* V_base = V + (size_t)seqlen_start * kv_stride
                              + (size_t)head_idx * head_dim_kv;

    half_t* O_ptr = O + (size_t)batch_idx * num_heads_q * seqlen_q * head_dim_q
                      + (size_t)head_idx * seqlen_q * head_dim_q;

    // ========================================================================
    // Shared Memory: scores only (Q is cached in registers)
    //
    // scores_lds: [16 heads x (16+1)] floats = 1,088 bytes
    // Padding by 1 eliminates bank conflicts (17 floats per head row)
    // ========================================================================
    constexpr int NUM_HEADS = 16;
    constexpr int SKV_PAD   = 17;   // 16 + 1 for bank conflict avoidance

    __shared__ __attribute__((aligned(128))) float scores_lds[NUM_HEADS * SKV_PAD];

    // ========================================================================
    // Stage 0: Cache Q in Registers
    //
    // Each thread loads its dim chunk directly from global memory.
    // No LDS needed, no barrier needed.
    //
    // Coalescing: 16 consecutive threads per head read 16 consecutive
    // bf16x8 chunks (128 bytes each) = 256 contiguous bytes per head.
    // 4 heads per warp = 1KB contiguous. Fully coalesced.
    //
    // Register cost: 8 floats (D=128) or 16 floats (D=256) = 8-16 VGPRs
    // ========================================================================
    float q_f[16];  // max dims_per_thread = 16 (for D=256)

    if (active) {
        bf16x8 qv0 = *(const bf16x8*)(&Q_ptr[dim_start]);
        #pragma unroll
        for (int i = 0; i < 8; i++) q_f[i] = static_cast<float>(qv0[i]);

        if (dims_per_thread > 8) {
            bf16x8 qv1 = *(const bf16x8*)(&Q_ptr[dim_start + 8]);
            #pragma unroll
            for (int i = 0; i < 8; i++) q_f[8 + i] = static_cast<float>(qv1[i]);
        }
    }

    // ========================================================================
    // Stage 1: QK^T via Vectorized Dot Product + Warp Shuffle Reduction
    //
    // For each kv position:
    //   1. Each thread loads K[kv, head, my_dims] with bf16x8 (coalesced)
    //   2. Computes partial dot product over its dims (8 or 16 FMAs)
    //   3. Butterfly reduction across 16 threads -> full dot product
    //   4. Thread 0 of each head writes score to scores_lds
    //
    // K load coalescing: identical to Q loads (consecutive threads read
    // consecutive dim chunks within each head, heads are contiguous).
    //
    // Total K loads: seqlen_kv x 1 bf16x8 per thread (D=128)
    //            or: seqlen_kv x 2 bf16x8 per thread (D=256)
    // ========================================================================
    for (int kv = 0; kv < seqlen_kv; kv++) {
        float partial = 0.0f;

        if (active) {
            const bhalf_t* K_kv = &K_base[kv * kv_stride + dim_start];

            bf16x8 kv0 = *(const bf16x8*)(K_kv);
            #pragma unroll
            for (int i = 0; i < 8; i++)
                partial += q_f[i] * static_cast<float>(kv0[i]);

            if (dims_per_thread > 8) {
                bf16x8 kv1 = *(const bf16x8*)(K_kv + 8);
                #pragma unroll
                for (int i = 0; i < 8; i++)
                    partial += q_f[8 + i] * static_cast<float>(kv1[i]);
            }
        }

        // Butterfly reduction: sum partial dot products across 16 threads
        // All 16 threads in the sub-group get the same final sum
        #pragma unroll
        for (int off = 8; off >= 1; off >>= 1) {
            partial += __shfl_xor(partial, off, 16);
        }

        // One thread per head writes the score to LDS
        if (dim_thread == 0) {
            scores_lds[head_local * SKV_PAD + kv] = partial;
        }
    }

    __syncthreads();  // -- Barrier 1: QK^T scores visible --

    // ========================================================================
    // Stage 2: Softmax (per head, across all kv positions)
    //
    // Thread mapping (same as QK^T): head = tid/16, kv = tid%16
    // 16-thread sub-groups compute softmax via warp shuffles.
    //
    // dim_thread serves double duty: in QK^T it indexes dim chunks,
    // here it indexes kv positions (both [0, 16)).
    // ========================================================================
    {
        const bool sm_valid = (dim_thread < seqlen_kv) && active;
        float raw = sm_valid ? scores_lds[head_local * SKV_PAD + dim_thread] * softmax_scale
                             : -INFINITY;

        // Max reduction within 16-thread sub-group
        float maxVal = raw;
        #pragma unroll
        for (int off = 8; off >= 1; off >>= 1) {
            maxVal = fmaxf(maxVal, __shfl_xor(maxVal, off, 16));
        }

        // Exp + sum (branchless: invalid has -INF -> exp=0)
        float exp_val = sm_valid ? expf(raw - maxVal) : 0.0f;
        float sumExp = exp_val;
        #pragma unroll
        for (int off = 8; off >= 1; off >>= 1) {
            sumExp += __shfl_xor(sumExp, off, 16);
        }

        // Normalize and write back
        float norm_val = (sumExp > 0.0f) ? (exp_val / sumExp) : 0.0f;
        scores_lds[head_local * SKV_PAD + dim_thread] = norm_val;
    }

    __syncthreads();  // -- Barrier 2: softmax weights visible --

    // ========================================================================
    // Stage 3: Attn x V via Vectorized FMA
    //
    // For each kv position:
    //   1. Load softmax weight from scores_lds (broadcast within head)
    //   2. Load V[kv, head, my_dims] with bf16x8 (coalesced)
    //   3. Weighted accumulate: acc[d] += weight * V[d]
    //
    // Softmax weights are cached in registers before the loop to avoid
    // repeated LDS reads (up to 16 weights per head).
    //
    // V load coalescing: identical to K loads — contiguous within head,
    // heads contiguous in memory. 1KB per warp per kv_pos.
    //
    // Register usage: acc[8 or 16] + w_cache[16] = 24-32 VGPRs
    // ========================================================================
    float acc[16] = {0};   // output accumulator, max dims_per_thread = 16

    // Cache all softmax weights in registers (read LDS once, reuse per dim)
    float w_cache[16];
    #pragma unroll
    for (int kv = 0; kv < 16; kv++) {
        w_cache[kv] = (kv < seqlen_kv)
            ? scores_lds[head_local * SKV_PAD + kv]
            : 0.0f;
    }

    if (active) {
        for (int kv = 0; kv < seqlen_kv; kv++) {
            const float w = w_cache[kv];
            const bhalf_t* V_kv = &V_base[kv * kv_stride + dim_start];

            bf16x8 vv0 = *(const bf16x8*)(V_kv);
            #pragma unroll
            for (int i = 0; i < 8; i++)
                acc[i] += w * static_cast<float>(vv0[i]);

            if (dims_per_thread > 8) {
                bf16x8 vv1 = *(const bf16x8*)(V_kv + 8);
                #pragma unroll
                for (int i = 0; i < 8; i++)
                    acc[8 + i] += w * static_cast<float>(vv1[i]);
            }
        }
    }

    // ========================================================================
    // Stage 4: Write Output with Vector Stores
    //
    // Convert FP32 accumulators to FP16 and write with 128-bit stores.
    // 16 consecutive threads per head -> contiguous 256-byte writes per head.
    // 4 heads per warp -> 1KB contiguous. Fully coalesced.
    // ========================================================================
    if (active) {
        // First 8 dims: 128-bit fp16x8 vector store
        fp16x8 out0;
        #pragma unroll
        for (int i = 0; i < 8; i++)
            out0[i] = static_cast<half_t>(acc[i]);
        *(fp16x8*)(&O_ptr[dim_start]) = out0;

        if (dims_per_thread > 8) {
            fp16x8 out1;
            #pragma unroll
            for (int i = 0; i < 8; i++)
                out1[i] = static_cast<half_t>(acc[8 + i]);
            *(fp16x8*)(&O_ptr[dim_start + 8]) = out1;
        }
    }
}
