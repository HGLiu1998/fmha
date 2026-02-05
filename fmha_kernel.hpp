#include <hip/hip_runtime.h>
#include <stdio.h>
#include <math.h>

using bhalf_t = __bf16;
using half_t  = _Float16;
using bf16x2 = __bf16 __attribute__((ext_vector_type(2)));
using bf16x4 = __bf16 __attribute__((ext_vector_type(4)));
using bf16x8 = __bf16 __attribute__((ext_vector_type(8)));
using floatx4 = float __attribute__((ext_vector_type(4)));

#define WARP_SIZE 64
#define BLOCK_SIZE 256
#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))

// ============================================================================
// FMHA Kernel using v_mfma_f32_16x16x16_bf16
// Grid: (1, num_heads, batch) - Each block handles ONE head
// Block: 256 threads (4 warps of 64 threads each)
// Spec: seqlen_q = 1, seqlen_kv = 1-16, head_dim = 128 or 256
// ============================================================================
__global__
__launch_bounds__(256, 1)
void fmha_forward_kernel(
    const bhalf_t* Q,        // [B, H_q, S_q, D] - bf16
    const bhalf_t* K,        // [B, H_kv, S_kv, D] - bf16
    const bhalf_t* V,        // [B, H_kv, S_kv, D] - bf16
    half_t* O,               // [B, H_q, S_q, D] - fp16
    const int batch,
    const int num_heads_q,
    const int num_heads_kv,
    const int seqlen_q,
    const int seqlen_kv,
    const int head_dim_q,
    const int head_dim_kv,
    const float softmax_scale)
{
    // ========================================================================
    // Thread and Block Mapping
    // ========================================================================
    // blockIdx.x: always 0 (not used)
    // blockIdx.y: head index [0, num_heads_q)
    // blockIdx.z: batch index [0, batch)
    const int batch_idx = blockIdx.z;
    const int head_idx = blockIdx.y;
    
    const int tid = threadIdx.x;          // [0, 256)
    const int warp_id = tid / WARP_SIZE;  // [0, 4) - 4 warps
    const int lane_id = tid % WARP_SIZE;  // [0, 64) - within warp
    
    // ========================================================================
    // Pointer Adjustment to Current Batch and Head
    // ========================================================================
    // Move to the correct batch and head
    const bhalf_t* Q_ptr = Q + batch_idx * num_heads_q * seqlen_q * head_dim_q 
                             + head_idx * seqlen_q * head_dim_q;
    const bhalf_t* K_ptr = K + batch_idx * num_heads_kv * seqlen_kv * head_dim_kv 
                             + head_idx * seqlen_kv * head_dim_kv;  // Assuming GQA: head_idx may need adjustment
    const bhalf_t* V_ptr = V + batch_idx * num_heads_kv * seqlen_kv * head_dim_kv 
                             + head_idx * seqlen_kv * head_dim_kv;
    half_t* O_ptr = O + batch_idx * num_heads_q * seqlen_q * head_dim_q 
                      + head_idx * seqlen_q * head_dim_q;
    
    // ========================================================================
    // Shared Memory Allocation (per head)
    // ========================================================================
    // Q: [1, head_dim] with padding for bank conflict avoidance
    // K: [seqlen_kv (max 16), head_dim] with padding
    // V: [seqlen_kv (max 16), head_dim] with padding (can reuse K space)
    // Scores: [seqlen_kv (max 16)] in registers or shared memory
    constexpr int max_seqlen_kv = 16;
    constexpr int max_head_dim = 256;
    constexpr int padding = 4;  // For bank conflict avoidance
    
    __shared__ __attribute__((aligned(128))) bhalf_t Qs[1 * (max_head_dim + padding)];
    
    // Use union to share memory between K and V (not accessed simultaneously)
    union {
        __attribute__((aligned(128))) bhalf_t Ks[max_seqlen_kv * (max_head_dim + padding)];
        __attribute__((aligned(128))) bhalf_t Vs[max_seqlen_kv * (max_head_dim + padding)];
    } KV_shared;
    
    __shared__ __attribute__((aligned(128))) float scores[max_seqlen_kv];  // QK^T results
    __shared__ __attribute__((aligned(128))) float softmax_out[max_seqlen_kv];  // After softmax
    
    // ========================================================================
    // PHASE 1: Load Q into Shared Memory
    // ========================================================================
    // Q is [1, head_dim], all 256 threads cooperate to load using 128-bit loads
    constexpr int vec_size = 8;  // bf16x8 = 128-bit
    const int vecs_per_dim = CEIL_DIV(head_dim_q, vec_size);
    const int vecs_per_thread_q = CEIL_DIV(vecs_per_dim, 256);
    
    #pragma unroll
    for (int v = 0; v < vecs_per_thread_q; ++v) {
        const int vec_idx = v * 256 + tid;  // Perfect coalescing
        if (vec_idx < vecs_per_dim) {
            const int d = vec_idx * vec_size;
            const int lds_offset = d;
            const int global_offset = d;
            
            *(bf16x8*)(&Qs[lds_offset]) = *(bf16x8*)(&Q_ptr[global_offset]);
        }
    }
    __syncthreads();
    
    // ========================================================================
    // PHASE 2: Compute Q @ K^T using MFMA (tiled along head_dim)
    // ========================================================================
    // We tile head_dim into chunks of 16 for the MFMA K-dimension
    // MFMA: v_mfma_f32_16x16x16_bf16 computes C[16x16] += A[16x16] * B[16x16]
    //   - M=16: rows (can be used for query tokens, but we only have 1)
    //   - N=16: cols (can be used for key tokens, up to 16)
    //   - K=16: reduction dimension (head_dim tiles)
    
    constexpr int mfma_k_dim = 16;  // Reduction dimension for MFMA
    const int dim_tiles = CEIL_DIV(head_dim_q, mfma_k_dim);
    
    // Initialize score accumulator in registers (one per lane for each key token)
    // For simplicity, we'll use warp 0 to compute, others can assist with loading
    float score_acc[max_seqlen_kv];
    #pragma unroll
    for (int i = 0; i < max_seqlen_kv; ++i) {
        score_acc[i] = 0.0f;
    }
    
    // Load K tile-by-tile along dimension and accumulate Q @ K^T
    for (int dim_tile = 0; dim_tile < dim_tiles; ++dim_tile) {
        const int d_start = dim_tile * mfma_k_dim;
        const int d_size = (d_start + mfma_k_dim <= head_dim_kv) ? mfma_k_dim : (head_dim_kv - d_start);
        
        // ====================================================================
        // Load K tile: [seqlen_kv, d_size] into shared memory
        // ====================================================================
        // All 256 threads cooperate to load K[seqlen_kv][d_start:d_start+d_size]
        const int total_k_vecs = seqlen_kv * CEIL_DIV(d_size, vec_size);
        const int vecs_per_thread_k = CEIL_DIV(total_k_vecs, 256);
        
        #pragma unroll
        for (int v = 0; v < vecs_per_thread_k; ++v) {
            const int vec_idx = v * 256 + tid;  // Perfect coalescing
            if (vec_idx < total_k_vecs) {
                const int vecs_per_token = CEIL_DIV(d_size, vec_size);
                const int token_idx = vec_idx / vecs_per_token;
                const int vec_in_token = vec_idx % vecs_per_token;
                const int d = d_start + vec_in_token * vec_size;
                
                if (token_idx < seqlen_kv && d < head_dim_kv) {
                    const int lds_offset = token_idx * (max_head_dim + padding) + (d - d_start);
                    const int global_offset = token_idx * head_dim_kv + d;
                    
                    // Handle partial vectors at the end
                    if (d + vec_size <= head_dim_kv) {
                        *(bf16x8*)(&KV_shared.Ks[lds_offset]) = *(bf16x8*)(&K_ptr[global_offset]);
                    } else {
                        // Scalar loads for partial vector
                        for (int i = 0; i < vec_size && (d + i) < head_dim_kv; ++i) {
                            KV_shared.Ks[lds_offset + i] = K_ptr[global_offset + i];
                        }
                    }
                }
            }
        }
        __syncthreads();
        
        // ====================================================================
        // Compute Q[1, d_size] @ K^T[d_size, seqlen_kv] using MFMA
        // ====================================================================
        // TODO: Use v_mfma_f32_16x16x16_bf16 intrinsic here
        // For now, use a simple dot product as placeholder
        
        if (warp_id == 0) {
            // Each lane computes dot product for one or more key tokens
            // Distribute seqlen_kv across 64 lanes
            for (int k = lane_id; k < seqlen_kv; k += WARP_SIZE) {
                float partial_sum = 0.0f;
                
                // Dot product over the current dimension tile
                #pragma unroll
                for (int d = 0; d < d_size; ++d) {
                    const int q_offset = d_start + d;
                    const int k_offset = k * (max_head_dim + padding) + d;
                    
                    float q_val = __bfloat162float(Qs[q_offset]);
                    float k_val = __bfloat162float(KV_shared.Ks[k_offset]);
                    partial_sum += q_val * k_val;
                }
                
                score_acc[k] += partial_sum;
            }
        }
        __syncthreads();
    }
    
    // ========================================================================
    // PHASE 3: Apply Softmax Scale and Compute Softmax
    // ========================================================================
    // Write scores to shared memory and mask invalid positions
    if (warp_id == 0) {
        for (int k = lane_id; k < max_seqlen_kv; k += WARP_SIZE) {
            if (k < seqlen_kv) {
                scores[k] = score_acc[k] * softmax_scale;
            } else {
                // Mask padded positions with -inf to exclude from softmax
                scores[k] = -INFINITY;
            }
        }
    }
    __syncthreads();
    
    // Online softmax: compute max, subtract, exp, sum, normalize
    __shared__ float max_score;
    __shared__ float sum_exp;
    
    if (tid == 0) {
        float max_val = -INFINITY;
        for (int k = 0; k < seqlen_kv; ++k) {
            max_val = fmaxf(max_val, scores[k]);
        }
        max_score = max_val;
    }
    __syncthreads();
    
    // Compute exp(score - max) and sum
    // Note: exp(-inf - max) = 0, so masked positions naturally become 0
    if (warp_id == 0) {
        for (int k = lane_id; k < max_seqlen_kv; k += WARP_SIZE) {
            if (k < seqlen_kv) {
                softmax_out[k] = expf(scores[k] - max_score);
            } else {
                softmax_out[k] = 0.0f;  // Explicitly zero out padded positions
            }
        }
    }
    __syncthreads();
    
    if (tid == 0) {
        float sum_val = 0.0f;
        for (int k = 0; k < seqlen_kv; ++k) {
            sum_val += softmax_out[k];
        }
        sum_exp = sum_val;
    }
    __syncthreads();
    
    // Normalize (only valid positions, padded positions remain 0)
    if (warp_id == 0) {
        for (int k = lane_id; k < seqlen_kv; k += WARP_SIZE) {
            softmax_out[k] /= sum_exp;
        }
    }
    __syncthreads();
    
    // ========================================================================
    // PHASE 4: Compute Output O = Softmax(QK^T) @ V
    // ========================================================================
    // Load V and compute weighted sum
    float output_acc[max_head_dim];
    #pragma unroll
    for (int d = 0; d < max_head_dim; ++d) {
        output_acc[d] = 0.0f;
    }
    
    // Tile along head_dim for V as well
    for (int dim_tile = 0; dim_tile < dim_tiles; ++dim_tile) {
        const int d_start = dim_tile * mfma_k_dim;
        const int d_size = (d_start + mfma_k_dim <= head_dim_kv) ? mfma_k_dim : (head_dim_kv - d_start);
        
        // Load V tile: [seqlen_kv, d_size]
        const int total_v_vecs = seqlen_kv * CEIL_DIV(d_size, vec_size);
        const int vecs_per_thread_v = CEIL_DIV(total_v_vecs, 256);
        
        #pragma unroll
        for (int v = 0; v < vecs_per_thread_v; ++v) {
            const int vec_idx = v * 256 + tid;
            if (vec_idx < total_v_vecs) {
                const int vecs_per_token = CEIL_DIV(d_size, vec_size);
                const int token_idx = vec_idx / vecs_per_token;
                const int vec_in_token = vec_idx % vecs_per_token;
                const int d = d_start + vec_in_token * vec_size;
                
                if (token_idx < seqlen_kv && d < head_dim_kv) {
                    const int lds_offset = token_idx * (max_head_dim + padding) + (d - d_start);
                    const int global_offset = token_idx * head_dim_kv + d;
                    
                    if (d + vec_size <= head_dim_kv) {
                        *(bf16x8*)(&KV_shared.Vs[lds_offset]) = *(bf16x8*)(&V_ptr[global_offset]);
                    } else {
                        for (int i = 0; i < vec_size && (d + i) < head_dim_kv; ++i) {
                            KV_shared.Vs[lds_offset + i] = V_ptr[global_offset + i];
                        }
                    }
                }
            }
        }
        __syncthreads();
        
        // Compute weighted sum: O[d] = sum_k(softmax[k] * V[k, d])
        if (warp_id == 0) {
            for (int d = 0; d < d_size; ++d) {
                float sum = 0.0f;
                
                // Each lane handles some dimensions
                if (lane_id < d_size) {
                    for (int k = 0; k < seqlen_kv; ++k) {
                        const int v_offset = k * (max_head_dim + padding) + d;
                        float v_val = __bfloat162float(KV_shared.Vs[v_offset]);
                        sum += softmax_out[k] * v_val;
                    }
                    output_acc[d_start + d] = sum;
                }
            }
        }
        __syncthreads();
    }
    
    // ========================================================================
    // PHASE 5: Write Output to Global Memory
    // ========================================================================
    // Convert to fp16 and write out
    if (warp_id == 0) {
        const int dims_per_thread = CEIL_DIV(head_dim_q, WARP_SIZE);
        
        for (int i = 0; i < dims_per_thread; ++i) {
            const int d = lane_id * dims_per_thread + i;
            if (d < head_dim_q) {
                O_ptr[d] = __float2half(output_acc[d]);
            }
        }
    }
}


__global__
__launch_bounds__(256, 1)
void fmha_forward_kernel_head_combined(
    const bhalf_t* Q,        // [B, H_q, S_q, D] - bf16
    const bhalf_t* K,        // [B, H_kv, S_kv, D] - bf16
    const bhalf_t* V,        // [B, H_kv, S_kv, D] - bf16
    half_t* O,               // [B, H_q, S_q, D] - fp16
    const int batch,
    const int num_heads_q,
    const int num_heads_kv,
    const int seqlen_q,
    const int seqlen_kv,
    const int head_dim_q,
    const int head_dim_kv,
    const float softmax_scale) 
{
    // Thread mapping: (batch, head, seq_position)
    const int batchIdx = blockIdx.z;
    //const int head_idx = blockIdx.y; No head. we reduce head dimension to seqlen.
    // move pointer to the start of the batch
    Q += batchIdx * num_heads_q * seqlen_q * head_dim_q;
    K += batchIdx * num_heads_kv * seqlen_kv * head_dim_kv;
    V += batchIdx * num_heads_kv * seqlen_kv * head_dim_kv;
    O += batchIdx * num_heads_q * seqlen_q * head_dim_q;
    
    const int tid = threadIdx.x;
    const int warpIdx = tid / 64; 
    const int laneIdx = tid % 64;

    // Process 16 heads at a time (MFMA handles 16 blocks)
    constexpr int heads_tile_size = 16;
    constexpr int k_tile_size = 4;  // Process 4 tokens per K/V tile
    
    // Determine which head group to process (for 32 heads: 2 groups)
    const int num_head_groups = CEIL_DIV(num_heads_q, heads_tile_size);

    // Head index within the current group (0-15)
    const int headIdx = laneIdx / 4;  // [0-15]
    const int laneInHead = laneIdx % 4;  // [0-3]
    
    // Adjust pointers to start of this head group
    const int M_iter = CEIL_DIV(seqlen_kv, k_tile_size);
    const int N_iter = CEIL_DIV(seqlen_q, 4);  // seqlen_q fixed to 1

    // Shared memory for 16 heads at a time
    // For 32 heads: process in 2 passes (or 2 waves in parallel with separate smem regions)
    // Q: 16 heads × 1 token × (head_dim + 4) padding
    // K: 16 heads × 4 tokens × (head_dim + 4) padding (tile_size=4)
    // Scores: 16 heads × 1 × 16 max_seqlen_k
    // Max memory (head_dim=256): 16×(256+4)×2 + 16×4×(256+4)×2 + 16×16×4 = 8.3KB + 33.3KB + 1KB = 42.6KB
    __shared__ __attribute__((aligned(128))) bhalf_t Qs[16 * 1 * (256 + 4)];  // Max: 16 heads, seqlen_q=1, head_dim=256
    __shared__ __attribute__((aligned(128))) bhalf_t Ks[16 * 4 * (256 + 4)];  // Max: 16 heads, tile_size=4, head_dim=256
    __shared__ __attribute__((aligned(128))) float scores[16 * 1 * 16];       // Max: 16 heads, seqlen_q=1, max_seqlen_k=16

    // PHASE 1 & 2: Iterate over head groups (1 group for 16 heads, 2 groups for 32 heads)
    // Each group processes 16 heads using MFMA 16-block instruction
    constexpr int vec_size = 8;  // Use 128-bit loads (8 bf16 values)
    
    // Since seqlen_q = 1, simplify calculations
    const int vecs_per_dim = CEIL_DIV(head_dim_q, vec_size);  // Vectors per head dimension (head_dim/8)
    
    // For 16 heads: all waves can participate or wave 0 only
    // For 32 heads: wave 0 & 2 handle heads 0-15, wave 1 & 3 handle heads 16-31
    const bool should_compute = (warpIdx == 0);  // For now, only wave 0 computes
    
    // Iterate over head groups (1 for 16 heads, 2 for 32 heads)
    for (int head_group = 0; head_group < num_head_groups; ++head_group) {
        // Calculate head offset for this group (0 for first group, 16 for second group)
        const int head_group_offset = head_group * heads_tile_size;
        
        // Adjust Q, K, V, O pointers to this head group
        const bhalf_t* Q_group = Q + head_group_offset * seqlen_q * head_dim_q;
        const bhalf_t* K_group = K + head_group_offset * seqlen_kv * head_dim_kv;
        const bhalf_t* V_group = V + head_group_offset * seqlen_kv * head_dim_kv;
        half_t* O_group = O + head_group_offset * seqlen_q * head_dim_q;
        
        // ====================================================================
        // Step 1: Load Q for all 16 heads in this group (parallel)
        // ====================================================================
        // All 256 threads cooperate to load Q[16 heads][head_dim] using 128-bit loads
        const int total_q_vecs = heads_tile_size * vecs_per_dim;  // 16 × (head_dim/8)
        const int vecs_per_thread = CEIL_DIV(total_q_vecs, 256);
        
        #pragma unroll
        for (int v = 0; v < vecs_per_thread; ++v) {
            const int vec_idx = tid * vecs_per_thread + v;
            if (vec_idx < total_q_vecs) {
                const int h = vec_idx / vecs_per_dim;  // Which head [0-15]
                const int d = (vec_idx % vecs_per_dim) * vec_size;  // Dimension offset [0, 8, 16, ...]
                
                // Global memory offset: Q_group[h][d] (seqlen_q=1, so no seq dimension)
                const int qGlobalOffset = h * head_dim_q + d;
                
                // LDS offset: Qs[h][d] with padding
                const int qLDSOffset = h * (head_dim_q + 4) + d;
                
                // 128-bit vectorized load from global to shared memory
                *(bf16x8*)(&Qs[qLDSOffset]) = *(bf16x8*)(&Q_group[qGlobalOffset]);
            }
        }
        
        // ====================================================================
        // Step 2: Load all K tiles for all 16 heads in this group (parallel)
        // ====================================================================
        for (int kv_tile = 0; kv_tile < M_iter; ++kv_tile) {
            const int token_start = kv_tile * k_tile_size;
            const int tokens_this_tile = (seqlen_kv - token_start < k_tile_size) ? 
                                         (seqlen_kv - token_start) : k_tile_size;
            
            // All 256 threads cooperate to load K[16 heads][tokens_this_tile][head_dim] using 128-bit loads
            const int total_k_vecs = heads_tile_size * tokens_this_tile * vecs_per_dim;
            const int vecs_per_thread = CEIL_DIV(total_k_vecs, 256);
            
            #pragma unroll
            for (int v = 0; v < vecs_per_thread; ++v) {
                const int vec_idx = tid * vecs_per_thread + v;
                if (vec_idx < total_k_vecs) {
                    // Decode: vec_idx → (head, token_in_tile, dim)
                    const int vecs_per_head_tile = tokens_this_tile * vecs_per_dim;
                    const int h = vec_idx / vecs_per_head_tile;  // Which head [0-15]
                    const int vec_in_head = vec_idx % vecs_per_head_tile;
                    const int t = vec_in_head / vecs_per_dim;     // Token within tile [0, tokens_this_tile)
                    const int d = (vec_in_head % vecs_per_dim) * vec_size;  // Dimension [0, 8, 16, ...]
                    
                    const int token_global = token_start + t;  // Global token index
                    
                    // Global memory offset: K_group[h][token_global][d]
                    const int kGlobalOffset = h * seqlen_kv * head_dim_kv + token_global * head_dim_kv + d;
                    
                    // LDS offset: Ks[h][t][d] with padding
                    const int kLDSOffset = h * k_tile_size * (head_dim_kv + 4) + t * (head_dim_kv + 4) + d;
                    
                    // 128-bit vectorized load from global to shared memory
                    *(bf16x8*)(&Ks[kLDSOffset]) = *(bf16x8*)(&K_group[kGlobalOffset]);

                }
            }
        }
        __syncthreads();  // Ensure all Q and K data for this group is loaded
        
        if (should_compute) {
            // ================================================================
            // PHASE 3: Compute Q @ K^T using MFMA for this group of 16 heads
            // ================================================================
            for (int kv_tile = 0; kv_tile < M_iter; ++kv_tile) {
                const int token_start = kv_tile * k_tile_size;
                const int tokens_this_tile = (seqlen_kv - token_start < k_tile_size) ? 
                                             (seqlen_kv - token_start) : k_tile_size;
                
                // TODO: Compute Q @ K^T using MFMA here
                // MFMA v_mfma_f32_4x4x4_16b_bf16 processes all 16 heads simultaneously
                // Input: Q[16 heads][head_dim] from LDS
                // Input: K[16 heads][tokens_this_tile][head_dim] from LDS
                // Output: scores[16 heads][tokens_this_tile] accumulated in registers
                
            }
            
            // TODO: Softmax computation for this group
            
            // TODO: Load V tiles and compute output O = Attention @ V for this group
            
        }  // end if (should_compute)
        
        __syncthreads();  // Sync before moving to next head group
        
    }  // end for (head_group)

}