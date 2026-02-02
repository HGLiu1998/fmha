/**
 * Flash Multi-Head Attention (FMHA) - Kernel Implementation
 * 
 * Computes: O = softmax(Q·K^T / √d) · V using online softmax for memory efficiency.
 * 
 * Grid: (seqlen_q/256, num_heads_q, batch)
 * Block: 256 threads
 * Each thread handles one query position across all KV positions.
 * 
 * Precision: bf16 inputs (Q/K/V), fp16 output (O), fp32 internal computation.
 */

#include <hip/hip_runtime.h>
#include <stdio.h>
#include <math.h>

using bhalf_t = __bf16;
using half_t  = _Float16;

#define WARP_SIZE 64
#define BLOCK_SIZE 256

/**
 * FMHA Forward Kernel
 * 
 * Tensors in BHSD layout: [batch, num_heads, seqlen, head_dim]
 * Supports Grouped Query Attention (GQA): num_heads_q >= num_heads_kv
 */
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
    const float softmax_scale
) {
    // Thread mapping: (batch, head, seq_position)
    const int batch_idx = blockIdx.z;
    const int head_idx = blockIdx.y;
    const int seq_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (batch_idx >= batch || head_idx >= num_heads_q || seq_idx >= seqlen_q) {
        return;
    }

    // GQA: Map query head to corresponding KV head
    // Standard MHA: num_heads_q == num_heads_kv (1:1 mapping)
    // GQA: Multiple Q heads share one KV head (e.g., 32:8 = 4:1)
    const int kv_head_idx = (head_idx * num_heads_kv) / num_heads_q;

    // Online softmax state (Flash Attention core)
    // Maintains: row_max = max(scores), row_sum = Σ exp(score - row_max)
    float row_max = -INFINITY;
    float row_sum = 0.0f;

    // Output accumulator (fp32 for precision)
    float acc[256] = {0.0f};

    // Validate head dimensions match
    if (head_dim_q != head_dim_kv) {
        return;
    }
    
    // Compute offset for Q[batch_idx, head_idx, seq_idx, :]
    const int q_offset = batch_idx * num_heads_q * seqlen_q * head_dim_q +
                         head_idx * seqlen_q * head_dim_q +
                         seq_idx * head_dim_q;

    // Main loop: Process all KV positions for this query
    // Fuses: attention scores (Q·K^T), softmax, and weighted sum (P·V)
    for (int kv_idx = 0; kv_idx < seqlen_kv; kv_idx++) {
        // Compute attention score: Q · K^T / √d
        float score = 0.0f;
        const int k_offset = batch_idx * num_heads_kv * seqlen_kv * head_dim_kv +
                            kv_head_idx * seqlen_kv * head_dim_kv +
                            kv_idx * head_dim_kv;

        // Dot product (bf16 -> fp32)
        for (int d = 0; d < head_dim_q; d++) {
            float q_val = __bfloat162float(Q[q_offset + d]);
            float k_val = __bfloat162float(K[k_offset + d]);
            score += q_val * k_val;
        }
        score *= softmax_scale;  // Scale by 1/√d

        // Online softmax update
        // When max changes, rescale previous values: exp(x-a) = exp(x-b)*exp(b-a)
        float prev_max = row_max;
        row_max = fmaxf(row_max, score);
        float scale = expf(prev_max - row_max);
        
        row_sum *= scale;  // Rescale sum
        float exp_score = expf(score - row_max);
        row_sum += exp_score;

        // Accumulate weighted V: acc = acc * scale + exp_score * V
        const int v_offset = batch_idx * num_heads_kv * seqlen_kv * head_dim_kv +
                            kv_head_idx * seqlen_kv * head_dim_kv +
                            kv_idx * head_dim_kv;

        for (int d = 0; d < head_dim_q; d++) {
            float v_val = __bfloat162float(V[v_offset + d]);
            acc[d] = acc[d] * scale + exp_score * v_val;
        }
    }

    // Finalize: normalize by softmax sum and write output
    // Output: O[d] = (Σ exp(score_j - max) * V[j,d]) / (Σ exp(score_i - max))
    float inv_sum = 1.0f / (row_sum + 1e-10f);

    const int o_offset = batch_idx * num_heads_q * seqlen_q * head_dim_q +
                        head_idx * seqlen_q * head_dim_q +
                        seq_idx * head_dim_q;

    // Write output (fp32 -> fp16)
    for (int d = 0; d < head_dim_q; d++) {
        O[o_offset + d] = __float2half(acc[d] * inv_sum);
    }
}