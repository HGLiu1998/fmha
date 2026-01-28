#include <hip/hip_runtime.h>
#include <stdio.h>
#include <math.h>

// Type alias for bfloat16
using bhalf_t = __bf16;
using half_t  = _Float16;

// Flash Attention kernel configuration
#define WARP_SIZE 64        // AMD GPU warp size (64 threads per wavefront)
#define BLOCK_SIZE 256      // Threads per block

/**
 * Flash Multi-Head Attention Forward Kernel
 *
 * Computes attention output: O = softmax(Q·K^T / √d) · V
 * using online softmax algorithm for memory efficiency.
 *
 * Thread Organization:
 * - Grid X: Sequence dimension (seqlen_q), blocks of BLOCK_SIZE threads
 * - Grid Y: Number of query heads (num_heads_q)
 * - Grid Z: Batch size
 * - Each thread processes one query position across all key/value positions
 *
 * Memory Precision:
 * - Inputs Q, K, V: bfloat16 (bf16) - 2 bytes, wider dynamic range than fp16
 * - Output O: float16 (fp16) - 2 bytes, sufficient precision for output
 * - Internal computation: float32 for numerical stability
 *
 * @param Q Query tensor [batch, num_heads_q, seqlen_q, head_dim_q] in bf16
 * @param K Key tensor [batch, num_heads_kv, seqlen_kv, head_dim_kv] in bf16
 * @param V Value tensor [batch, num_heads_kv, seqlen_kv, head_dim_kv] in bf16
 * @param O Output tensor [batch, num_heads_q, seqlen_q, head_dim_q] in fp16
 * @param batch Batch size
 * @param num_heads_q Number of query heads (for GQA, can be > num_heads_kv)
 * @param num_heads_kv Number of key/value heads
 * @param seqlen_q Query sequence length
 * @param seqlen_kv Key/value sequence length
 * @param head_dim_q Head dimension for Q (must equal head_dim_kv)
 * @param head_dim_kv Head dimension for K and V
 * @param softmax_scale Scaling factor for attention scores (typically 1/√d)
 */
__global__ void fmha_forward_kernel(
    const bhalf_t* Q,        // [batch, num_heads_q, seqlen_q, head_dim_q] - bf16
    const bhalf_t* K,        // [batch, num_heads_kv, seqlen_kv, head_dim_kv] - bf16
    const bhalf_t* V,        // [batch, num_heads_kv, seqlen_kv, head_dim_kv] - bf16
    half* O,                 // [batch, num_heads_q, seqlen_q, head_dim_q] - fp16
    const int batch,
    const int num_heads_q,
    const int num_heads_kv,
    const int seqlen_q,
    const int seqlen_kv,
    const int head_dim_q,
    const int head_dim_kv,
    const float softmax_scale
) {
    // ========================================================================
    // Thread Index Calculation
    // ========================================================================
    // Map 3D block/thread indices to batch, head, and sequence dimensions
    const int batch_idx = blockIdx.z;   // Which batch element (0 to batch-1)
    const int head_idx = blockIdx.y;    // Which query head (0 to num_heads_q-1)
    const int seq_idx = blockIdx.x * blockDim.x + threadIdx.x;  // Which query position

    // Early exit for threads beyond valid range
    if (batch_idx >= batch || head_idx >= num_heads_q || seq_idx >= seqlen_q) {
        return;
    }

    // ========================================================================
    // Grouped Query Attention (GQA) Head Mapping
    // ========================================================================
    // Maps query heads to key/value heads. For standard MHA, num_heads_q == num_heads_kv.
    // For GQA, multiple query heads share the same key/value heads.
    // Example: 32 query heads with 8 KV heads -> each KV head is shared by 4 query heads
    // Requirement: num_heads_q must be divisible by num_heads_kv
    const int kv_head_idx = (head_idx * num_heads_kv) / num_heads_q;

    // ========================================================================
    // Online Softmax State Initialization
    // ========================================================================
    // The online softmax algorithm maintains running max and sum to compute
    // softmax incrementally without storing the full attention matrix.
    // This is the core of Flash Attention's memory efficiency.
    //
    // Algorithm invariant: At step i, we maintain:
    //   row_max = max(score_0, ..., score_i)
    //   row_sum = Σ exp(score_j - row_max) for j=0..i
    //   acc[d] = Σ exp(score_j - row_max) * V[j,d] for j=0..i
    //
    // When row_max increases, we rescale both row_sum and acc to maintain correctness.
    float row_max = -INFINITY;  // Running maximum of attention scores
    float row_sum = 0.0f;       // Running sum of exp(score - row_max)

    // ========================================================================
    // Output Accumulator
    // ========================================================================
    // Accumulates the weighted sum of V values in float32 for precision.
    // Sized for maximum head dimension (256). Each element stores:
    //   acc[d] = Σ_j exp(score_j - row_max) * V[j, d]
    float acc[256] = {0.0f};

    // ========================================================================
    // Input Validation
    // ========================================================================
    // Q and K must have the same head dimension for valid dot product computation.
    // V's head dimension must also match to ensure proper accumulation.
    if (head_dim_q != head_dim_kv) {
        return;  // Invalid configuration, exit silently
    }
    
    // ========================================================================
    // Query Vector Offset Calculation
    // ========================================================================
    // Compute the starting index for this thread's query vector in BHSD layout
    // Layout: [batch, num_heads_q, seqlen_q, head_dim_q]
    const int q_offset = batch_idx * num_heads_q * seqlen_q * head_dim_q +
                         head_idx * seqlen_q * head_dim_q +
                         seq_idx * head_dim_q;

    // ========================================================================
    // Main Attention Loop: Iterate over all Key/Value positions
    // ========================================================================
    // For each query position, we compute attention over all key positions.
    // This loop implements the fused computation of:
    //   1. Attention scores: S = Q·K^T / √d
    //   2. Softmax: P = softmax(S)
    //   3. Output: O = P·V
    // All in a single pass without materializing the attention matrix.
    for (int kv_idx = 0; kv_idx < seqlen_kv; kv_idx++) {
        // --------------------------------------------------------------------
        // Step 1: Compute Attention Score (Q · K^T)
        // --------------------------------------------------------------------
        float score = 0.0f;

        // Calculate offset for K[batch_idx, kv_head_idx, kv_idx, :]
        const int k_offset = batch_idx * num_heads_kv * seqlen_kv * head_dim_kv +
                            kv_head_idx * seqlen_kv * head_dim_kv +
                            kv_idx * head_dim_kv;

        // Dot product: Q[seq_idx] · K[kv_idx]
        // Convert bf16 -> fp32 for computation, then accumulate
        for (int d = 0; d < head_dim_q; d++) {
            float q_val = __bfloat162float(Q[q_offset + d]);
            float k_val = __bfloat162float(K[k_offset + d]);
            score += q_val * k_val;
        }

        // Scale by 1/√d for numerical stability (prevents very large logits)
        score *= softmax_scale;

        // --------------------------------------------------------------------
        // Step 2: Online Softmax Update
        // --------------------------------------------------------------------
        // Online softmax algorithm: when we see a new score, we may need to
        // update the running max. If the max changes, we must rescale all
        // previous computations to maintain correctness.
        //
        // Mathematical foundation:
        //   exp(x - a) / Σ exp(x_i - a) = exp(x - b) / Σ exp(x_i - b)
        //   where we can convert between different bases a and b using:
        //   exp(x - a) = exp(x - b) * exp(b - a)

        float prev_max = row_max;  // Store old max
        row_max = fmaxf(row_max, score);  // Update to new max

        // Compute rescaling factor: exp(old_max - new_max)
        // If new_max > old_max, this is < 1, which scales down previous values
        // If new_max == old_max, this is 1, no rescaling needed
        float scale = expf(prev_max - row_max);

        // Rescale the running sum to account for the new max
        row_sum *= scale;

        // Compute current score's contribution with the new max
        float exp_score = expf(score - row_max);
        row_sum += exp_score;  // Add to running sum

        // --------------------------------------------------------------------
        // Step 3: Accumulate Weighted V Values
        // --------------------------------------------------------------------
        // Calculate offset for V[batch_idx, kv_head_idx, kv_idx, :]
        const int v_offset = batch_idx * num_heads_kv * seqlen_kv * head_dim_kv +
                            kv_head_idx * seqlen_kv * head_dim_kv +
                            kv_idx * head_dim_kv;

        // Update accumulator: acc = acc * scale + exp_score * V
        // - acc * scale: rescale previous accumulations due to max change
        // - exp_score * V: add current position's contribution
        for (int d = 0; d < head_dim_q; d++) {
            float v_val = __bfloat162float(V[v_offset + d]);
            acc[d] = acc[d] * scale + exp_score * v_val;
        }
    }  // End of KV loop

    // ========================================================================
    // Step 4: Finalize Softmax and Write Output
    // ========================================================================
    // At this point, acc contains: Σ exp(score_j - row_max) * V[j, d]
    // and row_sum contains: Σ exp(score_j - row_max)
    // To complete the softmax normalization, we divide acc by row_sum.
    //
    // Final output: O[d] = Σ_j softmax(Q·K^T)[j] * V[j, d]
    //                    = Σ_j (exp(score_j - max) / Σ exp(score_i - max)) * V[j, d]
    //                    = (Σ_j exp(score_j - max) * V[j, d]) / (Σ exp(score_i - max))
    //                    = acc[d] / row_sum

    // Compute inverse sum (add small epsilon for numerical safety)
    // In practice, row_sum should never be zero with valid inputs
    float inv_sum = 1.0f / (row_sum + 1e-10f);

    // Calculate output offset in BHSD layout
    const int o_offset = batch_idx * num_heads_q * seqlen_q * head_dim_q +
                        head_idx * seqlen_q * head_dim_q +
                        seq_idx * head_dim_q;

    // Write output: normalize accumulator and convert fp32 -> fp16
    for (int d = 0; d < head_dim_q; d++) {
        O[o_offset + d] = __float2half(acc[d] * inv_sum);
    }
}  // End of fmha_forward_kernel