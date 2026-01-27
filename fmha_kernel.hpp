#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdio.h>
#include <math.h>

// Flash Attention kernel configuration
#define WARP_SIZE 64
#define BLOCK_SIZE 256

// Attention computation with online softmax
__global__ void fmha_forward_kernel(
    const half* Q,           // [batch, num_heads_q, seqlen_q, head_dim_q]
    const half* K,           // [batch, num_heads_kv, seqlen_kv, head_dim_kv]
    const half* V,           // [batch, num_heads_kv, seqlen_kv, head_dim_kv]
    half* O,                 // [batch, num_heads_q, seqlen_q, head_dim_q]
    const int batch,
    const int num_heads_q,
    const int num_heads_kv,
    const int seqlen_q,
    const int seqlen_kv,
    const int head_dim_q,
    const int head_dim_kv,
    const float softmax_scale
) {
    const int batch_idx = blockIdx.z;
    const int head_idx = blockIdx.y;
    const int seq_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (batch_idx >= batch || head_idx >= num_heads_q || seq_idx >= seqlen_q) {
        return;
    }
    
    // For GQA: map query heads to key/value heads
    const int kv_head_idx = head_idx * num_heads_kv / num_heads_q;
    
    // Shared memory for attention scores and intermediate results
    __shared__ float s_scores[BLOCK_SIZE];
    __shared__ float s_max[BLOCK_SIZE];
    __shared__ float s_sum[BLOCK_SIZE];
    
    // Initialize running max and sum for online softmax
    float row_max = -INFINITY;
    float row_sum = 0.0f;
    
    // Output accumulator
    float acc[8] = {0.0f}; // Assuming head_dim up to 256, process in chunks
    
    // Q vector for current sequence position
    const int q_offset = batch_idx * num_heads_q * seqlen_q * head_dim_q +
                         head_idx * seqlen_q * head_dim_q +
                         seq_idx * head_dim_q;
    
    // Compute attention scores: Q @ K^T
    for (int kv_idx = 0; kv_idx < seqlen_kv; kv_idx++) {
        float score = 0.0f;
        
        const int k_offset = batch_idx * num_heads_kv * seqlen_kv * head_dim_kv +
                            kv_head_idx * seqlen_kv * head_dim_kv +
                            kv_idx * head_dim_kv;
        
        // Compute dot product Q[seq_idx] · K[kv_idx]
        for (int d = 0; d < head_dim_q && d < head_dim_kv; d++) {
            float q_val = __half2float(Q[q_offset + d]);
            float k_val = __half2float(K[k_offset + d]);
            score += q_val * k_val;
        }
        
        score *= softmax_scale;
        
        // Online softmax: update running max
        float prev_max = row_max;
        row_max = fmaxf(row_max, score);
        
        // Rescale previous sum
        float scale = expf(prev_max - row_max);
        row_sum *= scale;
        
        // Add current score contribution
        float exp_score = expf(score - row_max);
        row_sum += exp_score;
        
        // Update output with V weighted by attention
        const int v_offset = batch_idx * num_heads_kv * seqlen_kv * head_dim_kv +
                            kv_head_idx * seqlen_kv * head_dim_kv +
                            kv_idx * head_dim_kv;
        
        // Accumulate V values weighted by attention
        for (int d = 0; d < min(8, head_dim_q); d++) {
            if (d < head_dim_kv) {
                float v_val = __half2float(V[v_offset + d]);
                acc[d] = acc[d] * scale + exp_score * v_val;
            }
        }
    }
    
    // Normalize by sum
    float inv_sum = 1.0f / row_sum;
    
    // Write output
    const int o_offset = batch_idx * num_heads_q * seqlen_q * head_dim_q +
                        head_idx * seqlen_q * head_dim_q +
                        seq_idx * head_dim_q;
    
    for (int d = 0; d < min(8, head_dim_q); d++) {
        O[o_offset + d] = __float2half(acc[d] * inv_sum);
    }
    
    // Handle remaining dimensions if head_dim > 8
    if (head_dim_q > 8) {
        for (int d = 8; d < head_dim_q; d++) {
            float acc_val = 0.0f;
            float local_max = -INFINITY;
            float local_sum = 0.0f;
            
            for (int kv_idx = 0; kv_idx < seqlen_kv; kv_idx++) {
                float score = 0.0f;
                
                const int k_offset = batch_idx * num_heads_kv * seqlen_kv * head_dim_kv +
                                    kv_head_idx * seqlen_kv * head_dim_kv +
                                    kv_idx * head_dim_kv;
                
                for (int dim = 0; dim < head_dim_q && dim < head_dim_kv; dim++) {
                    float q_val = __half2float(Q[q_offset + dim]);
                    float k_val = __half2float(K[k_offset + dim]);
                    score += q_val * k_val;
                }
                
                score *= softmax_scale;
                
                float prev_max = local_max;
                local_max = fmaxf(local_max, score);
                float scale = expf(prev_max - local_max);
                local_sum *= scale;
                
                float exp_score = expf(score - local_max);
                local_sum += exp_score;
                
                const int v_offset = batch_idx * num_heads_kv * seqlen_kv * head_dim_kv +
                                    kv_head_idx * seqlen_kv * head_dim_kv +
                                    kv_idx * head_dim_kv;
                
                if (d < head_dim_kv) {
                    float v_val = __half2float(V[v_offset + d]);
                    acc_val = acc_val * scale + exp_score * v_val;
                }
            }
            
            O[o_offset + d] = __float2half(acc_val / local_sum);
        }
    }
}

// Host function to launch FMHA kernel
extern "C" void launch_fmha_forward(
    const half* d_Q,
    const half* d_K,
    const half* d_V,
    half* d_O,
    const int batch,
    const int num_heads_q,
    const int num_heads_kv,
    const int seqlen_q,
    const int seqlen_kv,
    const int head_dim_q,
    const int head_dim_kv,
    hipStream_t stream
) {
    // Compute softmax scale: 1/sqrt(head_dim)
    const float softmax_scale = 1.0f / sqrtf((float)head_dim_q);
    
    // Launch configuration
    dim3 block(BLOCK_SIZE);
    dim3 grid(
        (seqlen_q + BLOCK_SIZE - 1) / BLOCK_SIZE,  // x: sequence dimension
        num_heads_q,                                 // y: heads
        batch                                        // z: batch
    );
    
    hipLaunchKernelGGL(
        fmha_forward_kernel,
        grid, block, 0, stream,
        d_Q, d_K, d_V, d_O,
        batch, num_heads_q, num_heads_kv,
        seqlen_q, seqlen_kv,
        head_dim_q, head_dim_kv,
        softmax_scale
    );
}
