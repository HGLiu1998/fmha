# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a high-performance Flash Multi-Head Attention (FMHA) implementation for AMD GPUs using HIP with AMD's MFMA (Matrix Fused Multiply-Add) instructions. The implementation uses `v_mfma_f32_16x16x16_bf16` for efficient matrix operations and focuses on memory-efficient attention computation using the Flash Attention algorithm with online softmax.

**Key Features**:
- AMD MFMA 16x16x16 instructions for matrix operations
- BF16 precision inputs with FP32 accumulation and FP16 output
- One head per block kernel design for optimal parallelization
- Vectorized 128-bit memory loads (bf16x8) with perfect coalescing
- Shared memory tiling with bank conflict avoidance
- Support for variable sequence lengths (seqlen_kv: 1-16)

**Target Hardware**: AMD MI100 (gfx908), MI210/MI250 (gfx90a), MI300 (gfx942)

## Build Instructions

The project uses CMake but the CMakeLists.txt file needs to be created. Based on the README, the standard build process is:

```bash
# Create build directory
mkdir build
cd build

# Configure with CMake (requires CMakeLists.txt to be created)
cmake .. -DCMAKE_CXX_COMPILER=hipcc

# Build
make -j$(nproc)

# Run benchmark
./fmha_benchmark
```

Note: CMakeLists.txt is not yet present in the repository and needs to be created with:
- Target architectures: gfx90a (MI210/MI250), gfx908 (MI100), gfx942 (MI300)
- Compile fmha_host.cpp with hipcc
- Link against HIP runtime

## Code Architecture

### Core Components

**fmha_kernel.hpp** - HIP kernel implementation with MFMA
- `fmha_forward_kernel`: GPU kernel implementing Flash Attention forward pass using MFMA 16x16x16
- Each block handles exactly ONE head (grid: `(1, num_heads, batch)`)
- 256 threads per block (4 warps), all cooperate for data loading
- Uses `v_mfma_f32_16x16x16_bf16` instruction (planned, currently uses placeholder dot product)
- Online softmax for numerical stability and memory efficiency
- Supports Grouped Query Attention (GQA) where num_heads_q can differ from num_heads_kv
- Variable sequence length support with proper masking (seqlen_kv: 1-16)

**fmha_host.cpp** - Host code and benchmarking
- `FMHAConfig`: Configuration class for attention parameters
- `run_fmha_benchmark`: Benchmark runner with timing and performance metrics
- Main function sets up 4 test configurations from the specifications

### Memory Layout

All tensors use BHSD (Batch, Heads, Sequence, Dimension) layout:
- Q: `[batch, num_heads_q, seqlen_q, head_dim_q]`
- K: `[batch, num_heads_kv, seqlen_kv, head_dim_kv]`
- V: `[batch, num_heads_kv, seqlen_kv, head_dim_kv]`
- O: `[batch, num_heads_q, seqlen_q, head_dim_q]`

### Kernel Design Details

**Grid and Block Configuration**:
- Grid dimensions: `(1, num_heads, batch)`
  - X=1: Not used (reserved for future extensions)
  - Y=num_heads: Each block handles ONE head
  - Z=batch: Batch parallelism
- Block size: 256 threads (4 warps of 64 threads each)

**Kernel Execution Flow**:

1. **Phase 1: Load Q Tile** (lines ~90-115)
   - Load Q[1, head_dim] into shared memory
   - All 256 threads cooperate using vectorized bf16x8 loads (128-bit)
   - Access pattern: `vec_idx = v * 256 + tid` ensures perfect coalescing
   
2. **Phase 2: Compute Q @ K^T** (lines ~125-193)
   - Tile head_dim into chunks of 16 (MFMA K-dimension)
   - For each dimension tile:
     - Load K[seqlen_kv, 16] into shared memory (all threads cooperate)
     - Compute partial dot products (currently placeholder, MFMA planned)
     - Accumulate scores[seqlen_kv] across dimension tiles
   - Result: attention scores for all key tokens
   
3. **Phase 3: Apply Softmax** (lines ~195-240)
   - Scale scores by `1/sqrt(head_dim)`
   - Mask invalid positions (k >= seqlen_kv) with -inf
   - Numerically stable online softmax:
     - Compute max across all valid scores
     - Exp(score - max) for numerical stability
     - Sum all exp values
     - Normalize: softmax[k] = exp(score[k] - max) / sum
   
4. **Phase 4: Compute Attention @ V** (lines ~242-280)
   - Tile head_dim into chunks of 16
   - For each dimension tile:
     - Load V[seqlen_kv, 16] into shared memory (reuses K buffer via union)
     - Compute weighted sum: output[d] = Σ(softmax[k] × V[k,d])
     - Accumulate across dimension tiles
   - Result: output[1, head_dim]
   
5. **Phase 5: Write Output** (lines ~282-293)
   - Convert FP32 accumulator to FP16
   - Write output[head_dim] to global memory

**Memory Coalescing Strategy**:
The key insight for perfect coalescing is the access pattern:
```cpp
for (int v = 0; v < vecs_per_thread; ++v) {
    const int vec_idx = v * 256 + tid;  // Consecutive threads → consecutive memory
    // Load bf16x8 (128-bit aligned) from global memory
}
```
This ensures threads in a warp access consecutive 128-bit chunks, maximizing bandwidth.

**Shared Memory Layout**:
```cpp
__shared__ bhalf_t Qs[1 * (max_head_dim + 4)];        // 520 bytes (head_dim=256)
union {
    bhalf_t Ks[16 * (max_head_dim + 4)];              // 8,320 bytes
    bhalf_t Vs[16 * (max_head_dim + 4)];              // Reuses same space
} KV_shared;
__shared__ float scores[16];                           // 64 bytes
__shared__ float softmax_out[16];                      // 64 bytes
Total: ~9KB per head
```

Features:
- 128-byte alignment for optimal MI300 transactions
- 4-element padding to avoid bank conflicts
- Union for K/V buffers saves 50% shared memory

**MFMA 16x16x16 Instruction** (planned):
- Instruction: `v_mfma_f32_16x16x16_bf16`
- Computes: C[16×16] += A[16×16] × B[16×16]
- Input: BF16, Output: FP32 accumulator
- One wave (64 threads) computes entire 16×16 tile
- Currently using placeholder dot product, MFMA intrinsic to be added

### Supported Configurations

The implementation includes 18 test configurations (see fmha_host.cpp:262-279):
- **Batch size**: 30720 (fixed)
- **Sequence lengths**: seqlen_q=1 (fixed), seqlen_kv=1,2,4,8,16 (variable)
- **Head dimensions**: 128, 256
- **Number of heads**: 16, 32 (Q and KV heads equal in test configs)

Examples:
- Config 0: batch=30720, heads=16, seqlen_q=1, seqlen_kv=1, head_dim=128
- Config 4: batch=30720, heads=16, seqlen_q=1, seqlen_kv=16, head_dim=128
- Config 9: batch=30720, heads=16, seqlen_q=1, seqlen_kv=16, head_dim=256
- Config 17: batch=30720, heads=32, seqlen_q=1, seqlen_kv=16, head_dim=256

Run specific config: `./fmha_benchmark 0` (runs config 0)
Run all configs: `./fmha_benchmark` (runs all 18)

### Performance Notes

**Implemented Optimizations**:
✅ Vectorized 128-bit loads (bf16x8) for maximum memory bandwidth  
✅ Perfect memory coalescing with consecutive thread access pattern  
✅ Shared memory tiling with 128-byte alignment and bank conflict avoidance  
✅ Union-based K/V buffer sharing reduces shared memory by 50%  
✅ Online softmax with numerical stability  
✅ Masking for variable sequence lengths  

**Planned Optimizations**:
- Replace placeholder dot products with actual MFMA intrinsics
- Double buffering to overlap computation and data loading
- Use all 4 warps for parallel computation (currently only warp 0)
- Warp shuffle primitives for faster reductions
- Register blocking for MFMA matrices
- Support for longer sequences (seqlen_kv > 16) with iterative tiling
- Backward pass implementation for training

### Error Handling

The code uses `HIP_CHECK` macro (fmha_host.cpp:25-33) for HIP API error checking. All HIP calls should be wrapped with this macro.

## Key Implementation Details

**Mixed Precision Strategy**:
- Input: BF16 (`__bf16`) for Q, K, V
- Computation: FP32 accumulation for numerical accuracy
- Output: FP16 (`_Float16`) for O
- Conversions: `__bfloat162float()` and `__float2half()`

**Softmax Scale**: Computed as `1/sqrt(head_dim_q)` for numerical stability

**Vector Types**:
```cpp
using bf16x8 = __bf16 __attribute__((ext_vector_type(8)));  // 128-bit loads
```

**Important Macros**:
```cpp
#define WARP_SIZE 64           // AMD warp size
#define BLOCK_SIZE 256         // 4 warps per block
#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))  // Integer ceiling division
```

**Masking Logic**: 
Invalid positions (k >= seqlen_kv) are masked with -inf before softmax, ensuring they contribute 0 after exp() and don't affect the output.

**Attention Computation**: 
Fuses QK^T matmul, softmax, and attention-weighted V sum in a single pass with dimension tiling to fit in shared memory and utilize MFMA efficiently.
