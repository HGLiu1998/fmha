# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a high-performance Flash Multi-Head Attention (FMHA) implementation for AMD GPUs using HIP (Heterogeneous-Compute Interface for Portability). The implementation focuses on memory-efficient attention computation using the Flash Attention algorithm with online softmax.

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

**fmha_kernel.hpp** - HIP kernel implementation
- `fmha_forward_kernel`: GPU kernel implementing Flash Attention forward pass
- `launch_fmha_forward`: Host-side launcher function
- Uses online softmax for numerical stability and memory efficiency
- Supports Grouped Query Attention (GQA) where num_heads_q can differ from num_heads_kv

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

**Online Softmax Algorithm**:
The kernel implements numerically stable softmax without materializing the full attention matrix by maintaining running max and sum values. This is the core of Flash Attention's memory efficiency.

**Thread Organization** (fmha_kernel.hpp:25-31):
- Grid dimension X: sequence dimension (seqlen_q), blocks of 256 threads
- Grid dimension Y: number of query heads
- Grid dimension Z: batch size
- Each thread processes one query position

**Grouped Query Attention** (fmha_kernel.hpp:34):
Maps query heads to key/value heads using: `kv_head_idx = head_idx * num_heads_kv / num_heads_q`

**Dimension Handling** (fmha_kernel.hpp:46, 88-93, 109-150):
First 8 dimensions are processed in the main loop with accumulator array `acc[8]`. Dimensions beyond 8 are handled separately in a second loop. This is a simplification - production code would process all dimensions more efficiently.

### Supported Configurations

The implementation is tested with these specific configs (see fmha_host.cpp:189-201):
- Batch size: 30720
- Sequence lengths: seqlen_q=1, seqlen_kv=2
- Head dimensions: 128 or 256
- Number of heads: 16 or 32 (Q and KV heads equal in test configs)

### Performance Notes

The current implementation is a template/educational version. The README.md lists several optimization opportunities for production use:
- Tile-based processing for better cache utilization
- Shared memory optimization for Q/K/V tiles
- Warp-level primitives for reductions
- Multiple kernel variants specialized for different head dimensions
- Backward pass implementation for training

### Error Handling

The code uses `HIP_CHECK` macro (fmha_host.cpp:25-33) for HIP API error checking. All HIP calls should be wrapped with this macro.

## Key Implementation Details

**Softmax Scale** (fmha_kernel.hpp:169): Computed as `1/sqrt(head_dim_q)` for numerical stability

**FP16 Precision**: All tensors use half-precision (`half` type), with conversions to float32 for computation using `__half2float` and `__float2half`

**Attention Computation** (fmha_kernel.hpp:54-94): Fuses QK^T matmul, softmax, and attention-weighted V sum in a single pass through the KV sequence
