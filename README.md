# Flash Multi-Head Attention (FMHA) HIP Implementation

High-performance HIP implementation of Flash Multi-Head Attention for AMD GPUs.

## Specifications

This implementation supports the following configurations:

| Parameter | Values |
|-----------|--------|
| Batch Size | 30720 |
| Query Sequence Length | 1 |
| Key/Value Sequence Length | 2 |
| Query Head Dimension | 256, 128 |
| Key/Value Head Dimension | 256, 128 |
| Number of Query Heads | 16, 32 |
| Number of Key/Value Heads | 16, 32 |

### Tested Configurations

1. **Config 1**: batch=30720, seqlen_q=1, seqlen_kv=2, head_dim=256, num_heads=16
2. **Config 2**: batch=30720, seqlen_q=1, seqlen_kv=2, head_dim=128, num_heads=16
3. **Config 3**: batch=30720, seqlen_q=1, seqlen_kv=2, head_dim=256, num_heads=32
4. **Config 4**: batch=30720, seqlen_q=1, seqlen_kv=2, head_dim=128, num_heads=32

## Features

- **Flash Attention Algorithm**: Memory-efficient attention with online softmax
- **FP16 Precision**: Uses half-precision floating point for performance
- **Grouped Query Attention (GQA)**: Supports different number of Q and KV heads
- **Optimized for AMD GPUs**: Written in HIP for AMD hardware
- **Comprehensive Benchmarking**: Built-in performance measurement

## Architecture Support

Tested on:
- AMD MI100 (gfx908)
- AMD MI210/MI250 (gfx90a)
- AMD MI300 (gfx942)

## Prerequisites

- ROCm 5.0 or later
- HIP runtime
- CMake 3.16 or later
- C++17 compatible compiler

## Building

```bash
# Create build directory
mkdir build
cd build

# Configure with CMake
# Optional: Set GPU architecture
cmake .. -DCMAKE_CXX_COMPILER=hipcc

# For specific GPU targets, edit CMakeLists.txt and uncomment:
# - gfx90a for MI210/MI250
# - gfx908 for MI100
# - gfx942 for MI300

# Build
make -j$(nproc)
```

## Running

```bash
# Run the benchmark
./fmha_benchmark
```

The benchmark will:
1. Display GPU information
2. Run all 4 configurations
3. Report timing and performance metrics
4. Show sample output values

## Performance Metrics

The benchmark reports:
- **Average time per iteration** (milliseconds)
- **Throughput** (iterations per second)
- **Estimated TFLOPS** (approximate computational performance)

## Implementation Details

### Kernel Design

The `fmha_forward_kernel` implements:

1. **Online Softmax**: Numerically stable softmax computation without materializing the full attention matrix
2. **Fused Operations**: Combines QK^T, softmax, and attention-weighted sum in a single pass
3. **Grouped Query Attention**: Maps multiple query heads to fewer key/value heads

### Memory Layout

Tensors use the following memory layout:
- **Q**: `[batch, num_heads_q, seqlen_q, head_dim_q]`
- **K**: `[batch, num_heads_kv, seqlen_kv, head_dim_kv]`
- **V**: `[batch, num_heads_kv, seqlen_kv, head_dim_kv]`
- **O**: `[batch, num_heads_q, seqlen_q, head_dim_q]`

### Launch Configuration

- **Block size**: 256 threads
- **Grid dimensions**:
  - X: sequence dimension (seqlen_q)
  - Y: number of query heads
  - Z: batch size

## Customization

To modify the configurations, edit `fmha_host.cpp`:

```cpp
FMHAConfig(batch, num_heads_q, num_heads_kv, seqlen_q, seqlen_kv, head_dim_q, head_dim_kv)
```

## File Structure

```
fmha/
├── CMakeLists.txt      # Build configuration
├── fmha_kernel.hip     # HIP kernel implementation
├── fmha_host.cpp       # Host code and benchmark
└── README.md           # This file
```

## Optimization Opportunities

For production use, consider:

1. **Tile-based processing**: Process attention in tiles for better cache utilization
2. **Shared memory optimization**: Use shared memory for Q/K/V tiles
3. **Warp-level primitives**: Use warp shuffle for reductions
4. **Multiple kernel variants**: Specialize for different head dimensions
5. **Backward pass**: Implement gradients for training

## License

This is a template implementation for educational and development purposes.

## References

- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)
- [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691)
- [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245)
