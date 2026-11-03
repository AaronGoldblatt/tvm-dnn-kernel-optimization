# DNN Acceleration — TVM CPU/GPU Kernel Optimization

Optimization of deep-learning kernels using [Apache TVM](https://tvm.apache.org/), separating computation definitions from hardware execution schedules.

## Kernels and techniques

- 1D convolution on CPU and GPU.
- GPU matrix multiplication (GEMM).
- GPU depthwise separable convolution.
- Vectorization, loop reordering, unrolling, caching, and GPU thread scheduling.

## Recorded benchmarks

| Kernel | Speedup over unoptimized baseline |
| --- | ---: |
| CPU convolution | 141.69× |
| GPU convolution | 238.37× |
| GPU GEMM | 13.29× |
| GPU depthwise convolution | 1.23× |

The recorded GPU 1D convolution runtime decreased from 4.233 ms to 0.0178 ms. These are workload-specific comparisons against unoptimized TVM baselines, not end-to-end model speedups or universal comparisons against PyTorch. Results are historical measurements.

## Explore the implementation

The notebooks, implementations, and functionality tests document the scheduling experiments. FPGA-related files are experimental and are not part of the reported CPU/GPU results.

Useful references:
- [TVM schedule primitives](https://tvm.apache.org/docs/v0.10.0/how_to/work_with_schedules/schedule_primitives.html)
- [GPU convolution optimization](https://tvm.apache.org/docs/v0.10.0/how_to/optimize_operators/opt_conv_cuda.html)

## Attribution

Built on the ML-HW-SYS starter code and benchmark framework. Existing source credits and license notices are retained.
