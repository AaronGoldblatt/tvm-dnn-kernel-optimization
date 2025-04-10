import os
import tvm
from tvm import te


def make_conv1d_cpu_scheduler_baseline(M, N):
    A = te.placeholder((M,), name="A")
    W = te.placeholder((N,), name="W")

    k = te.reduce_axis((0, M + N - 1), "k")
    B = te.compute(
        (M + N - 1,),
        lambda n: te.sum(tvm.tir.if_then_else(
            tvm.tir.any(k < 0, k >= M, n - k < 0, n - k >= N),
            tvm.tir.const(0.0, "float32"),
            A[k] * W[n - k]), axis=k),
        name="B",
    )

    s = te.create_schedule(B.op)

    return s, A, W, B


def make_conv1d_cpu_scheduler(M, N):
    A = te.placeholder((M,), name="A")
    W = te.placeholder((N,), name="W")
    
    pad = N - 1
    
    A_padded = te.compute(
        (M + 2 * pad,),
        lambda i: tvm.tir.if_then_else(
            tvm.tir.any(i < pad, i >= M + pad),
            tvm.tir.const(0.0, "float32"),
            A[i - pad]
        ),
        name="A_padded"
    )

    k = te.reduce_axis((0, N), "k")
    B = te.compute(
        (M + N - 1,),
        lambda n: te.sum(
            A_padded[n + pad - k] * W[k],
            axis=k
        ),
        name="B",
    )

    s = te.create_schedule(B.op)
    
    A_local = s.cache_read(A_padded, "local", [B])
    W_local = s.cache_read(W, "local", [B])
    
    n = B.op.axis[0]
    
    vector_size = 16
    n, n_vector_elements = s[B].split(n, factor=vector_size)
    
    unroll_size = 16
    k_outer, k_inner = s[B].split(k, factor=unroll_size)
    
    s[B].reorder(
        n,
        k_outer,
        n_vector_elements,
        k_inner
    )
    
    s[A_local].compute_at(s[B], k_outer)
    s[W_local].compute_at(s[B], k_outer)
            
    s[B].vectorize(n_vector_elements)
    
    s[B].unroll(k_inner)
    
    return s, A, W, B


def make_conv1d_gpu_scheduler_baseline(M, N):
    A = te.placeholder((M,), name="A")
    W = te.placeholder((N,), name="W")

    k = te.reduce_axis((0, M + N - 1), "k")
    B = te.compute(
        (M + N - 1,),
        lambda n: te.sum(tvm.tir.if_then_else(
            tvm.tir.any(k < 0, k >= M, n - k < 0, n - k >= N),
            tvm.tir.const(0.0, "float32"),
            A[k] * W[n - k]), axis=k),
        name="B",
    )

    s = te.create_schedule(B.op)
    
    n = B.op.axis[0]
    
    block_size = 256
    n_block, n_thread = s[B].split(n, factor=block_size)
    
    s[B].bind(n_block, te.thread_axis("blockIdx.x"))
    s[B].bind(n_thread, te.thread_axis("threadIdx.x"))

    return s, A, W, B


def make_conv1d_gpu_scheduler(M, N):
    A = te.placeholder((M,), name="A")
    W = te.placeholder((N,), name="W")
    
    pad = N - 1
    
    A_padded = te.compute(
        (M + 2 * pad,),
        lambda i: tvm.tir.if_then_else(
            tvm.tir.any(i < pad, i >= M + pad),
            tvm.tir.const(0.0, "float32"),
            A[i - pad]
        ),
        name="A_padded"
    )

    k = te.reduce_axis((0, N), "k")
    B = te.compute(
        (M + N - 1,),
        lambda n: te.sum(
            A_padded[n + pad - k] * W[k],
            axis=k
        ),
        name="B",
    )

    s = te.create_schedule(B.op)
    
    n = B.op.axis[0]
    
    block_size = 64
    n_block, n_thread = s[B].split(n, factor=block_size)
    
    unroll_size = 8
    k_outer, k_unroll = s[B].split(k, factor=unroll_size)
    
    s[B].bind(n_block, te.thread_axis("blockIdx.x"))
    s[B].bind(n_thread, te.thread_axis("threadIdx.x"))
    
    pad_axis = s[A_padded].op.axis[0]
    pad_outer, pad_inner = s[A_padded].split(pad_axis, factor=block_size)
    
    s[A_padded].bind(pad_outer, te.thread_axis("blockIdx.y"))
    s[A_padded].bind(pad_inner, te.thread_axis("threadIdx.y"))
    
    s[B].reorder(
        n_block,
        n_thread,
        k_outer,
        k_unroll
    )
    
    s[B].unroll(k_unroll)
            
    return s, A, W, B


def make_gemm_gpu_scheduler_baseline(M, K, N):
    A = te.placeholder((M, K), name="A")
    B = te.placeholder((K, N), name="B")

    # TVM Matrix Multiplication using TE
    k = te.reduce_axis((0, K), "k")
    A = te.placeholder((M, K), name="A")
    B = te.placeholder((K, N), name="B")
    C = te.compute((M, N), lambda x, y: te.sum(A[x, k] * B[k, y], axis=k), name="C")
    # Default schedule
    s = te.create_schedule(C.op)

    # the i-th block is indexed by blockIdx.x.
    # the number of threads in each block is blockDim.x
    # and the i-th thread within a block is indexed by threadIdx.x
    # overall index of a thread can be calculated as
    # 𝑖=blockIdx.x×blockDim.x+threadIdx.x
    block_x = te.thread_axis("blockIdx.y")
    block_y = te.thread_axis("blockIdx.x")

    x, y = s[C].op.axis
    (k,) = s[C].op.reduce_axis
    s[C].bind(y, block_y)
    s[C].bind(x, block_x)

    return s, A, B, C


def make_gemm_gpu_scheduler(M, K, N):
    A = te.placeholder((M, K), name="A")
    B = te.placeholder((K, N), name="B")

    k = te.reduce_axis((0, K), "k")
    C = te.compute((M, N), lambda x, y: te.sum(A[x, k] * B[k, y], axis=k), name="C")

    s = te.create_schedule(C.op)

    block_size = 32
    thread_size = 32

    row, col = s[C].op.axis

    reduction_outer, reduction_inner = s[C].split(k, factor=block_size)

    block_row, thread_row = s[C].split(row, factor=block_size)
    block_col, thread_col = s[C].split(col, factor=block_size)

    s[C].bind(block_row, te.thread_axis("blockIdx.y"))
    s[C].bind(block_col, te.thread_axis("blockIdx.x"))

    A_local = s.cache_read(A, "local", [C])
    B_local = s.cache_read(B, "local", [C])

    s[A_local].compute_at(s[C], reduction_outer)
    s[B_local].compute_at(s[C], reduction_outer)

    s[C].reorder(
        reduction_outer,
        thread_row,
        thread_col,
        reduction_inner
    )
   
    s[C].bind(thread_row, te.thread_axis((0, thread_size), "threadIdx.y"))
    s[C].bind(thread_col, te.thread_axis((0, thread_size), "threadIdx.x"))

    # s[A_local].storage_align(s[A_local].op.axis[0], thread_size, 0)
    # s[B_local].storage_align(s[B_local].op.axis[0], thread_size, 0)

    s[C].unroll(reduction_inner)

    return s, A, B, C


def make_dwsp_conv2d_gpu_scheduler_baseline(B, C, H, W, K):
    assert K % 2 == 1
    inp = te.placeholder((B, C, H, W), name="A")
    ker = te.placeholder((C, 1, K, K), name="W")

    kernel_height = te.reduce_axis((0, K), name="kernel_height")
    kernel_width = te.reduce_axis((0, K), name="kernel_width")

    out = te.compute(
        (B, C, H, W),
        lambda batch, channel, height, width: te.sum(
            tvm.tir.if_then_else(
                tvm.tir.any(
                    height + kernel_height - (K//2) < 0, height + kernel_height - (K//2) >= H,
                    width + kernel_width - (K//2) < 0, width + kernel_width - (K//2) >= W
                ),
                tvm.tir.const(0.0, "float32"),
                inp[batch, channel, height + kernel_height - (K//2), width + kernel_width - (K//2)] * ker[channel, 0, kernel_height, kernel_width]
            ),
            axis=[kernel_height, kernel_width]
        ),
        name="output"
    )

    s = te.create_schedule(out.op)

    block_height = te.thread_axis("blockIdx.y")
    block_width = te.thread_axis("blockIdx.x")

    batch_axis, channel_axis, height_axis, width_axis = out.op.axis
    s[out].bind(height_axis, block_height)
    s[out].bind(width_axis, block_width)

    return s, inp, ker, out


def make_dwsp_conv2d_gpu_scheduler(B, C, H, W, K):
    assert K % 2 == 1
    inp = te.placeholder((B, C, H, W), name="inp")
    ker = te.placeholder((C, 1, K, K), name="ker")

    height_pad = (K - 1) // 2
    width_pad = height_pad

    inp_padded = te.compute(
        (B, C, H + K - 1, W + K - 1),
        lambda batch, channel, height, width: tvm.tir.if_then_else(
            tvm.tir.any(height < height_pad, height >= H + height_pad, width < width_pad, width >= W + width_pad),
            tvm.tir.const(0.0, "float32"),
            inp[batch, channel, height - height_pad, width - width_pad]
        ),
        name="inp_padded"
    )

    kernel_height = te.reduce_axis((0, K), name='kernel_height')
    kernel_width = te.reduce_axis((0, K), name='kernel_width')

    out = te.compute(
        (B, C, H, W),
        lambda batch, channel, height, width: te.sum(
            inp_padded[batch, channel, height + kernel_height, width + kernel_width] * ker[channel, 0, kernel_height, kernel_width],
            axis=[kernel_height, kernel_width]
        ),
        name="out"
    )

    s = te.create_schedule(out.op)
    
    tile_height = 4
    tile_width = 4

    batch, channel, height, width = s[out].op.axis
    height_outer, height_inner = s[out].split(height, factor=tile_height)
    width_outer, width_inner = s[out].split(width, factor=tile_width)

    s[out].reorder(
        batch,
        channel,
        height_outer,
        width_outer,
        height_inner,
        width_inner
    )
    
    s[out].bind(height_outer, te.thread_axis("blockIdx.x"))
    s[out].bind(width_outer, te.thread_axis("blockIdx.y"))
    
    s[out].bind(height_inner, te.thread_axis("threadIdx.x"))
    s[out].bind(width_inner, te.thread_axis("threadIdx.y"))
    
    batch_padded, channel_padded, height_padded, width_padded = s[inp_padded].op.axis
    height_padded_outer, height_padded_inner = s[inp_padded].split(height_padded, factor=tile_height)
    width_padded_outer, width_padded_inner = s[inp_padded].split(width_padded, factor=tile_width)

    s[inp_padded].reorder(
        batch_padded,
        channel_padded,
        height_padded_outer,
        width_padded_outer,
        height_padded_inner,
        width_padded_inner
    )
    
    s[inp_padded].bind(height_padded_outer, te.thread_axis("blockIdx.z"))
    s[inp_padded].bind(width_padded_outer, te.thread_axis("blockIdx.y"))
    
    s[inp_padded].bind(height_padded_inner, te.thread_axis("threadIdx.z"))
    s[inp_padded].bind(width_padded_inner, te.thread_axis("threadIdx.y"))

    return s, inp, ker, out