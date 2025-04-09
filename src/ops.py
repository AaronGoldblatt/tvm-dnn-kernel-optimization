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
    
    num_cores = 4
    n_parallel, n = s[B].split(B.op.axis[0], nparts=num_cores)
    
    l2_cache_tile = 2048
    n_l2_blocks, n = s[B].split(n, factor=l2_cache_tile)
    
    l1_cache_tile = 256
    n_l1_blocks, n = s[B].split(n, factor=l1_cache_tile)
    
    avx_512_elems = 16
    n_vector_blocks, n_vector_elements = s[B].split(n, factor=avx_512_elems)
    
    k_cache_blocks, k_within_block = s[B].split(k, factor=32)
    
    s[A_local].compute_at(s[B], k_cache_blocks)
    s[W_local].compute_at(s[B], k_cache_blocks)
    
    s[B].reorder(
        n_parallel,
        n_l2_blocks,
        n_l1_blocks,
        k_cache_blocks,
        n_vector_blocks,
        k_within_block,
        n_vector_elements
    )
    
    s[B].parallel(n_parallel)
    
    s[B].vectorize(n_vector_elements)
    
    s[B].unroll(k_within_block)
    
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
        
    A_shared = s.cache_read(A_padded, "shared", [B])
    W_shared = s.cache_read(W, "shared", [B])
    
    A_local = s.cache_read(A_shared, "local", [B])
    W_local = s.cache_read(W_shared, "local", [B])
    
    n = B.op.axis[0]
    
    block_size = 256
    n_block, n_thread = s[B].split(n, factor=block_size)
    
    tile_size = 32
    k_cache_blocks, k_within_block = s[B].split(k, factor=tile_size)
    
    vector_size = 4
    n_thread, n_vector = s[B].split(n_thread, factor=vector_size)
    
    pad_axis = s[A_padded].op.axis[0]
    pad_outer, pad_inner = s[A_padded].split(pad_axis, factor=256)
    s[A_padded].bind(pad_outer, te.thread_axis("blockIdx.x"))
    s[A_padded].bind(pad_inner, te.thread_axis("threadIdx.x"))
    
    s[B].reorder(
        n_block,
        k_cache_blocks,
        n_thread,
        k_within_block,
        n_vector
    )
    
    s[B].bind(n_block, te.thread_axis("blockIdx.x"))
    s[B].bind(n_thread, te.thread_axis("threadIdx.x"))
    
    s[A_shared].compute_at(s[B], k_cache_blocks)
    s[W_shared].compute_at(s[B], k_cache_blocks)
    
    s[A_local].compute_at(s[B], k_within_block)
    s[W_local].compute_at(s[B], k_within_block)
    
    s[A_shared].storage_align(s[B].op.axis[0], 32, 0)
    s[W_shared].storage_align(s[B].op.axis[0], 32, 0)
    
    s[B].unroll(k_within_block)
    
    s[B].vectorize(n_vector)
    
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
    A = te.placeholder((M, K), name="A")
    B = te.placeholder((K, N), name="B")
    C = te.compute((M, N), lambda x, y: te.sum(A[x, k] * B[k, y], axis=k), name="C")

    s = te.create_schedule(C.op)

    block_size = 32
    thread_size = 32

    row, col = s[C].op.axis

    reduction_outer, reduction_inner = s[C].split(k, nparts=(K + block_size - 1) // block_size)

    block_row, thread_row, block_col, thread_col = s[C].tile(
        row,
        col,
        x_nparts=(M + block_size - 1) // block_size,
        y_nparts=(N + block_size - 1) // block_size
    )

    s[C].reorder(
        block_row,
        block_col,
        reduction_outer,
        thread_row,
        thread_col,
        reduction_inner
    )

    A_shared = s.cache_read(A, "shared", [C])
    B_shared = s.cache_read(B, "shared", [C])
    A_local = s.cache_read(A_shared, "local", [C])
    B_local = s.cache_read(B_shared, "local", [C])

    s[A_shared].compute_at(s[C], reduction_outer)
    s[B_shared].compute_at(s[C], reduction_outer)
    s[A_local].compute_at(s[C], reduction_inner)
    s[B_local].compute_at(s[C], reduction_inner)

    s[C].bind(block_row, te.thread_axis("blockIdx.y"))
    s[C].bind(block_col, te.thread_axis("blockIdx.x"))
    s[C].bind(thread_row, te.thread_axis((0, thread_size), "threadIdx.y"))
    s[C].bind(thread_col, te.thread_axis((0, thread_size), "threadIdx.x"))

    s[A_shared].storage_align(s[A_shared].op.axis[0], thread_size, 0)
    s[B_shared].storage_align(s[B_shared].op.axis[0], thread_size, 0)

    s[C].unroll(reduction_inner)

    s[C].vectorize(thread_col)

    return s, A, B, C


def make_dwsp_conv2d_gpu_scheduler_baseline(B, C, H, W, K):
    assert K % 2 == 1
    inp = te.placeholder((B, C, H, W), name="A")
    ker = te.placeholder((C, 1, K, K), name="W")

    # TODO: fill-in start
    out = None
    s = None
    # TODO: fill-in end

    return s, inp, ker, out


def make_dwsp_conv2d_gpu_scheduler(B, C, H, W, K):
    assert K % 2 == 1
    inp = te.placeholder((B, C, H, W), name="A")
    ker = te.placeholder((C, 1, K, K), name="W")

    # TODO: fill-in start
    out = None
    s = None
    # TODO: fill-in end

    return s, inp, ker, out