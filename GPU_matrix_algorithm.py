import typing

import pyopencl as cl
import numpy
import time
import os
import gevent
from gevent import monkey
import queue as q
monkey.patch_all()

kernel_code_multiply = """
__kernel void matrix_multiply(__global int* a, __global int* b, __global int* c, int N) {
    int row = get_global_id(0);
    int col = get_global_id(1);

    if (row < N && col < N) {
        int sum = 0;
        for (int i = 0; i < N; i++) {
            sum += a[row * N + i] * b[i * N + col];
        }
        c[row * N + col] = sum;
    }
}
"""

kernel_code_add = """
__kernel void matrix_add(__global const float* a, __global const float* b, __global float* c) {
    int gid_x = get_global_id(0);
    int gid_y = get_global_id(1);
    int index = gid_x * %d + gid_y;
    c[index] = a[index] + b[index];
}
"""
MATRICES_FILE_FOLDER = "matrices"


class OpenCLBufferManager:
    def __init__(self, buffers: typing.Sequence[cl.Buffer]):
        self.buffers = buffers

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for buffer in self.buffers:
            buffer.relase()

platform = cl.get_platforms()[0]
#
# print(f"platform.get_devices():{len(platform.get_devices())}")
# device = platform.get_devices()[0]
# context1 = cl.Context([device])
# queue = cl.CommandQueue(context1)
#
# N = 2048
#
# a = numpy.random.randint(0, 255, size=(N, N), dtype=numpy.int32)
# b = numpy.random.randint(0, 255, size=(N, N), dtype=numpy.int32)
#
# a_buffer = cl.Buffer(context1, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=a)
# b_buffer = cl.Buffer(context1, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=b)
# c_buffer = cl.Buffer(context1, cl.mem_flags.WRITE_ONLY, a.nbytes)
#
#
# program = cl.Program(context1, kernel_code_multiply).build()
# start = time.time()
#
# program.matrix_multiply(queue, a.shape, None, a_buffer, b_buffer, c_buffer, numpy.int32(N))
#
# c = numpy.empty((N, N), dtype=numpy.int32)
# cl.enqueue_copy(queue, c, c_buffer)
# end = time.time()
#
# print(c)
# print(end - start)

# op(add or multiple), matrixA, matrixB, i, j, k ,block_size, scale
my_queue: q.Queue[typing.Tuple[int, numpy.ndarray, numpy.ndarray, int, int, int, int, int]] = q.Queue()
add_queue: q.Queue[typing.Tuple[int, numpy.ndarray, int, int, int, int, int]] = q.Queue()


def multiple_fox_gpu(context: cl.Context, task:typing.Tuple) -> None:
    _, matrixA, matrixB, i, j, k, block_size, scale = task
    a_left = i * block_size
    a_right = (i + 1) * block_size
    a_top = k * block_size
    a_bottom = (k + 1) * block_size
    b_left = a_top
    b_right = a_bottom
    b_top = j * block_size
    b_bottom = (j + 1) * block_size
    c_left = a_left
    c_right = a_right
    c_top = b_top
    c_bottom = b_bottom
    A = numpy.ndarray((a_right - a_left, a_bottom - a_top ), dtype=numpy.int32)
    B = numpy.ndarray((b_right - b_left, b_bottom - b_top), dtype=numpy.int32)


def GPU_multiple_coroutine(context: cl.Context, b_size: int) -> None:
    c_queue = cl.CommandQueue(context)
    A = numpy.ndarray((b_size, b_size), dtype=numpy.int32)
    B = numpy.ndarray((b_size, b_size), dtype=numpy.int32)
    a_buffer = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=A)
    b_buffer = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=B)
    c_buffer = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, A.nbytes)
    with OpenCLBufferManager(buffers=[a_buffer, b_buffer, c_buffer]):
        if my_queue.empty():
            return
        task = my_queue.get()
        if task:
            # is multiple
            _, matrixA, matrixB, i, j, k, block_size, scale = task
            a_left = i * block_size
            a_right = (i + 1) * block_size
            a_top = k * block_size
            a_bottom = (k + 1) * block_size
            b_left = a_top
            b_right = a_bottom
            b_top = j * block_size
            b_bottom = (j + 1) * block_size
            c_left = a_left
            c_right = a_right
            c_top = b_top
            c_bottom = b_bottom
            cl.enqueue_copy(c_queue, a_buffer, matrixA[a_left:a_right, a_top:a_bottom])
            cl.enqueue_copy(c_queue, b_buffer, matrixB[a_left:a_right, a_top:a_bottom])
            program = cl.Program(context, kernel_code_multiply).build()
            program.matrix_multiply(c_queue, A.shape, None, a_buffer, b_buffer, c_buffer, b_size)
            matriC = numpy.ndarray((b_size, b_size), dtype=numpy.int32)
            cl.enqueue_copy(c_queue, matriC, c_buffer)
            add_queue.put((1, matriC, i, j, k, block_size, scale))



if __name__ == "__main__":

    global DEFAULT_SCALES
    for scale in DEFAULT_SCALES:
        for t in range(10):
            A = numpy.loadtxt(f"{MATRICES_FILE_FOLDER}{os.path.sep}{scale}-{t}-A.csv", delimiter=',')
            B = numpy.loadtxt(f"{MATRICES_FILE_FOLDER}{os.path.sep}{scale}-{t}-B.csv", delimiter=',')
            num_gpu: int = len(platform.get_devices())
            for i in range(num_gpu):
                device = platform.get_devices()[0]

