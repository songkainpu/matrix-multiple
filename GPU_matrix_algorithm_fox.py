import typing

import pyopencl as cl
import numpy
import time
import os
import gevent
from gevent import monkey
import queue as q
import threading
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
MATRICES_FILE_FOLDER = "matrices"


class OpenCLBufferManager:
    def __init__(self, buffers: typing.Sequence[cl.Buffer]):
        self.buffers = buffers

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # for buffer in self.buffers:
        #     buffer.relase()
        pass

platform = cl.get_platforms()[0]

my_queue: q.Queue[typing.Tuple[int, numpy.ndarray, numpy.ndarray, int, int, int, int, int]] = q.Queue()
add_queue: q.Queue[typing.Tuple[int, numpy.ndarray, int, int, int, int, int]] = q.Queue()
lock = threading.Lock()

def GPU_multiple_coroutine(context: cl.Context, b_size: int) -> None:
    c_queue = cl.CommandQueue(context)
    A = numpy.ndarray((b_size, b_size), dtype=numpy.int32)
    B = numpy.ndarray((b_size, b_size), dtype=numpy.int32)
    a_buffer = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=A)
    b_buffer = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=B)
    program = cl.Program(context, kernel_code_multiply).build()
    kernel = cl.Kernel(program, 'matrix_multiply')
    with OpenCLBufferManager(buffers=[a_buffer, b_buffer]):
        while True:
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
                cl.enqueue_copy(c_queue, a_buffer, numpy.ascontiguousarray(a=matrixA[a_left:a_right, a_top:a_bottom]))
                cl.enqueue_copy(c_queue, b_buffer, numpy.ascontiguousarray(a=matrixB[b_left:b_right, b_top:b_bottom]))
                c_buffer = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, matrixA.nbytes)
                kernel.set_arg(0, a_buffer)
                kernel.set_arg(1, b_buffer)
                kernel.set_arg(2, c_buffer)
                kernel.set_arg(3, numpy.int32(block_size))
                event = cl.enqueue_nd_range_kernel(queue=c_queue, kernel=kernel, global_work_size=matrixA.shape,
                                                       local_work_size=(min(block_size, 128), 1))
                matriC = numpy.ndarray((b_size, b_size), dtype=numpy.int32)
                event.wait()
                cl.enqueue_copy(c_queue, matriC, c_buffer)
                add_queue.put((1, matriC, i, j, k, block_size, scale))


DEFAULT_SCALES: typing.List[int] = [16, 32, 64, 128, 256, 512, 1024, 2048]


if __name__ == "__main__":
    for scale in DEFAULT_SCALES:
        device = platform.get_devices()[0]
        time_list: typing.List[float] = []
        for t in range(10):
            A: numpy.ndarray = numpy.loadtxt(f"{MATRICES_FILE_FOLDER}{os.path.sep}{scale}-{t}-A.csv", delimiter=',', dtype=numpy.int32)
            B: numpy.ndarray = numpy.loadtxt(f"{MATRICES_FILE_FOLDER}{os.path.sep}{scale}-{t}-B.csv", delimiter=',', dtype=numpy.int32)
            num_gpu: int = len(platform.get_devices())
            sub_matrix_count = max(2, num_gpu//2)
            if sub_matrix_count != 1 and sub_matrix_count %2 == 1:
                sub_matrix_count -= 1
            block_size = scale // sub_matrix_count
            count = 0
            for k in range(sub_matrix_count):
                for i in range(sub_matrix_count):
                    for j in range(sub_matrix_count):
                        count += 1
                        my_queue.put(item=(0, A, B, i, j, k, block_size, scale))
            gpu_coroutine_list: typing.List[gevent.Greenlet] = []
            # print(f"num_gpu:{num_gpu}")
            for i in range(num_gpu):
                device = platform.get_devices()[i]
                context = cl.Context([device])
                coroutine = gevent.spawn(GPU_multiple_coroutine, **{
                    "context": context,
                    "b_size": block_size
                })
                gpu_coroutine_list.append(coroutine)
            result: numpy.ndarray = numpy.zeros_like(a=A)
            start_time = time.time()
            time_add = 0
            for coroutine in gpu_coroutine_list:
                coroutine.start()
            while count != 0:
                count -= 1
                add_task = add_queue.get(timeout=1000)
                add_start_time = time.time()
                _, matrix_c, i, j, k, block_size, scale = add_task
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
                result[c_left:c_right, c_top:c_bottom] += matrix_c
                time_add += (time.time() - add_start_time)
            # print(f"result:{result}")
            # print(f"numpy.dot:{numpy.dot(A,B)}")
            end_time = time.time()
            time_list.append(end_time - start_time)
            print(f"time_add:{time_add}")

        print(f"scale:{scale} fox GPU time_list:{time_list}")
        print(f"scale:{scale} fox GPU avg:{sum(time_list) / len(time_list)}")








