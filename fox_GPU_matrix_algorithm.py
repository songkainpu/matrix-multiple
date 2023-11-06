import pyopencl as cl
import numpy
import time
import typing

DEFAULT_SCALES: typing.List[int] = [16, 32, 64, 128, 256, 512, 1024, 2048]
platform = cl.get_platforms()[0]
print(f"platform.get_devices():{len(platform.get_devices())}")
device = platform.get_devices()[0]
context = cl.Context([device])
queue = cl.CommandQueue(context)

N = 2048

a = numpy.random.randint(0, 255, size=(N, N), dtype=numpy.int32)
b = numpy.random.randint(0, 255, size=(N, N), dtype=numpy.int32)

a_buffer = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=a)
b_buffer = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=b)
c_buffer = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, a.nbytes)

kernel_code = """
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

program = cl.Program(context, kernel_code).build()
start = time.time()

program.matrix_multiply(queue, a.shape, None, a_buffer, b_buffer, c_buffer, numpy.int32(N))

c = numpy.empty((N, N), dtype=numpy.int32)
cl.enqueue_copy(queue, c, c_buffer)
end = time.time()

print(c)
print(end - start)
