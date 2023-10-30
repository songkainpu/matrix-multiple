import pyopencl as cl
import numpy
import time

platform = cl.get_platforms()[0]
device = platform.get_devices()[0]
context = cl.Context([device])
queue = cl.CommandQueue(context)

N = 2048

a = numpy.random.rand(N, N).astype(numpy.float32)
b = numpy.random.rand(N, N).astype(numpy.float32)

a_buffer = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=a)
b_buffer = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=b)
c_buffer = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, a.nbytes)

kernel_code = """
__kernel void matrix_multiply(__global float* a, __global float* b, __global float* c, int N) {
    int row = get_global_id(0);
    int col = get_global_id(1);

    if (row < N && col < N) {
        float sum = 0.0;
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

c = numpy.empty((N, N), dtype=numpy.float32)
cl.enqueue_copy(queue, c, c_buffer)
end = time.time()

print(c)
print(end - start)
