import pyopencl as cl
import numpy
import time
import typing
import os

DEFAULT_SCALES: typing.List[int] = [16, 32, 64, 128, 256, 512, 1024, 2048]
# DEFAULT_SCALES: typing.List[int] = [2048]
platform = cl.get_platforms()[0]
print(f"platform.get_devices():{len(platform.get_devices())}")
device = platform.get_devices()[0]
context = cl.Context([device])
queue = cl.CommandQueue(context)

MATRICES_FILE_FOLDER = "matrices"

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
if __name__ == "__main__":
    # global DEFAULT_SCALES
    total_gpu_num = device.get_info(cl.device_info.MAX_COMPUTE_UNITS)
    print(f"total_gpu_num:{total_gpu_num}")
    MAX_COMPUTE_UNITS = device.get_info(cl.device_info.MAX_COMPUTE_UNITS)
    print(f"max_work_item_sizes:{MAX_COMPUTE_UNITS}")

    for scale in DEFAULT_SCALES:
        time_list: typing.List[float] = []
        program = cl.Program(context, kernel_code).build()
        kernel = cl.Kernel(program, 'matrix_multiply')
        for t in range(10):
            # A = numpy.random.randint(0, 255, size=(scale, scale), dtype=numpy.int32)
            # B = numpy.random.randint(0, 255, size=(scale, scale), dtype=numpy.int32)
            A = numpy.loadtxt(f"{MATRICES_FILE_FOLDER}{os.path.sep}{scale}-{t}-A.csv", delimiter=',', dtype=numpy.int32)
            B = numpy.loadtxt(f"{MATRICES_FILE_FOLDER}{os.path.sep}{scale}-{t}-B.csv", delimiter=',', dtype=numpy.int32)
            start = time.time()
            num_gpu: int = len(platform.get_devices())
            a_buffer = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=A)
            b_buffer = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=B)
            c_buffer = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, A.nbytes)
            kernel.set_arg(0, a_buffer)
            kernel.set_arg(1, b_buffer)
            kernel.set_arg(2, c_buffer)
            kernel.set_arg(3, numpy.int32(scale))
            event =cl.enqueue_nd_range_kernel(queue=queue, kernel=kernel, global_work_size=A.shape, local_work_size=(min(scale, 128), 1))
            # event = program.matrix_multiply(queue, A.shape, None, a_buffer, b_buffer, c_buffer, numpy.int32(scale), wait_for=None)
            event.wait()
            # kernel_work_group_size = kernel.get_work_group_info(cl.kernel_work_group_info.WORK_GROUP_SIZE, device)
            # print(f"kernel_work_group_size:{kernel_work_group_size}")
            c = numpy.empty((scale, scale), dtype=numpy.int32)
            cl.enqueue_copy(queue, c, c_buffer)
            end = time.time()
            time_list.append(end - start)
        print(f"scale:{scale} GPU time:{time_list}")
        print(f"scale:{scale} GPU time avg:{sum(time_list) / len(time_list)}")


