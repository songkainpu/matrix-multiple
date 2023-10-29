import multiprocessing
import traceback
from multiprocessing import synchronize, shared_memory
import argparse
import time
import typing
import logging
import os

import numpy
from matrix_operation import matrix_multiple
from multiprocessing.pool import Pool
multiprocessing.set_start_method('fork')
MATRICES_FILE_FOLDER = "matrices"

logger: logging.Logger = logging.getLogger(name=__name__)
DEFAULT_SCALES: typing.List[int] = [16, 32, 64, 128, 256, 512, 1024, 2048]
# DEFAULT_SCALES: typing.List[int] = [16]
SHARE_MEMO_NAME = "matrix calculate"

parser = argparse.ArgumentParser(description='This is a demo to compute matrices by fox algorithm, '
                                             'The demo will compute the matrices randomly generated in the size of'
                                             ' [16, 32, 64, 128, 256, 512, 1024, 2048]')
SCALE_THRESHOLD = 4
# spilt to 4 matrices 2 *2
SPILT_SIZE = 2
lock = multiprocessing.Lock()


def _generate_matrices(scales: typing.Sequence[int] = None,):
    if scales is None:
        scales = DEFAULT_SCALES
    for scale in scales:
        for i in range(10):
            result = (numpy.random.randint(0, 255, size=(scale, scale), dtype=numpy.int32),
                      numpy.random.randint(0, 255, size=(scale, scale), dtype=numpy.int32))
            numpy.savetxt(fname=f"{MATRICES_FILE_FOLDER}{os.path.sep}{scale}-{i}-A.csv",
                          X=result[0], delimiter=',', fmt='%d')
            numpy.savetxt(fname=f"{MATRICES_FILE_FOLDER}{os.path.sep}{scale}-{i}-B.csv",
                          X=result[0], delimiter=',', fmt='%d')


def _process_matrix_result(modification_func):
    def decorator(func):
        def wrapper(*args, **kwargs):
            scale = kwargs.get("scale")
            left1 = kwargs.get("left1")
            top1 = kwargs.get("top1")
            right1 = kwargs.get("right1")
            bottom1 = kwargs.get("bottom1")
            left2 = kwargs.get("left2")
            top2 = kwargs.get("top2")
            right2 = kwargs.get("right2")
            bottom2 = kwargs.get("bottom2")
            if right1 is None:
                right1 = scale
                kwargs['right1'] = scale
            if bottom1 is None:
                bottom1 = scale
                kwargs['bottom1'] = scale
            if right2 is None:
                right2 = scale
                kwargs['right2'] = scale
            if bottom2 is None:
                bottom2 = scale
                kwargs['bottom2'] = scale
            shm1 = multiprocessing.shared_memory.SharedMemory(name=f"{SHARE_MEMO_NAME}:m1")
            M1: numpy.ndarray = numpy.ndarray((scale, scale), dtype=numpy.int32, buffer=shm1.buf)
            shm2 = multiprocessing.shared_memory.SharedMemory(name=f"{SHARE_MEMO_NAME}:m2")
            M2: numpy.ndarray = numpy.ndarray((scale, scale), dtype=numpy.int32, buffer=shm1.buf)
            numpy.frombuffer()
            kwargs['M1'] = M1
            kwargs['M2'] = M2
            result = func(*args, **kwargs)


def _compute(scale: int, left1: int = 0, right1: int = None, top1: int = 0,
             bottom1: int = None, left2: int = 0, right2: int = None, top2: int = 0,
             bottom2: int = None, pool: multiprocessing.Pool = None):
    print(f"scale:{scale}, left1:{left1}, left2:{left2}, right1:{right1}, right2:{right2}")
    if right1 is None:
        right1 = scale
    if bottom1 is None:
        bottom1 = scale
    if right2 is None:
        right2 = scale
    if bottom2 is None:
        bottom2 = scale

    cur_scale = bottom1 - top1 + 1
    A = matrices_map[scale][0]
    B = matrices_map[scale][1]
    A = A[left1:right1, top1:bottom1]
    B = B[left2:right2, top2:bottom2]
    if not pool:
        result = matrix_multiple(matrix1=A, matrix2=B)
        return result
    pool = multiprocessing.Pool(processes=4)
    spilt_size = cur_scale // 2
    Base = {
        "scale": scale,
        "pool": pool,
    }
    X11 = {
        "left1": left1,
        "right1": left1 + spilt_size,
        "top1": top1,
        "bottom1": top1 + spilt_size
    }
    X12 = {
        "left1": left1 + spilt_size,
        "right1": right1,
        "top1": top1,
        "bottom1": top1 + spilt_size,
    }
    X21 = {
        "left1": left1,
        "right1": left1 + spilt_size,
        "top1": top1 + spilt_size,
        "bottom1": bottom1
    }
    X22 = {
        "left1": left1 + spilt_size,
        "right1": right1,
        "top1": top1 + spilt_size,
        "bottom1": bottom1
    }

    Y11 = {
        "left2": left2,
        "right2": left2 + spilt_size,
        "top2": top2,
        "bottom2": top2 + spilt_size
    }
    Y12 = {
        "left2": left2 + spilt_size,
        "right2": right2,
        "top2": top2,
        "bottom2": top2 + spilt_size,
    }
    Y21 = {
        "left1": left2,
        "right1": left2 + spilt_size,
        "top1": top2 + spilt_size,
        "bottom1": bottom2
    }
    Y22 = {
        "left2": left2 + spilt_size,
        "right2": right2,
        "top2": top2 + spilt_size,
        "bottom2": bottom2
    }
    M1_processes_list = [
        # X11 * Y11
        pool.apply_async(_compute, kwds={
            **Base,
            **X11,
            **Y11,
            "idx": 0
        }),
        # X12 * Y22
        pool.apply_async(_compute, kwds={
            **Base,
            **X11,
            **Y12,
            "idx": 1
        }),
        # X21
        pool.apply_async(_compute, kwds={
            **Base,
            **X21,
            **Y11,
            "idx": 2
        }),
        # X22
        pool.apply_async(_compute, kwds={
            **Base,
            **X21,
            **Y12,
            "idx": 3
        })
    ]
    M2_processes_list = [
        # X12 * Y21
        pool.apply_async(_compute, kwds={
            **Base,
            **X12,
            **Y21,
            "idx": 0
        }),
        # X12 * Y22
        pool.apply_async(_compute, kwds={
            **Base,
            **X12,
            **Y22,
            "idx": 1
        }),
        # X22 * Y21
        pool.apply_async(_compute, kwds={
            **Base,
            **X22,
            **Y21,
            "idx": 2
        }),
        # X22 * Y22
        pool.apply_async(_compute, kwds={
            **Base,
            **X22,
            **Y22,
            "idx": 3
        })
    ]
    for result in M1_processes_list:
        result.wait()
    for result in M2_processes_list:
        result.wait()
    # M1_two_dim: typing.List[typing.Optional[numpy.ndarray]] = [result.get() for result in M1_processes_list]
    # M2_two_dim: typing.List[typing.Optional[numpy.ndarray]] = [result.get() for result in M2_processes_list]


def __compute(i: int, j: int, k: int, block_size: int, scale: int):

    # f"{i},{j},{k},B"
    # A = matrices_map[scale][0]
    # B = matrices_map[scale][1]
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
    shm_A = multiprocessing.shared_memory.SharedMemory(name=f"{i},{j},{k},A")
    A = numpy.ndarray((a_right - a_left, a_bottom - a_top ), dtype=numpy.int32, buffer=shm_A.buf)
    shm_B = multiprocessing.shared_memory.SharedMemory(name=f"{i},{j},{k},B")
    B = numpy.ndarray((b_right - b_left, b_bottom - b_top), dtype=numpy.int32, buffer=shm_B.buf)
    cur_result = matrix_multiple(matrix1=A, matrix2=B)
    shm_A.close()
    shm_B.close()
    shm = multiprocessing.shared_memory.SharedMemory(name=SHARE_MEMO_NAME)
    shared_result_matrix = numpy.ndarray((scale, scale), dtype=numpy.int32, buffer=shm.buf)
    with lock:
        shared_result_matrix[c_left:c_right, c_top:c_bottom] += cur_result
    shm.close()





def _compute_matrices():
    global DEFAULT_SCALES
    for scale in DEFAULT_SCALES:
        directed_time = []
        fox_time = []
        for t in range(10):
            pool: multiprocessing.Pool = multiprocessing.Pool(processes=4)
            start_time = time.time()
            A = numpy.loadtxt(f"{MATRICES_FILE_FOLDER}{os.path.sep}{scale}-{t}-A.csv", delimiter=',')
            B = numpy.loadtxt(f"{MATRICES_FILE_FOLDER}{os.path.sep}{scale}-{t}-B.csv", delimiter=',')
            directed_computed = matrix_multiple(matrix1=A,
                                                matrix2=B)
            end_time = time.time()
            directed_time.append(end_time-start_time)
            # context: multiprocessing.context.ForkContext = multiprocessing.get_context(method='spawn')
            result = numpy.zeros(shape=(scale, scale), dtype=numpy.int32)
            shm = shared_memory.SharedMemory(name=SHARE_MEMO_NAME, create=True, size=result.nbytes)
            shared_result_matrix = numpy.ndarray(result.shape, dtype=numpy.int32, buffer=shm.buf)
            shared_result_matrix[:] = result
            # _compute(scale=scale)
            # 定义矩阵的规模
            # block_size = 4  # 块大小
            # 初始化结果矩阵 C
            # 定义子矩阵的数量
            # sub_matrix_count = scale // block_size
            sub_matrix_count = 4
            block_size = scale // sub_matrix_count
            process_list = []
            start_time = time.time()
            shm_list = []
            for k in range(sub_matrix_count):
                for i in range(sub_matrix_count):
                    for j in range(sub_matrix_count):
                        try:
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
                            tem_A = A[a_left:a_right, a_top:a_bottom]
                            tem_B = B[b_left:b_right, b_top:b_bottom]

                            tem_A_shm = shared_memory.SharedMemory(name=f"{i},{j},{k},A", create=True, size=tem_A.nbytes)
                            shared_result_matrix_A = numpy.ndarray(tem_A.shape, dtype=numpy.int32, buffer=tem_A_shm.buf)
                            shared_result_matrix_A[:] = tem_A
                            tem_B_shm = shared_memory.SharedMemory(name=f"{i},{j},{k},B", create=True, size=tem_B.nbytes)
                            shared_result_matrix_B = numpy.ndarray(tem_B.shape, dtype=numpy.int32, buffer=tem_B_shm.buf)
                            shared_result_matrix_B[:] = tem_B
                            shm_list.append(tem_A_shm)
                            shm_list.append(tem_B_shm)
                            process_task = pool.apply_async(func=__compute, kwds={
                                "i": i,
                                "j": j,
                                "k": k,
                                "block_size": block_size,
                                "scale": scale,
                            })
                            process_list.append(process_task)
                        except:
                            traceback.print_exc()
                            print(f"i:{i},j:{j},k:{k}")
            for process_task in process_list:
                process_task.wait()
                process_task.get()
            end_time = time.time()
            fox_time.append(end_time-start_time)
            for share_memo in shm_list:
                share_memo.close()
                share_memo.unlink()
            main_thread_shm = multiprocessing.shared_memory.SharedMemory(name=SHARE_MEMO_NAME)
            shared_result_matrix = numpy.ndarray((scale, scale), dtype=numpy.int32, buffer=main_thread_shm.buf)
            # print("Main Thread - Shared Array:", directed_computed)

            # print("Main Thread - Shared Array:", shared_result_matrix)
            shm.close()
            shm.unlink()
        print(f"scale:{scale}, directed_time:{directed_time}, fox_time:{fox_time}")
        print(f"scale:{scale}, directed_time avg:{sum(directed_time) / len(directed_time)}, fox_time avg:{sum(fox_time) /len(fox_time)}")
    # import numpy as np
    #
    # # 定义矩阵的规模和块大小
    # n = 4  # 矩阵的维度
    # block_size = 2  # 块大小
    #
    # # 随机生成两个矩阵 A 和 B
    # A = np.random.rand(n, n)
    # B = np.random.rand(n, n)
    #
    # # 初始化结果矩阵 C
    # C = np.zeros((n, n))
    # # 使用Fox算法进行矩阵乘法
    # for k in range(n):
    #     for i in range(n):
    #         for j in range(n):
    #             # 计算子块的坐标
    #             row = (i + k) % n
    #             col = (j + k) % n
    #             # 计算局部块的乘积
    #             local_sum = np.dot(A[i, row::block_size], B[col::block_size, j])
    #             # 在C中累积局部块的乘积
    #             C[i, j] += local_sum
    #
    # print("Matrix C (Result of Matrix Multiplication):")
    # print(C)
    #
    # pass


def __main() -> None:
    # matrices_map = _generate_matrices()

    _compute_matrices()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    p = multiprocessing.Process(target=__main)
    p.start()
    p.join()
