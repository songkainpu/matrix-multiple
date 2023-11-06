import multiprocessing
import threading
import traceback
from multiprocessing import synchronize, shared_memory
import argparse
import time
import typing
import logging
import os
import functools

import numpy
from matrix_operation import matrix_multiple, matrix_multiple_native
from multiprocessing.pool import Pool

multiprocessing.set_start_method('fork')
MATRICES_FILE_FOLDER = "matrices"

logger: logging.Logger = logging.getLogger(name=__name__)
DEFAULT_SCALES: typing.List[int] = [16, 32, 64, 128, 256, 512, 1024, 2048]
# DEFAULT_SCALES: typing.List[int] = [2048]

# DEFAULT_SCALES: typing.List[int] = [16]
SHARE_MEMO_NAME = "matrix calculate"

parser = argparse.ArgumentParser(description='This is a demo to compute matrices by fox algorithm, '
                                             'The demo will compute the matrices randomly generated in the size of'
                                             ' [16, 32, 64, 128, 256, 512, 1024, 2048]')
SCALE_THRESHOLD = 4
# spilt to 4 matrices 2 *2
SPILT_SIZE = 2
lock = multiprocessing.Lock()


def _generate_matrices(scales: typing.Sequence[int] = None, ):
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
    A = numpy.ndarray((a_right - a_left, a_bottom - a_top), dtype=numpy.int32, buffer=shm_A.buf)
    shm_B = multiprocessing.shared_memory.SharedMemory(name=f"{i},{j},{k},B")
    B = numpy.ndarray((b_right - b_left, b_bottom - b_top), dtype=numpy.int32, buffer=shm_B.buf)
    result: numpy.ndarray = numpy.zeros(shape=(A.shape[0], B.shape[1]), dtype=numpy.int32)
    cur_result = matrix_multiple(matrix1=A, matrix2=B, result=result)
    # cur_result = fox_compute_synchronize(matrix1=A, matrix2=B,result=result)
    shm_A.close()
    shm_B.close()
    shm = multiprocessing.shared_memory.SharedMemory(name=SHARE_MEMO_NAME)
    shared_result_matrix = numpy.ndarray((scale, scale), dtype=numpy.int32, buffer=shm.buf)
    with lock:
        shared_result_matrix[c_left:c_right, c_top:c_bottom] += cur_result
    shm.close()


@functools.cache
def compute_2_pow(exponent: int) -> int:
    return 2 ** exponent


def fox_compute_synchronize(matrix1: numpy.ndarray, matrix2: numpy.ndarray,
                            result: numpy.ndarray, sub_matrix_count: int = 2):
    row1, common, col2 = matrix1.shape[0], matrix1.shape[1], matrix2.shape[1]
    new_dim = compute_2_pow((max(row1, common, col2) - 1).bit_length())
    if new_dim != row1 or new_dim != common or new_dim != col2:
        print(f"不相等")
        new_matrix = numpy.zeros((new_dim, new_dim))
        new_matrix[:row1, :common] = matrix1
        matrix1 = new_matrix
        new_matrix = numpy.zeros((new_dim, new_dim))
        new_matrix[:common, :col2] = matrix2
        matrix2 = new_matrix
    if new_dim <= sub_matrix_count:
        return matrix_multiple(matrix1=matrix1[:row1, :common],
                               matrix2=matrix2[:common, :col2],
                               result=result)
    else:
        block_size = new_dim // sub_matrix_count
        for k in range(sub_matrix_count):
            for i in range(sub_matrix_count):
                for j in range(sub_matrix_count):
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
                    tem_A = matrix1[a_left:a_right, a_top:a_bottom]
                    tem_B = matrix2[b_left:b_right, b_top:b_bottom]
                    fox_compute_synchronize(matrix1=tem_A, matrix2=tem_B, result=result[c_left:c_right, c_top:c_bottom])
        return result


def _compute_matrices():
    global DEFAULT_SCALES
    for scale in DEFAULT_SCALES:
        directed_time = []
        fox_time = []
        for t in range(10):
            pool: multiprocessing.Pool = multiprocessing.Pool(processes=4)
            A = numpy.loadtxt(f"{MATRICES_FILE_FOLDER}{os.path.sep}{scale}-{t}-A.csv",
                              delimiter=',', dtype=numpy.int32)
            B = numpy.loadtxt(f"{MATRICES_FILE_FOLDER}{os.path.sep}{scale}-{t}-B.csv",
                              delimiter=',', dtype=numpy.int32)
            result = numpy.zeros(shape=(scale, scale), dtype=numpy.int32)
            shm = shared_memory.SharedMemory(name=SHARE_MEMO_NAME, create=True, size=result.nbytes)
            shared_result_matrix = numpy.ndarray(result.shape, dtype=numpy.int32, buffer=shm.buf)
            shared_result_matrix[:] = result
            sub_matrix_count = 2
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
                            tem_A = A[a_left:a_right, a_top:a_bottom]
                            tem_B = B[b_left:b_right, b_top:b_bottom]

                            tem_A_shm = shared_memory.SharedMemory(name=f"{i},{j},{k},A", create=True,
                                                                   size=tem_A.nbytes)
                            shared_result_matrix_A = numpy.ndarray(tem_A.shape, dtype=numpy.int32, buffer=tem_A_shm.buf)
                            shared_result_matrix_A[:] = tem_A
                            tem_B_shm = shared_memory.SharedMemory(name=f"{i},{j},{k},B", create=True,
                                                                   size=tem_B.nbytes)
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
                        except Exception as e:
                            traceback.print_exc()
                            print(f"i:{i},j:{j},k:{k}")
            for process_task in process_list:
                process_task.wait()
                process_task.get()
            end_time = time.time()
            fox_time.append(end_time - start_time)
            for share_memo in shm_list:
                share_memo.close()
                share_memo.unlink()
            main_thread_shm = multiprocessing.shared_memory.SharedMemory(name=SHARE_MEMO_NAME)
            shared_result_matrix = numpy.ndarray((scale, scale), dtype=numpy.int32, buffer=main_thread_shm.buf)
            shm.close()
            shm.unlink()
        print(f"scale:{scale}, fox_time:{fox_time}")
        print(f"scale:{scale},fox_time avg:{sum(fox_time) / len(fox_time)}")


def compute_single() -> None:
    global DEFAULT_SCALES
    for scale in DEFAULT_SCALES:
        synchronize_time = []
        for t in range(10):
            A = numpy.loadtxt(f"{MATRICES_FILE_FOLDER}{os.path.sep}{scale}-{t}-A.csv",
                              delimiter=',', dtype=numpy.int32)
            B = numpy.loadtxt(f"{MATRICES_FILE_FOLDER}{os.path.sep}{scale}-{t}-B.csv",
                              delimiter=',', dtype=numpy.int32)
            start_time = time.time()
            result: numpy.ndarray = numpy.zeros_like(A)
            directed_result = matrix_multiple(matrix1=A, matrix2=B, result=result)
            end_time = time.time()
            synchronize_time.append(end_time - start_time)
        print(f"scale:{scale}, synchronize_time:{synchronize_time}")
        print(f"scale:{scale}, synchronize_time avg:{sum(synchronize_time) / len(synchronize_time)}")


def __main() -> None:
    # matrices_map = _generate_matrices()

    _compute_matrices()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    p = multiprocessing.Process(target=__main)
    p2 = multiprocessing.Process(target=compute_single)
    p.start()
    p.join()
