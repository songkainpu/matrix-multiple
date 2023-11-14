import csv
import time
import numpy as np
import os
import multiprocessing
from multiprocessing import Process, Array, shared_memory
from multiprocessing.shared_memory import SharedMemory
from multiprocessing import Pool

MATRICES_FILE_FOLDER = "matrices"
# Function to perform coppersmith-winograd on a pair of matrices

def coppersmith_winograd(matrix1, matrix2):
    size = len(matrix1)
    if size == 1:
        return np.dot(matrix1, matrix2)

    half_size = size // 2

    # Divide matrices into submatrices
    a11 = matrix1[:half_size, :half_size]
    a12 = matrix1[:half_size, half_size:]
    a21 = matrix1[half_size:, :half_size]
    a22 = matrix1[half_size:, half_size:]

    b11 = matrix2[:half_size, :half_size]
    b12 = matrix2[:half_size, half_size:]
    b21 = matrix2[half_size:, :half_size]
    b22 = matrix2[half_size:, half_size:]

    # Recursively compute submatrix products
    m1 = coppersmith_winograd(a11 + a22, b11 + b22)
    m2 = coppersmith_winograd(a21 + a22, b11)
    m3 = coppersmith_winograd(a11, b12 - b22)
    m4 = coppersmith_winograd(a22, b21 - b11)
    m5 = coppersmith_winograd(a11 + a12, b22)
    m6 = coppersmith_winograd(a21 - a11, b11 + b12)
    m7 = coppersmith_winograd(a12 - a22, b21 + b22)

    # Compute the resulting submatrices
    result = np.zeros((size, size))

    result[:half_size, :half_size] = m1 + m4 - m5 + m7
    result[:half_size, half_size:] = m3 + m5
    result[half_size:, :half_size] = m2 + m4
    result[half_size:, half_size:] = m1 - m2 + m3 + m6

    return result

# Function for sequential processing of all groups
def sequential_processing_all_groups():
    matrix_size = [16, 32, 64, 128, 256, 512, 1024, 2048]
    for t in range(10):
        total_time = 0
        for size in matrix_size :
            A = np.loadtxt(fname=f"{MATRICES_FILE_FOLDER}{os.path.sep}{size}-{t}-A.csv",
                           delimiter=',', dtype=np.int32)
            B = np.loadtxt(fname=f"{MATRICES_FILE_FOLDER}{os.path.sep}{size}-{t}-B.csv",
                           delimiter=',', dtype=np.int32)

            start_time = time.time()
            result = coppersmith_winograd(A, B)
            end_time = time.time()
            total_time = end_time - start_time
            print(f"Processing {t}matricx{size} took {total_time} seconds.")


# [B-n.2] Sequential processing of all groups

def coppersmith_winograd_multiply(A_shm, B_shm, result_shm, size):
    A = np.ndarray((size, size), dtype=np.int64, buffer=A_shm.buf)
    B = np.ndarray((size, size), dtype=np.int64, buffer=B_shm.buf)
    result = np.ndarray((size, size), dtype=np.int64, buffer=result_shm.buf)

    stack = [(0, 0, size)]

    while stack:
        start_row, start_col, current_size = stack.pop()

        if current_size == 1:
            result[start_row, start_col] += A[start_row, start_col] * B[start_row, start_col]
        else:
            h = current_size // 2

            # Step 1: Calculate the auxiliary matrices
            S1 = B[start_row:start_row + h, start_col + h] - B[start_row + h:start_row + current_size, start_col + h]
            S2 = A[start_row:start_row + h, start_col:start_col + h] + A[start_row:start_row + h, start_col + h]
            S3 = A[start_row + h:start_row + current_size, start_col:start_col + h] + A[start_row + h:start_row + current_size, start_col + h]
            S4 = B[start_row + h:start_row + current_size, start_col] - B[start_row:start_row + h, start_col]

            # Step 2: Recursive calls
            stack.append((start_row, start_col, h))
            stack.append((start_row, start_col + h, h))
            stack.append((start_row + h, start_col, h))
            stack.append((start_row + h, start_col + h, h))
            stack.append((start_row, start_col, h))
            stack.append((start_row, start_col, h))
            stack.append((start_row, start_col, h))

            # Step 3: Calculate the result matrix
            result[start_row:start_row + h, start_col:start_col + h] += S2 * S4
            result[start_row:start_row + h, start_col + h:start_col + current_size] += S1 + S2
            result[start_row + h:start_row + current_size, start_col:start_col + h] += S3 + S4
            result[start_row + h:start_row + current_size, start_col + h:start_col + current_size] += S1 - S3

    A_shm.close()
    B_shm.close()
    result_shm.close()
def parallel_coppersmith_winograd_multiply(size):
    A = np.loadtxt(fname=f"{MATRICES_FILE_FOLDER}{os.path.sep}{size}-0-A.csv", delimiter=',', dtype=np.int64)
    B = np.loadtxt(fname=f"{MATRICES_FILE_FOLDER}{os.path.sep}{size}-0-B.csv", delimiter=',', dtype=np.int64)

    A_shm = shared_memory.SharedMemory(create=True, size=A.nbytes)
    B_shm = shared_memory.SharedMemory(create=True, size=B.nbytes)
    result_shm = shared_memory.SharedMemory(create=True, size=A.nbytes)

    np.copyto(np.ndarray((size, size), dtype=np.int64, buffer=A_shm.buf), A)
    np.copyto(np.ndarray((size, size), dtype=np.int64, buffer=B_shm.buf), B)

    start_time = time.time()

    with Pool(processes=1) as pool:
        pool.starmap(coppersmith_winograd_multiply, [(A_shm, B_shm, result_shm, size//4)] * 4)

    end_time = time.time()
    total_time = end_time - start_time
    print(f"Process {4} Matrix size {size}:  took {total_time} seconds")

    A_shm.close()
    B_shm.close()
    result_shm.close()

if __name__ == "__main__":
    sequential_processing_all_groups()
    matrix_sizes = [16, 32, 64, 128, 256, 512, 1024, 2048]
    for size in matrix_sizes:
        parallel_coppersmith_winograd_multiply(size)