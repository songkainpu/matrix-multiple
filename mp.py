import os
import numpy as np
import csv
import time
from multiprocessing import Process, Array, shared_memory
from multiprocessing.shared_memory import SharedMemory


def shift_matrix_left(matrix, shifts):
    return np.roll(matrix, -shifts, axis = 1)


def shift_matrix_up(matrix, shifts):
    return np.roll(matrix, -shifts, axis = 0)


def matrix_multiply_normal(A, B):
    n = A.shape[0]
    result = np.zeros((n,n), dtype=int)

    for i in range(n):
        for j in range(n):
            for k in range(n):
                result[i][j] += A[i][k] * B[k][j]

    return result

def matrix_multiply_worker(A_shared_name, B_shared_name, C_shared_name, row_start, row_end, col_start, col_end, size, num_shifts):
    A_shared = shared_memory.SharedMemory(name=A_shared_name)
    B_shared = shared_memory.SharedMemory(name=B_shared_name)
    C_shared = shared_memory.SharedMemory(name=C_shared_name)

    A = np.ndarray((size, size), dtype=int, buffer=A_shared.buf)
    B = np.ndarray((size, size), dtype=int, buffer=B_shared.buf)
    C = np.ndarray((size, size), dtype=int, buffer=C_shared.buf)

    for _ in range(size // (row_end - row_start)):
        A_block = A[row_start:row_end, :]
        B_block = B[:, col_start:col_end]
        C_block = matrix_multiply_normal(A_block, B_block)
        C[row_start:row_end, col_start:col_end] += C_block

        A[row_start:row_end, :] = np.roll(A[row_start:row_end, :], -num_shifts, axis=1)
        B[:, col_start:col_end] = np.roll(B[:, col_start:col_end], -num_shifts, axis=0)

    A_shared.close()
    B_shared.close()
    C_shared.close()



def cannon_matrix_multiply(A, B, num_processes):
    size = A.shape[0]
    virtual_grid_size = int(np.ceil(np.sqrt(num_processes)))
    block_size = size // virtual_grid_size
    num_shifts = block_size

    A_shared = shared_memory.SharedMemory(create=True, size=A.nbytes)
    B_shared = shared_memory.SharedMemory(create=True, size=B.nbytes)
    C_shared = shared_memory.SharedMemory(create=True, size=A.nbytes)

    A_np = np.ndarray(A.shape, dtype=A.dtype, buffer=A_shared.buf)
    B_np = np.ndarray(B.shape, dtype=B.dtype, buffer=B_shared.buf)
    C_np = np.ndarray(A.shape, dtype=A.dtype, buffer=C_shared.buf)

    A_np[:] = A[:]
    B_np[:] = B[:]
    C_np[:] = np.zeros_like(A, dtype=int )

    for i in range(virtual_grid_size):
        A_np[i * block_size:(i + 1) * block_size, :] = shift_matrix_left(A_np[i * block_size:(i + 1) * block_size, :],
                                                                         i)
        B_np[:, i * block_size:(i + 1) * block_size] = shift_matrix_up(B_np[:, i * block_size:(i + 1) * block_size], i)

    processes = []
    for i in range(virtual_grid_size):
        for j in range(virtual_grid_size):

            row_start = i * block_size
            col_start = j * block_size
            row_end = min((i + 1) * block_size, size)
            col_end = min((j + 1) * block_size, size)

            p = Process(target=matrix_multiply_worker,
                        args=(A_shared.name, B_shared.name, C_shared.name,
                              row_start, row_end, col_start, col_end, size, num_shifts))
            processes.append(p)

    start_time = time.time()
    for p in processes:
        p.start()
    for p in processes:
        p.join()
    end_time = time.time()

    A_shared.close()
    B_shared.close()
    C_shared.close()
    A_shared.unlink()
    B_shared.unlink()
    C_shared.unlink()

    return end_time - start_time

def load_matrix(filename):
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        matrix = list(reader)
    return np.array(matrix, dtype=int)
def pad_matrix(matrix, block_size):
    (rows, cols) = matrix.shape
    pad_rows = block_size - rows % block_size if rows % block_size != 0 else 0
    pad_cols = block_size - cols % block_size if cols % block_size != 0 else 0

    if pad_rows > 0 or pad_cols > 0:
        new_matrix = np.pad(matrix, ((0, pad_rows), (0, pad_cols)), mode='constant', constant_values=0)
        return new_matrix
    return matrix


def adjust_matrices(A, B, num_processes):

    block_size = int(num_processes)

    if A.shape[0] % block_size != 0 or A.shape[1] % block_size != 0:
        A = pad_matrix(A, block_size)
        B = pad_matrix(B, block_size)

    return A, B

def save_results(matrix_sizes, processes, group):
    results = {'Size/Processes': matrix_sizes}
    output_filename = f'MpResult{group}.csv'
    for num_processes in processes:
        results[num_processes] = []
        for size in matrix_sizes:
            A_filename = f"./matrices/{size}-{group}-A.csv"
            B_filename = f"./matrices/{size}-{group}-B.csv"
            A = load_matrix(A_filename)
            B = load_matrix(B_filename)
            A, B = adjust_matrices(A, B, int(np.ceil(np.sqrt(num_processes))))
            elapsed_time = cannon_matrix_multiply(A, B, num_processes)
            results[num_processes].append(elapsed_time)

    with open(output_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Size/Processes'] + processes)
        for size in matrix_sizes:
            row = [size]
            for num_processes in processes:
                row.append(results[num_processes][matrix_sizes.index(size)])
            writer.writerow(row)



if __name__ == '__main__':
    matrix_sizes = [16, 32, 64, 128, 256, 512, 1024, 2048]
    processes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

    for group in range(10):
        save_results(matrix_sizes, processes, group)




