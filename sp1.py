import numpy as np
import csv
import os

matrix_sizes = [16, 32, 64, 128, 256, 512, 1024, 2048]

def load_matrix(filename):
    with open(filename, 'r') as f:
        return np.array(list(csv.reader(f, delimiter=',')), dtype=int)


def matrix_multiply(A, B):
    rows_A, cols_A = len(A), len(A[0])
    rows_B, cols_B = len(B), len(B[0])
    result = [[0 for j in range(cols_B)] for i in range(rows_A)]

    for i in range(rows_A):
        for j in range(cols_B):
            for k in range(cols_A):
                result[i][j] += A[i][k] * B[k][j]

    return result


def get_cpu_time():
    pid = os.getpid()
    process = psutil.Process(pid)
    cpu_times = process.cpu_times()
    return cpu_times.user + cpu_times.system


def perform_sequential_multiplication_for_size(size, group_idx):
    A = load_matrix(f"./matrices/{size}-{group_idx}-A.csv")
    B = load_matrix(f"./matrices/{size}-{group_idx}-B.csv")
    N = 1
    start_time = get_cpu_time()
    for _ in range(N):
        matrix_multiply(A, B)
    end_time = get_cpu_time()
    average_duration = (end_time - start_time) / N
    return average_duration

def perform_sequential_multiplication():
    results = []
    cumulative_time = 0
    for size in matrix_sizes:
        time_taken = perform_sequential_multiplication_for_size(size, 0)
        print(time_taken)
        cumulative_time += time_taken
        print(cumulative_time)
        results.append([size, time_taken, cumulative_time])
    with open("sequential_results0_comparison.csv", 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Pair Index", "Sequential Time", "Cumulative Sequential Time"])
        writer.writerows(results)

if __name__ == "__main__":
  perform_sequential_multiplication()