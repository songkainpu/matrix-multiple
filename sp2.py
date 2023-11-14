import sp1
import csv

def save_results_to_csv(results, filename):
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Group Index", "Sequential Time", "Cumulative Sequential Time"])
        writer.writerows(results)

cumulative_time = 0
results = []

for group_idx in range(10):
    total_time_for_group = 0
    for size in sp1.matrix_sizes:
        total_time_for_group += sp1.perform_sequential_multiplication_for_size(size, group_idx)

    cumulative_time += total_time_for_group
    results.append([group_idx, total_time_for_group, cumulative_time])

save_results_to_csv(results, "sq_results1.csv")

