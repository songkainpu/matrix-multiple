import numpy as np
import matplotlib.pyplot as plt

times = np.array([
    [0.127967119, 0.179242134, 0.776365042, 0.160832882, 0.814480066, 0.830382824, 0.27385807, 0.723299026],
    [0.266499043, 0.364447117, 0.914302111, 0.313441992, 0.984185457, 1.048563004, 0.515036106, 1.445970297],
    [0.548137903, 0.568401098, 1.09218812, 0.520857096, 1.174209356, 1.28737998, 0.838598251, 2.046667337],
    [1.928364992, 1.67140317, 1.723715425, 1.048856974, 1.679509242, 1.843058983, 1.333377361, 2.86286521],
    [11.29569697, 7.356743574, 5.342885613, 4.206408024, 4.02903125, 4.527469714, 3.487377405, 5.25445199],
    [91.4071939, 47.75320172, 33.17227697, 27.45306683, 20.36184282, 30.32542141, 19.79085112, 20.81529689],
    [745.9166832, 378.2094796, 250.5539483, 201.3350809, 151.209339, 216.1334522, 137.928381, 144.2181687],
    [6159.733769, 3184.486736, 2002.285802, 1626.19035, 1376.821132, 1706.766847, 1151.189481, 1124.558037]
])
processes = [1, 2, 3, 4, 5, 6, 9, 16]
original_values = [16*16, 32*32, 64*64, 128*128, 256*256, 512*512, 1024*1024, 2048*2048]

workloads = [sum(original_values[:i+1]) for i in range(len(original_values))]
speedup_wl = times[:, 0, np.newaxis] / times

fig, axs = plt.subplots(1, 2)
for i, proc in enumerate(processes):
    axs[0].plot(workloads, speedup_wl[i, :], label=f'Processes {proc}')
axs[0].set_xlabel('Workload')
axs[0].set_ylabel('Speedup')
axs[0].set_title('Speedup vs Workload for Different Processes')
axs[0].legend()

speedup_proc = times[:, 0, np.newaxis] / times
for i, wl in enumerate(workloads):
    axs[1].plot(processes, speedup_proc[:, i], label=f'Workload {wl}')
axs[1].set_xlabel('Processes')
axs[1].set_ylabel('Speedup')
axs[1].set_title('Speedup vs Processes for Different Workloads')
axs[1].legend()

plt.show()
