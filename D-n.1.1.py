import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata

original_values = [16*16, 32*32, 64*64, 128*128, 256*256, 512*512, 1024*1024, 2048*2048]

wl = [sum(original_values[:i+1]) for i in range(len(original_values))]
print(wl)
processes = [1, 2, 3, 4, 5, 6]
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


fig, axs = plt.subplots(1, 3, figsize=(18, 6))

for i, proc in enumerate(processes):
    axs[0].plot(wl, times[:, i], label=f'Process {proc}')
axs[0].set_title('Processing time vs Workload of Group 0')
axs[0].set_xlabel('Workload')
axs[0].set_ylabel('Time')
axs[0].legend()

for i, wl_value in enumerate(wl):
    axs[1].plot(processes, times[i, :], label=f'Workload {wl_value}')
axs[1].set_title('Processing time vs Processes of Group 0')
axs[1].set_xlabel('Processes')
axs[1].set_ylabel('Time')

X, Y = np.meshgrid(processes, wl)
points = np.array([[p, w] for p in processes for w in wl])
values = times.T.flatten()
Z = griddata(points, values, (X, Y), method='linear')
ax2 = fig.add_subplot(133, projection='3d')
for i, proc in enumerate(processes):
    ax2.scatter([proc]*len(wl), wl, times[:, i])
ax2.plot_surface(X, Y, Z, rstride=1, cstride=1, alpha=0.5, cmap='viridis')
ax2.set_title('3D View: Processing time with Processors and Workload')
ax2.set_xlabel('Processors')
ax2.set_ylabel('Workload')
ax2.set_zlabel('Time')

plt.tight_layout()
plt.show()
