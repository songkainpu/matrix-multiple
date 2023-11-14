import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata

wl = [1, 2, 3, 4, 5, 6, 7, 8,9,10]
processes = [1, 2, 3, 4, 5, 6, 9, 16]
times = np.array([
    [6159.733769, 3184.486736, 2002.285802, 1626.19035, 1376.821132, 1706.766847, 1151.189481, 1124.558037],
    [12420.02975, 6404.086216, 3975.577107, 3239.92326, 2738.425516, 3392.55608, 2285.054808, 2260.17576],
    [18586.46068, 9609.483621, 5970.709931, 4897.202517, 4146.16735, 5107.139716, 3468.222396, 3369.265316],
    [24618.42088, 12833.73736, 7930.835352, 6475.406219, 5503.15106, 6801.944605, 4611.007026, 4474.485044],
    [30635.93258, 16015.79745, 9956.477376, 8139.783128, 6846.358096, 8466.383136, 5738.110781, 5578.53747],
    [36940.81276, 19129.36802, 11983.28165, 9781.615079, 8209.544789, 10137.74263, 6876.226574, 6727.149566],
    [42916.65973, 22355.36004, 13963.16094, 11444.17597, 9588.015589, 11815.84869, 8016.223545, 7874.196165],
    [49151.41401, 25469.5907, 15939.24857, 13026.5847, 10975.64738, 13471.69284, 9152.475106, 9004.688701],
    [55415.67232, 28614.44686, 17945.40059, 14611.96698, 12393.30862, 15154.5874, 10319.04899, 10138.74572],
    [61703.12495, 31854.884, 19888.65491, 16270.33424, 13737.61127, 16836.8919, 11498.75528, 11284.38465]
])

fig, axs = plt.subplots(1, 3, figsize=(18, 6))

for i, proc in enumerate(processes):
    axs[0].plot(wl, times[:, i], label=f'Process {proc}')
axs[0].set_title('Processing time vs Workload of All groups')
axs[0].set_xlabel('Workload')
axs[0].set_ylabel('Time')
axs[0].legend()

for i, wl_value in enumerate(wl):
    axs[1].plot(processes, times[i, :], label=f'Workload {wl_value}')
axs[1].set_title('Processing time vs Processes of All groups')
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
