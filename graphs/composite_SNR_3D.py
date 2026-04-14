import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from fuzzy_logic import compute_composite_snr_db

snr_trad_db = 40

# Ranges
distance = np.linspace(1, 100, 50)
speed = np.linspace(0, 50, 50)

D, S = np.meshgrid(distance, speed)

Z = np.zeros_like(D)

for i in range(D.shape[0]):
    for j in range(D.shape[1]):
        Z[i, j] = compute_composite_snr_db(
            snr_trad_db,
            D[i, j],
            S[i, j]
        )

# Plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(D, S, Z)

ax.set_xlabel("Distance (m)")
ax.set_ylabel("Speed (m/s)")
ax.set_zlabel("Composite SNR (dB)")
ax.set_title("Composite SNR vs Distance & Speed")

plt.show()