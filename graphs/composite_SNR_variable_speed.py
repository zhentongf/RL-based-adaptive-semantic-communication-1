import numpy as np
import matplotlib.pyplot as plt
from fuzzy_logic import compute_composite_snr_db

snr_trad_db = 40
distance_m = 100

speed = np.linspace(0, 50, 100)
snr_values = []

for v in speed:
    snr = compute_composite_snr_db(
        snr_trad_db,
        distance_m,
        v
    )
    snr_values.append(snr)

plt.plot(speed, snr_values)
plt.xlabel("Speed (m/s)")
plt.ylabel("Composite SNR (dB)")
plt.title("Composite SNR vs Speed (Distance=100m)")
plt.grid(True)
plt.show()