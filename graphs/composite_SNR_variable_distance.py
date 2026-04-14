import numpy as np
import matplotlib.pyplot as plt
from fuzzy_logic import compute_composite_snr_db

snr_trad_db = 40
rel_speed_ms = 50

distance = np.linspace(1, 100, 100)
snr_values = []

for d in distance:
    snr = compute_composite_snr_db(
        snr_trad_db,
        d,
        rel_speed_ms
    )
    snr_values.append(snr)

plt.plot(distance, snr_values)
plt.xlabel("Distance (m)")
plt.ylabel("Composite SNR (dB)")
plt.title("Composite SNR vs Distance (Speed=50 m/s)")
plt.grid(True)
plt.show()