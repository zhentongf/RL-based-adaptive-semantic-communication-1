import pandas as pd
import numpy as np
from fuzzy_logic import decide_use_nn

# -----------------------------
# 1. Vehicle definitions
# -----------------------------
# Speeds in m/s
speeds = {
    "A": 5,
    "B": 10,
    "C": 15,
    "D": 20,
    "E": 25
}

# Time steps (seconds)
times = [1, 2, 3, 4, 5]

# -----------------------------
# 2. Compute positions
# -----------------------------
# position = speed * time
positions = {
    car: [speed * t for t in times]
    for car, speed in speeds.items()
}
# print("Positions (m):")
# for car, pos in positions.items():
#     print(f"{car}: {pos}")

# -----------------------------
# 3. SNR model
# -----------------------------
# Linear decay:
# distance = 0 m  -> SNR = 40 dB
# distance = 100 m -> SNR = -10 dB
def compute_snr(distance):
    return 40 - 0.5 * distance

# -----------------------------
# 4. Generate data
# -----------------------------
rows = []

# Pairs: A-B, A-C, A-D, A-E
other_cars = ["B", "C", "D", "E"]

for other in other_cars:
    for i, t in enumerate(times):
        # Distance between car A and the other car
        distance = abs(positions[other][i] - positions["A"][i])

        # Relative speed
        rel_speed = abs(speeds[other] - speeds["A"])

        # SNR value
        snr = compute_snr(distance)

        rows.append([
            snr,
            distance,
            rel_speed
        ])

# -----------------------------
# 5. Create DataFrame
# -----------------------------
df = pd.DataFrame(
    rows,
    columns=[
        "snr_values",
        "distance_values",
        "rel_speed_values"
    ]
)

# -----------------------------
# 6. Normalize data (0–1)
# -----------------------------
# SNR: min = -10, max = 40
df["snr_values_norm"] = (df["snr_values"] - (-10)) / (40 - (-10))

# Distance: min = 0, max = 100
df["distance_values_norm"] = (df["distance_values"] - 0) / (100 - 0)

# Relative speed: min = 0, max = 25
df["rel_speed_values_norm"] = (df["rel_speed_values"] - 0) / (25 - 0)

# -----------------------------
# 7. Compute composite SNR and fuzzy decision
# -----------------------------
# --- helper: compute composite SNR in dB ---
def compute_composite_snr_db(snr_trad_db, distance_m, rel_speed_ms,
                             d_ref=1.0, d_min=1.0,
                             n=2.0,            # path-loss exponent
                             v_ref=10.0,       # reference speed (m/s)
                             w_d=1.0, w_v=1.0, # weights for distance and speed penalties
                             clip_min_db=-30.0, clip_max_db=40.0):
    """
    Returns composite SNR in dB.
    snr_trad_db: conventional SNR in dB (float or numpy scalar)
    distance_m: distance in meters (float)
    rel_speed_ms: relative speed in m/s (can be negative; function uses abs)
    Other params are tunable.
    """
    # sanitize inputs
    d = max(distance_m, d_min)
    v = abs(rel_speed_ms)

    # distance penalty: 10 * n * log10(d / d_ref) scaled by w_d
    pen_d_db = w_d * 10.0 * n * np.log10(d / d_ref)

    # speed penalty: soft-log penalty, scaled by w_v
    pen_v_db = w_v * 10.0 * np.log10(1.0 + (v / v_ref))

    composite = float(snr_trad_db) - float(pen_d_db) - float(pen_v_db)

    # clip to avoid extreme values
    composite = max(min(composite, clip_max_db), clip_min_db)
    return composite

# composite SNR in dB
df["composite_snr_db"] = df.apply(
    lambda row: compute_composite_snr_db(
        row["snr_values"],
        row["distance_values"],
        row["rel_speed_values"]
    ),
    axis=1
)
# Fuzzy decision
df["use_nn"] = df.apply(
    lambda row: decide_use_nn(
        row["composite_snr_db"],
        row["distance_values_norm"],
        row["rel_speed_values_norm"]
    ),
    axis=1
)

# -----------------------------
# 8. Save to CSV
# -----------------------------
df.to_csv("sim_data.csv", index=False)

print("sim_data.csv generated successfully!")
print(f"Total rows: {len(df)}")
