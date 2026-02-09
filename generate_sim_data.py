import pandas as pd

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
print("Positions (m):")
for car, pos in positions.items():
    print(f"{car}: {pos}")

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
# 7. Save to CSV
# -----------------------------
df.to_csv("sim_data.csv", index=False)

print("sim_data.csv generated successfully!")
print(f"Total rows: {len(df)}")
