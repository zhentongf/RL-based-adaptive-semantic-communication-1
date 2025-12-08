import torch
import numpy as np
import torch.serialization
from rl_agent import RLAgent

# ---------------------------
# Load trained RL model
# ---------------------------

rl_model_path = './saved_models/rl_agent_snr_threshold.pkl'

# Create RLAgent instance exactly as trained
rl_agent = RLAgent(
    state_dim=3,
    action_dim=10,
    lr=0.001,
    gamma=0.99,
    epsilon_start=0.01,   # use low epsilon for testing (no random actions)
    epsilon_end=0.01,
    epsilon_decay=1.0,    # no decay
    device='cpu'
)

# Load model file
rl_agent.load_model(rl_model_path)
print("RL model loaded.")

# ---------------------------
# User input testing
# ---------------------------

print("\nEnter 3 values (traditional_SNR, distance_m, relative_speed_m_s):")
snr = float(input("Traditional SNR (dB): "))
distance = float(input("Distance (meters): "))
speed = float(input("Relative speed (m/s): "))

state = np.array([snr, distance, speed])

# RL selects action (threshold index)
action = rl_agent.select_action(state, training=False)

# Convert action to threshold value
snr_threshold = rl_agent.get_snr_threshold(action)

print("\n=== RL Agent Decision ===")
print(f"Selected Action Index: {action}")
print(f"Predicted SNR Threshold: {snr_threshold:.2f} dB")
