# encoding: utf-8
"""
Deep Reinforcement Learning Agent for Adaptive SNR Threshold Selection
Uses Deep Q-Network (DQN) to learn optimal SNR threshold based on:
- State: (traditional SNR, distance, relative speed)
- Action: SNR threshold selection
- Reward: Based on classification accuracy and PSNR
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import os


class ReplayBuffer:
    """Experience replay buffer for DQN"""
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """Add experience to buffer"""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """Sample a batch of experiences"""
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return np.array(state), np.array(action), np.array(reward), np.array(next_state), np.array(done)
    
    def __len__(self):
        return len(self.buffer)


class DQN(nn.Module):
    """Deep Q-Network for SNR threshold selection"""
    def __init__(self, state_dim=3, action_dim=10, hidden_dim=128):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, action_dim)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class RLAgent:
    """Reinforcement Learning Agent for SNR threshold optimization"""
    def __init__(self, state_dim=3, action_dim=10, lr=0.001, gamma=0.99,
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995,
                 memory_size=10000, batch_size=64, target_update=10,
                 device='cuda'):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update
        self.device = device
        self.update_counter = 0
        
        # Define action space: SNR threshold values (in dB)
        # Range from 0 to 20 dB with action_dim steps
        self.snr_threshold_min = 0.0
        self.snr_threshold_max = 20.0
        self.snr_thresholds = np.linspace(
            self.snr_threshold_min, 
            self.snr_threshold_max, 
            action_dim
        )
        
        # Normalization parameters for state space
        # Traditional SNR: 0-20 dB, Distance: 1-200 m, Relative Speed: 0-50 m/s
        self.state_mean = np.array([10.0, 100.5, 25.0])
        self.state_std = np.array([5.77, 57.45, 14.43])  # Approximately std for uniform distributions
        
        # Initialize networks
        self.q_network = DQN(state_dim, action_dim).to(device)
        self.target_network = DQN(state_dim, action_dim).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # Replay buffer
        self.memory = ReplayBuffer(memory_size)
        
        # State normalization bounds
        self.state_min = np.array([0.0, 1.0, 0.0])
        self.state_max = np.array([20.0, 200.0, 50.0])
    
    def normalize_state(self, state):
        """Normalize state to [-1, 1] range"""
        state = np.array(state)
        # Normalize to [0, 1] first
        state_norm = (state - self.state_min) / (self.state_max - self.state_min)
        # Then normalize to [-1, 1]
        state_norm = 2 * state_norm - 1
        return state_norm
    
    def denormalize_state(self, state_norm):
        """Denormalize state from [-1, 1] range"""
        state_norm = np.array(state_norm)
        # First map from [-1, 1] to [0, 1]
        state = (state_norm + 1) / 2
        # Then map to original range
        state = state * (self.state_max - self.state_min) + self.state_min
        return state
    
    def select_action(self, state, training=True):
        """Select action using epsilon-greedy policy"""
        if training and random.random() < self.epsilon:
            # Random action
            action = random.randint(0, self.action_dim - 1)
        else:
            # Greedy action
            state_norm = self.normalize_state(state)
            state_tensor = torch.FloatTensor(state_norm).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
                action = q_values.argmax().item()
        
        return action
    
    def get_snr_threshold(self, action):
        """Get SNR threshold value from action index"""
        return self.snr_thresholds[action]
    
    def compute_reward(self, accuracy, psnr, use_nn=False, snr_threshold=None):
        """
        Compute reward based on accuracy and PSNR
        Higher accuracy and PSNR give higher rewards
        Penalize if using NN when SNR is high (inefficient)
        """
        # Base reward from accuracy and PSNR
        acc_reward = accuracy * 100  # Scale accuracy to 0-100
        psnr_reward = psnr / 40.0 * 100  # Normalize PSNR (assuming max ~40 dB)
        
        # Combined reward (weighted)
        reward = 0.7 * acc_reward + 0.3 * psnr_reward
        
        # Penalty for unnecessary NN usage (if SNR >> threshold, using NN is inefficient)
        if use_nn and snr_threshold is not None:
            # This penalty will be applied in the environment
            pass
        
        return reward
    
    def store_transition(self, state, action, reward, next_state, done):
        """Store transition in replay buffer"""
        self.memory.push(state, action, reward, next_state, done)
    
    def train_step(self):
        """Train the DQN on a batch of experiences"""
        if len(self.memory) < self.batch_size:
            return None
        
        # Sample batch from replay buffer
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # Normalize states
        states_norm = np.array([self.normalize_state(s) for s in states])
        next_states_norm = np.array([self.normalize_state(s) for s in next_states])
        
        # Convert to tensors
        states_tensor = torch.FloatTensor(states_norm).to(self.device)
        actions_tensor = torch.LongTensor(actions).to(self.device)
        rewards_tensor = torch.FloatTensor(rewards).to(self.device)
        next_states_tensor = torch.FloatTensor(next_states_norm).to(self.device)
        dones_tensor = torch.FloatTensor(dones).to(self.device)
        
        # Current Q values
        q_values = self.q_network(states_tensor)
        q_value = q_values.gather(1, actions_tensor.unsqueeze(1)).squeeze(1)
        
        # Next Q values from target network
        with torch.no_grad():
            next_q_values = self.target_network(next_states_tensor)
            next_q_value = next_q_values.max(1)[0]
            target_q_value = rewards_tensor + (1 - dones_tensor) * self.gamma * next_q_value
        
        # Compute loss
        loss = nn.MSELoss()(q_value, target_q_value)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        # Update target network
        self.update_counter += 1
        if self.update_counter % self.target_update == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Decay epsilon
        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay
        
        return loss.item()
    
    def save_model(self, filepath):
        """Save the Q-network state"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'snr_thresholds': self.snr_thresholds,
        }, filepath)
        print(f"RL model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load the Q-network state"""
        if os.path.exists(filepath):
            checkpoint = torch.load(filepath, map_location=self.device)
            self.q_network.load_state_dict(checkpoint['q_network'])
            self.target_network.load_state_dict(checkpoint['target_network'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.epsilon = checkpoint.get('epsilon', self.epsilon_end)
            self.snr_thresholds = checkpoint.get('snr_thresholds', self.snr_thresholds)
            print(f"RL model loaded from {filepath}")
        else:
            print(f"No RL model found at {filepath}, starting from scratch")

