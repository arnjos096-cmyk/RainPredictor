import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

# ==========================================
# 1. THE NEURAL NETWORK (THE DQN BRAIN)
# ==========================================
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_size)
            # No activation at the end, we want raw Q-Values
        )

    def forward(self, x):
        return self.network(x)


# ==========================================
# 2. THE REPLAY BUFFER (THE MEMORY)
# ==========================================
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)

    def __len__(self):
        return len(self.buffer)


# ==========================================
# 3. THE ENVIRONMENT (THE REAL WORLD)
# ==========================================
class WeatherDecisionEnv:
    def __init__(self):
        self.state_size = 12 # 11 satellite features + 1 LSTM predicted rainfall
        self.action_size = 4
        """
        Actions:
        0: Normal Operations
        1: Agricultural Delay (Shut off sprinklers)
        2: Grid Backup (Spin up coal generators)
        3: Disaster Evacuation
        """

    def step(self, action, true_rainfall, predicted_rainfall, cloud_top_height, cape):
        """
        Calculates the reward based on the chosen action and the ACTUAL weather that occurred.
        """
        reward = 0
        
        if action == 0: # Normal Operations
            if true_rainfall > 35:
                reward = -100 # Disaster! We did nothing during a massive storm
            elif true_rainfall > 10:
                reward = -20 # Heavy rain, we should have warned people
            else:
                reward = 1 # Everything is normal, good job doing nothing
                
        elif action == 1: # Agriculture Delay
            if 0.5 < true_rainfall < 15:
                reward = 15 # Perfect! We saved water because it rained gently
            elif true_rainfall <= 0.5:
                reward = -5 # We shut off sprinklers but it didn't rain (crops dry)
            else:
                reward = -10 # It flooded, agriculture delay was irrelevant
                
        elif action == 2: # Grid Backup
            if cloud_top_height > 10 and true_rainfall > 5:
                reward = 25 # Saved the grid! Solar panels were blocked but we had backup
            else:
                reward = -15 # Wasted money spinning up coal generators on a sunny day
                
        elif action == 3: # Disaster Evacuation
            if true_rainfall > 35:
                reward = 100 # Saved lives! Excellent decision
            elif true_rainfall > 20:
                reward = 10 # Better safe than sorry
            else:
                reward = -50 # Massive false alarm! Cost the city millions

        return float(reward)


# ==========================================
# 4. THE DQN AGENT
# ==========================================
class DQNAgent:
    def __init__(self, state_size, action_size, device='cpu'):
        self.state_size = state_size
        self.action_size = action_size
        self.device = device
        
        # Hyperparameters
        self.gamma = 0.99           # Discount factor
        self.epsilon = 1.0          # Exploration rate
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.learning_rate = 1e-3
        self.batch_size = 64
        
        # Networks
        self.policy_net = DQN(state_size, action_size).to(device)
        self.target_net = DQN(state_size, action_size).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.memory = ReplayBuffer(capacity=50000)

    def select_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size) # Explore
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
        return torch.argmax(q_values).item() # Exploit

    def update(self):
        if len(self.memory) < self.batch_size:
            return 0
            
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        # Compute current Q values
        current_q_values = self.policy_net(states).gather(1, actions)
        
        # Compute next max Q values from Target Network
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0].unsqueeze(1)
            
        # Bellman Equation
        expected_q_values = rewards + (self.gamma * next_q_values * (1 - dones))
        
        # Loss and Backprop
        criterion = nn.MSELoss()
        loss = criterion(current_q_values, expected_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
