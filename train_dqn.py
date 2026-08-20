import torch
import numpy as np
import pickle
from torch.utils.data import DataLoader, TensorDataset
from model import INSAT_Rainfall_XAI_LSTM
from dqn_agent import DQNAgent, WeatherDecisionEnv
import matplotlib.pyplot as plt

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Load the frozen LSTM
    print("Loading pre-trained INSAT LSTM...")
    lstm = INSAT_Rainfall_XAI_LSTM(input_size=11, hidden_size=64, num_layers=2, dropout=0.2).to(device)
    lstm.load_state_dict(torch.load('model.pth', map_location=device, weights_only=False))
    lstm.eval()

    # 2. Load the Dataset and Scaler
    print("Loading data...")
    X_train = torch.load('X_train.pt', weights_only=False)
    y_train = torch.load('y_train.pt', weights_only=False)
    
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)

    # We will simulate the environment by iterating through the dataset sequentially
    # Treat the sequence of X_train as a timeline.
    
    # Extract features from scaled dataset directly
    # The scaler expects 11 features.
    
    # 3. Initialize DQN Agent and Environment
    env = WeatherDecisionEnv()
    agent = DQNAgent(state_size=env.state_size, action_size=env.action_size, device=device)
    
    episodes = 5
    batch_size = 256 # Data chunk size for simulation
    loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=False)
    
    rewards_history = []
    
    print("Starting DQN Training...")
    
    for e in range(episodes):
        total_reward = 0
        step_count = 0
        
        # Initial dummy state
        current_state = np.zeros(env.state_size)
        
        for X_b, y_b in loader:
            X_b = X_b.to(device)
            y_b = y_b.to(device)
            
            # Predict rainfall for the next hour using LSTM
            with torch.no_grad():
                preds, _ = lstm(X_b)
                
            preds = preds.cpu().numpy()
            y_true = y_b.cpu().numpy()
            X_b_np = X_b.cpu().numpy() # Shape: (batch, 24, 11)
            
            # Inverse transform predictions and true values to get actual mm/h
            dummy_pred = np.zeros((len(preds), 11))
            dummy_true = np.zeros((len(y_true), 11))
            dummy_pred[:, 10] = preds.squeeze()
            dummy_true[:, 10] = y_true.squeeze()
            
            preds_mm = np.maximum(0.0, scaler.inverse_transform(dummy_pred)[:, 10])
            true_mm = np.maximum(0.0, scaler.inverse_transform(dummy_true)[:, 10])
            
            # Unscale the 24th hour satellite features to get actual cloud_top_height and cape
            last_hour_features = X_b_np[:, -1, :] # (batch, 11)
            last_hour_unscaled = scaler.inverse_transform(last_hour_features)
            
            # Iterate through the batch to simulate hourly steps
            for i in range(len(preds)):
                # Construct state: 11 satellite features from the current hour + 1 predicted rainfall
                next_state = np.zeros(env.state_size)
                next_state[:11] = last_hour_features[i] # Keep scaled features for Neural Net stability
                next_state[11] = preds_mm[i] / 150.0 # Normalize prediction
                
                # DQN selects an action based on current state
                action = agent.select_action(current_state)
                
                # Environment calculates reward based on the action and the TRUE weather that happened
                cth = last_hour_unscaled[i, 2] # Cloud top height (index 2)
                cape = last_hour_unscaled[i, 3] # CAPE (index 3)
                
                reward = env.step(action, true_mm[i], preds_mm[i], cth, cape)
                total_reward += reward
                
                # In this continuous simulation, it's never really 'done' until the dataset ends
                done = (i == len(preds)-1 and step_count == len(loader)-1)
                
                # Store experience
                agent.memory.add(current_state, action, reward, next_state, done)
                
                # Train DQN
                agent.update()
                
                current_state = next_state
            
            step_count += 1
            
        agent.update_target_network()
        agent.decay_epsilon()
        
        print(f"Episode {e+1}/{episodes} - Total Reward: {total_reward:.1f} - Epsilon: {agent.epsilon:.3f}")
        rewards_history.append(total_reward)
        
    print("DQN Training Complete!")
    
    # Save DQN
    torch.save(agent.policy_net.state_dict(), 'dqn_policy.pth')
    print("Saved dqn_policy.pth")
    
    # Plot Learning Curve
    plt.figure(figsize=(10, 5))
    plt.plot(rewards_history, label='Cumulative Reward', marker='o')
    plt.title('DQN Learning Curve (Reward over Episodes)')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.grid(True)
    plt.savefig('dqn_learning_curve.png')
    print("Saved dqn_learning_curve.png")

if __name__ == '__main__':
    main()
