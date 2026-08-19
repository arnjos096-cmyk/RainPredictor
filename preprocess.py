import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
import pickle
import os

def create_sequences(data, target_col_idx, seq_length=24):
    """
    Creates sequences of length seq_length to predict the next time step's target.
    data: np.array of shape (num_samples, num_features)
    target_col_idx: the index of the column we want to predict
    """
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length, target_col_idx]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def preprocess_data(input_file='synthetic_weather_data.csv', seq_length=24):
    print(f"Loading data from {input_file}...")
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"{input_file} not found. Please run generate_data.py first.")
        
    df = pd.read_csv(input_file, index_col='date', parse_dates=True)
    
    # 1. Temporal Embeddings
    df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24.0)
    df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24.0)
    
    days_in_year = 365.25
    df['day_sin'] = np.sin(2 * np.pi * df.index.dayofyear / days_in_year)
    df['day_cos'] = np.cos(2 * np.pi * df.index.dayofyear / days_in_year)
    
    # 2. Rolling Averages (Trend Momentum)
    df['pressure_roll_3'] = df['pressure'].rolling(window=3).mean().bfill()
    df['pressure_roll_6'] = df['pressure'].rolling(window=6).mean().bfill()
    df['humidity_roll_3'] = df['humidity'].rolling(window=3).mean().bfill()
    df['humidity_roll_6'] = df['humidity'].rolling(window=6).mean().bfill()

    # Move target column to the end for easier indexing
    target_col = 'rainfall_mm'
    features = [c for c in df.columns if c != target_col] + [target_col]
    df = df[features]
    target_col_idx = len(features) - 1
    
    # Scale data
    print("Scaling data...")
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)
    
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
        
    print(f"Creating sliding windows of length {seq_length}...")
    X, y = create_sequences(scaled_data, target_col_idx, seq_length)
    
    # Train / Validation Split (80% / 20%)
    split_idx = int(len(X) * 0.8)
    
    X_train, y_train = X[:split_idx], y[:split_idx]
    X_val, y_val = X[split_idx:], y[split_idx:]
    
    # Convert to PyTorch tensors
    print("Converting to PyTorch tensors...")
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
    
    # Save tensors
    print("Saving tensors...")
    torch.save(X_train_tensor, 'X_train.pt')
    torch.save(y_train_tensor, 'y_train.pt')
    torch.save(X_val_tensor, 'X_val.pt')
    torch.save(y_val_tensor, 'y_val.pt')
    
    print(f"Train shapes: X={X_train_tensor.shape}, y={y_train_tensor.shape}")
    print(f"Val shapes: X={X_val_tensor.shape}, y={y_val_tensor.shape}")
    print("Preprocessing complete!")

if __name__ == '__main__':
    preprocess_data()
