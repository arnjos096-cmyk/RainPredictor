import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
import pickle
import os

INSAT_FEATURE_COLS = [
    'tir1_temp',
    'wv_channel',
    'cloud_top_height',
    'cape_index',
    'pressure',
    'humidity',
    'temperature',
    'moisture_conv',
    'wind_speed',
    'wind_shear',
    'rainfall_mm'
]

def create_sequences(data, target_col_idx=10, seq_length=24):
    """
    Creates sliding sequences of 24 hourly steps to predict the next time step's rainfall rate.
    data: np.array of shape (num_samples, 11)
    target_col_idx: index 10 (rainfall_mm)
    """
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length, target_col_idx]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def preprocess_data(input_file='synthetic_weather_data.csv', seq_length=24):
    print(f"Loading INSAT satellite dataset from {input_file}...")
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"{input_file} not found. Please run generate_data.py first.")
        
    df = pd.read_csv(input_file, index_col='date', parse_dates=True)
    
    # Ensure columns match standard INSAT feature ordering
    df = df[INSAT_FEATURE_COLS]
    target_col_idx = INSAT_FEATURE_COLS.index('rainfall_mm')
    
    # Scale data
    print("Fitting MinMaxScaler across 11 INSAT channels...")
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)
    
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print("Saved scaler.pkl successfully.")
        
    print(f"Creating 24-hour sliding windows (seq_length={seq_length})...")
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
    print("INSAT Data Preprocessing complete!")

if __name__ == '__main__':
    preprocess_data()

