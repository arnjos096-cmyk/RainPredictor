import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import pickle
import numpy as np
from sklearn.metrics import f1_score, mean_absolute_error
import pandas as pd
import os
from model import EnhancedRainfallLSTM

def custom_evaluate(y_true, y_pred, threshold_mm=0.1):
    y_true_bin = (y_true > threshold_mm).astype(int)
    y_pred_bin = (y_pred > threshold_mm).astype(int)
    
    f1 = f1_score(y_true_bin, y_pred_bin, zero_division=0)
    mae = mean_absolute_error(y_true, y_pred)
    return f1, mae

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, device='cpu'):
    model.to(device)
    train_losses = []
    val_losses = []
    
    print("Starting training...")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * X_batch.size(0)
            
        epoch_train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)
        
        # Validation
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                running_val_loss += loss.item() * X_batch.size(0)
                
        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        val_losses.append(epoch_val_loss)
        
        print(f"Epoch [{epoch+1}/{num_epochs}] Train Loss: {epoch_train_loss:.6f} | Val Loss: {epoch_val_loss:.6f}")
        
    return train_losses, val_losses

def evaluate_and_plot(model, X_val, y_val, scaler, target_col_idx=10, seq_length=24, device='cpu'):
    print("Evaluating 7-day slice...")
    model.eval()
    
    # 7 days = 168 hours
    slice_size = 168
    # Start somewhere in the validation set
    start_idx = 1000 
    
    X_slice = X_val[start_idx:start_idx+slice_size].to(device)
    y_slice_true = y_val[start_idx:start_idx+slice_size].cpu().numpy()
    
    with torch.no_grad():
        y_slice_pred = model(X_slice).cpu().numpy()
        
    # We need to inverse transform the scaled values back to real mm
    # Scaler expects shape (n_samples, n_features). We only care about target.
    # Create dummy arrays to use inverse_transform
    dummy_true = np.zeros((len(y_slice_true), 11))
    dummy_pred = np.zeros((len(y_slice_pred), 11))
    
    dummy_true[:, target_col_idx] = y_slice_true.squeeze()
    dummy_pred[:, target_col_idx] = y_slice_pred.squeeze()
    
    y_true_unscaled = scaler.inverse_transform(dummy_true)[:, target_col_idx]
    y_pred_unscaled = scaler.inverse_transform(dummy_pred)[:, target_col_idx]
    
    # Plot Actual vs Predicted
    plt.figure(figsize=(12, 6))
    plt.plot(y_true_unscaled, label='Actual Rainfall (mm)', alpha=0.8, color='blue', linewidth=2)
    plt.plot(y_pred_unscaled, label='Predicted Rainfall (mm)', alpha=0.8, color='red', linestyle='dashed', linewidth=2)
    plt.title('7-Day Slice: Actual vs Predicted Rainfall')
    plt.xlabel('Hours')
    plt.ylabel('Rainfall (mm)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('actual_vs_predicted.png')
    print("Saved actual_vs_predicted.png")
    
    f1, mae = custom_evaluate(y_true_unscaled, y_pred_unscaled)
    print(f"\n--- Model Evaluation ---")
    print(f"Validation F1-Score (Rain/No Rain): {f1:.4f}")
    print(f"Validation MAE (Volume): {mae:.4f} mm\n")
    
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load tensors
    print("Loading data...")
    X_train = torch.load('X_train.pt')
    y_train = torch.load('y_train.pt')
    X_val = torch.load('X_val.pt')
    y_val = torch.load('y_val.pt')
    
    # Create DataLoaders
    batch_size = 64
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Model Setup
    model = EnhancedRainfallLSTM(actual_input_size=19, legacy_input_size=11, hidden_size=64, num_layers=2, dropout=0.2)
    criterion = nn.HuberLoss(delta=1.0)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train
    train_losses, val_losses = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, device=device)
    
    # Save Loss Curve
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss_curve.png')
    print("Saved loss_curve.png")
    
    # Load Scaler to inverse transform
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
        
    evaluate_and_plot(model, X_val, y_val, scaler, target_col_idx=10, seq_length=24, device=device)
    print("Pipeline complete.")

if __name__ == '__main__':
    main()
