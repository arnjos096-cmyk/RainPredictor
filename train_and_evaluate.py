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
from model import INSAT_Rainfall_XAI_LSTM

def compute_meteorological_scores(y_true, y_pred, threshold_mm=35.5):
    """
    Computes IMD / ISRO standard verification metrics for Heavy Rainfall Nowcasting:
    - Hits (TP), False Alarms (FP), Misses (FN), Correct Negatives (TN)
    - POD (Probability of Detection) = TP / (TP + FN)
    - FAR (False Alarm Ratio) = FP / (TP + FP)
    - CSI (Critical Success Index / Threat Score) = TP / (TP + FP + FN)
    - ETS (Equitable Threat Score)
    - F1-Score & MAE
    """
    y_true_bin = (y_true >= threshold_mm).astype(int)
    y_pred_bin = (y_pred >= threshold_mm).astype(int)
    
    tp = np.sum((y_true_bin == 1) & (y_pred_bin == 1))
    fp = np.sum((y_true_bin == 0) & (y_pred_bin == 1))
    fn = np.sum((y_true_bin == 1) & (y_pred_bin == 0))
    tn = np.sum((y_true_bin == 0) & (y_pred_bin == 0))
    total = len(y_true_bin)
    
    pod = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    far = fp / (tp + fp) if (tp + fp) > 0 else 0.0
    csi = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    
    # Random chance hits for ETS
    ar = ((tp + fp) * (tp + fn)) / total if total > 0 else 0.0
    ets = (tp - ar) / (tp + fp + fn - ar) if (tp + fp + fn - ar) > 0 else 0.0
    
    f1 = f1_score(y_true_bin, y_pred_bin, zero_division=0)
    mae = mean_absolute_error(y_true, y_pred)
    
    return {
        "TP": int(tp),
        "FP": int(fp),
        "FN": int(fn),
        "TN": int(tn),
        "POD": float(pod),
        "FAR": float(far),
        "CSI": float(csi),
        "ETS": float(ets),
        "F1": float(f1),
        "MAE": float(mae)
    }

def train_model(model, train_loader, val_loader, optimizer, num_epochs=12, device='cpu'):
    model.to(device)
    train_losses = []
    val_losses = []
    
    print(f"Training INSAT_Rainfall_XAI_LSTM with Convective Weighted Loss for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs, _ = model(X_batch)
            
            # Positive sample weighting for heavy & extreme rainfall events
            weights = torch.where(y_batch > 0.005, 25.0, 1.0)
            loss = torch.mean(weights * ((outputs - y_batch) ** 2))
            
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
                outputs, _ = model(X_batch)
                weights = torch.where(y_batch > 0.005, 25.0, 1.0)
                loss = torch.mean(weights * ((outputs - y_batch) ** 2))
                running_val_loss += loss.item() * X_batch.size(0)
                
        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        val_losses.append(epoch_val_loss)
        
        print(f"Epoch [{epoch+1:02d}/{num_epochs:02d}] Train Loss: {epoch_train_loss:.6f} | Val Loss: {epoch_val_loss:.6f}")
        
    return train_losses, val_losses

def evaluate_full_dataset(model, X_val, y_val, scaler, target_col_idx=10, device='cpu'):
    print("\n--- Running Full Validation Set Meteorological Evaluation ---")
    model.eval()
    
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=256, shuffle=False)
    all_preds = []
    
    with torch.no_grad():
        for X_batch, _ in val_loader:
            X_batch = X_batch.to(device)
            preds, _ = model(X_batch)
            all_preds.append(preds.cpu().numpy())
            
    y_pred_scaled = np.vstack(all_preds)
    y_true_scaled = y_val.cpu().numpy()
    
    # Inverse transform
    dummy_true = np.zeros((len(y_true_scaled), 11))
    dummy_pred = np.zeros((len(y_pred_scaled), 11))
    
    dummy_true[:, target_col_idx] = y_true_scaled.squeeze()
    dummy_pred[:, target_col_idx] = y_pred_scaled.squeeze()
    
    y_true_unscaled = np.maximum(0.0, scaler.inverse_transform(dummy_true)[:, target_col_idx])
    y_pred_unscaled = np.maximum(0.0, scaler.inverse_transform(dummy_pred)[:, target_col_idx])
    
    # Evaluation at Heavy Rain threshold (>35.5 mm/h)
    scores_heavy = compute_meteorological_scores(y_true_unscaled, y_pred_unscaled, threshold_mm=35.5)
    # General Rain threshold (>0.1 mm/h)
    scores_rain = compute_meteorological_scores(y_true_unscaled, y_pred_unscaled, threshold_mm=0.1)
    
    print("\n=======================================================")
    print("      ISRO SIH260006 METEOROLOGICAL VERIFICATION")
    print("=======================================================")
    print(f"Overall MAE (Precipitation Volume): {scores_rain['MAE']:.3f} mm/h")
    print(f"Rain / No-Rain F1-Score:            {scores_rain['F1']:.4f}")
    print("--- Heavy Rain Events (>35.5 mm/h) Verification ---")
    print(f"Probability of Detection (POD):    {scores_heavy['POD']:.4f}")
    print(f"False Alarm Ratio (FAR):           {scores_heavy['FAR']:.4f}")
    print(f"Critical Success Index (CSI):      {scores_heavy['CSI']:.4f}")
    print(f"Equitable Threat Score (ETS):      {scores_heavy['ETS']:.4f}")
    print(f"High-Impact F1-Score:              {scores_heavy['F1']:.4f}")
    print(f"Contingency Matrix: TP={scores_heavy['TP']}, FP={scores_heavy['FP']}, FN={scores_heavy['FN']}, TN={scores_heavy['TN']}")
    print("=======================================================\n")
    
    # Plot 7-Day Zoomed Slice (168 hours)
    start_idx = 1200
    slice_len = 168
    plt.figure(figsize=(14, 6), dpi=150)
    plt.plot(y_true_unscaled[start_idx:start_idx+slice_len], label='Actual INSAT Hydro-Estimator (mm/h)', color='#38bdf8', linewidth=2.2)
    plt.plot(y_pred_unscaled[start_idx:start_idx+slice_len], label='XAI BiLSTM Nowcast (mm/h)', color='#ef4444', linestyle='--', linewidth=2.0)
    plt.title('INSAT-3D/3DR Satellite Nowcasting: 7-Day Actual vs Predicted Rainfall Rate', fontsize=14, fontweight='bold', pad=12)
    plt.xlabel('Lead Time (Hours)', fontsize=12)
    plt.ylabel('Rainfall Intensity (mm/h)', fontsize=12)
    plt.legend(frameon=True, facecolor='#ffffff', edgecolor='#cbd5e1')
    plt.grid(True, alpha=0.3, linestyle=':')
    plt.tight_layout()
    plt.savefig('actual_vs_predicted.png')
    print("Saved actual_vs_predicted.png")
    
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load tensors
    print("Loading INSAT preprocessed tensors...")
    X_train = torch.load('X_train.pt', weights_only=False)
    y_train = torch.load('y_train.pt', weights_only=False)
    X_val = torch.load('X_val.pt', weights_only=False)
    y_val = torch.load('y_val.pt', weights_only=False)
    
    batch_size = 128
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize INSAT XAI Model
    model = INSAT_Rainfall_XAI_LSTM(input_size=11, hidden_size=64, num_layers=2, dropout=0.2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.002)
    
    train_losses, val_losses = train_model(model, train_loader, val_loader, optimizer, num_epochs=12, device=device)
    
    # Save Model Weights
    torch.save(model.state_dict(), 'model.pth')
    print("Saved model.pth successfully.")
    
    # Save Loss Curve
    plt.figure(figsize=(10, 5), dpi=150)
    plt.plot(train_losses, label='Train Weighted MSE Loss', color='#38bdf8', linewidth=2.0)
    plt.plot(val_losses, label='Validation Loss', color='#f59e0b', linewidth=2.0)
    plt.title('INSAT-3D/3DR XAI BiLSTM Training & Validation Loss Curve', fontsize=13, fontweight='bold', pad=10)
    plt.xlabel('Epochs', fontsize=11)
    plt.ylabel('Weighted MSE Loss', fontsize=11)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('loss_curve.png')
    print("Saved loss_curve.png")
    
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
        
    evaluate_full_dataset(model, X_val, y_val, scaler, target_col_idx=10, device=device)
    print("Training and Evaluation Pipeline Complete!")

if __name__ == '__main__':
    main()

