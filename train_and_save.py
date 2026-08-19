import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import pickle
import os
from model import EnhancedRainfallLSTM

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("Loading preprocessed tensors...")
    X_train = torch.load('X_train.pt', weights_only=False)
    y_train = torch.load('y_train.pt', weights_only=False)
    X_val = torch.load('X_val.pt', weights_only=False)
    y_val = torch.load('y_val.pt', weights_only=False)
    
    batch_size = 128
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    model = EnhancedRainfallLSTM(actual_input_size=11, legacy_input_size=11, hidden_size=64, num_layers=2, dropout=0.2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.002)
    
    num_epochs = 15
    print(f"Training EnhancedRainfallLSTM with positive rain sample weighting for {num_epochs} epochs...")
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        running_train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            
            # Weighted loss: positive rain instances weighted 20x to counter extreme class imbalance
            weights = torch.where(y_batch > 0.005, 25.0, 1.0)
            loss = torch.mean(weights * ((outputs - y_batch) ** 2))
            
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item() * X_batch.size(0)
            
        train_loss = running_train_loss / len(train_loader.dataset)
        
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                weights = torch.where(y_batch > 0.005, 25.0, 1.0)
                loss = torch.mean(weights * ((outputs - y_batch) ** 2))
                running_val_loss += loss.item() * X_batch.size(0)
                
        val_loss = running_val_loss / len(val_loader.dataset)
        print(f"Epoch [{epoch+1}/{num_epochs}] Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'model.pth')
            
    print(f"Training complete! Best Val Loss: {best_val_loss:.6f}. Saved model.pth successfully.")

if __name__ == '__main__':
    main()
