import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from model import Peephole_Conv_LSTM

def objective(trial):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Suggest Hyperparameters dynamically
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    dropout = trial.suggest_float("dropout", 0.1, 0.4)
    hidden_size = trial.suggest_categorical("hidden_size", [32, 64, 128])
    # Locked to 1 layer because custom Peephole is not nested for multi-layer yet
    num_layers = 1 
    
    # Load data
    X_train = torch.load('X_train.pt', weights_only=False)
    y_train = torch.load('y_train.pt', weights_only=False)
    X_val = torch.load('X_val.pt', weights_only=False)
    y_val = torch.load('y_val.pt', weights_only=False)
    
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=256, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=256, shuffle=False)
    
    # 2. Instantiate Architecture
    model = Peephole_Conv_LSTM(input_size=11, 
                               hidden_size=hidden_size, 
                               num_layers=num_layers, 
                               dropout=dropout).to(device)
                                 
    criterion = nn.HuberLoss(delta=1.0)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # 3. Train for a few epochs to evaluate
    epochs = 2 
    for epoch in range(epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            preds, _ = model(X_batch)
            
            # Apply Convective Weighted Loss
            weights = torch.where(y_batch > 0.1, 10.0, 1.0)
            loss = (criterion(preds, y_batch) * weights).mean()
            
            loss.backward()
            optimizer.step()
            
    # 4. Validation step
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            preds, _ = model(X_batch)
            weights = torch.where(y_batch > 0.1, 10.0, 1.0)
            batch_loss = (criterion(preds, y_batch) * weights).mean()
            val_loss += batch_loss.item()
            
    return val_loss / len(val_loader)

if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=5) # Kept to 5 trials due to custom RNN loop speed
    
    print("\n=== Best Hyperparameters ===")
    print(study.best_params)
