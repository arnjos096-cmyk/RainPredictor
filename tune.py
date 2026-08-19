import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from model import EnhancedRainfallLSTM

def objective(trial):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Suggest Hyperparameters dynamically
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    hidden_size = trial.suggest_categorical("hidden_size", [32, 64, 128])
    num_layers = trial.suggest_int("num_layers", 1, 3)
    
    # Load data (Assuming preprocessed data is loaded)
    X_train = torch.load('X_train.pt', weights_only=False)
    y_train = torch.load('y_train.pt', weights_only=False)
    X_val = torch.load('X_val.pt', weights_only=False)
    y_val = torch.load('y_val.pt', weights_only=False)
    
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=64, shuffle=False)
    
    # 2. Instantiate Architecture
    model = EnhancedRainfallLSTM(actual_input_size=19, legacy_input_size=11, 
                                 hidden_size=hidden_size, num_layers=num_layers, 
                                 dropout=dropout).to(device)
                                 
    criterion = nn.HuberLoss(delta=1.0)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # 3. Train for a few epochs to evaluate
    epochs = 3 
    for epoch in range(epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(X_batch.to(device)), y_batch.to(device))
            loss.backward()
            optimizer.step()
            
    # 4. Validation step
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            val_loss += criterion(model(X_batch.to(device)), y_batch.to(device)).item()
            
    return val_loss / len(val_loader)

if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=5) # Kept small for quick run
    
    print("\n=== Best Hyperparameters ===")
    print(study.best_params)
