import torch
import pickle
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from model import INSAT_Rainfall_XAI_LSTM
from train_and_evaluate import compute_meteorological_scores

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

X_train = torch.load('X_train.pt', weights_only=False)
y_train = torch.load('y_train.pt', weights_only=False)
X_val = torch.load('X_val.pt', weights_only=False)
y_val = torch.load('y_val.pt', weights_only=False)

model = INSAT_Rainfall_XAI_LSTM(input_size=11, hidden_size=64, num_layers=1, dropout=0.3).to(device)
model.load_state_dict(torch.load('model.pth', map_location=device, weights_only=False))
model.eval()

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

def get_loss(X, y):
    loader = DataLoader(TensorDataset(X, y), batch_size=256, shuffle=False)
    criterion = torch.nn.HuberLoss(delta=1.0)
    total_loss = 0.0
    with torch.no_grad():
        for X_b, y_b in loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            preds, _ = model(X_b)
            # weights = torch.where(y_b > 0.005, 5.0, 1.0) # approx
            loss = criterion(preds, y_b)
            total_loss += loss.item() * X_b.size(0)
    return total_loss / len(X)

train_loss = get_loss(X_train, y_train)
val_loss = get_loss(X_val, y_val)
print(f"Train Huber Loss: {train_loss:.6f}")
print(f"Val Huber Loss: {val_loss:.6f}")

if train_loss < (val_loss * 0.5):
    print("STATUS: OVERFITTED (Train loss is significantly lower than Val loss)")
elif abs(train_loss - val_loss) < 0.001:
    print("STATUS: NOT OVERFITTED (Losses are close)")
else:
    print("STATUS: UNKNOWN")
