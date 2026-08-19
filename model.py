import torch
import torch.nn as nn

class RainfallLSTM(nn.Module):
    def __init__(self, input_size=11, hidden_size=64, num_layers=2, dropout=0.2):
        super(RainfallLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layer
        # batch_first=True means the input should be (batch_size, seq_len, input_size)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                            batch_first=True, dropout=dropout if num_layers > 1 else 0)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Output layer to predict a single continuous value (rainfall_mm)
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        # Initialize hidden state and cell state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        # out: tensor of shape (batch_size, seq_length, hidden_size)
        out, _ = self.lstm(x, (h0, c0))
        
        # Decode the hidden state of the last time step
        # out[:, -1, :] gets the last time step across the sequence
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        
        # Since we scaled rainfall with MinMaxScaler (0-1), we can use a ReLU 
        # to ensure no negative rain predictions, though the model might naturally learn this.
        out = torch.relu(out)
        return out

if __name__ == '__main__':
    # Test instantiation
    model = RainfallLSTM()
    print("Model architecture:")
    print(model)
    
    # Test with dummy data
    dummy_input = torch.randn(32, 24, 11) # Batch size: 32, Seq Len: 24, Features: 11
    dummy_output = model(dummy_input)
    print(f"Output shape: {dummy_output.shape}")
