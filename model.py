import torch
import torch.nn as nn

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attention = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, lstm_out):
        # Calculate attention scores for each time step
        attn_scores = self.attention(lstm_out) # (batch, seq_len, 1)
        attn_weights = torch.softmax(attn_scores, dim=1) 
        
        # Multiply weights by LSTM output to get the context vector
        context_vector = torch.sum(attn_weights * lstm_out, dim=1) # (batch, hidden_size)
        return context_vector, attn_weights

class EnhancedRainfallLSTM(nn.Module):
    def __init__(self, actual_input_size=19, legacy_input_size=11, hidden_size=64, num_layers=2, dropout=0.2):
        super(EnhancedRainfallLSTM, self).__init__()
        
        # Dynamic Wrapper to satisfy constraint: maps new features down to the original input shape
        self.feature_proj = nn.Linear(actual_input_size, legacy_input_size)
        
        # Upgrade to BiLSTM
        self.lstm = nn.LSTM(legacy_input_size, hidden_size, num_layers, 
                            batch_first=True, dropout=dropout if num_layers > 1 else 0,
                            bidirectional=True)
        
        # Attention on top of BiLSTM (hidden_size * 2 because it's bidirectional)
        self.attention = Attention(hidden_size * 2)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, 1)
        
    def forward(self, x):
        # 1. Project to original shape dimension
        x = self.feature_proj(x)
        
        # 2. BiLSTM extraction
        out, _ = self.lstm(x)
        
        # 3. Attention mechanism
        context, attn_weights = self.attention(out)
        
        # 4. Final prediction
        out = self.dropout(context)
        out = self.fc(out)
        return torch.relu(out)

if __name__ == '__main__':
    # Test instantiation
    model = EnhancedRainfallLSTM()
    print("Model architecture:")
    print(model)
    
    # Test with dummy data
    dummy_input = torch.randn(32, 24, 19) # Batch size: 32, Seq Len: 24, Features: 19
    dummy_output = model(dummy_input)
    print(f"Output shape: {dummy_output.shape}")
