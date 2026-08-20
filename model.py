import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class TemporalAttention(nn.Module):
    """
    Temporal Self-Attention mechanism that calculates importance weights for each hourly timestep.
    Provides temporal explainability (XAI) by revealing which past hours triggered the prediction.
    """
    def __init__(self, hidden_size):
        super(TemporalAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1, bias=False)
        )

    def forward(self, lstm_out):
        # lstm_out shape: (batch, seq_len, hidden_size)
        attn_scores = self.attention(lstm_out) # (batch, seq_len, 1)
        attn_weights = torch.softmax(attn_scores, dim=1) # (batch, seq_len, 1)
        context_vector = torch.sum(attn_weights * lstm_out, dim=1) # (batch, hidden_size)
        return context_vector, attn_weights.squeeze(-1) # (batch, hidden_size), (batch, seq_len)


class CausalConv1d(nn.Module):
    """
    1D Causal Convolution. 
    Pads the sequence on the left so the kernel never looks into the future.
    """
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super(CausalConv1d, self).__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=self.padding, dilation=dilation)

    def forward(self, x):
        # x is (batch, channels, seq_len)
        x = self.conv(x)
        if self.padding > 0:
            x = x[:, :, :-self.padding] # Remove right padding to ensure causality
        return x


class PeepholeLSTMCell(nn.Module):
    """
    Custom LSTM Cell with Peephole connections.
    Gates (Forget, Input, Output) can directly look at the memory cell state (c_t).
    """
    def __init__(self, input_size, hidden_size):
        super(PeepholeLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        self.W_f = nn.Linear(input_size + hidden_size, hidden_size)
        self.V_f = nn.Parameter(torch.Tensor(hidden_size))
        
        self.W_i = nn.Linear(input_size + hidden_size, hidden_size)
        self.V_i = nn.Parameter(torch.Tensor(hidden_size))
        
        self.W_c = nn.Linear(input_size + hidden_size, hidden_size)
        
        self.W_o = nn.Linear(input_size + hidden_size, hidden_size)
        self.V_o = nn.Parameter(torch.Tensor(hidden_size))
        
        self.init_weights()

    def init_weights(self):
        for name, param in self.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_uniform_(param)
        nn.init.uniform_(self.V_f, -0.1, 0.1)
        nn.init.uniform_(self.V_i, -0.1, 0.1)
        nn.init.uniform_(self.V_o, -0.1, 0.1)
        # Forget gate bias = 1.0
        nn.init.constant_(self.W_f.bias, 1.0)

    def forward(self, x, states):
        h_prev, c_prev = states
        
        xh = torch.cat([x, h_prev], dim=1)
        
        f_t = torch.sigmoid(self.W_f(xh) + self.V_f * c_prev)
        i_t = torch.sigmoid(self.W_i(xh) + self.V_i * c_prev)
        c_tilde = torch.tanh(self.W_c(xh))
        
        c_t = f_t * c_prev + i_t * c_tilde
        
        o_t = torch.sigmoid(self.W_o(xh) + self.V_o * c_t)
        h_t = o_t * torch.tanh(c_t)
        
        return h_t, c_t


class Peephole_Conv_LSTM(nn.Module):
    """
    Causal Real-Time Nowcasting Architecture:
    1D Causal Convolution -> Peephole LSTM -> Temporal Attention
    """
    def __init__(self, input_size=11, hidden_size=64, num_layers=1, dropout=0.2):
        super(Peephole_Conv_LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # 1. Local Pattern Extraction (Causal)
        self.conv1d = CausalConv1d(in_channels=input_size, out_channels=hidden_size, kernel_size=3)
        
        # 2. Peephole Recurrence
        self.peephole_cell = PeepholeLSTMCell(input_size=hidden_size, hidden_size=hidden_size)
        
        # 3. Unidirectional Attention
        self.attention = TemporalAttention(hidden_size)
        self.dropout = nn.Dropout(dropout)
        
        # Output Regression
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        
        x_conv = x.permute(0, 2, 1)
        conv_out = self.conv1d(x_conv) # (batch, hidden_size, seq_len)
        conv_out = conv_out.permute(0, 2, 1) # Back to (batch, seq_len, hidden_size)
        
        device = x.device
        h_t = torch.zeros(batch_size, self.hidden_size).to(device)
        c_t = torch.zeros(batch_size, self.hidden_size).to(device)
        
        outputs = []
        for t in range(seq_len):
            h_t, c_t = self.peephole_cell(conv_out[:, t, :], (h_t, c_t))
            outputs.append(h_t.unsqueeze(1))
            
        lstm_out = torch.cat(outputs, dim=1)
        
        context, attn_weights = self.attention(lstm_out)
        out = self.dropout(context)
        out = self.fc(out)
        return torch.relu(out), attn_weights

    def explain_instance(self, x_single_tensor, feature_names=None):
        self.eval()
        x = x_single_tensor.clone().detach().requires_grad_(True)
        if x.dim() == 2:
            x = x.unsqueeze(0)
            
        pred, attn_weights = self.forward(x)
        
        pred.backward()
        gradients = x.grad.data.abs()
        
        temporal_importance = attn_weights.detach().cpu().numpy()[0]
        weighted_grads = gradients[0].cpu().numpy() * temporal_importance[:, None]
        feature_importance_raw = np.sum(weighted_grads, axis=0)
        
        total_sum = np.sum(feature_importance_raw)
        if total_sum > 0:
            feature_importance_pct = (feature_importance_raw / total_sum) * 100.0
        else:
            feature_importance_pct = np.ones(self.input_size) * (100.0 / self.input_size)
            
        return {
            "predicted_rainfall_mm": float(pred.item()),
            "temporal_attention": [round(float(w), 4) for w in temporal_importance],
            "feature_importance": [round(float(pct), 2) for pct in feature_importance_pct]
        }

# Aliases
INSAT_Rainfall_XAI_LSTM = Peephole_Conv_LSTM
EnhancedRainfallLSTM = Peephole_Conv_LSTM
RainfallLSTM = Peephole_Conv_LSTM

if __name__ == '__main__':
    model = Peephole_Conv_LSTM()
    print("Peephole Conv1D LSTM Initialized:")
    print(model)
    dummy_x = torch.randn(1, 24, 11)
    explanation = model.explain_instance(dummy_x)
    print("Test Output Pred mm:", explanation["predicted_rainfall_mm"])
