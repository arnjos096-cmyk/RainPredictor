import numpy as np

# ==============================================================================
# Pure NumPy Inference Engine (Zero-PyTorch dependency for serverless/Vercel)
# ==============================================================================

class NumpyPeepholeConvLSTM:
    """
    High-Performance Pure NumPy Implementation of Causal Peephole Conv-LSTM with Attention.
    Matches PyTorch forward and explainability outputs to float precision without torch/CUDA bloat.
    """
    def __init__(self, weights_dict=None):
        self.w = weights_dict or {}
        self.input_size = 11
        self.hidden_size = 64

    def load_weights(self, weights_path_or_dict):
        if isinstance(weights_path_or_dict, str):
            data = np.load(weights_path_or_dict)
            self.w = {k: data[k] for k in data.files}
        elif isinstance(weights_path_or_dict, dict):
            self.w = weights_path_or_dict

    def _sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))

    def _tanh(self, x):
        return np.tanh(x)

    def _relu(self, x):
        return np.maximum(0, x)

    def _softmax(self, x, axis=-1):
        e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return e_x / np.sum(e_x, axis=axis, keepdims=True)

    def forward(self, x):
        # x shape: (batch, seq_len=24, channels=11)
        if x.ndim == 2:
            x = x[np.newaxis, :, :]
            
        batch_size, seq_len, in_channels = x.shape
        hidden_size = self.hidden_size
        
        # 1. Causal Conv1D: kernel_size=3, padding=2
        x_p = np.pad(x.transpose(0, 2, 1), ((0, 0), (0, 0), (2, 0)), mode='constant') # (batch, 11, seq_len+2)
        
        W_conv = self.w['conv1d.conv.weight'] # (64, 11, 3)
        b_conv = self.w['conv1d.conv.bias']   # (64,)
        
        conv_out = np.zeros((batch_size, hidden_size, seq_len), dtype=np.float32)
        for t in range(seq_len):
            window = x_p[:, :, t:t+3] # (batch, 11, 3)
            conv_out[:, :, t] = np.tensordot(window, W_conv, axes=([1, 2], [1, 2])) + b_conv
            
        conv_out = conv_out.transpose(0, 2, 1) # (batch, seq_len, 64)
        
        # 2. Peephole LSTM
        W_f = self.w['peephole_cell.W_f.weight'] # (64, 128)
        b_f = self.w['peephole_cell.W_f.bias']   # (64,)
        V_f = self.w['peephole_cell.V_f']        # (64,)
        
        W_i = self.w['peephole_cell.W_i.weight'] # (64, 128)
        b_i = self.w['peephole_cell.W_i.bias']   # (64,)
        V_i = self.w['peephole_cell.V_i']        # (64,)
        
        W_c = self.w['peephole_cell.W_c.weight'] # (64, 128)
        b_c = self.w['peephole_cell.W_c.bias']   # (64,)
        
        W_o = self.w['peephole_cell.W_o.weight'] # (64, 128)
        b_o = self.w['peephole_cell.W_o.bias']   # (64,)
        V_o = self.w['peephole_cell.V_o']        # (64,)
        
        h_t = np.zeros((batch_size, hidden_size), dtype=np.float32)
        c_t = np.zeros((batch_size, hidden_size), dtype=np.float32)
        
        lstm_outputs = []
        for t in range(seq_len):
            x_t = conv_out[:, t, :] # (batch, 64)
            xh = np.concatenate([x_t, h_t], axis=1) # (batch, 128)
            
            f_t = self._sigmoid(xh @ W_f.T + b_f + V_f * c_t)
            i_t = self._sigmoid(xh @ W_i.T + b_i + V_i * c_t)
            c_tilde = self._tanh(xh @ W_c.T + b_c)
            
            c_t = f_t * c_t + i_t * c_tilde
            o_t = self._sigmoid(xh @ W_o.T + b_o + V_o * c_t)
            h_t = o_t * self._tanh(c_t)
            lstm_outputs.append(h_t[:, np.newaxis, :])
            
        lstm_out = np.concatenate(lstm_outputs, axis=1) # (batch, seq_len, 64)
        
        # 3. Temporal Attention
        W_att0 = self.w['attention.attention.0.weight'] # (32, 64)
        b_att0 = self.w['attention.attention.0.bias']   # (32,)
        W_att2 = self.w['attention.attention.2.weight'] # (1, 32)
        
        att_hidden = self._tanh(lstm_out @ W_att0.T + b_att0) # (batch, seq_len, 32)
        att_scores = att_hidden @ W_att2.T # (batch, seq_len, 1)
        attn_weights = self._softmax(att_scores, axis=1) # (batch, seq_len, 1)
        
        context = np.sum(attn_weights * lstm_out, axis=1) # (batch, 64)
        
        # 4. FC Linear + ReLU
        W_fc = self.w['fc.weight'] # (1, 64)
        b_fc = self.w['fc.bias']   # (1,)
        
        out = context @ W_fc.T + b_fc
        pred = self._relu(out)
        return pred, attn_weights.squeeze(-1)

    def explain_instance(self, x_np, feature_names=None):
        if x_np.ndim == 2:
            x_np = x_np[np.newaxis, :, :] # (1, 24, 11)
            
        pred, attn_weights = self.forward(x_np)
        pred_val = float(pred[0, 0])
        temporal_importance = attn_weights[0] # (24,)
        
        # Numerical Gradient Estimation for Explainable Feature Attributions
        eps = 1e-4
        grads = np.zeros_like(x_np)
        for t in range(x_np.shape[1]):
            for c in range(x_np.shape[2]):
                x_plus = x_np.copy()
                x_plus[0, t, c] += eps
                pred_plus, _ = self.forward(x_plus)
                
                x_minus = x_np.copy()
                x_minus[0, t, c] -= eps
                pred_minus, _ = self.forward(x_minus)
                
                grads[0, t, c] = abs((pred_plus[0, 0] - pred_minus[0, 0]) / (2 * eps))
                
        weighted_grads = grads[0] * temporal_importance[:, None]
        feature_importance_raw = np.sum(weighted_grads, axis=0)
        total_sum = np.sum(feature_importance_raw)
        
        if total_sum > 0:
            feature_importance_pct = (feature_importance_raw / total_sum) * 100.0
        else:
            feature_importance_pct = np.ones(x_np.shape[2]) * (100.0 / x_np.shape[2])
            
        return {
            "predicted_rainfall_mm": pred_val,
            "temporal_attention": [round(float(w), 4) for w in temporal_importance],
            "feature_importance": [round(float(pct), 2) for pct in feature_importance_pct]
        }


# ==============================================================================
# PyTorch Architecture Definition (Preserved for offline training & research)
# ==============================================================================
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class TemporalAttention(nn.Module):
        def __init__(self, hidden_size):
            super(TemporalAttention, self).__init__()
            self.attention = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.Tanh(),
                nn.Linear(hidden_size // 2, 1, bias=False)
            )

        def forward(self, lstm_out):
            attn_scores = self.attention(lstm_out)
            attn_weights = torch.softmax(attn_scores, dim=1)
            context_vector = torch.sum(attn_weights * lstm_out, dim=1)
            return context_vector, attn_weights.squeeze(-1)

    class CausalConv1d(nn.Module):
        def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
            super(CausalConv1d, self).__init__()
            self.padding = (kernel_size - 1) * dilation
            self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=self.padding, dilation=dilation)

        def forward(self, x):
            x = self.conv(x)
            if self.padding > 0:
                x = x[:, :, :-self.padding]
            return x

    class PeepholeLSTMCell(nn.Module):
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
        def __init__(self, input_size=11, hidden_size=64, num_layers=1, dropout=0.2):
            super(Peephole_Conv_LSTM, self).__init__()
            self.input_size = input_size
            self.hidden_size = hidden_size
            
            self.conv1d = CausalConv1d(in_channels=input_size, out_channels=hidden_size, kernel_size=3)
            self.peephole_cell = PeepholeLSTMCell(input_size=hidden_size, hidden_size=hidden_size)
            self.attention = TemporalAttention(hidden_size)
            self.dropout = nn.Dropout(dropout)
            self.fc = nn.Linear(hidden_size, 1)

        def forward(self, x):
            batch_size, seq_len, _ = x.size()
            x_conv = x.permute(0, 2, 1)
            conv_out = self.conv1d(x_conv)
            conv_out = conv_out.permute(0, 2, 1)
            
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

    INSAT_Rainfall_XAI_LSTM = Peephole_Conv_LSTM
    EnhancedRainfallLSTM = Peephole_Conv_LSTM
    RainfallLSTM = Peephole_Conv_LSTM

except ImportError:
    # PyTorch not installed in runtime environment (e.g. Vercel)
    Peephole_Conv_LSTM = NumpyPeepholeConvLSTM
    INSAT_Rainfall_XAI_LSTM = NumpyPeepholeConvLSTM
    EnhancedRainfallLSTM = NumpyPeepholeConvLSTM
    RainfallLSTM = NumpyPeepholeConvLSTM
