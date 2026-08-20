import torch
import torch.nn as nn
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


class INSAT_Rainfall_XAI_LSTM(nn.Module):
    """
    ISRO SIH260006 Explainable AI Model for High-Impact Rainfall Nowcasting using INSAT-3D/3DR Satellite Data.
    Features:
    - Dynamic feature projection layer supporting 11 INSAT/meteorological channels
    - Bidirectional LSTM for synoptic temporal feature extraction
    - Temporal Attention for hourly timeline explainability
    - Feature Attribution module for XAI predictor ranking
    """
    def __init__(self, input_size=11, hidden_size=64, num_layers=2, dropout=0.2):
        super(INSAT_Rainfall_XAI_LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # BiLSTM feature extraction
        self.lstm = nn.LSTM(
            input_size, 
            hidden_size, 
            num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Temporal attention over BiLSTM bidirectional hidden state (hidden_size * 2)
        self.attention = TemporalAttention(hidden_size * 2)
        self.dropout = nn.Dropout(dropout)
        
        # Output Regression Head (Predicts precipitation rate mm/hr)
        self.fc = nn.Linear(hidden_size * 2, 1)
        
    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        out, _ = self.lstm(x)
        context, attn_weights = self.attention(out)
        out = self.dropout(context)
        out = self.fc(out)
        return torch.relu(out), attn_weights

    def explain_instance(self, x_single_tensor, feature_names=None):
        """
        XAI Explainability Function:
        Computes both Temporal Attention Weights and Feature Attribution Scores.
        """
        self.eval()
        x = x_single_tensor.clone().detach().requires_grad_(True)
        if x.dim() == 2:
            x = x.unsqueeze(0) # (1, 24, 11)
            
        pred, attn_weights = self.forward(x)
        
        # Integrated / Saliency Gradient for Feature Attribution
        pred.backward()
        gradients = x.grad.data.abs() # (1, 24, 11)
        
        # Temporal attention array across 24 hours
        temporal_importance = attn_weights.detach().cpu().numpy()[0] # (24,)
        
        # Feature importance: weight gradient by attention across timesteps
        weighted_grads = gradients[0].cpu().numpy() * temporal_importance[:, None]
        feature_importance_raw = np.sum(weighted_grads, axis=0) # (11,)
        
        # Normalize to percentage sum = 100
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

# Aliases for backward compatibility
EnhancedRainfallLSTM = INSAT_Rainfall_XAI_LSTM
RainfallLSTM = INSAT_Rainfall_XAI_LSTM

if __name__ == '__main__':
    model = INSAT_Rainfall_XAI_LSTM()
    print("ISRO INSAT-3D/3DR XAI Architecture Initialized:")
    print(model)
    dummy_x = torch.randn(1, 24, 11)
    explanation = model.explain_instance(dummy_x)
    print("Test Explanation Output:")
    print("Pred mm:", explanation["predicted_rainfall_mm"])
    print("Temporal Attention (24h):", len(explanation["temporal_attention"]), "steps")
    print("Feature Importance (%):", explanation["feature_importance"])
