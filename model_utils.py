# --- START OF FILE model_utils.py ---
import copy
import torch.nn as nn
import torch.nn.functional as F
import torch
import math

def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation {activation} not supported")

class PositionalEncoding(nn.Module):
    """Standard Sinusoidal Positional Encoding"""
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [seq_len, batch, dim]
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)