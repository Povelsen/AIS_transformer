import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """Injects positional information into the input embeddings."""
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe) # Registers 'pe' as a buffer

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, d_model]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class VesselTransformer(nn.Module):
    """
    The main Transformer model for vessel trajectory prediction.
    Uses a Transformer Encoder architecture.
    """
    def __init__(self, input_features, d_model, nhead, num_layers, dim_feedforward, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        
        # 1. Input Embedding
        # Projects 4 features (Lat, Lon, SOG, COG) to d_model
        self.input_embed = nn.Linear(input_features, d_model)
        
        # 2. Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # 3. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True # Expects (batch_size, seq_len, features)
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 4. Output Head
        # Predicts all 4 input features for the next time step
        self.output_head = nn.Linear(d_model, input_features) 
        
        self.src_mask = None

    def _generate_square_subsequent_mask(self, sz, device):
        """Generates a causal (look-ahead) mask."""
        mask = (torch.triu(torch.ones(sz, sz, device=device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src):
        """
        Args:
            src: Tensor, shape [batch_size, seq_len, input_features]
        """
        
        # Create causal mask if needed
        if self.src_mask is None or self.src_mask.size(0) != src.size(1):
            self.src_mask = self._generate_square_subsequent_mask(src.size(1), src.device)
            
        # 1. Embed input: (B, S, F) -> (B, S, D)
        src_embed = self.input_embed(src) * math.sqrt(self.d_model)
        
        # 2. Add positional encoding
        # Must transpose to (S, B, D) for pos_encoder, then transpose back to (B, S, D)
        src_pos = self.pos_encoder(src_embed.transpose(0, 1)).transpose(0, 1)
        
        # 3. Pass through Transformer Encoder
        output = self.transformer_encoder(src_pos, self.src_mask) # (B, S, D)
        
        # 4. Pass through output head
        prediction = self.output_head(output) # (B, S, F)
        return prediction
