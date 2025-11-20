# --- START OF FILE model.py ---
import math
import torch
import torch.nn as nn

# Import our custom Encoder/Decoder
from encoder import TransformerEncoder, TransformerEncoderLayer
from decoder import TransformerDecoder, TransformerDecoderLayer
from model_utils import PositionalEncoding

class VesselTransformer(nn.Module):
    """
    AIS Trajectory Predictor using Custom Encoder/Decoder
    """
    def __init__(
        self,
        input_features: int,
        d_model: int,
        nhead: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        dim_feedforward: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        
        # Embeddings
        self.encoder_embed = nn.Linear(input_features, d_model)
        self.decoder_embed = nn.Linear(input_features, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # --- CUSTOM ENCODER ---
        encoder_layer = TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            pre_norm=True  # Using Pre-Norm is usually better for training stability
        )
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers)

        # --- CUSTOM DECODER ---
        decoder_layer = TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            pre_norm=True
        )
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers)

        # Output Head
        self.output_head = nn.Linear(d_model, input_features)

    def _generate_square_subsequent_mask(self, sz, device):
        """Generates the causal mask so decoder doesn't see the future"""
        mask = (torch.triu(torch.ones(sz, sz, device=device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src, tgt):
        # src: [Batch, Seq_Len, Features]
        # tgt: [Batch, Target_Len, Features]
        
        # 1. Embed + Positional Encoding
        # Permute to [Seq_Len, Batch, Dim] because custom layers usually expect Sequence first
        src = self.encoder_embed(src).permute(1, 0, 2) 
        src = self.pos_encoder(src)
        
        tgt = self.decoder_embed(tgt).permute(1, 0, 2)
        tgt = self.pos_encoder(tgt)

        # 2. Encoder
        memory = self.encoder(src)

        # 3. Decoder
        # Create Causal Mask for Target
        tgt_mask = self._generate_square_subsequent_mask(tgt.size(0), tgt.device)
        
        output = self.decoder(
            tgt=tgt,
            memory=memory,
            tgt_mask=tgt_mask
        )

        # 4. Output Head
        # Permute back to [Batch, Seq, Dim]
        output = output.permute(1, 0, 2)
        prediction = self.output_head(output)
        
        return prediction