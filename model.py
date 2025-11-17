import math
from typing import Optional

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Injects positional information into the input embeddings."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Shape (1, max_len, d_model) to match batch-first inputs
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encodings to ``x`` (expects batch-first tensors)."""
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class VesselTransformer(nn.Module):
    """
    Transformer Encoder-Decoder for vessel trajectory prediction.
    
    Architecture:
    - Encoder: Processes historical trajectory (context)
    - Decoder: Generates future predictions autoregressively with cross-attention to encoder
    
    This is the classic seq2seq Transformer architecture from "Attention Is All You Need".
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
        self.input_features = input_features

        # Input embeddings for encoder and decoder
        self.encoder_embed = nn.Linear(input_features, d_model)
        self.decoder_embed = nn.Linear(input_features, d_model)

        # Positional encodings
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.pos_decoder = PositionalEncoding(d_model, dropout)

        # Transformer Encoder-Decoder
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )

        # Output head
        self.output_head = nn.Linear(d_model, input_features)

        self._tgt_mask: Optional[torch.Tensor] = None

    def _generate_square_subsequent_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        """Generates a causal mask for the decoder (prevents looking ahead)."""
        mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float("-inf"))
        return mask

    def forward(
        self, 
        src: torch.Tensor, 
        tgt: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for training.
        
        Args:
            src: Source sequence (history), shape [batch_size, src_len, input_features]
            tgt: Target sequence (teacher forcing), shape [batch_size, tgt_len, input_features]
            tgt_mask: Optional target mask for decoder
        
        Returns:
            predictions: Predicted future trajectory, shape [batch_size, tgt_len, input_features]
        """
        
        # Generate causal mask for decoder if not provided
        if tgt_mask is None:
            tgt_len = tgt.size(1)
            device = tgt.device
            if self._tgt_mask is None or self._tgt_mask.size(0) != tgt_len:
                self._tgt_mask = self._generate_square_subsequent_mask(tgt_len, device)
            tgt_mask = self._tgt_mask

        # Embed and add positional encoding for encoder
        src_embed = self.encoder_embed(src) * math.sqrt(self.d_model)
        src_pos = self.pos_encoder(src_embed)

        # Embed and add positional encoding for decoder
        tgt_embed = self.decoder_embed(tgt) * math.sqrt(self.d_model)
        tgt_pos = self.pos_decoder(tgt_embed)

        # Pass through transformer
        transformer_out = self.transformer(
            src=src_pos,
            tgt=tgt_pos,
            tgt_mask=tgt_mask,
        )

        # Generate predictions
        predictions = self.output_head(transformer_out)

        return predictions

    def encode(self, src: torch.Tensor) -> torch.Tensor:
        """
        Encode the source sequence (for inference).
        
        Args:
            src: Source sequence, shape [batch_size, src_len, input_features]
        
        Returns:
            memory: Encoded representation, shape [batch_size, src_len, d_model]
        """
        src_embed = self.encoder_embed(src) * math.sqrt(self.d_model)
        src_pos = self.pos_encoder(src_embed)
        
        # Use the transformer's encoder
        memory = self.transformer.encoder(src_pos)
        return memory

    def decode_step(
        self, 
        tgt: torch.Tensor, 
        memory: torch.Tensor,
    ) -> torch.Tensor:
        """
        Decode one step (for autoregressive inference).
        
        Args:
            tgt: Target sequence so far, shape [batch_size, tgt_len, input_features]
            memory: Encoded source from encode(), shape [batch_size, src_len, d_model]
        
        Returns:
            prediction: Next prediction, shape [batch_size, tgt_len, input_features]
        """
        tgt_len = tgt.size(1)
        device = tgt.device
        
        # Generate causal mask
        tgt_mask = self._generate_square_subsequent_mask(tgt_len, device)
        
        # Embed and add positional encoding
        tgt_embed = self.decoder_embed(tgt) * math.sqrt(self.d_model)
        tgt_pos = self.pos_decoder(tgt_embed)
        
        # Decode
        decoder_out = self.transformer.decoder(
            tgt=tgt_pos,
            memory=memory,
            tgt_mask=tgt_mask,
        )
        
        # Generate prediction
        prediction = self.output_head(decoder_out)
        return prediction


# Backward compatibility: keep old name but use encoder-decoder
class VesselTransformerEncoderOnly(VesselTransformer):
    """
    Backward-compatible wrapper that now reuses the encoder-decoder ``VesselTransformer``.

    Older code instantiating ``VesselTransformerEncoderOnly`` will still receive a full
    seq2seq model, but it must now provide both ``src`` (history) and ``tgt`` (decoder input)
    when calling ``forward``. This keeps the API surface aligned with the modern model while
    avoiding two divergent implementations.
    """

    def __init__(
        self,
        input_features: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(
            input_features=input_features,
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )

    def forward(
        self,
        src: torch.Tensor,
        tgt: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if tgt is None:
            raise ValueError(
                "VesselTransformerEncoderOnly now uses the encoder-decoder architecture. "
                "Pass both 'src' (history) and 'tgt' (decoder input) just like VesselTransformer."
            )
        return super().forward(src, tgt, tgt_mask)