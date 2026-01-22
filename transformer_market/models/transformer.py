# transformer_market/models/transformer.py

'''
Responsabilidad del archivo

Este archivo solo define el modelo, nada más:
    entrada: (batch, W=60, F=13)
    salida: (batch, 1) → regresión P5
    sin entrenamiento
    sin lectura de datos
    sin métricas

Arquitectura final (v1 cerrada)
    X (60 x 13)
    ↓ Linear projection
    E (60 x 64)
    ↓ Positional encoding
    ↓ TransformerEncoder (2 layers, 4 heads)
    ↓ Mean pooling over time
     ↓ Linear head
    y_hat (1)

Hiperparámetros
    d_model = 64
    n_heads = 4
    n_layers = 2
    dim_ff = 128
    dropout = 0.1
    pooling = mean
    output = regresión

Esto es perfecto para ~500 samples de train.

'''

from __future__ import annotations

import logging
import math
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class PositionalEncoding(nn.Module):
    """
    Positional encoding sinusoidal clásico (Vaswani et al., 2017).
    
    Añade información temporal absoluta sin aprender parámetros.
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    
    Args:
        d_model: dimensión del modelo (embedding)
        max_len: máxima longitud de secuencia (default: 5000)
    """

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        if d_model <= 0:
            raise ValueError(f"d_model debe ser > 0 (recibido: {d_model})")
        if max_len <= 0:
            raise ValueError(f"max_len debe ser > 0 (recibido: {max_len})")

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()

        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)
        logger.debug(f"PositionalEncoding initialized: d_model={d_model}, max_len={max_len}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Suma posicional encoding a la entrada.
        
        Args:
            x: tensor de entrada (batch, seq_len, d_model)
        
        Returns:
            tensor con positional encoding sumado: (batch, seq_len, d_model)
        """
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]


class TransformerRegressor(nn.Module):
    """
    Transformer Encoder para regresión temporal (predicción de retornos P5).
    
    Arquitectura:
        Input: (batch, W=60, F=13) → features técnicas
        ↓ Linear projection → (batch, 60, d_model)
        ↓ Positional encoding
        ↓ TransformerEncoder (n_layers, n_heads)
        ↓ Mean pooling over time → (batch, d_model)
        ↓ Linear head → (batch,) regresión
    
    Parámetros clave:
        d_model: 64 (dimensión del embedding)
        n_heads: 4 (attention heads)
        n_layers: 2 (encoder layers)
        dim_ff: 128 (feedforward dimension)
        dropout: 0.1
        
    Args:
        n_features: número de features de entrada (default: 13)
        d_model: dimensión del modelo (default: 64)
        n_heads: número de heads en multi-head attention (default: 4)
        n_layers: número de encoder layers (default: 2)
        dim_ff: dimensión feedforward (default: 128)
        dropout: dropout rate (default: 0.1)
    
    Raises:
        ValueError: si d_model no es divisible por n_heads, o parámetros inválidos
    """

    def __init__(
        self,
        n_features: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dim_ff: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        # Validaciones
        if n_features <= 0:
            raise ValueError(f"n_features debe ser > 0 (recibido: {n_features})")
        if d_model <= 0:
            raise ValueError(f"d_model debe ser > 0 (recibido: {d_model})")
        if n_heads <= 0:
            raise ValueError(f"n_heads debe ser > 0 (recibido: {n_heads})")
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) debe ser divisible por n_heads ({n_heads})")
        if not (0 <= dropout < 1):
            raise ValueError(f"dropout debe estar en [0, 1) (recibido: {dropout})")

        logger.info(f"TransformerRegressor: n_features={n_features}, d_model={d_model}, n_heads={n_heads}, n_layers={n_layers}")

        # Proyección de features -> espacio del modelo
        self.input_proj = nn.Linear(n_features, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)

        # Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
            activation="relu",
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
        )

        # Pooling temporal: mean
        self.pool = lambda x: x.mean(dim=1)

        # Head de regresión
        self.head = nn.Linear(d_model, 1)

        self._init_weights()
        logger.debug("Model weights initialized with Xavier uniform")

    def _init_weights(self):
        """Inicialización Xavier para estabilidad."""
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass del modelo.
        
        Args:
            x: tensor de entrada (batch, seq_len, n_features)
        
        Returns:
            predicciones de regresión (batch,)
        
        Raises:
            AssertionError: si shapes de entrada no son válidos
        """
        assert len(x.shape) == 3, f"x debe ser 3D (batch, seq, features), recibido {x.shape}"
        assert x.size(2) > 0, f"n_features debe ser > 0, recibido {x.size(2)}"
        
        x = self.input_proj(x)          # (batch, seq_len, d_model)
        x = self.pos_encoder(x)
        x = self.encoder(x)             # (batch, seq_len, d_model)
        x = self.pool(x)                # (batch, d_model)
        out = self.head(x)              # (batch, 1)
        return out.squeeze(-1)           # (batch,)

