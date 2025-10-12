"""
Vision encoders for processing images.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class CNNEncoder(nn.Module):
    """CNN encoder for processing images."""
    
    def __init__(
        self,
        input_channels: int = 3,
        output_dim: int = 512,
        channels: List[int] = [32, 64, 128, 256],
        kernel_sizes: List[int] = [3, 3, 3, 3],
        strides: List[int] = [2, 2, 2, 2],
        padding: List[int] = [1, 1, 1, 1],
        activation: str = "relu",
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.input_channels = input_channels
        self.output_dim = output_dim
        self.channels = channels
        self.kernel_sizes = kernel_sizes
        self.strides = strides
        self.padding = padding
        
        # Build encoder layers
        self.encoder = nn.ModuleList()
        in_channels = input_channels
        
        for i, (out_channels, kernel_size, stride, pad) in enumerate(
            zip(channels, kernel_sizes, strides, padding)
        ):
            self.encoder.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU() if activation == "relu" else nn.GELU(),
                    nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
                )
            )
            in_channels = out_channels
        
        # Calculate output size after convolutions
        self.conv_output_size = self._get_conv_output_size()
        
        # Final projection layer
        self.projection = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(self.conv_output_size, output_dim),
            nn.ReLU() if activation == "relu" else nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
        logger.info(f"CNNEncoder created: {input_channels} -> {output_dim}")
        logger.info(f"Channels: {channels}")
        logger.info(f"Conv output size: {self.conv_output_size}")
    
    def _get_conv_output_size(self) -> int:
        """Calculate the output size after convolution layers."""
        # Create a dummy input to calculate output size
        dummy_input = torch.zeros(1, self.input_channels, 128, 128)
        
        with torch.no_grad():
            x = dummy_input
            for layer in self.encoder:
                x = layer(x)
            
            # Calculate flattened size
            conv_output_size = x.numel() // x.size(0)
        
        return conv_output_size
    
    def _init_weights(self, m):
        """Initialize weights."""
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Apply convolution layers
        for layer in self.encoder:
            x = layer(x)
        
        # Apply projection
        x = self.projection(x)
        
        return x


class ViTEncoder(nn.Module):
    """Vision Transformer encoder (simplified)."""
    
    def __init__(
        self,
        input_channels: int = 3,
        output_dim: int = 512,
        patch_size: int = 8,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_patches = (128 // patch_size) ** 2  # Assuming 128x128 input
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(
            input_channels, embed_dim, 
            kernel_size=patch_size, stride=patch_size
        )
        
        # Position embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.projection = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
        logger.info(f"ViTEncoder created: {input_channels} -> {output_dim}")
        logger.info(f"Patch size: {patch_size}, Embed dim: {embed_dim}")
    
    def _init_weights(self, m):
        """Initialize weights."""
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # (B, embed_dim, H//patch_size, W//patch_size)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        
        # Add position embedding
        x = x + self.pos_embed[:, :x.size(1), :]
        
        # Apply transformer blocks
        for block in self.transformer_blocks:
            x = block(x)
        
        # Global average pooling
        x = x.mean(dim=1)  # (B, embed_dim)
        
        # Output projection
        x = self.projection(x)  # (B, output_dim)
        
        return x


class TransformerBlock(nn.Module):
    """Transformer block for ViT."""
    
    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float, dropout: float):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Self-attention
        norm_x = self.norm1(x)
        attn_out, _ = self.attn(norm_x, norm_x, norm_x)
        x = x + attn_out
        
        # MLP
        norm_x = self.norm2(x)
        mlp_out = self.mlp(norm_x)
        x = x + mlp_out
        
        return x


def create_encoder(encoder_type: str, **kwargs) -> nn.Module:
    """Create encoder based on type."""
    if encoder_type.lower() == "cnn":
        return CNNEncoder(**kwargs)
    elif encoder_type.lower() == "vit":
        return ViTEncoder(**kwargs)
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")


if __name__ == "__main__":
    # Test encoders
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Test CNN encoder
    cnn_encoder = CNNEncoder(input_channels=3, output_dim=512).to(device)
    dummy_input = torch.randn(2, 3, 128, 128).to(device)
    cnn_output = cnn_encoder(dummy_input)
    print(f"CNN output shape: {cnn_output.shape}")
    
    # Test ViT encoder
    vit_encoder = ViTEncoder(input_channels=3, output_dim=512).to(device)
    vit_output = vit_encoder(dummy_input)
    print(f"ViT output shape: {vit_output.shape}")