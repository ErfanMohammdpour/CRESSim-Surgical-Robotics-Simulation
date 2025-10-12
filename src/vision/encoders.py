"""
Vision encoders for processing RGB images.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple
import math


class CNNEncoder(nn.Module):
    """
    CNN encoder similar to IMPALA architecture.
    """
    
    def __init__(
        self,
        input_channels: int = 3,
        hidden_dim: int = 256,
        channels: list = [32, 64, 128, 256],
        kernel_sizes: list = [3, 3, 3, 3],
        strides: list = [2, 2, 2, 2],
        padding: list = [1, 1, 1, 1],
        activation: str = "relu",
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.input_channels = input_channels
        self.hidden_dim = hidden_dim
        self.activation = getattr(F, activation)
        
        # Build convolutional layers
        conv_layers = []
        in_channels = input_channels
        
        for i, (out_channels, kernel_size, stride, pad) in enumerate(
            zip(channels, kernel_sizes, strides, padding)
        ):
            conv_layers.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad),
                nn.BatchNorm2d(out_channels),
                nn.ReLU() if activation == "relu" else nn.ELU(),
                nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
            ])
            in_channels = out_channels
        
        self.conv_layers = nn.Sequential(*conv_layers)
        
        # Calculate output size
        self.conv_output_size = self._get_conv_output_size()
        
        # Final projection layer
        self.projection = nn.Linear(self.conv_output_size, hidden_dim)
    
    def _get_conv_output_size(self) -> int:
        """Calculate the output size of convolutional layers."""
        # Create a dummy input to calculate output size
        dummy_input = torch.zeros(1, self.input_channels, 84, 84)
        with torch.no_grad():
            dummy_output = self.conv_layers(dummy_input)
        return dummy_output.view(1, -1).size(1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # x shape: (batch_size, channels, height, width)
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.projection(x)
        return x


class ViTEncoder(nn.Module):
    """
    Vision Transformer encoder (ViT-tiny).
    """
    
    def __init__(
        self,
        input_size: int = 84,
        patch_size: int = 8,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        hidden_dim: int = 256
    ):
        super().__init__()
        
        self.input_size = input_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_patches = (input_size // patch_size) ** 2
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(
            3, embed_dim, 
            kernel_size=patch_size, 
            stride=patch_size
        )
        
        # Positional embedding
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, embed_dim)
        )
        
        # Class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Output projection
        self.projection = nn.Linear(embed_dim, hidden_dim)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.patch_embed.weight, std=0.02)
        nn.init.constant_(self.patch_embed.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        batch_size = x.size(0)
        
        # Patch embedding
        x = self.patch_embed(x)  # (B, embed_dim, H/patch_size, W/patch_size)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        
        # Add class token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Add positional embedding
        x = x + self.pos_embed
        
        # Transformer encoder
        x = self.transformer(x)
        
        # Use class token for final representation
        x = x[:, 0]  # (B, embed_dim)
        
        # Project to hidden dimension
        x = self.projection(x)
        
        return x


class MultiModalEncoder(nn.Module):
    """
    Multi-modal encoder that combines visual and auxiliary inputs.
    """
    
    def __init__(
        self,
        visual_encoder: nn.Module,
        aux_dim: int = 4,
        hidden_dim: int = 256,
        fusion_method: str = "concat"  # "concat", "add", "attention"
    ):
        super().__init__()
        
        self.visual_encoder = visual_encoder
        self.aux_dim = aux_dim
        self.hidden_dim = hidden_dim
        self.fusion_method = fusion_method
        
        # Auxiliary input processing
        self.aux_encoder = nn.Sequential(
            nn.Linear(aux_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim)
        )
        
        # Fusion layer
        if fusion_method == "concat":
            self.fusion = nn.Linear(hidden_dim * 2, hidden_dim)
        elif fusion_method == "add":
            self.fusion = nn.Identity()
        elif fusion_method == "attention":
            self.attention = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)
            self.fusion = nn.Linear(hidden_dim, hidden_dim)
        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")
    
    def forward(self, image: torch.Tensor, aux: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Encode visual input
        visual_features = self.visual_encoder(image)
        
        # Encode auxiliary input
        aux_features = self.aux_encoder(aux)
        
        # Fuse features
        if self.fusion_method == "concat":
            combined = torch.cat([visual_features, aux_features], dim=-1)
            output = self.fusion(combined)
        elif self.fusion_method == "add":
            output = self.fusion(visual_features + aux_features)
        elif self.fusion_method == "attention":
            # Reshape for attention
            visual_features = visual_features.unsqueeze(1)
            aux_features = aux_features.unsqueeze(1)
            
            # Self-attention between visual and aux features
            attended, _ = self.attention(visual_features, aux_features, aux_features)
            output = self.fusion(attended.squeeze(1))
        
        return output


def create_encoder(config: Dict[str, Any]) -> nn.Module:
    """Create encoder based on configuration."""
    encoder_type = config.get('encoder_type', 'cnn')
    
    if encoder_type == 'cnn':
        cnn_config = config.get('cnn_encoder', {})
        visual_encoder = CNNEncoder(
            input_channels=3,
            hidden_dim=config.get('hidden_dim', 256),
            channels=cnn_config.get('channels', [32, 64, 128, 256]),
            kernel_sizes=cnn_config.get('kernel_sizes', [3, 3, 3, 3]),
            strides=cnn_config.get('strides', [2, 2, 2, 2]),
            padding=cnn_config.get('padding', [1, 1, 1, 1]),
            activation=config.get('activation', 'relu'),
            dropout=config.get('dropout', 0.1)
        )
    
    elif encoder_type == 'vit':
        vit_config = config.get('vit_encoder', {})
        visual_encoder = ViTEncoder(
            input_size=84,
            patch_size=vit_config.get('patch_size', 8),
            embed_dim=vit_config.get('embed_dim', 256),
            num_heads=vit_config.get('num_heads', 8),
            num_layers=vit_config.get('num_layers', 6),
            mlp_ratio=vit_config.get('mlp_ratio', 4.0),
            dropout=config.get('dropout', 0.1),
            hidden_dim=config.get('hidden_dim', 256)
        )
    
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")
    
    # Create multi-modal encoder
    multimodal_encoder = MultiModalEncoder(
        visual_encoder=visual_encoder,
        aux_dim=4,  # [suction_state, liquid_mass, contaminant_mass, collisions]
        hidden_dim=config.get('hidden_dim', 256),
        fusion_method=config.get('fusion_method', 'concat')
    )
    
    return multimodal_encoder


class EncoderWithPretraining(nn.Module):
    """
    Encoder with optional pretraining on segmentation task.
    """
    
    def __init__(
        self,
        encoder: nn.Module,
        pretrain_config: Optional[Dict[str, Any]] = None
    ):
        super().__init__()
        
        self.encoder = encoder
        self.pretrain_config = pretrain_config or {}
        
        # Segmentation head for pretraining
        if self.pretrain_config.get('enabled', False):
            self.segmentation_head = nn.Sequential(
                nn.ConvTranspose2d(256, 128, 4, 2, 1),
                nn.ReLU(),
                nn.ConvTranspose2d(128, 64, 4, 2, 1),
                nn.ReLU(),
                nn.ConvTranspose2d(64, 32, 4, 2, 1),
                nn.ReLU(),
                nn.Conv2d(32, 3, 3, 1, 1)  # 3 classes: background, tissue, dangerous_areas
            )
        else:
            self.segmentation_head = None
    
    def forward(self, image: torch.Tensor, aux: torch.Tensor) -> torch.Tensor:
        """Forward pass for main task."""
        return self.encoder(image, aux)
    
    def forward_pretrain(self, image: torch.Tensor) -> torch.Tensor:
        """Forward pass for pretraining."""
        if self.segmentation_head is None:
            raise RuntimeError("Pretraining not enabled")
        
        # Get visual features from encoder
        visual_encoder = self.encoder.visual_encoder
        features = visual_encoder.conv_layers(image)
        
        # Apply segmentation head
        segmentation = self.segmentation_head(features)
        
        return segmentation
