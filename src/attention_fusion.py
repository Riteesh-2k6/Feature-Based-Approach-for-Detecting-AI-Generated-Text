# attention_fusion.py
import torch
import torch.nn as nn


class MultiScaleChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()

        # Global average pooling
        self.avg_pool = nn.AdaptiveAvgPool1d(1)

        # Multi-scale channels
        self.fc1 = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
        )

        # Local context (1D conv)
        self.local_conv = nn.Conv1d(
            channels, channels, kernel_size=3, padding=1, groups=channels, bias=False
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Args:
            x: (batch_size, channels)
        """
        # Add channel dimension for pooling
        x_expanded = x.unsqueeze(2)  # (batch_size, channels, 1)

        # Global context
        global_context = self.avg_pool(x_expanded).squeeze(2)
        global_att = self.fc1(global_context)

        # Local context
        local_att = self.local_conv(x_expanded).squeeze(2)

        # Combine and activate
        attention = self.sigmoid(global_att + local_att)

        return x * attention


class AttentionalFeatureFusion(nn.Module):
    def __init__(self, feature_dim, embedding_dim, fused_dim=256):
        super().__init__()

        # Project features and embeddings to same dimension
        self.feature_proj = nn.Linear(feature_dim, fused_dim)
        self.embedding_proj = nn.Linear(embedding_dim, fused_dim)

        # Multi-scale attention
        self.attention = MultiScaleChannelAttention(fused_dim)

        # Layer normalization
        self.norm = nn.LayerNorm(fused_dim)

    def forward(self, features, embeddings):
        """
        Args:
            features: (batch_size, feature_dim)
            embeddings: (batch_size, embedding_dim)

        Returns:
            fused: (batch_size, fused_dim)
        """
        # Project to common dimension
        feat_proj = self.feature_proj(features)
        emb_proj = self.embedding_proj(embeddings)

        # Initial fusion (element-wise sum)
        initial_fusion = feat_proj + emb_proj

        # Apply attention
        attended = self.attention(initial_fusion)

        # Weighted fusion
        fusion_weight = torch.sigmoid(attended)
        fused = fusion_weight * feat_proj + (1 - fusion_weight) * emb_proj

        # Normalize
        fused = self.norm(fused)

        return fused


class IterativeAFF(nn.Module):
    def __init__(self, feature_dim, embedding_dim, fused_dim=256, num_iterations=2):
        super().__init__()

        # First level fusion
        self.first_fusion = AttentionalFeatureFusion(
            feature_dim, embedding_dim, fused_dim
        )

        # Iterative refinement
        self.iterations = nn.ModuleList(
            [
                AttentionalFeatureFusion(fused_dim, fused_dim, fused_dim)
                for _ in range(num_iterations - 1)
            ]
        )

    def forward(self, features, embeddings):
        # First fusion
        fused = self.first_fusion(features, embeddings)

        # Iterative refinement
        for fusion_layer in self.iterations:
            fused = fusion_layer(fused, fused)

        return fused


# Usage
fusion = IterativeAFF(feature_dim=54, embedding_dim=768, fused_dim=256)
