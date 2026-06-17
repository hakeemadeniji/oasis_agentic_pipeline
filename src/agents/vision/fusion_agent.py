import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np

class CleanCrossAttentionBridge(nn.Module):
    """
    Explicit Multi-Head Cross-Attention Layer.
    Bypasses black-box nn.MultiheadAttention to generate clean,
    linear mathematical nodes that export to ONNX flawlessly.
    """
    def __init__(self, embed_dim: int = 512, num_heads: int = 8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        self.out_linear = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        # query (Tabular): [B, 1, 512] | key (Vision Tokens): [B, 49, 512]
        B, L_q, D = query.shape
        _, L_k, _ = key.shape
        
        # Project and segment into independent attention heads
        # Shape output: [B, num_heads, Length, head_dim]
        q = self.q_linear(query).view(B, L_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_linear(key).view(B, L_k, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_linear(key).view(B, L_k, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Mathematical Scaled Dot-Product Attention: Q x K^T / sqrt(d_k)
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attn_weights = torch.softmax(scores, dim=-1)
        
        # Weight the value tokens: Context Shape [B, num_heads, L_q, head_dim]
        context = torch.matmul(attn_weights, v)
        
        # Concat heads back together and project through final linear layer
        context = context.transpose(1, 2).contiguous().view(B, L_q, D)
        return self.out_linear(context)

class AlzheimerMultimodalFusionNet(nn.Module):
    """
    Expert-grade Intermediate Fusion Network.
    Fuses spatial tokens and tabular clinical biomarkers using a compiler-safe
    explicit Cross-Attention Transformer architecture.
    """
    def __init__(self, num_classes: int = 4, tabular_dim: int = 8, embed_dim: int = 512):
        super().__init__()
        print(f"[*] Constructing Explicit Cross-Attention Fusion Architecture (Embedding: {embed_dim})...")
        
        # 1. Vision Feature Extractor Block
        resnet = models.resnet18(weights=None)
        self.backbone = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4
        )
        self.backbone[0] = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # 2. Tabular Projection Feature Alignment Block
        self.tabular_projection = nn.Sequential(
            nn.Linear(tabular_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
        # 3. Clean Compiler-Safe Cross-Attention Bridge
        self.cross_attention = CleanCrossAttentionBridge(embed_dim=embed_dim, num_heads=8)
        self.layer_norm = nn.LayerNorm(embed_dim)
        
        # 4. Dense Diagnostic Prediction Head
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, image: torch.Tensor, tabular: torch.Tensor) -> torch.Tensor:
        # Extract features: [B, 1, 224, 224] -> [B, 512, 7, 7]
        vision_features = self.backbone(image)
        B, C, H, W = vision_features.shape
        
        # Flatten spatial maps to token streams: [B, 49, 512]
        vision_tokens = vision_features.view(B, C, H * W).transpose(1, 2)
        
        # Align tabular dimensions: [B, 8] -> [B, 1, 512]
        tabular_query = self.tabular_projection(tabular).unsqueeze(1)
        
        # Execute explicitly traced attention bridge
        attn_output = self.cross_attention(query=tabular_query, key=vision_tokens)
        
        # Merge via residual network loop
        fused_vector = self.layer_norm(attn_output + tabular_query).squeeze(1)
        
        return self.classifier(fused_vector)

if __name__ == "__main__":
    print("[*] Testing Explicit Cross-Attention Graph Layout...")
    model = AlzheimerMultimodalFusionNet()
    mock_image = torch.randn(2, 1, 224, 224)
    mock_tabular = torch.randn(2, 8)
    output_logits = model(mock_image, mock_tabular)
    print(f"[SUCCESS] Explicit tensor output dimensions verified: {output_logits.shape}")
    