# model.py
# Python 3.6 compatible
import torch
import torch.nn as nn
import torch.nn.functional as F


# =====================================================
# Patch Embedding
# =====================================================
class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, embed_dim=768, patch_size=16, img_size=256):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, embed_dim)
        )

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        # x: B,C,H,W
        B = x.size(0)

        x = self.proj(x)                 # B,768,16,16
        x = x.flatten(2).transpose(1, 2) # B,256,768

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # B,257,768

        x = x + self.pos_embed
        return x


# =====================================================
# Standard Transformer Encoder (single modality)
# =====================================================
class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim=768, num_heads=12, depth=12, mlp_ratio=4.0):
        super(TransformerEncoder, self).__init__()

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleDict({
                "norm1": nn.LayerNorm(embed_dim),
                "attn": nn.MultiheadAttention(embed_dim, num_heads),
                "norm2": nn.LayerNorm(embed_dim),
                "mlp": nn.Sequential(
                    nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
                    nn.GELU(),
                    nn.Linear(int(embed_dim * mlp_ratio), embed_dim)
                )
            }))

    def forward(self, x):
        # x: B,N,768
        x = x.transpose(0, 1)  # N,B,768

        for layer in self.layers:
            x2 = layer["norm1"](x)
            attn_out, _ = layer["attn"](x2, x2, x2)
            x = x + attn_out

            x2 = layer["norm2"](x)
            x = x + layer["mlp"](x2)

        x = x.transpose(0, 1)  # B,N,768
        return x


# =====================================================
# DCMF: Decision-oriented Cross-Modal Fusion
# (CLS-level attention)
# =====================================================
class DCMF(nn.Module):
    def __init__(self, embed_dim=768, num_heads=12):
        super(DCMF, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads)

    def forward(self, cls_ms, cls_rgb, cls_tir):
        # each: B,1,768
        cls = torch.cat([cls_ms, cls_rgb, cls_tir], dim=1)  # B,3,768
        cls = cls.transpose(0, 1)  # 3,B,768

        fused, _ = self.attn(cls, cls, cls)
        fused = fused.mean(dim=0, keepdim=True)  # 1,B,768
        fused = fused.transpose(0, 1)  # B,1,768
        return fused


# =====================================================
# MulD Tokenizer
# =====================================================
class MulDTokenizer(nn.Module):
    def __init__(self, in_dim=10, embed_dim=768):
        super(MulDTokenizer, self).__init__()
        self.proj = nn.Linear(in_dim, embed_dim)

    def forward(self, x):
        # B,10
        x = self.proj(x)
        return x.unsqueeze(1)  # B,1,768


# =====================================================
# CLS-guided Low-Rank Attention (核心创新)
# Only CLS produces Query
# =====================================================
class CLSGuidedBlock(nn.Module):
    def __init__(self, embed_dim=768, mlp_ratio=4.0):
        super(CLSGuidedBlock, self).__init__()

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim)
        )

    def forward(self, x):
        # x: B,N,768
        B, N, C = x.shape

        x_norm = self.norm1(x)

        cls = x_norm[:, 0:1, :]          # B,1,768

        Q = self.q_proj(cls)             # B,1,768
        K = self.k_proj(x_norm)          # B,N,768
        V = self.v_proj(x_norm)          # B,N,768

        attn = torch.matmul(Q, K.transpose(1, 2)) / (C ** 0.5)  # B,1,N
        attn = F.softmax(attn, dim=-1)

        cls_update = torch.matmul(attn, V)  # B,1,768

        # Residual (only CLS updated)
        cls = cls + cls_update

        # FFN
        cls = cls + self.mlp(self.norm2(cls))

        # replace CLS
        x = torch.cat([cls, x[:, 1:, :]], dim=1)

        return x


# =====================================================
# Decision Transformer (2 layers)
# =====================================================
class DecisionTransformer(nn.Module):
    def __init__(self, embed_dim=768, depth=2):
        super(DecisionTransformer, self).__init__()
        self.layers = nn.ModuleList(
            [CLSGuidedBlock(embed_dim) for _ in range(depth)]
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


# =====================================================
# Full Model
# =====================================================
class MultiModalDecisionViT(nn.Module):
    def __init__(self, embed_dim=768):
        super(MultiModalDecisionViT, self).__init__()

        # Patch embedding
        self.ms_embed = PatchEmbedding(5, embed_dim)
        self.rgb_embed = PatchEmbedding(3, embed_dim)
        self.tir_embed = PatchEmbedding(1, embed_dim)

        # Single-modality encoders
        self.ms_encoder = TransformerEncoder(embed_dim)
        self.rgb_encoder = TransformerEncoder(embed_dim)
        self.tir_encoder = TransformerEncoder(embed_dim)

        # Cross-modal fusion
        self.dcmf = DCMF(embed_dim)

        # MulD
        self.muld_token = MulDTokenizer(10, embed_dim)

        # Decision transformer
        self.decision_encoder = DecisionTransformer(embed_dim, depth=2)

        # Regression head
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 1)
        )

    def forward(self, ms, rgb, tir, muld):
        # -------------------------------------------------
        # Patch embedding
        # -------------------------------------------------
        ms = self.ms_embed(ms)     # B,257,768
        rgb = self.rgb_embed(rgb)
        tir = self.tir_embed(tir)

        # -------------------------------------------------
        # Single-modality encoding
        # -------------------------------------------------
        ms = self.ms_encoder(ms)
        rgb = self.rgb_encoder(rgb)
        tir = self.tir_encoder(tir)

        # CLS extraction
        cls_ms = ms[:, 0:1, :]
        cls_rgb = rgb[:, 0:1, :]
        cls_tir = tir[:, 0:1, :]

        # -------------------------------------------------
        # DCMF
        # -------------------------------------------------
        cls_fused = self.dcmf(cls_ms, cls_rgb, cls_tir)

        # -------------------------------------------------
        # Patch fusion (concat patches)
        # -------------------------------------------------
        patches = torch.cat([
            ms[:, 1:, :],
            rgb[:, 1:, :],
            tir[:, 1:, :]
        ], dim=1)  # B,768 patches total (256*3)

        # -------------------------------------------------
        # MulD Decision Anchor
        # -------------------------------------------------
        muld_token = self.muld_token(muld)  # B,1,768

        # Final tokens
        x = torch.cat([cls_fused, patches, muld_token], dim=1)

        # -------------------------------------------------
        # Decision Transformer
        # -------------------------------------------------
        x = self.decision_encoder(x)

        # -------------------------------------------------
        # Regression
        # -------------------------------------------------
        cls_final = x[:, 0]
        out = self.head(cls_final)

        return out