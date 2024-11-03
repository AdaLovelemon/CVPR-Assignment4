import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, embed_dim, dropout_rate=None):
        super(PatchEmbedding, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate is not None else nn.Identity()

    def forward(self, x):
        x = self.proj(x)  # (B, embed_dim, H/patch_size, W/patch_size)
        x = x.flatten(2)  # (B, embed_dim, num_patches)
        x = x.transpose(1, 2)  # (B, num_patches, embed_dim)
        x = self.dropout(x)
        return x

class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout_rate=None):
        super(TransformerEncoderLayer, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout_rate)
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate is not None else nn.Identity()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            self.dropout,
            nn.Linear(mlp_dim, embed_dim),
            self.dropout
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        attn_output, _ = self.attn(x, x, x)
        x = x + attn_output
        x = self.norm1(x)
        mlp_output = self.mlp(x)
        x = x + mlp_output
        x = self.norm2(x)
        return x

class VisionTransformer(nn.Module):
    def __init__(self, in_channels, patch_size, image_size, embed_dim, num_heads, num_blocks, num_classes, dropout_rate=None):
        super(VisionTransformer, self).__init__()
        self.patch_embed = PatchEmbedding(
            img_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            dropout_rate=dropout_rate
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.patch_embed.num_patches, embed_dim))
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate is not None else nn.Identity()
        self.transformer = nn.ModuleList([
            TransformerEncoderLayer(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_dim=embed_dim * 4,
                dropout_rate=dropout_rate
            ) for _ in range(num_blocks)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        B = x.size(0)
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.dropout(x)

        for layer in self.transformer:
            x = layer(x)

        x = self.norm(x)
        cls_token_final = x[:, 0]
        x = self.head(cls_token_final)
        return x

