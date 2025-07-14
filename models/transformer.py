import torch
import torch.nn as nn


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor):
        B, P, D = x.shape
        H = self.heads
        qkv = self.qkv(x)  # [B, P, 3*D]
        qkv = qkv.reshape(B, P, 3, H, D // H).permute(
            2, 0, 3, 1, 4
        )  # [3, B, H, P, D_head]
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, H, P, D_head]
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, H, P, P]
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = attn @ v  # [B, H, P, D_head]
        out = out.transpose(1, 2).reshape(B, P, D)
        return self.proj(out)


class TransformerEncoderBlock(nn.Module):
    def __init__(
        self, dim: int, heads: int = 4, mlp_ratio: float = 4.0, dropout: float = 0.1
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads, dropout)
        self.drop1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor):
        x = x + self.drop1(self.attn(self.norm1(x)))
        x = x + self.mlp(self.norm2(x))
        return x
