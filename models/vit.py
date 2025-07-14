import torch
import torch.nn as nn

from .patch_embed import PatchEmbedding
from .transformer import TransformerEncoderBlock


class MiniViT(nn.Module):
    def __init__(
        self,
        img_size=32,
        patch_size=4,
        in_channels=3,
        embed_dim=128,
        depth=4,
        heads=4,
        mlp_ratio=4.0,
        num_classes=10,
        dropout=0.1,
    ):
        super().__init__()

        self.patch_embed = PatchEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
        )
        self.num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))

        self.pos_drop = nn.Dropout(p=0.1)

        self.blocks = nn.ModuleList(
            [
                TransformerEncoderBlock(embed_dim, heads, mlp_ratio, dropout)
                for _ in range(depth)
            ]
        )

        self.norm = nn.LayerNorm(embed_dim)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim), nn.Linear(embed_dim, num_classes)
        )

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x, return_cls=False):
        B = x.shape[0]
        x = self.patch_embed(x)

        # expand [cls] token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        x = x + self.pos_embed[:, : x.size(1), :]
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)

        cls_out = x[:, 0]
        return cls_out if return_cls else self.mlp_head(cls_out)


if __name__ == "__main__":
    model = MiniViT()
    x = torch.randn(2, 3, 32, 32)
    out = model(x)
    print(out.shape)  # [2, 10]
