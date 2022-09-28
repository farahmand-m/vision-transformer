"""
The implementation has been adapted from the ViT-PyTorch repository.
https://github.com/lucidrains/vit-pytorch
"""

import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange


class MLP(nn.Sequential):

    def __init__(self, width, hidden_dim, dropout):
        super().__init__(
            nn.Linear(width, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, width),
            nn.Dropout(dropout)
        )


class MultiHeadAttention(nn.Module):

    def __init__(self, width, heads, dim_head, dropout):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == width)
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Linear(width, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, width),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class EncoderBlock(nn.Module):

    def __init__(self, width, heads, dim_head, mlp_width, dropout):
        super().__init__()
        self.attn = MultiHeadAttention(width, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = MLP(width, mlp_width, dropout=dropout)
        self.norm_1 = nn.LayerNorm(width)
        self.norm_2 = nn.LayerNorm(width)

    def forward(self, x):
        x = self.attn(self.norm_1(x)) + x
        x = self.mlp(self.norm_2(x)) + x
        return x


class TransformerEncoder(nn.Module):

    def __init__(self, width, depth, heads, dim_head, mlp_width, dropout):
        super().__init__()
        self.encoder_blocks = nn.ModuleList()
        for i in range(depth):
            block = EncoderBlock(width, heads, dim_head, mlp_width, dropout)
            self.encoder_blocks.append(block)

    def forward(self, x):
        for block in self.encoder_blocks:
            x = block(x)
        return x


class ViT(nn.Module):

    def __init__(self, image_size, patch_size, num_classes, width, depth, heads, mlp_width, channels=3, dim_head=64, dropout=0, embedding_dropout=0):
        super().__init__()
        image_height, image_width = image_size if isinstance(image_size, tuple) else (image_size, image_size)
        patch_height, patch_width = patch_size if isinstance(patch_size, tuple) else (patch_size, patch_size)
        # A Simple Safety Check
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        # These will come in handy.
        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        # Image Patches / Embedding Projections
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.Linear(patch_dim, width),
        )
        # Learnable Embeddings
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, width))
        self.class_token = nn.Parameter(torch.randn(1, 1, width))
        self.embedding_dropout = nn.Dropout(embedding_dropout)
        # Transformer Encoder Network
        self.transformer = TransformerEncoder(width, depth, heads, dim_head, mlp_width, dropout)
        # Class Prediction Head
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, num_classes)
        )

    def forward(self, images):
        # Breaking the Images into Patches and Projecting them onto an Embedding Space
        x = self.to_patch_embedding(images)
        b, n, d = x.shape
        # Appending the Class Token to the Patch Sequences
        class_tokens = repeat(self.class_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((class_tokens, x), dim=1)
        # Adding Positional Embeddings to the Tokens
        x += self.pos_embedding[:, :(n + 1)]
        x = self.embedding_dropout(x)
        # Processing the Sequences using a Transformer
        x = self.transformer(x)
        # Extracting the Elements Associated with the Class Tokens
        x = x[:, 0]
        # Predicting the Image Classes
        return self.mlp_head(x)
