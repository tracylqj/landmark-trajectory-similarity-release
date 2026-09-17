import math

import torch
from torch import nn
from torch.nn import functional as F


class TransformerLayer(nn.Module):
    def __init__(self, dim=128, heads=4, ffn_dim=512):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, dim),
        )

    def forward(self, x, valid):
        batch, length, dim = x.shape
        q, k, v = self.qkv(x).view(
            batch, length, 3, self.heads, dim // self.heads
        ).permute(2, 0, 3, 1, 4)
        attention = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=valid[:, None, None, :],
            dropout_p=0.0,
        )
        attention = attention.transpose(1, 2).reshape(batch, length, dim)
        x = self.norm1(x + self.out(attention))
        x = self.norm2(x + self.ff(x))
        return x * valid[..., None]


class TrajectoryEncoder(nn.Module):
    def __init__(self, landmark_table, dim=128, heads=4, ffn_dim=512, max_length=512):
        super().__init__()
        table = torch.as_tensor(landmark_table, dtype=torch.float32)
        self.register_buffer("table", table)
        self.input = nn.Linear(table.shape[1], dim, bias=False)
        self.layers = nn.ModuleList([TransformerLayer(dim, heads, ffn_dim)])

        position = torch.arange(max_length)[:, None]
        scale = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim))
        encoding = torch.zeros(max_length, dim)
        encoding[:, 0::2] = torch.sin(position * scale)
        encoding[:, 1::2] = torch.cos(position * scale)
        self.register_buffer("pe", encoding)

    def forward(self, sequences, lengths):
        if sequences.shape[1] > self.pe.shape[0]:
            raise ValueError("trajectory length exceeds positional encoding capacity")
        valid = torch.arange(sequences.shape[1], device=sequences.device)[None] < lengths[:, None]
        x = (self.input(self.table[sequences]) + self.pe[: sequences.shape[1]]) * valid[..., None]
        for layer in self.layers:
            x = layer(x, valid)
        vector = x.sum(dim=1) / lengths[:, None]
        return vector, x

