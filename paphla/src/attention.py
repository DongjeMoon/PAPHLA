import math

import torch
import torch.nn as nn
from torch_geometric.utils import to_dense_batch


class PMA(nn.Module):
    def __init__(self, dim, num_heads, num_seeds=1):
        super().__init__()
        self.num_seeds = num_seeds
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_normal_(self.S)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.ff = nn.Linear(dim, dim)
        self.ln0 = nn.LayerNorm(dim)
        self.ln1 = nn.LayerNorm(dim)
        self.relu = nn.ReLU()
    
    def forward(self, batch):
        X, mask = to_dense_batch(batch.x, batch.batch, 0)
        S = self.S.repeat(X.size(0), 1, 1)
        attn_out, attn_w = self.attn(S, X, X, key_padding_mask=~mask)
        H = self.ln0(S + attn_out)
        O = H + self.relu(self.ff(H))
        O = self.ln1(O)
        return O, attn_w


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads, attn_dropout, dropout):
        super().__init__()

        self.num_heads = num_heads
        self.dk = dim // num_heads

        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)

        self.out = nn.Linear(dim, dim)

        # post-norm
        self.ln = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, 4*dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(4*dim, dim)
        )

        self.attn_dropout = nn.Dropout(attn_dropout)
        self.dropout = nn.Dropout(dropout)
        self.tau = nn.Parameter(torch.zeros(1))

    def forward(self, batch):
        q, mask = to_dense_batch(batch.x, batch.batch, 0)
        k = v = batch.hla

        B, Np, D = q.shape
        Nh = k.shape[1]

        attn_mask = mask.unsqueeze(-1).expand(-1, -1, Nh)
        attn_mask = attn_mask.unsqueeze(1)

        _q = self.q(q)  # (B, Np, D)
        _k = self.k(k)
        _v = self.v(v)

        Q = _q.view(B, Np, self.num_heads, self.dk).transpose(1, 2)
        K = _k.view(B, Nh, self.num_heads, self.dk).transpose(1, 2)
        V = _v.view(B, Nh, self.num_heads, self.dk).transpose(1, 2)
        logits = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.dk)

        scale = self.tau.exp()
        logits = logits * scale

        if attn_mask is not None:
            if attn_mask.dim() == 3:
                attn_mask = attn_mask.unsqueeze(1)
            logits = logits.masked_fill(~attn_mask, -1e4)

        A = torch.softmax(logits, dim=-1)
        A = self.attn_dropout(A)

        ctx = torch.matmul(A, V)
        ctx = ctx.transpose(1, 2).contiguous().view(B, Np, D)

        x = self.out(ctx)
        q = q + self.dropout(x)
        q = self.ln(q + self.dropout(self.ff(q)))

        batch.x = q[mask]

        return batch, A