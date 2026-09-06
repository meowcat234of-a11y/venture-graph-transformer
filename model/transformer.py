import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention import MultiHeadAttention

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

class SwiGLU(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, hidden_dim):
        super().__init__()
        self.attn = MultiHeadAttention(dim, heads)
        self.ff = SwiGLU(dim, hidden_dim)
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)

    def forward(self, x, pos=0, kv_cache=None):
        x = x + self.attn(self.norm1(x), pos, kv_cache)
        x = x + self.ff(self.norm2(x))
        return x

class Transformer(nn.Module):
    def __init__(self, vocab_size, dim, layers, heads, hidden_dim):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([TransformerBlock(dim, heads, hidden_dim) for _ in range(layers)])
        self.norm = RMSNorm(dim)
        self.out = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, x, pos=0, caches=None):
        h = self.emb(x)
        for i, layer in enumerate(self.layers):
            cache = caches[i] if caches else None
            h = layer(h, pos, cache)
        return self.out(self.norm(h))
