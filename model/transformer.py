import torch
import torch.nn as nn
from .attention import MultiHeadAttention

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.mean(x ** 2, dim=-1, keepdim=True)
        return x * torch.rsqrt(norm + self.eps) * self.weight

class SwiGLU(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(nn.functional.silu(self.w1(x)) * self.w3(x))

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, hidden_dim):
        super().__init__()
        self.attention = MultiHeadAttention(dim, num_heads)
        self.feed_forward = SwiGLU(dim, hidden_dim)
        self.attention_norm = RMSNorm(dim)
        self.ffn_norm = RMSNorm(dim)

    def forward(self, x, start_pos=0, kv_cache=None):
        h = x + self.attention(self.attention_norm(x), start_pos, kv_cache)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out

class Transformer(nn.Module):
    def __init__(self, vocab_size, dim, num_layers, num_heads, hidden_dim):
        super().__init__()
        self.tok_embeddings = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([TransformerBlock(dim, num_heads, hidden_dim) for _ in range(num_layers)])
        self.norm = RMSNorm(dim)
        self.output = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, tokens, start_pos=0, kv_caches=None):
        h = self.tok_embeddings(tokens)
        
        for i, layer in enumerate(self.layers):
            cache = kv_caches[i] if kv_caches is not None else None
            h = layer(h, start_pos, cache)
            
        return self.output(self.norm(h))
