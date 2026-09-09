import math

import torch
import torch.nn as nn


class KVCache:
    def __init__(self):
        self.key = None
        self.value = None

    @property
    def length(self):
        return 0 if self.key is None else self.key.size(2)

    def update(self, key, value):
        if self.key is None:
            self.key, self.value = key, value
        else:
            if self.key.shape[:2] != key.shape[:2] or self.key.shape[-1] != key.shape[-1]:
                raise ValueError("cache shape is incompatible with incoming keys")
            self.key = torch.cat((self.key, key), dim=2)
            self.value = torch.cat((self.value, value), dim=2)
        return self.key, self.value


class RoPE(nn.Module):
    def __init__(self, dim, max_len=2048):
        super().__init__()
        freqs = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_len)
        freqs = torch.outer(t, freqs)
        emb = torch.cat((freqs, freqs), dim=-1)

        self.register_buffer("cos", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, q, k, pos):
        seq_len = q.shape[2]
        if pos < 0 or pos + seq_len > self.cos.size(2):
            raise ValueError("position range exceeds configured RoPE length")
        c = self.cos[:, :, pos : pos + seq_len]
        s = self.sin[:, :, pos : pos + seq_len]

        def rotate_half(x):
            x1, x2 = x.chunk(2, dim=-1)
            return torch.cat((-x2, x1), dim=-1)

        return (q * c) + (rotate_half(q) * s), (k * c) + (rotate_half(k) * s)


class MultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        if dim % num_heads:
            raise ValueError("dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        if self.head_dim % 2:
            raise ValueError("head dimension must be even for RoPE")

        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)
        self.rope = RoPE(self.head_dim)

    def forward(self, x, pos=0, kv_cache=None):
        bsz, seq, _ = x.shape

        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(bsz, seq, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq, self.num_heads, self.head_dim).transpose(1, 2)

        start = kv_cache.length if kv_cache is not None else pos
        q, k = self.rope(q, k, start)

        if kv_cache is not None:
            k, v = kv_cache.update(k, v)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        key_positions = torch.arange(k.size(2), device=x.device)
        if kv_cache is None:
            key_positions = key_positions + start
        query_positions = start + torch.arange(seq, device=x.device)
        causal = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
        scores.masked_fill_(~causal[None, None], float("-inf"))

        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)

        out = out.transpose(1, 2).contiguous().view(bsz, seq, -1)
        return self.out(out)
