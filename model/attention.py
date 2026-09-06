import torch
import torch.nn as nn
import math

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
        c = self.cos[:, :, pos:pos+seq_len]
        s = self.sin[:, :, pos:pos+seq_len]
        
        def rotate_half(x):
            x1, x2 = x.chunk(2, dim=-1)
            return torch.cat((-x2, x1), dim=-1)
            
        return (q * c) + (rotate_half(q) * s), (k * c) + (rotate_half(k) * s)

class MultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)
        self.rope = RoPE(self.head_dim)

    def forward(self, x, pos=0, kv_cache=None):
        bsz, seq, _ = x.shape
        
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(bsz, seq, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq, self.num_heads, self.head_dim).transpose(1, 2)
        
        q, k = self.rope(q, k, pos)
        
        if kv_cache is not None:
            k_c, v_c = kv_cache
            k_c[:, :, pos:pos+seq] = k
            v_c[:, :, pos:pos+seq] = v
            k = k_c[:, :, :pos+seq]
            v = v_c[:, :, :pos+seq]

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        mask = torch.tril(torch.ones(seq, seq, device=x.device))
        scores.masked_fill_(mask == 0, float("-inf"))
        
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        
        out = out.transpose(1, 2).contiguous().view(bsz, seq, -1)
        return self.out(out)
