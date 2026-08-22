import torch
import torch.nn as nn
import math

class RoPE(nn.Module):
    def __init__(self, dim, max_seq_len=2048):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq_len).type_as(inv_freq)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, x, seq_len):
        return self.cos_cached[:, :, :seq_len, ...], self.sin_cached[:, :, :seq_len, ...]

def apply_rotary_emb(xq, xk, cos, sin):
    q_half1, q_half2 = xq.chunk(2, dim=-1)
    k_half1, k_half2 = xk.chunk(2, dim=-1)
    
    q_rot = torch.cat([-q_half2, q_half1], dim=-1)
    k_rot = torch.cat([-k_half2, k_half1], dim=-1)
    
    xq_out = (xq * cos) + (q_rot * sin)
    xk_out = (xk * cos) + (k_rot * sin)
    return xq_out, xk_out

class MultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)
        
        self.rope = RoPE(self.head_dim)

    def forward(self, x, start_pos=0, kv_cache=None):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        
        xq = xq.view(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        xk = xk.view(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        xv = xv.view(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        
        cos, sin = self.rope(xq, seqlen + start_pos)
        cos, sin = cos[:, :, start_pos:start_pos+seqlen], sin[:, :, start_pos:start_pos+seqlen]
        
        xq, xk = apply_rotary_emb(xq, xk, cos, sin)
        
        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            k_cache[:, :, start_pos:start_pos+seqlen] = xk
            v_cache[:, :, start_pos:start_pos+seqlen] = xv
            xk = k_cache[:, :, :start_pos+seqlen]
            xv = v_cache[:, :, :start_pos+seqlen]

        scores = torch.matmul(xq, xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = torch.tril(torch.ones(seqlen, seqlen)).to(x.device)
        scores = scores.masked_fill(mask == 0, float("-inf"))
        
        attn = torch.softmax(scores, dim=-1)
        output = torch.matmul(attn, xv)
        
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)
