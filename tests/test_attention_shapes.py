import torch
import unittest
from model.attention import MultiHeadAttention

class TestAttentionShapes(unittest.TestCase):
    def test_multihead_attention(self):
        dim = 256
        num_heads = 8
        seq_len = 32
        bsz = 2
        
        mha = MultiHeadAttention(dim, num_heads)
        x = torch.randn(bsz, seq_len, dim)
        
        out = mha(x)
        self.assertEqual(out.shape, (bsz, seq_len, dim))

if __name__ == "__main__":
    unittest.main()
