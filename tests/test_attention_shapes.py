import torch
import unittest
from model.attention import MultiHeadAttention
from model.transformer import Transformer

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

    def test_cached_decode_matches_full_decode(self):
        torch.manual_seed(0)
        model = Transformer(vocab_size=32, dim=16, layers=2, heads=4, hidden_dim=32).eval()
        tokens = torch.tensor([[1, 2, 3, 4, 5]])

        full_logits = model(tokens)
        caches = model.init_caches()
        model(tokens[:, :3], caches=caches)
        cached_logits = model(tokens[:, 3:], caches=caches)

        self.assertTrue(torch.allclose(cached_logits, full_logits[:, 3:], atol=1e-5))

if __name__ == "__main__":
    unittest.main()
