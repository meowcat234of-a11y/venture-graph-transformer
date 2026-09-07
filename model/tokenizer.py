class ByteTokenizer:
    vocab_size = 256

    def __init__(self, vocab_size=256):
        if vocab_size < self.vocab_size:
            raise ValueError("byte-level tokenization requires a vocabulary of at least 256 tokens")
        self.vocab_size = vocab_size

    def encode(self, text):
        return list(text.encode("utf-8"))

    def decode(self, tokens):
        token_bytes = bytes(tokens)
        return token_bytes.decode("utf-8", errors="replace")


SimpleTokenizer = ByteTokenizer
