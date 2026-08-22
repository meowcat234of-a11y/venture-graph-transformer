class SimpleTokenizer:
    def __init__(self, vocab_size=50000):
        self.vocab_size = vocab_size

    def encode(self, text):
        return [ord(c) for c in text]  # naive implementation

    def decode(self, tokens):
        return "".join([chr(t) for t in tokens])
